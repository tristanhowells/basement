"""
AFL betting model — decision-tree family
========================================
Trains Random Forest and XGBoost classifiers to predict game outcomes,
then backtests a betting strategy where we only bet when our estimated
probability exceeds the market's implied probability (positive EV filter).

Walk-forward validation: train on all data before year Y, test on year Y.
Staking: flat 1-unit and fractional Kelly.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────
odds = pd.read_excel("data/afl/odds_data.xlsx")
odds["Date"] = pd.to_datetime(odds["Date"])
odds["year"] = odds["Date"].dt.year
odds = odds.sort_values("Date").reset_index(drop=True)

odds["home_win"]  = (odds["Home Score"] > odds["Away Score"]).astype(int)
odds["margin"]    = odds["Home Score"] - odds["Away Score"]
odds["is_finals"] = (odds["Play Off Game?"] == "Y").astype(int)

# Drop rows without closing odds (needed for EV filter)
odds = odds.dropna(subset=["Home Odds Close", "Away Odds Close"]).reset_index(drop=True)

# Market-implied probabilities
odds["overround"]      = 1 / odds["Home Odds Close"] + 1 / odds["Away Odds Close"]
odds["market_prob"]    = (1 / odds["Home Odds Close"]) / odds["overround"]   # normalised

# ─────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
VENUE_HWR = {}   # filled below in rolling loop; also pre-compute global stats
venue_global = odds.groupby("Venue")["home_win"].mean().to_dict()

# Rolling team state — computed row-by-row to prevent leakage
team_state = defaultdict(lambda: {
    "wins": 0, "games": 0, "streak": 0,
    "margins": [], "h2h": defaultdict(lambda: [0, 0]),
})

features, targets = [], []

for idx, row in odds.iterrows():
    h, a = row["Home Team"], row["Away Team"]
    sh, sa = team_state[h], team_state[a]

    # ── Market features ──────────────────────────────────────────────
    mkt_prob = row["market_prob"]
    overround = row["overround"]
    home_odds_close = row["Home Odds Close"]
    away_odds_close = row["Away Odds Close"]

    # Odds movement: positive = home steamed in (shortened)
    home_open  = row.get("Home Odds Open") if pd.notna(row.get("Home Odds Open")) else home_odds_close
    away_open  = row.get("Away Odds Open") if pd.notna(row.get("Away Odds Open")) else away_odds_close
    home_move  = float(home_open - home_odds_close)   # positive = odds shortened (backed)
    away_move  = float(away_open - away_odds_close)
    move_diff  = home_move - away_move                # relative steam

    # ── Form / streak ────────────────────────────────────────────────
    h_wr   = sh["wins"] / sh["games"] if sh["games"] > 0 else 0.5
    a_wr   = sa["wins"] / sa["games"] if sa["games"] > 0 else 0.5
    h_str  = sh["streak"]
    a_str  = sa["streak"]
    h_form = float(np.mean(sh["margins"][-5:])) if sh["margins"] else 0.0
    a_form = float(np.mean(sa["margins"][-5:])) if sa["margins"] else 0.0

    # ── H2H ─────────────────────────────────────────────────────────
    h2h = sh["h2h"][a]  # [home_wins, games]
    h2h_rate = h2h[0] / h2h[1] if h2h[1] > 0 else 0.5
    h2h_n    = h2h[1]

    # ── Venue ────────────────────────────────────────────────────────
    venue_hwr = venue_global.get(row["Venue"], 0.566)

    # ── Line market ──────────────────────────────────────────────────
    home_line = float(row["Home Line Close"]) if pd.notna(row.get("Home Line Close")) else 0.0
    total_line = float(row["Total Score Close"]) if pd.notna(row.get("Total Score Close")) else 167.0

    # ── Derived ──────────────────────────────────────────────────────
    # Model's "prior" before any form: venue_hwr adjusted by market
    venue_vs_market = venue_hwr - mkt_prob
    form_diff   = h_form - a_form
    streak_diff = h_str - a_str
    wr_diff     = h_wr - a_wr

    feat = {
        # Market
        "market_prob":     mkt_prob,
        "overround":       overround,
        "home_odds_close": home_odds_close,
        "log_odds_ratio":  np.log(home_odds_close / away_odds_close),

        # Odds movement
        "home_move":       home_move,
        "away_move":       away_move,
        "move_diff":       move_diff,
        "abs_home_move":   abs(home_move),

        # Form
        "h_win_rate":      h_wr,
        "a_win_rate":      a_wr,
        "wr_diff":         wr_diff,
        "h_streak":        h_str,
        "a_streak":        a_str,
        "streak_diff":     streak_diff,
        "h_form5":         h_form,
        "a_form5":         a_form,
        "form_diff":       form_diff,

        # H2H
        "h2h_rate":        h2h_rate,
        "h2h_n":           min(h2h_n, 10),

        # Context
        "venue_hwr":       venue_hwr,
        "venue_vs_market": venue_vs_market,
        "home_line":       home_line,
        "total_line":      total_line,
        "is_finals":       int(row["is_finals"]),
        "year":            int(row["year"]),
    }
    features.append(feat)
    targets.append(row["home_win"])

    # Update state AFTER recording features (no leakage)
    won = row["home_win"] == 1
    mar = row["margin"]
    for team, win, m in [(h, won, mar), (a, not won, -mar)]:
        s = team_state[team]
        s["wins"]  += int(win)
        s["games"] += 1
        s["streak"] = (max(0, s["streak"]) + 1) if win else (min(0, s["streak"]) - 1)
        s["margins"].append(m)
    # H2H update
    team_state[h]["h2h"][a][1] += 1
    team_state[a]["h2h"][h][1] += 1
    if won:
        team_state[h]["h2h"][a][0] += 1
    else:
        team_state[a]["h2h"][h][0] += 1

df = pd.DataFrame(features)
df["home_win"]     = targets
df["Date"]         = odds["Date"].values
df["home_odds"]    = odds["Home Odds Close"].values
df["away_odds"]    = odds["Away Odds Close"].values
df["market_prob"]  = odds["market_prob"].values
df["year"]         = odds["year"].values

FEATURE_COLS = [c for c in df.columns if c not in
                ("home_win", "Date", "home_odds", "away_odds", "year")]

print(f"Dataset: {len(df)} rows, {len(FEATURE_COLS)} features")
print(f"Years: {df['year'].min()}–{df['year'].max()}")
print(f"Baseline (always predict home): {df['home_win'].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────
# 3. WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────────
TRAIN_START  = 2009
TEST_YEARS   = list(range(2015, 2024))   # need 5+ years of training data first
MIN_EDGE     = 0.03    # minimum (model_prob - market_prob) to bet
KELLY_FRAC   = 0.25    # fractional Kelly multiplier

models = {
    "DecisionTree":  DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, random_state=42),
    "RandomForest":  RandomForestClassifier(n_estimators=300, max_depth=6,
                                             min_samples_leaf=30, random_state=42),
    "GradientBoost": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                 learning_rate=0.05, subsample=0.8,
                                                 min_samples_leaf=30, random_state=42),
    "XGBoost":       xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8,
                                        use_label_encoder=False, eval_metric="logloss",
                                        random_state=42, verbosity=0),
}

results = {name: [] for name in models}
all_bets = []  # for combined analysis

for test_year in TEST_YEARS:
    train_mask = df["year"] < test_year
    test_mask  = df["year"] == test_year

    X_train = df.loc[train_mask, FEATURE_COLS].values
    y_train = df.loc[train_mask, "home_win"].values
    X_test  = df.loc[test_mask,  FEATURE_COLS].values
    y_test  = df.loc[test_mask,  "home_win"].values

    test_rows = df[test_mask].copy()

    for name, base_model in models.items():
        # Calibrate probabilities using isotonic regression on training data
        model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        # ── EV filter: only bet when model edge > threshold ──────────
        home_edge = probs - test_rows["market_prob"].values
        away_edge = (1 - probs) - (1 - test_rows["market_prob"].values)

        bet_home  = home_edge > MIN_EDGE
        bet_away  = away_edge > MIN_EDGE
        any_bet   = bet_home | bet_away

        # ── Staking ──────────────────────────────────────────────────
        flat_pl, kelly_pl = 0.0, 0.0
        flat_bets, kelly_bets = 0, 0

        for i, (_, row) in enumerate(test_rows.iterrows()):
            ho, ao = row["home_odds"], row["away_odds"]
            actual = row["home_win"]
            mp = probs[i]

            if bet_home[i]:
                # flat 1 unit
                flat_pl += (ho - 1) if actual == 1 else -1
                flat_bets += 1
                # Kelly: f = (p*(b+1) - 1) / b  where b = odds-1
                b = ho - 1
                f = max(0, (mp * (b + 1) - 1) / b) * KELLY_FRAC
                kelly_pl += f * ((ho - 1) if actual == 1 else -1)
                kelly_bets += 1

            elif bet_away[i]:
                flat_pl += (ao - 1) if actual == 0 else -1
                flat_bets += 1
                b = ao - 1
                p_away = 1 - mp
                f = max(0, (p_away * (b + 1) - 1) / b) * KELLY_FRAC
                kelly_pl += f * ((ao - 1) if actual == 0 else -1)
                kelly_bets += 1

        n_games = len(y_test)
        roi_flat  = flat_pl  / flat_bets  if flat_bets  > 0 else 0.0
        roi_kelly = kelly_pl / kelly_bets if kelly_bets > 0 else 0.0

        results[name].append({
            "year":        test_year,
            "n_games":     n_games,
            "n_bets":      flat_bets,
            "bet_pct":     flat_bets / n_games,
            "flat_pl":     flat_pl,
            "roi_flat":    roi_flat,
            "kelly_pl":    kelly_pl,
            "roi_kelly":   roi_kelly,
            "brier":       brier_score_loss(y_test, probs),
            "auc":         roc_auc_score(y_test, probs),
        })

        if name == "XGBoost":
            # Store bets for deeper analysis
            for i, (_, row) in enumerate(test_rows.iterrows()):
                ho, ao = row["home_odds"], row["away_odds"]
                actual = row["home_win"]
                mp = probs[i]
                if bet_home[i]:
                    all_bets.append({"year": test_year, "side": "home",
                                     "odds": ho, "model_prob": mp,
                                     "market_prob": row["market_prob"],
                                     "edge": mp - row["market_prob"],
                                     "won": actual == 1})
                elif bet_away[i]:
                    all_bets.append({"year": test_year, "side": "away",
                                     "odds": ao, "model_prob": 1 - mp,
                                     "market_prob": 1 - row["market_prob"],
                                     "edge": (1 - mp) - (1 - row["market_prob"]),
                                     "won": actual == 0})

    print(f"  {test_year}: trained on {train_mask.sum()} games, tested on {test_mask.sum()}")

# ─────────────────────────────────────────────────────────────────────
# 4. RESULTS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("WALK-FORWARD BACKTEST RESULTS")
print(f"Min edge threshold: {MIN_EDGE:.0%} | Kelly fraction: {KELLY_FRAC:.0%}")
print("="*70)

for name, rows in results.items():
    rdf = pd.DataFrame(rows)
    total_bets  = rdf["n_bets"].sum()
    total_flat  = rdf["flat_pl"].sum()
    total_kelly = rdf["kelly_pl"].sum()
    avg_roi_flat  = total_flat  / total_bets if total_bets > 0 else 0
    avg_roi_kelly = total_kelly / total_bets if total_bets > 0 else 0
    pos_years = (rdf["flat_pl"] > 0).sum()

    print(f"\n── {name} ──")
    print(f"  Total bets     : {total_bets} / {rdf['n_games'].sum()} games "
          f"({total_bets/rdf['n_games'].sum():.1%})")
    print(f"  Flat P&L       : {total_flat:+.1f} units  |  ROI: {avg_roi_flat:+.3f} ({avg_roi_flat*100:+.1f}%)")
    print(f"  Kelly P&L      : {total_kelly:+.2f} units  |  ROI: {avg_roi_kelly:+.3f} ({avg_roi_kelly*100:+.1f}%)")
    print(f"  Profitable years: {pos_years}/{len(rdf)}")
    print(f"  Mean Brier     : {rdf['brier'].mean():.5f}  |  Mean AUC: {rdf['auc'].mean():.4f}")
    print(f"\n  Year-by-year (flat):")
    for _, r in rdf.iterrows():
        bar = "▓" * int(abs(r["flat_pl"])) if abs(r["flat_pl"]) < 40 else ("▓"*39 + "+")
        sign = "+" if r["flat_pl"] >= 0 else ""
        print(f"    {int(r['year'])}: {r['n_bets']:>3} bets  "
              f"P&L {sign}{r['flat_pl']:>6.1f}  ROI {r['roi_flat']:>+.3f}  {bar}")

# ─────────────────────────────────────────────────────────────────────
# 5. XGB BET QUALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────
if all_bets:
    bdf = pd.DataFrame(all_bets)
    print("\n" + "="*70)
    print("XGBoost BET QUALITY BREAKDOWN")
    print("="*70)
    print(f"Total bets: {len(bdf)} | Win rate: {bdf['won'].mean():.3f}")
    print(f"Avg odds:   {bdf['odds'].mean():.3f}")
    print(f"Avg edge:   {bdf['edge'].mean():.4f}")

    # By edge bucket
    bdf["edge_bucket"] = pd.cut(bdf["edge"], bins=[0.03, 0.06, 0.09, 0.12, 0.20, 1.0],
                                 labels=["3-6%","6-9%","9-12%","12-20%",">20%"])
    print("\nBy edge bucket:")
    print(bdf.groupby("edge_bucket", observed=True).agg(
        n=("won","count"),
        win_rate=("won","mean"),
        avg_odds=("odds","mean"),
        roi=("won", lambda x: ((bdf.loc[x.index,"odds"] - 1) * x - (1 - x)).mean())
    ).round(3).to_string())

    # By side
    print("\nBy side:")
    print(bdf.groupby("side").agg(
        n=("won","count"),
        win_rate=("won","mean"),
        avg_odds=("odds","mean"),
    ).round(3).to_string())

# ─────────────────────────────────────────────────────────────────────
# 6. DECISION TREE — INTERPRETABLE RULES
# ─────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("DECISION TREE — TOP RULES (full training set, depth=4)")
print("="*70)
X_all = df[FEATURE_COLS].values
y_all = df["home_win"].values
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
dt.fit(X_all, y_all)
print(export_text(dt, feature_names=FEATURE_COLS, max_depth=4))

# Feature importances from XGBoost (full fit)
print("\n" + "="*70)
print("XGBOOST FEATURE IMPORTANCES (full training set)")
print("="*70)
xgb_full = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               use_label_encoder=False, eval_metric="logloss",
                               random_state=42, verbosity=0)
xgb_full.fit(X_all, y_all)
imp = pd.Series(xgb_full.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(imp.head(15).round(4).to_string())

print("\nDone.")
