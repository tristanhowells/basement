"""
build_training_dataset.py
=========================
Reads raw JSON race files produced by scrape_racenet.py and builds a
flattened, feature-engineered dataset ready for DNN or RL training.

Output:
    dataset/races.jsonl          — one JSON object per runner per race
    dataset/dataset.csv          — same data as CSV (optional)

Usage:
    python build_training_dataset.py [--data-dir data] [--out-dir dataset]

Feature engineering done here:
  - Numeric finish position (1 = winner)
  - Win flag (binary label)
  - Top-3 flag (binary label for place)
  - Recent form features: avg finish pos last 3/5/10 runs
  - Recent form at same distance ± 100m
  - Recent form on same track condition
  - Days since last run
  - Win rate in last 10 starts
  - Average win odds last 5 runs
"""

import argparse
import csv
import json
import re
import statistics
from datetime import date, datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_float(s) -> Optional[float]:
    if s is None:
        return None
    try:
        return float(re.sub(r"[^0-9.\-]", "", str(s)))
    except (ValueError, TypeError):
        return None


def parse_int(s) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(re.sub(r"[^0-9\-]", "", str(s)))
    except (ValueError, TypeError):
        return None


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def safe_mean(vals: list) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return statistics.mean(clean) if clean else None


def finish_pos_numeric(pos: Optional[str]) -> Optional[int]:
    """Convert finish position string to int. Handles 'WNR','1st','DNF', etc."""
    if pos is None:
        return None
    s = str(pos).strip().upper()
    if s in ("DNF", "SCR", "DQ", "PU", "UR", "NS", "NP", "DISQ", "W/D"):
        return 999  # effectively last
    m = re.match(r"^(\d+)", s)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Form feature extraction
# ---------------------------------------------------------------------------

def extract_form_features(form_history: list, race_distance: Optional[int],
                           track_condition: Optional[str]) -> dict:
    """Compute rolling statistics from a horse's form history."""

    positions = [finish_pos_numeric(r.get("finish_position")) for r in form_history]
    positions = [p for p in positions if p is not None and p < 999]

    odds_list = [parse_float(r.get("win_odds")) for r in form_history]
    odds_list = [o for o in odds_list if o is not None]

    dates = [parse_date(r.get("date")) for r in form_history]

    features = {}

    # --- Recent avg finish position ---
    for n in (3, 5, 10):
        last_n = positions[:n]
        features[f"avg_finish_pos_last{n}"] = safe_mean(last_n)

    # --- Win rate last 10 ---
    wins = sum(1 for p in positions[:10] if p == 1)
    features["win_rate_last10"] = wins / min(len(positions), 10) if positions else None

    # --- Top-3 rate last 10 ---
    top3 = sum(1 for p in positions[:10] if p <= 3)
    features["top3_rate_last10"] = top3 / min(len(positions), 10) if positions else None

    # --- Days since last run ---
    today = date.today()
    valid_dates = [d for d in dates if d is not None]
    if valid_dates:
        last_run = valid_dates[0]  # form history is typically newest-first
        features["days_since_last_run"] = (today - last_run).days
    else:
        features["days_since_last_run"] = None

    # --- Avg odds last 5 ---
    features["avg_win_odds_last5"] = safe_mean(odds_list[:5])

    # --- Form at similar distance ---
    if race_distance is not None:
        dist_positions = [
            finish_pos_numeric(r.get("finish_position"))
            for r in form_history
            if (parse_int(r.get("distance_m")) or 0)
            and abs((parse_int(r.get("distance_m")) or 0) - race_distance) <= 100
        ]
        dist_positions = [p for p in dist_positions if p is not None and p < 999]
        features["avg_finish_pos_similar_dist"] = safe_mean(dist_positions[:5])
        features["runs_at_similar_dist"] = len(dist_positions)
    else:
        features["avg_finish_pos_similar_dist"] = None
        features["runs_at_similar_dist"] = None

    # --- Form on same track condition ---
    if track_condition:
        cond_key = track_condition.lower().split()[0]  # e.g. "good", "soft", "heavy"
        cond_positions = [
            finish_pos_numeric(r.get("finish_position"))
            for r in form_history
            if (r.get("track_condition") or "").lower().startswith(cond_key)
        ]
        cond_positions = [p for p in cond_positions if p is not None and p < 999]
        features["avg_finish_pos_same_condition"] = safe_mean(cond_positions[:5])
        features["runs_on_same_condition"] = len(cond_positions)
    else:
        features["avg_finish_pos_same_condition"] = None
        features["runs_on_same_condition"] = None

    # --- Total career starts ---
    features["career_starts"] = len(form_history)
    features["career_wins"] = sum(
        1 for r in form_history
        if finish_pos_numeric(r.get("finish_position")) == 1
    )

    return features


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------

def flatten_race_file(path: Path) -> list[dict]:
    """Load one race JSON and return a list of flat dicts (one per runner)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    runners = data.get("runners", [])

    race_date = meta.get("date")
    venue = meta.get("venue")
    race_number = meta.get("race_number")
    distance_m = parse_int(meta.get("distance_m") or "")
    track_condition = meta.get("track_condition")
    prize_money = parse_int(meta.get("prize_money") or "")
    url = meta.get("url", "")

    rows = []
    for runner in runners:
        finish_pos = finish_pos_numeric(runner.get("finish_position"))
        win_odds = parse_float(runner.get("win_odds"))
        barrier = parse_int(runner.get("barrier") or "")
        weight_str = runner.get("weight") or ""
        weight_kg = parse_float(re.sub(r"[^0-9.]", "", weight_str) or "")

        form_feats = extract_form_features(
            runner.get("form_history", []),
            distance_m,
            track_condition,
        )

        row = {
            # Identifiers
            "race_date": race_date,
            "venue": venue,
            "race_number": race_number,
            "url": url,
            # Horse
            "horse_name": runner.get("horse_name"),
            "jockey": runner.get("jockey"),
            "trainer": runner.get("trainer"),
            # Race context
            "distance_m": distance_m,
            "track_condition": track_condition,
            "prize_money": prize_money,
            "barrier": barrier,
            "weight_kg": weight_kg,
            "win_odds": win_odds,
            # Labels
            "finish_position": finish_pos,
            "is_winner": 1 if finish_pos == 1 else 0,
            "is_top3": 1 if (finish_pos is not None and finish_pos <= 3) else 0,
            "margin": runner.get("margin"),
            # Form features (rolling stats)
            **form_feats,
        }
        rows.append(row)

    return rows


def build_dataset(data_dir: Path, out_dir: Path, write_csv: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "races.jsonl"
    csv_path = out_dir / "dataset.csv"

    race_files = sorted(data_dir.rglob("race_*.json"))
    print(f"Found {len(race_files)} race files in {data_dir}")

    all_rows: list[dict] = []
    for path in race_files:
        try:
            rows = flatten_race_file(path)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"  ERROR processing {path}: {exc}")

    print(f"Total runner records: {len(all_rows)}")

    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Written: {jsonl_path}")

    # Write CSV
    if write_csv and all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Written: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


if __name__ == "__main__":
    if _is_jupyter():
        # Running inside a Jupyter notebook — set defaults directly here
        build_dataset(
            Path("data"),
            Path("dataset"),
            write_csv=True,
        )
    else:
        parser = argparse.ArgumentParser(
            description="Build flat training dataset from scraped race JSON files."
        )
        parser.add_argument("--data-dir", type=str, default="data",
                            help="Directory containing scraped race JSON files")
        parser.add_argument("--out-dir", type=str, default="dataset",
                            help="Output directory for training files")
        parser.add_argument("--no-csv", action="store_true",
                            help="Skip writing CSV file")
        args = parser.parse_args()

        build_dataset(
            Path(args.data_dir),
            Path(args.out_dir),
            write_csv=not args.no_csv,
        )
