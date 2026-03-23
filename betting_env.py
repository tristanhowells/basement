"""
Two-team win market betting environment for reinforcement learning.

Simulates sports betting markets (AFL, NRL, Tennis, Basketball, etc.) with:
  - Synthetic decimal odds at configurable overround (~105%)
  - Betfair-style commission on winning bets
  - Continuous action space: team selection + stake proportion
  - Episode ends on bust (<$1) or target (>$10,000)

Observation space (5 features):
  [implied_prob_a, implied_prob_b, log_balance_norm, step_fraction, recent_win_rate]

Action space (2 continuous actions, each in [0, 1]):
  action[0]: team selection  — <0.5 = Team A, >=0.5 = Team B
  action[1]: stake proportion — fraction of current balance to wager
              stake < min_stake is treated as a "no-bet" (skip market)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class BettingEnv(gym.Env):
    """
    Two-team win market RL environment.

    Parameters
    ----------
    starting_balance : float
        Balance at episode start. Default $100.
    max_steps : int
        Maximum bets per episode before truncation. Default 500.
    overround : float
        Market overround (e.g. 1.05 = 105%). Creates house edge.
    commission_rate : float
        Betfair-style commission on winning bet *profit* only. Default 5%.
    min_stake : float
        Minimum bet stake. Actions producing a lower stake are treated
        as "no-bet" (no balance change, small neutral reward). Default $1.
    bust_threshold : float
        Balance below this ends episode as a bust. Default $1.
    win_threshold : float
        Balance above this ends episode as a win. Default $10,000.
    bust_penalty : float
        Extra reward term applied on bust termination. Default -100.
    win_bonus : float
        Extra reward term applied on target reached. Default +100.
    survival_shaping_scale : float
        Scale for the per-step survival shaping penalty. At each bet step a
        quadratic proximity-to-bust term is subtracted from the reward:
            penalty = -survival_shaping_scale * danger²
        where danger = 1 − (balance − bust_threshold) / (starting_balance − bust_threshold),
        clipped to [0, 1].  Near starting balance the penalty is ~0; near bust
        it approaches -survival_shaping_scale.  Default 0.5.
    min_true_prob : float
        Minimum true win probability for Team A. Controls how extreme
        favorites can be. Default 0.30 (suits most two-team sports).
        Use 0.25 for tennis (bigger favorites allowed).
    max_true_prob : float
        Maximum true win probability for Team A. Default 0.70.
    recent_window : int
        Number of past bets included in the rolling win-rate observation.
        Default 20.
    reward_scale : float
        Scales step reward (log return). Default 10.0 keeps rewards in
        a reasonable range while preserving sign and magnitude ordering.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        starting_balance: float = 100.0,
        max_steps: int = 500,
        overround: float = 1.05,
        commission_rate: float = 0.05,
        min_stake: float = 1.0,
        bust_threshold: float = 1.0,
        win_threshold: float = 10_000.0,
        bust_penalty: float = -100.0,
        win_bonus: float = 100.0,
        survival_shaping_scale: float = 0.5,
        min_true_prob: float = 0.30,
        max_true_prob: float = 0.70,
        recent_window: int = 20,
        reward_scale: float = 10.0,
        render_mode=None,
    ):
        super().__init__()

        assert 0.0 < min_true_prob < max_true_prob < 1.0, (
            "Need 0 < min_true_prob < max_true_prob < 1"
        )
        assert commission_rate >= 0.0, "Commission rate must be non-negative"
        assert overround >= 1.0, "Overround must be >= 1.0"

        self.starting_balance = starting_balance
        self.max_steps = max_steps
        self.overround = overround
        self.commission_rate = commission_rate
        self.min_stake = min_stake
        self.bust_threshold = bust_threshold
        self.win_threshold = win_threshold
        self.bust_penalty = bust_penalty
        self.win_bonus = win_bonus
        self.survival_shaping_scale = survival_shaping_scale
        self.min_true_prob = min_true_prob
        self.max_true_prob = max_true_prob
        self.recent_window = recent_window
        self.reward_scale = reward_scale
        self.render_mode = render_mode

        # Precompute log bounds for observation normalisation
        self._log_bust = np.log(bust_threshold / starting_balance)       # ~-4.6 at defaults
        self._log_win = np.log(win_threshold / starting_balance)          # ~+4.6 at defaults

        # ── Observation space ───────────────────────────────────────────
        # [implied_prob_a, implied_prob_b, log_balance_norm,
        #  step_fraction, recent_win_rate]
        low = np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0,  1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Action space ────────────────────────────────────────────────
        # [team_selection, stake_proportion] both in [0, 1]
        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state (initialised in reset)
        self.balance: float = starting_balance
        self.current_step: int = 0
        self.odds_a: float = 2.0
        self.odds_b: float = 2.0
        self.true_prob_a: float = 0.5
        self._recent_outcomes: deque = deque(maxlen=recent_window)

    # ────────────────────────────────────────────────────────────────────
    # Market helpers
    # ────────────────────────────────────────────────────────────────────

    def _generate_market(self) -> None:
        """Sample a new market: true probabilities → decimal odds with overround."""
        self.true_prob_a = self.np_random.uniform(
            self.min_true_prob, self.max_true_prob
        )
        true_prob_b = 1.0 - self.true_prob_a

        # Proportional overround: each implied prob is scaled by overround factor,
        # so sum of implied probs = overround (e.g. 1.05).
        # odds = 1 / (true_prob * overround)
        # Example: true_prob_a=0.61, overround=1.05
        #   → odds_a = 1/(0.61*1.05) ≈ 1.56, odds_b = 1/(0.39*1.05) ≈ 2.44
        self.odds_a = 1.0 / (self.true_prob_a * self.overround)
        self.odds_b = 1.0 / (true_prob_b * self.overround)

    def _resolve_market(self) -> bool:
        """Resolve the market. Returns True if Team A wins."""
        return self.np_random.random() < self.true_prob_a

    # ────────────────────────────────────────────────────────────────────
    # Gymnasium interface
    # ────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.starting_balance
        self.current_step = 0
        self._recent_outcomes.clear()
        self._generate_market()
        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        team_action = float(np.clip(action[0], 0.0, 1.0))
        stake_action = float(np.clip(action[1], 0.0, 1.0))

        # ── Generate market for this step ────────────────────────────
        self._generate_market()

        # ── Resolve the bet ──────────────────────────────────────────
        old_balance = self.balance
        intended_stake = stake_action * self.balance
        no_bet = intended_stake < self.min_stake

        if no_bet:
            # Agent chose not to bet (or can't afford min stake)
            bet_won = False
            stake = 0.0
            selected_odds = 0.0
            bet_on_a = None
            # Small neutral reward signal — agent is not penalised for sitting out
            reward = 0.0
        else:
            # Clip stake to available balance (can't over-bet)
            stake = min(intended_stake, self.balance)
            bet_on_a = team_action < 0.5
            selected_odds = self.odds_a if bet_on_a else self.odds_b

            team_a_wins = self._resolve_market()
            bet_won = (bet_on_a == team_a_wins)

            if bet_won:
                gross_profit = stake * (selected_odds - 1.0)
                commission = gross_profit * self.commission_rate
                net_profit = gross_profit - commission
                self.balance += net_profit
            else:
                self.balance -= stake
                self.balance = max(self.balance, 0.0)  # floor at zero

            # Step reward: log return, scaled for training stability.
            # log return handles the full $1–$10,000 balance range naturally:
            #   $100 → $125 ≈ +0.22   (×reward_scale → +2.2)
            #   $100 → $75  ≈ -0.29   (×reward_scale → -2.9)
            #   $5000 → $5250 ≈ +0.049 (×reward_scale → +0.49)
            log_return = np.log(
                (self.balance + 1e-8) / (old_balance + 1e-8)
            )
            reward = float(log_return * self.reward_scale)

            # Survival shaping: quadratic penalty that grows as balance
            # approaches bust_threshold.  Provides a continuous danger signal
            # so the agent doesn't wait for the terminal bust penalty to learn
            # conservative play near ruin.
            #   danger = 0  at starting_balance (or above)
            #   danger = 1  at bust_threshold
            danger = 1.0 - (self.balance - self.bust_threshold) / (
                self.starting_balance - self.bust_threshold
            )
            danger = float(np.clip(danger, 0.0, 1.0))
            reward -= self.survival_shaping_scale * danger ** 2

        self._recent_outcomes.append(1 if bet_won else 0)
        self.current_step += 1

        # ── Check termination ────────────────────────────────────────
        terminated = False
        terminal_reason = None

        if self.balance < self.bust_threshold:
            reward += self.bust_penalty
            terminated = True
            terminal_reason = "bust"
        elif self.balance >= self.win_threshold:
            reward += self.win_bonus
            terminated = True
            terminal_reason = "target_reached"

        truncated = (not terminated) and (self.current_step >= self.max_steps)
        if truncated:
            terminal_reason = "max_steps"

        info = {
            "balance": self.balance,
            "terminal_reason": terminal_reason,
            "bet_won": bet_won,
            "stake": stake,
            "selected_odds": selected_odds,
            "odds_a": self.odds_a,
            "odds_b": self.odds_b,
            "true_prob_a": self.true_prob_a,
            "no_bet": no_bet,
        }
        if terminated or truncated:
            info["final_balance"] = self.balance

        if self.render_mode == "human":
            self._render_human(action, stake, bet_won, selected_odds, terminal_reason)

        return self._get_obs(), float(reward), terminated, truncated, info

    # ────────────────────────────────────────────────────────────────────
    # Observation
    # ────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        implied_prob_a = 1.0 / self.odds_a        # market implied prob (after overround)
        implied_prob_b = 1.0 / self.odds_b

        # Log balance normalised to [-1, 1] across the bust→win range
        log_bal = np.log((self.balance + 1e-8) / self.starting_balance)
        log_bal_norm = float(
            np.clip(log_bal / max(abs(self._log_bust), abs(self._log_win)), -1.0, 1.0)
        )

        step_frac = self.current_step / self.max_steps

        recent_win_rate = (
            sum(self._recent_outcomes) / len(self._recent_outcomes)
            if self._recent_outcomes
            else 0.5  # neutral prior at episode start
        )

        return np.array(
            [implied_prob_a, implied_prob_b, log_bal_norm, step_frac, recent_win_rate],
            dtype=np.float32,
        )

    # ────────────────────────────────────────────────────────────────────
    # Rendering
    # ────────────────────────────────────────────────────────────────────

    def _render_human(self, action, stake, bet_won, selected_odds, terminal_reason):
        team = "A" if action[0] < 0.5 else "B"
        result = "WIN " if bet_won else "LOSE"
        print(
            f"Step {self.current_step:>4d} | "
            f"Odds A:{self.odds_a:.2f} B:{self.odds_b:.2f} | "
            f"Bet {team} @ {selected_odds:.2f} | "
            f"Stake ${stake:>8.2f} | {result} | "
            f"Balance ${self.balance:>10.2f}"
            + (f" [{terminal_reason}]" if terminal_reason else "")
        )

    def render(self):
        if self.render_mode == "human":
            pass  # rendering happens inside step()


# ────────────────────────────────────────────────────────────────────────────
# Sport-specific presets
# ────────────────────────────────────────────────────────────────────────────

SPORT_PRESETS = {
    # Competitive team sports — tight probability range
    "afl":        {"min_true_prob": 0.30, "max_true_prob": 0.70, "overround": 1.05},
    "nrl":        {"min_true_prob": 0.30, "max_true_prob": 0.70, "overround": 1.05},
    "basketball": {"min_true_prob": 0.30, "max_true_prob": 0.70, "overround": 1.05},
    # Tennis — bigger range; big favorites common; slightly tighter margin
    "tennis":     {"min_true_prob": 0.20, "max_true_prob": 0.80, "overround": 1.04},
    # Mixed / generic
    "generic":    {"min_true_prob": 0.30, "max_true_prob": 0.70, "overround": 1.05},
}


def make_env(sport: str = "generic", **kwargs) -> BettingEnv:
    """Convenience factory. `sport` key overrides probability range and overround."""
    preset = SPORT_PRESETS.get(sport, SPORT_PRESETS["generic"])
    preset.update(kwargs)  # caller kwargs take precedence
    return BettingEnv(**preset)
