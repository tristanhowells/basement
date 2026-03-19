"""
Evaluation and visualisation for the trained betting RL agent.

Loads a saved model and runs N evaluation episodes, then produces:
  1. Balance trajectory plot (sample episodes)
  2. Final balance distribution (histogram)
  3. Episode outcome pie chart (positive / negative / timeout)
  4. Action heatmap (stake proportion vs team selection frequency)
  5. Printed statistics summary

Usage
-----
    python evaluate.py                                  # loads models/best_model.zip
    python evaluate.py --model models/final_model.zip  # specific model
    python evaluate.py --episodes 500 --render          # render to terminal
    python evaluate.py --sport tennis --episodes 200
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless — saves to files rather than showing windows
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO, SAC

from betting_env import BettingEnv, SPORT_PRESETS


# ────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ────────────────────────────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    sport: str = "generic",
    n_episodes: int = 500,
    commission_rate: float = 0.05,
    overround: float = 1.05,
    starting_balance: float = 100.0,
    max_steps: int = 500,
    algo: str = "ppo",
    render: bool = False,
    plot_dir: str = "plots",
    seed: int = 0,
) -> dict:

    os.makedirs(plot_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────
    AlgoCls = PPO if algo.lower() == "ppo" else SAC
    model = AlgoCls.load(model_path)
    print(f"Loaded model from: {model_path}")

    # ── Build env ────────────────────────────────────────────────────
    preset = SPORT_PRESETS.get(sport, SPORT_PRESETS["generic"]).copy()
    preset.update(
        {
            "commission_rate": commission_rate,
            "overround": overround,
            "starting_balance": starting_balance,
            "max_steps": max_steps,
            "render_mode": "human" if render else None,
        }
    )
    env = BettingEnv(**preset)

    # ── Data collectors ──────────────────────────────────────────────
    outcomes = {"target_reached": 0, "bust": 0, "max_steps": 0}
    final_balances: list[float] = []
    episode_lengths: list[int] = []

    # Store detailed trajectories for a sample of episodes to plot
    n_trajectories = min(50, n_episodes)
    sample_trajectories: list[list[float]] = []

    # Action histograms
    team_actions: list[float] = []
    stake_actions: list[float] = []

    rng = np.random.default_rng(seed)

    print(f"\nEvaluating {n_episodes} episodes (sport={sport}, algo={algo.upper()})...")
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        balance_history = [starting_balance]
        steps = 0
        terminal_reason = "max_steps"

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            team_actions.append(float(action[0]))
            stake_actions.append(float(action[1]))
            balance_history.append(info["balance"])
            steps += 1

            if done:
                terminal_reason = info.get("terminal_reason", "max_steps")
                final_bal = info.get("final_balance", info["balance"])

        outcomes[terminal_reason] = outcomes.get(terminal_reason, 0) + 1
        final_balances.append(final_bal)
        episode_lengths.append(steps)

        if ep < n_trajectories:
            sample_trajectories.append(balance_history)

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}/{n_episodes} complete")

    env.close()

    # ── Statistics ───────────────────────────────────────────────────
    n = n_episodes
    pos = outcomes.get("target_reached", 0)
    neg = outcomes.get("bust", 0)
    tmt = outcomes.get("max_steps", 0)

    stats = {
        "n_episodes": n,
        "positive_episodes": pos,
        "negative_episodes": neg,
        "timeout_episodes": tmt,
        "positive_rate": pos / n,
        "negative_rate": neg / n,
        "mean_final_balance": np.mean(final_balances),
        "median_final_balance": np.median(final_balances),
        "std_final_balance": np.std(final_balances),
        "mean_episode_length": np.mean(episode_lengths),
    }

    print(f"\n{'='*60}")
    print(f"  Evaluation Results  ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Positive (target reached) : {pos:>5d}  ({pos/n*100:.2f}%)")
    print(f"  Negative (bust)           : {neg:>5d}  ({neg/n*100:.2f}%)")
    print(f"  Timeout (max_steps)       : {tmt:>5d}  ({tmt/n*100:.2f}%)")
    print(f"  Mean final balance        : ${stats['mean_final_balance']:.2f}")
    print(f"  Median final balance      : ${stats['median_final_balance']:.2f}")
    print(f"  Std final balance         : ${stats['std_final_balance']:.2f}")
    print(f"  Mean episode length       : {stats['mean_episode_length']:.1f} bets")
    print(f"{'='*60}\n")

    # ── Plots ────────────────────────────────────────────────────────
    _plot_trajectories(sample_trajectories, starting_balance,
                       preset.get("win_threshold", 10_000),
                       preset.get("bust_threshold", 1.0),
                       plot_dir)

    _plot_balance_distribution(final_balances, plot_dir)

    _plot_outcome_pie(outcomes, plot_dir)

    _plot_action_distributions(team_actions, stake_actions, plot_dir)

    print(f"Plots saved to: {plot_dir}/")
    return stats


# ────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────────────────

def _plot_trajectories(
    trajectories: list[list[float]],
    starting_balance: float,
    win_threshold: float,
    bust_threshold: float,
    plot_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for traj in trajectories:
        ax.plot(traj, alpha=0.3, linewidth=0.8, color="steelblue")

    ax.axhline(win_threshold, color="green", linewidth=1.5,
               linestyle="--", label=f"Win target (${win_threshold:,.0f})")
    ax.axhline(bust_threshold, color="red", linewidth=1.5,
               linestyle="--", label=f"Bust threshold (${bust_threshold:.0f})")
    ax.axhline(starting_balance, color="grey", linewidth=1.0,
               linestyle=":", label=f"Start (${starting_balance:.0f})")

    ax.set_yscale("log")
    ax.set_xlabel("Bet number (step)")
    ax.set_ylabel("Balance ($, log scale)")
    ax.set_title(f"Balance Trajectories — {len(trajectories)} sample episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(plot_dir, "trajectories.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_balance_distribution(final_balances: list[float], plot_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale histogram
    ax = axes[0]
    ax.hist(final_balances, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Final Balance ($)")
    ax.set_ylabel("Episode count")
    ax.set_title("Final Balance Distribution (linear)")
    ax.grid(True, alpha=0.3)

    # Log scale histogram — better shows spread across orders of magnitude
    ax2 = axes[1]
    log_balances = np.log10(np.maximum(final_balances, 0.01))
    ax2.hist(log_balances, bins=50, color="darkorange", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("log₁₀(Final Balance)")
    ax2.set_ylabel("Episode count")
    ax2.set_title("Final Balance Distribution (log scale)")
    # Add reference ticks
    for exp, label in [(0, "$1"), (2, "$100"), (3, "$1k"), (4, "$10k")]:
        ax2.axvline(exp, color="grey", linewidth=0.8, linestyle="--")
        ax2.text(exp + 0.05, ax2.get_ylim()[1] * 0.9, label,
                 fontsize=8, color="grey")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "balance_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_outcome_pie(outcomes: dict, plot_dir: str) -> None:
    labels = []
    sizes = []
    colours = []
    colour_map = {
        "target_reached": "seagreen",
        "bust": "tomato",
        "max_steps": "steelblue",
    }
    for key, colour in colour_map.items():
        v = outcomes.get(key, 0)
        if v > 0:
            labels.append(f"{key.replace('_', ' ').title()} ({v})")
            sizes.append(v)
            colours.append(colour)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, colors=colours, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 12})
    ax.set_title("Episode Outcome Distribution", fontsize=14)

    path = os.path.join(plot_dir, "outcomes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_action_distributions(
    team_actions: list[float],
    stake_actions: list[float],
    plot_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Team selection distribution
    ax = axes[0]
    ax.hist(team_actions, bins=40, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.axvline(0.5, color="red", linewidth=1.5, linestyle="--", label="Decision boundary")
    n_a = sum(1 for t in team_actions if t < 0.5)
    n_b = len(team_actions) - n_a
    ax.set_xlabel("Team selection action (< 0.5 = Team A)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Team Selection Distribution\nTeam A: {n_a} ({n_a/len(team_actions)*100:.1f}%)  "
                 f"Team B: {n_b} ({n_b/len(team_actions)*100:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stake proportion distribution
    ax2 = axes[1]
    ax2.hist(stake_actions, bins=40, color="coral", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Stake proportion (fraction of current balance)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Stake Proportion Distribution\n"
                  f"Mean: {np.mean(stake_actions):.3f}   "
                  f"Median: {np.median(stake_actions):.3f}")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "action_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ────────────────────────────────────────────────────────────────────────────
# Benchmark: random agent
# ────────────────────────────────────────────────────────────────────────────

def benchmark_random(
    sport: str = "generic",
    n_episodes: int = 500,
    commission_rate: float = 0.05,
    overround: float = 1.05,
    starting_balance: float = 100.0,
    max_steps: int = 500,
    seed: int = 0,
) -> dict:
    """
    Run a random-action agent for comparison baseline.
    Useful to confirm the trained agent beats chance / the market edge.
    """
    preset = SPORT_PRESETS.get(sport, SPORT_PRESETS["generic"]).copy()
    preset.update(
        {"commission_rate": commission_rate, "overround": overround,
         "starting_balance": starting_balance, "max_steps": max_steps}
    )
    env = BettingEnv(**preset)
    outcomes = {"target_reached": 0, "bust": 0, "max_steps": 0}
    final_balances: list[float] = []
    rng = np.random.default_rng(seed)

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        terminal_reason = info.get("terminal_reason", "max_steps")
        outcomes[terminal_reason] = outcomes.get(terminal_reason, 0) + 1
        final_balances.append(info.get("final_balance", info["balance"]))

    env.close()
    n = n_episodes
    pos = outcomes.get("target_reached", 0)
    neg = outcomes.get("bust", 0)
    print(f"\n── Random Agent Benchmark ({n} episodes) ──")
    print(f"  Positive: {pos} ({pos/n*100:.2f}%)")
    print(f"  Negative: {neg} ({neg/n*100:.2f}%)")
    print(f"  Mean final balance: ${np.mean(final_balances):.2f}")
    return {"positive_rate": pos / n, "negative_rate": neg / n,
            "mean_final_balance": float(np.mean(final_balances))}


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate betting RL agent")
    parser.add_argument("--model", default="models/best_model.zip",
                        help="Path to saved model (.zip)")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--sport", default="generic",
                        choices=list(SPORT_PRESETS.keys()))
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of evaluation episodes")
    parser.add_argument("--commission", type=float, default=0.05)
    parser.add_argument("--overround", type=float, default=1.05)
    parser.add_argument("--balance", type=float, default=100.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--render", action="store_true",
                        help="Print each step to terminal")
    parser.add_argument("--benchmark", action="store_true",
                        help="Also run random-agent benchmark for comparison")
    parser.add_argument("--plot-dir", default="plots")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    stats = evaluate(
        model_path=args.model,
        sport=args.sport,
        n_episodes=args.episodes,
        commission_rate=args.commission,
        overround=args.overround,
        starting_balance=args.balance,
        max_steps=args.max_steps,
        algo=args.algo,
        render=args.render,
        plot_dir=args.plot_dir,
        seed=args.seed,
    )

    if args.benchmark:
        rand_stats = benchmark_random(
            sport=args.sport,
            n_episodes=args.episodes,
            commission_rate=args.commission,
            overround=args.overround,
            starting_balance=args.balance,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        print(f"\n── Trained vs Random comparison ──")
        print(f"  Positive rate — Trained: {stats['positive_rate']*100:.2f}%  "
              f"Random: {rand_stats['positive_rate']*100:.2f}%")
        print(f"  Negative rate — Trained: {stats['negative_rate']*100:.2f}%  "
              f"Random: {rand_stats['negative_rate']*100:.2f}%")
        print(f"  Mean balance  — Trained: ${stats['mean_final_balance']:.2f}  "
              f"Random: ${rand_stats['mean_final_balance']:.2f}")
