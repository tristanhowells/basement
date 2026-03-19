"""
Training script for the two-team betting market RL agent.

Algorithm: PPO (Proximal Policy Optimisation) via Stable-Baselines3.
  - PPO handles continuous action spaces well out of the box.
  - Vectorised environments run multiple episodes in parallel for efficiency.
  - A custom callback tracks and logs episode outcomes throughout training.

Usage
-----
    python train.py                          # train with defaults
    python train.py --sport tennis           # use tennis odds range
    python train.py --steps 2_000_000       # longer run
    python train.py --commission 0.0        # no commission
    python train.py --algo sac              # use SAC instead of PPO

Outputs
-------
    models/best_model.zip          — best model by mean episode reward
    logs/                          — TensorBoard event files
    models/final_model.zip         — model at end of training
"""

import argparse
import os
from collections import deque

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from betting_env import BettingEnv, make_env, SPORT_PRESETS


# ────────────────────────────────────────────────────────────────────────────
# Monitoring callback
# ────────────────────────────────────────────────────────────────────────────

class BettingMonitorCallback(BaseCallback):
    """
    Tracks episode outcomes across all vectorised environments and logs to
    TensorBoard. Prints a summary every `log_interval` episodes.

    Outcome categories
    ------------------
    positive  — balance reached win_threshold (e.g. $10,000)
    negative  — balance fell below bust_threshold (e.g. $1)
    timeout   — episode hit max_steps without either terminal event
    """

    def __init__(self, log_interval: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval

        self.episode_count = 0
        self.positive_episodes = 0   # target reached
        self.negative_episodes = 0   # bust
        self.timeout_episodes = 0    # max_steps

        self.final_balances: list[float] = []
        self._recent_rewards: deque = deque(maxlen=log_interval)

    def _on_step(self) -> bool:
        # SB3 VecEnv passes info dicts and done flags per parallel env
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.episode_count += 1
            terminal = info.get("terminal_reason", "max_steps")
            final_balance = info.get("final_balance", 0.0)
            self.final_balances.append(final_balance)

            if terminal == "target_reached":
                self.positive_episodes += 1
            elif terminal == "bust":
                self.negative_episodes += 1
            else:
                self.timeout_episodes += 1

            # Log scalars to TensorBoard
            self.logger.record("betting/episode_count", self.episode_count)
            self.logger.record("betting/positive_episodes", self.positive_episodes)
            self.logger.record("betting/negative_episodes", self.negative_episodes)
            self.logger.record("betting/timeout_episodes", self.timeout_episodes)
            self.logger.record("betting/final_balance", final_balance)

            if self.episode_count >= 1:
                pos_rate = self.positive_episodes / self.episode_count
                neg_rate = self.negative_episodes / self.episode_count
                self.logger.record("betting/positive_rate", pos_rate)
                self.logger.record("betting/negative_rate", neg_rate)

            # Print summary at interval
            if self.verbose >= 1 and self.episode_count % self.log_interval == 0:
                n = self.episode_count
                pos = self.positive_episodes
                neg = self.negative_episodes
                tmt = self.timeout_episodes
                recent_bal = np.mean(self.final_balances[-self.log_interval :])
                print(
                    f"\n{'─'*60}\n"
                    f"  Episodes completed : {n}\n"
                    f"  Positive (target)  : {pos:>6d}  ({pos/n*100:.1f}%)\n"
                    f"  Negative (bust)    : {neg:>6d}  ({neg/n*100:.1f}%)\n"
                    f"  Timeout (max_steps): {tmt:>6d}  ({tmt/n*100:.1f}%)\n"
                    f"  Avg final balance  : ${recent_bal:.2f} (last {self.log_interval})\n"
                    f"{'─'*60}"
                )

        return True  # returning False would stop training

    def get_summary(self) -> dict:
        n = max(self.episode_count, 1)
        return {
            "total_episodes": self.episode_count,
            "positive_episodes": self.positive_episodes,
            "negative_episodes": self.negative_episodes,
            "timeout_episodes": self.timeout_episodes,
            "positive_rate": self.positive_episodes / n,
            "negative_rate": self.negative_episodes / n,
            "mean_final_balance": float(np.mean(self.final_balances)) if self.final_balances else 0.0,
        }


# ────────────────────────────────────────────────────────────────────────────
# Environment factory for make_vec_env
# ────────────────────────────────────────────────────────────────────────────

def _make_env_fn(env_kwargs: dict):
    """Returns a factory function compatible with make_vec_env."""
    def _init():
        env = BettingEnv(**env_kwargs)
        env = Monitor(env)
        return env
    return _init


# ────────────────────────────────────────────────────────────────────────────
# Main training function
# ────────────────────────────────────────────────────────────────────────────

def train(
    sport: str = "generic",
    total_steps: int = 1_000_000,
    n_envs: int = 8,
    algo: str = "ppo",
    commission_rate: float = 0.05,
    overround: float = 1.05,
    starting_balance: float = 100.0,
    max_steps: int = 500,
    bust_penalty: float = -100.0,
    win_bonus: float = 100.0,
    log_dir: str = "logs",
    model_dir: str = "models",
    seed: int = 42,
    verbose: int = 1,
) -> None:

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Build environment kwargs from sport preset + overrides
    preset = SPORT_PRESETS.get(sport, SPORT_PRESETS["generic"]).copy()
    preset.update(
        {
            "commission_rate": commission_rate,
            "overround": overround,
            "starting_balance": starting_balance,
            "max_steps": max_steps,
            "bust_penalty": bust_penalty,
            "win_bonus": win_bonus,
        }
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Betting RL Training")
        print(f"  Sport          : {sport}")
        print(f"  Algorithm      : {algo.upper()}")
        print(f"  Parallel envs  : {n_envs}")
        print(f"  Total steps    : {total_steps:,}")
        print(f"  Overround      : {overround*100:.1f}%")
        print(f"  Commission     : {commission_rate*100:.1f}%")
        print(f"  Starting bal   : ${starting_balance}")
        print(f"  Max steps/ep   : {max_steps}")
        print(f"  Bust threshold : ${preset.get('bust_threshold', 1.0)}")
        print(f"  Win threshold  : ${preset.get('win_threshold', 10_000)}")
        print(f"{'='*60}\n")

    # ── Training environments ────────────────────────────────────────
    train_env = make_vec_env(
        _make_env_fn(preset),
        n_envs=n_envs,
        seed=seed,
    )

    # ── Evaluation environment (single, for EvalCallback) ───────────
    eval_env = make_vec_env(
        _make_env_fn(preset),
        n_envs=1,
        seed=seed + 9999,
    )

    # ── Callbacks ───────────────────────────────────────────────────
    monitor_cb = BettingMonitorCallback(log_interval=200, verbose=verbose)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(50_000 // n_envs, 1),   # evaluate every ~50k environment steps
        n_eval_episodes=100,
        deterministic=True,
        verbose=0,
    )

    callbacks = [monitor_cb, eval_cb]

    # ── Model ────────────────────────────────────────────────────────
    algo_upper = algo.upper()

    if algo_upper == "PPO":
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,           # steps per env before update
            batch_size=64,
            n_epochs=10,
            gamma=0.99,             # discount factor
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,          # entropy bonus encourages exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log=log_dir,
            seed=seed,
            verbose=0,
        )

    elif algo_upper == "SAC":
        # SAC: off-policy, more sample-efficient but requires more memory.
        # Suited for longer training runs or if PPO underfits.
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[128, 128]),
            tensorboard_log=log_dir,
            seed=seed,
            verbose=0,
        )

    else:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose 'ppo' or 'sac'.")

    # ── Train ────────────────────────────────────────────────────────
    if verbose:
        print(f"Training {algo_upper} for {total_steps:,} steps across {n_envs} envs...\n")

    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        progress_bar=True,
    )

    # ── Save ─────────────────────────────────────────────────────────
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    if verbose:
        print(f"\nFinal model saved to: {final_path}.zip")

    # ── Training summary ─────────────────────────────────────────────
    summary = monitor_cb.get_summary()
    print(f"\n{'='*60}")
    print("  Training Complete — Episode Outcome Summary")
    print(f"{'='*60}")
    print(f"  Total episodes     : {summary['total_episodes']:,}")
    print(f"  Positive (target)  : {summary['positive_episodes']:,}  "
          f"({summary['positive_rate']*100:.2f}%)")
    print(f"  Negative (bust)    : {summary['negative_episodes']:,}  "
          f"({summary['negative_rate']*100:.2f}%)")
    print(f"  Timeout (max_steps): {summary['timeout_episodes']:,}")
    print(f"  Mean final balance : ${summary['mean_final_balance']:.2f}")
    print(f"{'='*60}\n")

    train_env.close()
    eval_env.close()

    return model, monitor_cb


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train betting RL agent")
    parser.add_argument("--sport", default="generic",
                        choices=list(SPORT_PRESETS.keys()),
                        help="Sport market type (affects odds range and overround)")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Total environment steps for training")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"],
                        help="RL algorithm")
    parser.add_argument("--commission", type=float, default=0.05,
                        help="Commission rate on winning bets (0.05 = 5%%)")
    parser.add_argument("--overround", type=float, default=1.05,
                        help="Market overround (1.05 = 105%%)")
    parser.add_argument("--balance", type=float, default=100.0,
                        help="Starting balance per episode")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max bets per episode")
    parser.add_argument("--bust-penalty", type=float, default=-100.0,
                        help="Terminal reward penalty for busting")
    parser.add_argument("--win-bonus", type=float, default=100.0,
                        help="Terminal reward bonus for reaching target")
    parser.add_argument("--log-dir", default="logs",
                        help="TensorBoard log directory")
    parser.add_argument("--model-dir", default="models",
                        help="Directory to save models")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        sport=args.sport,
        total_steps=args.steps,
        n_envs=args.n_envs,
        algo=args.algo,
        commission_rate=args.commission,
        overround=args.overround,
        starting_balance=args.balance,
        max_steps=args.max_steps,
        bust_penalty=args.bust_penalty,
        win_bonus=args.win_bonus,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        verbose=1,
    )
