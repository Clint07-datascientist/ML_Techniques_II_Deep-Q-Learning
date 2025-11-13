# train.py  — Tuner B: Experiment Set D (ε linear decay over exploration_fraction)
import argparse
import os
from datetime import datetime

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback


class CSVLogCallback(BaseCallback):
    """
    Appends episode metrics to a CSV: timesteps, episode_reward, episode_length.
    Works with Monitor/VecEnv via ep_info_buffer.
    """
    def __init__(self, log_csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_csv_path = log_csv_path
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        with open(self.log_csv_path, "w") as f:
            f.write("timesteps,episode_reward,episode_length\n")

    def _on_step(self) -> bool:
        for ep_info in self.model.ep_info_buffer:
            r = ep_info.get("r", np.nan)
            l = ep_info.get("l", np.nan)
            with open(self.log_csv_path, "a") as f:
                f.write(f"{self.num_timesteps},{r},{l}\n")
        return True


def build_env(env_id: str, seed: int, frame_stack: int = 4):
    """
    Standard Atari preprocessing with frame stacking and reward clipping.
    """
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        wrapper_kwargs={"clip_reward": True}
    )
    env = VecFrameStack(env, n_stack=frame_stack)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--policy", type=str, default="CnnPolicy", choices=["CnnPolicy", "MlpPolicy"])
    parser.add_argument("--total_timesteps", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)

    # ---- Set D hyperparameters (ε decays over a large fraction of training) ----
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon_start", type=float, default=1.0)   # SB3: exploration_initial_eps
    parser.add_argument("--epsilon_end", type=float, default=0.05)    # SB3: exploration_final_eps
    parser.add_argument("--epsilon_decay", type=float, default=0.80)  # SB3: exploration_fraction ∈ (0,1]

    # Replay/target
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learning_starts", type=int, default=50_000)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--target_update_interval", type=int, default=10_000)

    # Logging / naming
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    run_stamp = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", "setD", f"run_{run_stamp}")
    tb_dir = os.path.join(log_dir, "tb")
    csv_path = os.path.join(log_dir, "metrics.csv")
    os.makedirs(log_dir, exist_ok=True)

    env = build_env(args.env_id, args.seed, frame_stack=4)

    model = DQN(
        policy=args.policy,
        env=env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.epsilon_start,
        exploration_final_eps=args.epsilon_end,
        exploration_fraction=args.epsilon_decay,  # <-- Set D mapping
        tensorboard_log=tb_dir,
        verbose=1,
        seed=args.seed,
    )

    csv_cb = CSVLogCallback(csv_path)

    model.learn(total_timesteps=args.total_timesteps, callback=csv_cb, progress_bar=True)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"dqn_setD_{run_stamp}.zip")
    model.save(model_path)
    print(f"\nSaved model => {model_path}")
    print(f"Logs (TB+CSV) => {log_dir}")

    env.close()


if __name__ == "__main__":
    main()
