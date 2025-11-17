"""
play.py
=======
Greedy evaluation script for a trained DQN agent on PongNoFrameskip-v4.

Usage:
    python play.py --model_path models/dqn_model.zip --episodes 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import shimmy  # registers ALE/Pong environments with Gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

ENV_ID = "PongNoFrameskip-v4"
DEFAULT_MODEL_PATH = Path("models") / "dqn_model.zip"
DEFAULT_SEED = 0


def build_env():
    """Match the training wrappers exactly (make_atari_env + VecFrameStack)."""
    env = make_atari_env(ENV_ID, n_envs=1, seed=DEFAULT_SEED)
    env = VecFrameStack(env, n_stack=4)
    return env


def play_agent(model_path: Path, episodes: int) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train one via train.py first."
        )

    print(f"Loading DQN model from {model_path}")
    model = DQN.load(model_path)
    env = build_env()

    try:
        for episode in range(1, episodes + 1):
            obs = env.reset()
            total_reward = 0.0
            done = False

            print(f"\nEpisode {episode}/{episodes}")
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                env.render()

                total_reward += float(rewards[0])
                done = bool(dones[0])

            print(f"Episode reward: {total_reward:.2f}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained DQN Pong agent.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the .zip model file.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of greedy episodes to render.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    play_agent(args.model_path, args.episodes)