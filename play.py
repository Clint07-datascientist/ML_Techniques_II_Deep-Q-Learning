"""
play.py - Play Pong using a trained DQN model (Task 2)
Loads the best saved model and plays a few episodes using greedy policy.
"""

import gymnasium as gym
from stable_baselines3 import DQN
import ale_py
import time

def make_env():
    """Create the Pong environment with human-render mode."""
    env = gym.make("ALE/Pong-v5", render_mode="human")
    return env

def play(model_path, episodes=3):
    """
    Play Pong using a trained DQN model.

    Args:
        model_path (str): Path to the saved DQN model (.zip file)
        episodes (int): Number of episodes to play
    """

    print(f"\nLoading model from: {model_path}")
    model = DQN.load(model_path)

    env = make_env()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        print(f"\n=== Episode {ep + 1} ===")

        while not done:
            # Greedy policy = deterministic=True
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Render frame
            env.render()

            # Slow down for visibility (optional)
            time.sleep(0.01)

        print(f"Episode {ep + 1} Reward: {total_reward}")

    env.close()
    print("\nFinished playing!")

if __name__ == "__main__":
    # Path to best model from training
    model_path = "./models/exp10.zip"

    play(model_path, episodes=3)