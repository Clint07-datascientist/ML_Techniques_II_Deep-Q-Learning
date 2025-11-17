# Script for playing with the agent (Task 2)
# Script for playing with the agent (Task 2)
"""
play.py - Evaluate and visualize a trained DQN agent on Atari Pong
This script loads a trained model and displays the agent playing
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import time
import ale_py
# --- Configuration ---
ENV_ID = "ALE/Pong-v5"        # Must be the SAME environment used in train.py
MODEL_PATH = "dqn_baseline.zip" # Path to the saved model (or best_model.zip)
N_EVAL_EPISODES = 5          # Number of episodes to run for visualization

def make_env(render_mode="human"):
    """Create and wrap the Pong environment identically to train.py, but with rendering."""
    # Note: Using render_mode="human" here will display the game in a new window.
    # The default Atari wrappers (which you excluded in train.py by not using 
    # make_atari_env or VecFrameStack) are NOT applied here, matching your training.
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = Monitor(env) # Match the Monitor wrapper from train.py
    return env
def play_agent(model_path):
    """Loads the model and runs episodes with visualization."""
    
    # 1. Load the Trained Model
    try:
        # Load the model without setting the environment yet
        model = DQN.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure '{model_path}' exists after running train.py.")
        return

    # 2. Setup Evaluation Environment with Rendering
    # Create the environment for visualization and set it for prediction
    env = make_env(render_mode="human")
    model.set_env(env)
    
    # Set the model to evaluation mode (disables exploration and ensures deterministic actions)
    model.policy.set_training_mode(False)
    
    # 3. Run and Visualize Episodes
    print(f"\nStarting {N_EVAL_EPISODES} evaluation episodes. Close the rendering window to stop.")

    mean_reward = 0
    
    for episode in range(N_EVAL_EPISODES):
        # Reset returns (observation, info) in Gymnasium
        obs, info = env.reset() 
        done = False
        total_reward = 0
        
        print(f"--- Episode {episode + 1} ---")
        
        while not done:
            # Agent selects action (deterministic=True is the GreedyQPolicy equivalent)
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment (returns obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Rendering is handled by the 'render_mode="human"' argument in gym.make()
            # We can pause slightly to see the frames more clearly if needed
            # time.sleep(0.01)

        print(f"Episode finished with reward: {total_reward:.2f}")
        mean_reward += total_reward

    # 4. Clean up
    mean_reward /= N_EVAL_EPISODES
    print(f"\nMean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f}")
    env.close()

if __name__ == "__main__":
    play_agent(MODEL_PATH)