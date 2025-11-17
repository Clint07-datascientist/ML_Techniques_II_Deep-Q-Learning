# Script for playing with the agent (Task 2)
"""
play.py - Evaluate and visualize a trained DQN agent on Atari Pong
This script loads a trained model and displays the agent playing
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    AtariWrapper,
    ClipRewardEnv, 
    EpisodicLifeEnv, 
    MaxAndSkipEnv, 
    NoopResetEnv
)
# We also need the nature_cnn wrapper from SB3's utils to handle the image format
from stable_baselines3.common.vec_env import VecTransposeImage
import time
import ale_py
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
# --- Configuration ---
ENV_ID = "ALE/Pong-v5"        # Must be the SAME environment used in train.py
MODEL_PATH = "Exp_1_Baseline.zip" # Path to the saved model (or best_model.zip)
N_EVAL_EPISODES = 5          # Number of episodes to run for visualization

# --- Environment Setup (Manually stacking ALL required wrappers) ---
def make_atari_env_stacked(env_id="ALE/Pong-v5", seed=0, num_stack=4, render_mode=None):
    """
    Creates the fully wrapped, frame-stacked environment by explicitly 
    applying the full list of Atari wrappers to guarantee the correct 
    observation space (4, 84, 84).
    """
    
    # 1. Define the function that creates the base environment and applies the wrappers
    def make_env():
        env = gym.make(env_id, render_mode=render_mode)
        
        # Apply the SAME wrappers that make_atari_env would, but manually:
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        
        # **This is the critical one for the observation space:**
        # It handles grayscale, resizing to 84x84, and frame stacking preparation
        env = AtariWrapper(env)
        
        env = ClipRewardEnv(env)
        
        return env

    # 2. Vectorize the environment
    env = DummyVecEnv([make_env])
    
    # 3. Apply Frame Stacking (This makes the 4 channel depth)
    env = VecFrameStack(env, n_stack=num_stack)
    
    # 4. Apply Transpose (SB3 expects the channel dimension first: (4, 84, 84))
    # This step is often implicitly handled, but makes it explicit and safe.
    env = VecTransposeImage(env)
    
    return env

# ----------------------------------------------------------------------
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
    env = make_atari_env_stacked(render_mode="human")
    # Set the model to evaluation mode (disables exploration and ensures deterministic actions)
    model.policy.set_training_mode(False)
    
    # 3. Run and Visualize Episodes
    print(f"\nStarting {N_EVAL_EPISODES} evaluation episodes. Close the rendering window to stop.")

    mean_reward = 0
    
    for episode in range(N_EVAL_EPISODES):
        # Reset returns (observation, info) in Gymnasium
        obs = env.reset() 
        done = False
        total_reward = 0
        
        print(f"--- Episode {episode + 1} ---")
        
        while not done:
            # Agent selects action
            action, _ = model.predict(obs, deterministic=True)
            
            # 🚨 FINAL FIX: Unpack 4 values: obs, reward, done_tuple, info
            # The VecEnv returns 'done' as a tuple/array (done_tuple), not separated terminated/truncated.
            obs, reward, done_tuple, info = env.step(action)
            
            # Use the 'done_tuple' array to check if the episode is finished
            done = any(done_tuple)
            total_reward += reward
            # Rendering is handled by the 'render_mode="human"' argument in gym.make()
            # We can pause slightly to see the frames more clearly if needed
            # time.sleep(0.01)

        # Convert the total_reward array to a float scalar before formatting
        print(f"Episode finished with reward: {total_reward.item():.2f}")
        mean_reward += total_reward

    # 4. Clean up
    mean_reward /= N_EVAL_EPISODES
    print(f"\nMean reward over {N_EVAL_EPISODES} episodes: {mean_reward.item():.2f}")
    env.close()

if __name__ == "__main__":
    play_agent(MODEL_PATH)