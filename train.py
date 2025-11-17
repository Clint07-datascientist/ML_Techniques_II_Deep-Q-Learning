# Script for training the agent (Task 1)
# Script for training the agent (Task 1)

"""
train.py - Train a DQN agent on Atari Pong
This script trains a Deep Q-Network agent using Stable Baselines3
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import ale_py
def make_env():
    """Create and wrap the Pong environment"""
    env = gym.make("ALE/Pong-v5", render_mode=None)
    env = Monitor(env)
    return env
def train_dqn(hyperparams, model_name="dqn_baseline"):
    """
    Train a DQN agent with specified hyperparameters
    
    Args:
        hyperparams: Dictionary containing hyperparameters
        model_name: Name to save the model
    """
    # Create environment
    env = make_env()
    eval_env = make_env()
    
    # Create callbacks directory
    os.makedirs("./logs/", exist_ok=True)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/checkpoints/",
        name_prefix="dqn_checkpoint"
    )
    
    print(f"\n{'='*60}")
    print(f"Training DQN with hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    # Create DQN model
    model = DQN(
        "MlpPolicy",  # CNN policy is better for image-based Atari games
        env,
        learning_rate=hyperparams['learning_rate'],
        buffer_size=hyperparams['buffer_size'],
        learning_starts=hyperparams['learning_starts'],
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        train_freq=hyperparams['train_freq'],
        gradient_steps=hyperparams['gradient_steps'],
        target_update_interval=hyperparams['target_update_interval'],
        exploration_fraction=hyperparams['exploration_fraction'],
        exploration_initial_eps=hyperparams['exploration_initial_eps'],
        exploration_final_eps=hyperparams['exploration_final_eps'],
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )
    # Train the agent
    print("\nStarting training...")
    model.learn(
        total_timesteps=hyperparams['total_timesteps'],
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final model
    model.save(model_name)
    print(f"\nModel saved as {model_name}.zip")
    
    # Final evaluation
    print("\nEvaluating trained agent...")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    eval_env.close()
    
    return model, mean_reward, std_reward


if __name__ == "__main__":
    # Hyperparameter configurations to test
    
    # Configuration 1: Baseline
    hyperparams_1 = {
        'learning_rate': 1e-4,
        'buffer_size': 100000,
        'learning_starts': 500000,
        'batch_size': 32,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 100,
        'exploration_fraction': 0.1,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.05,
        'total_timesteps': 100000
    }
    
    # Configuration 2: Higher learning rate, faster exploration decay
    hyperparams_2 = {
        'learning_rate': 5e-4,
        'buffer_size': 100000,
        'learning_starts': 500000,
        'batch_size': 64,
        'gamma': 0.99,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 100,
        'exploration_fraction': 0.2,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.01,
        'total_timesteps': 100000
    }
    
    # Configuration 3: More conservative (lower learning rate, slower exploration)
    hyperparams_3 = {
        'learning_rate': 1e-5,
        'buffer_size': 100000,
        'learning_starts': 500000,
        'batch_size': 32,
        'gamma': 0.95,
        'train_freq': 4,
        'gradient_steps': 1,
        'target_update_interval': 100,
        'exploration_fraction': 0.3,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.1,
        'total_timesteps': 100000
    }
    
    # Configuration 4: Larger batch, higher gamma
    hyperparams_4 = {
        'learning_rate': 2.5e-4,
        'buffer_size': 100000,
        'learning_starts': 500000,
        'batch_size': 128,
        'gamma': 0.99,
        'train_freq': 8,
        'gradient_steps': 2,
        'target_update_interval': 100,
        'exploration_fraction': 0.15,
        'exploration_initial_eps': 1.0,
        'exploration_final_eps': 0.02,
        'total_timesteps': 100000
    }
    
    # Choose which configuration to run (change this to test different configs)
    selected_config = hyperparams_1
    
    # Train the model
    model, mean_reward, std_reward = train_dqn(
        selected_config, 
        model_name="dqn_baseline"
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    