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
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import os
import ale_py
def make_atari_env_stacked(env_id="ALE/Pong-v5", n_envs=1, seed=0, num_stack=4):
    """Create and wrap the Pong environment"""
    env = make_atari_env(env_id, n_envs=n_envs, seed=seed)
    env = VecFrameStack(env, n_stack=num_stack)
    return env
def train_dqn(env_id,hyperparams, model_name="dqn_baseline"):
    """
    Train a DQN agent with specified hyperparameters
    
    Args:
        hyperparams: Dictionary containing hyperparameters
        model_name: Name to save the model
    """
    # Create environment

    env = make_atari_env_stacked(env_id)
    eval_env = make_atari_env_stacked(env_id, seed=42)
    
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
    # --- Define Experiments ---
    ATARI_ENV_ID = "ALE/Pong-v5" 
    TOTAL_TIMESTEPS = 100000 # SET TO 100,000 TIMESTEPS

    # BASE CONFIGURATION ADAPTED FOR 100K STEPS 
    BASE_PARAMS = {
        'learning_rate': 1e-4, 
        'buffer_size': 30000, 
        'learning_starts': 5000, # CRITICAL: Start training at step 5,000
        'batch_size': 32, 
        'gamma': 0.99, 
        'train_freq': 4, 
        'gradient_steps': 1, 
        'target_update_interval': 1000, 
        'exploration_fraction': 0.2, # Epsilon decays over 20,000 steps (20% of 100k)
        'exploration_initial_eps': 1.0, 
        'exploration_final_eps': 0.05, 
        'total_timesteps': TOTAL_TIMESTEPS 
    }

    # Define the 10 hyperparameter experiments
    hyperparams_experiments = {
        "Exp_1_Baseline": BASE_PARAMS,
        "Exp_2_High_LR": {**BASE_PARAMS, 'learning_rate': 5e-4}, 
        "Exp_3_Low_LR": {**BASE_PARAMS, 'learning_rate': 1e-5}, 
        "Exp_4_Low_Gamma": {**BASE_PARAMS, 'gamma': 0.9}, 
        "Exp_5_High_Gamma": {**BASE_PARAMS, 'gamma': 0.999}, 
        "Exp_6_Large_Batch": {**BASE_PARAMS, 'batch_size': 128}, 
        "Exp_7_Fast_Exploration": {**BASE_PARAMS, 'exploration_fraction': 0.1, 'exploration_final_eps': 0.01}, # Decay over 10k steps
        "Exp_8_Slow_Exploration": {**BASE_PARAMS, 'exploration_fraction': 0.8, 'exploration_final_eps': 0.1}, # Decay over 80k steps
        "Exp_9_Low_LR_Low_Gamma": {**BASE_PARAMS, 'learning_rate': 1e-5, 'gamma': 0.9}, 
        "Exp_10_High_LR_High_Gamma": {**BASE_PARAMS, 'learning_rate': 5e-4, 'gamma': 0.999}, 
    }
    
    results = []

    # Run all 10 experiments
    print("="*60)
    print(f"Starting Hyperparameter Tuning on {ATARI_ENV_ID} (100k steps each)")
    print("="*60)

    for name, params in hyperparams_experiments.items():
        result = train_dqn(
            env_id=ATARI_ENV_ID,
            hyperparams=params,
            model_name=name
        )
        results.append(result)
        print("-" * 60)

    # Print summary table structure for required documentation
    print("\n\n" + "#" * 70)
    print("Hyperparameter Tuning Summary for Documentation")
    print("#" * 70)
    print("| Experiment | Hyperparameter Set | Mean Reward +/- Std Dev | Noted Behavior |")
    print("|:---:|:---|:---:|:---|")
    
    for r in results:
        
        # 💡 CRITICAL FIX: Check if the item 'r' is a dictionary. 
        # If it's a tuple or anything else, skip it and print an error message.
        if not isinstance(r, dict):
            print(f"| CORRUPTED | Data structure failure | N/A | Entry was not a dictionary (likely a tuple). |")
            continue  # Skips the current iteration and moves to the next good one
            
        eps_start = r['hyperparams']['exploration_initial_eps']
        eps_end = r['hyperparams']['exploration_final_eps']
        eps_decay_proxy = r['hyperparams']['exploration_fraction'] 

        param_set_str = (
            f"lr={r['hyperparams']['learning_rate']}, "
            f"gamma={r['hyperparams']['gamma']}, "
            f"batch={r['hyperparams']['batch_size']}, "
            f"eps_start={eps_start}, eps_end={eps_end}, "
            f"eps_decay_frac={eps_decay_proxy}"
        )
        
        behavior = "OBSERVE TENSORBOARD AND FILL THIS IN" 
        
        print(
            f"| {r['model_name']} | {param_set_str} | "
            f"{r['mean_reward']:.2f} +/- {r['std_reward']:.2f} | "
            f"{behavior} |"
        )
    
    print("\nNOTE: Run this script and use the TensorBoard logs in ./logs/tensorboard/ to fill out the 'Noted Behavior' column.")