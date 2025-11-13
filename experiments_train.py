"""
train.py - Train a DQN agent on Atari Pong
Modified by Nicolle for first 5 Hyperparameter Experiments
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import ale_py

# ENVIRONMENT CREATION
def make_env():
    """Create and wrap the Pong environment"""
    env = gym.make("ALE/Pong-v5", render_mode=None)
    env = Monitor(env)
    return env

# TRAINING FUNCTION
def train_dqn(hyperparams, model_name="dqn_experiment"):
    """Train a DQN agent with specified hyperparameters."""
    env = make_env()
    eval_env = make_env()

    os.makedirs("./logs/", exist_ok=True)
    os.makedirs("./logs/checkpoints/", exist_ok=True)
    os.makedirs("./models/", exist_ok=True)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
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

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=hyperparams['learning_rate'],
        buffer_size=10000,
        learning_starts=50000,
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        train_freq=4,
        gradient_steps=1,
        target_update_interval=100,
        exploration_fraction=0.1,
        exploration_initial_eps=hyperparams['exploration_initial_eps'],
        exploration_final_eps=hyperparams['exploration_final_eps'],
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )

    # Train model
    print("\nStarting training...")
    model.learn(
        total_timesteps=hyperparams['total_timesteps'],
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save trained model
    model_path = f"./models/{model_name}.zip"
    model.save(model_path)
    print(f"\nModel saved as {model_path}")

    # Evaluate performance
    print("\nEvaluating trained agent...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=5,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    eval_env.close()

    return model, mean_reward, std_reward

# MAIN EXECUTION: TEN EXPERIMENTS
if __name__ == "__main__":
    experiments = {
    "exp1": {"learning_rate": 1e-4, "batch_size": 32, "gamma": 0.99,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "total_timesteps": 100_000},
    "exp2": {"learning_rate": 5e-5, "batch_size": 64, "gamma": 0.98,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.02, "total_timesteps": 100_000},
    "exp3": {"learning_rate": 5e-4, "batch_size": 32, "gamma": 0.995,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "total_timesteps": 100_000},
    "exp4": {"learning_rate": 1e-5, "batch_size": 128, "gamma": 0.95,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.1, "total_timesteps": 100_000},
    "exp5": {"learning_rate": 2.5e-4, "batch_size": 64, "gamma": 0.99,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.02, "total_timesteps": 100_000},
    "exp6": {"learning_rate": 1e-4, "batch_size": 16, "gamma": 0.98,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "total_timesteps": 150_000},
    "exp7": {"learning_rate": 5e-5, "batch_size": 32, "gamma": 0.995,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "total_timesteps": 150_000},
    "exp8": {"learning_rate": 5e-4, "batch_size": 128, "gamma": 0.99,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.02, "total_timesteps": 150_000},
    "exp9": {"learning_rate": 2.5e-4, "batch_size": 32, "gamma": 0.98,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.05, "total_timesteps": 150_000},
    "exp10":{"learning_rate": 1e-4, "batch_size": 64, "gamma": 0.995,
             "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01, "total_timesteps": 200_000},
}
       

    results = {}

    # Run all 10 experiments
    for exp_name, params in experiments.items():
        print(f"\n===== Running {exp_name} =====")
        model, mean_reward, std_reward = train_dqn(params, model_name=exp_name)
        results[exp_name] = mean_reward

    # Find best performing model
    best_exp = max(results, key=results.get)
    print(f"\n Best performing experiment: {best_exp} with mean reward = {results[best_exp]:.2f}")

    # Save best model as 'best_of_5.zip'
    best_model_path = f"./models/{best_exp}.zip"
    os.rename(best_model_path, "./models/best_of_10.zip")
    print("Best model saved as './models/best_of_10.zip'")

    print("\nAll experiments completed successfully!")
