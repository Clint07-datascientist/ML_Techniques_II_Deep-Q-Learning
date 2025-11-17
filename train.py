from __future__ import annotations

import os
import shutil
from pathlib import Path

import gymnasium as gym  # required for registration side effects
import shimmy  # ensures ALE/Pong envs are available through Gymnasium
import ale_py  # ensure ALE namespace is registered with Gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
# Gymnasium Atari uses ALE namespace (Gym v0.x used NoFrameskip-v4)
ENV_ID = "ALE/Pong-v5"

# Ensure ROMs can be found when running without shell env configured
if "ALE_PY_ROM_DIR" not in os.environ:
    try:
        import sys as _sys

        _default_roms = Path(_sys.prefix) / "Lib" / "site-packages" / "AutoROM" / "roms"
        if _default_roms.exists():
            os.environ["ALE_PY_ROM_DIR"] = str(_default_roms)
    except Exception:
        pass
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"
BEST_DIR = LOG_DIR / "best_model"
CHECKPOINTS_DIR = LOG_DIR / "checkpoints"

for path in (MODEL_DIR, TENSORBOARD_DIR, BEST_DIR, CHECKPOINTS_DIR):
    path.mkdir(parents=True, exist_ok=True)

# Shared Atari DQN defaults
COMMON_HP = dict(
    buffer_size=100_000,
    learning_starts=50_000,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=10_000,
    exploration_fraction=0.12,
    exploration_initial_eps=1.0,
)


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------
def make_env(seed: int = 0):
    """Create Pong env via SB3 helper + 4-frame stacking."""
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_dqn(hyperparams: dict, model_name: str) -> float:
    """Train and evaluate a single experiment."""
    env = make_env(seed=hyperparams.get("seed", 0))
    eval_env = make_env(seed=hyperparams.get("seed", 0) + 997)

    best_path = BEST_DIR / model_name
    checkpoint_path = CHECKPOINTS_DIR / model_name
    best_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_path),
        log_path=str(LOG_DIR / f"{model_name}_eval"),
        eval_freq=50_000,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(checkpoint_path),
        name_prefix=model_name,
    )

    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        batch_size=hyperparams["batch_size"],
        gamma=hyperparams["gamma"],
        exploration_final_eps=hyperparams["exploration_final_eps"],
        tensorboard_log=str(TENSORBOARD_DIR),
        verbose=1,
        **COMMON_HP,
    )

    model.learn(
        total_timesteps=hyperparams["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model_path = MODEL_DIR / model_name
    model.save(str(model_path))
    print(f"Saved {model_path}.zip")

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=5, deterministic=True
    )
    print(f"{model_name} → mean reward {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    eval_env.close()
    return mean_reward


# -----------------------------------------------------------------------------
# Experiment definitions (all unique hyperparameters)
# -----------------------------------------------------------------------------
EXPERIMENTS = {
    "exp1": dict(learning_rate=1.0e-4, batch_size=64, gamma=0.99, exploration_final_eps=0.02, total_timesteps=200_000),
    "exp2": dict(learning_rate=7.5e-5, batch_size=32, gamma=0.97, exploration_final_eps=0.05, total_timesteps=150_000),
    "exp3": dict(learning_rate=2.0e-4, batch_size=128, gamma=0.995, exploration_final_eps=0.01, total_timesteps=220_000),
    "exp4": dict(learning_rate=3.0e-4, batch_size=96, gamma=0.992, exploration_final_eps=0.02, total_timesteps=220_000),
    "exp5": dict(learning_rate=5.0e-5, batch_size=48, gamma=0.98, exploration_final_eps=0.08, total_timesteps=160_000),
    "exp6": dict(learning_rate=1.5e-4, batch_size=32, gamma=0.995, exploration_final_eps=0.015, total_timesteps=250_000),
    "exp7": dict(learning_rate=2.5e-4, batch_size=64, gamma=0.997, exploration_final_eps=0.005, total_timesteps=250_000),
    "exp8": dict(learning_rate=7.0e-5, batch_size=128, gamma=0.99, exploration_final_eps=0.03, total_timesteps=180_000),
    "exp9": dict(learning_rate=3.5e-4, batch_size=64, gamma=0.985, exploration_final_eps=0.04, total_timesteps=230_000),
    "exp10": dict(learning_rate=9.0e-5, batch_size=96, gamma=0.993, exploration_final_eps=0.015, total_timesteps=190_000),
}


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    results = {}

    for exp_name, params in EXPERIMENTS.items():
        mean_reward = train_dqn(params, model_name=exp_name)
        results[exp_name] = mean_reward

    best_exp = max(results, key=results.get)
    print("\n" + "=" * 60)
    print(f"Best experiment: {best_exp} with mean reward {results[best_exp]:.2f}")

    src_model = MODEL_DIR / f"{best_exp}.zip"
    best_of_10 = MODEL_DIR / "best_of_10.zip"
    default_play_model = MODEL_DIR / "dqn_model.zip"
    shutil.copyfile(src_model, best_of_10)
    shutil.copyfile(src_model, default_play_model)
    print(f"Copied {src_model.name} to {best_of_10.name} and {default_play_model.name}")
    print("Training run complete. Use play.py to watch the best model.")