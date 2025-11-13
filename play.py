# play.py — Greedy evaluation & windowed render
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

def build_vec_env(env_id: str, seed: int, frame_stack: int = 4):
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
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    # Vec env for model inference (preprocessing, stacking)
    venv = build_vec_env(args.env_id, args.seed, frame_stack=4)

    # Separate human-render env for display only
    render_env = gym.make(args.env_id, render_mode="human")
    render_obs, _ = render_env.reset(seed=args.seed)

    model = DQN.load(args.model_path)

    for ep in range(args.episodes):
        obs = venv.reset()                 # VecEnv reset -> obs only
        render_obs, _ = render_env.reset(seed=args.seed + ep)
        done = False
        ep_reward = 0.0

        while True:
            # Greedy action
            action, _ = model.predict(obs, deterministic=True)
            # Step vectorized env (returns arrays)
            obs, rewards, dones, infos = venv.step(action)
            ep_reward += float(rewards[0])

            # Step the human-render env with the same action (int)
            robs, r, terminated, truncated, info_r = render_env.step(int(action))
            render_env.render()

            if dones[0]:
                break

        print(f"Episode {ep+1}: reward={ep_reward:.2f}")

    venv.close()
    render_env.close()

if __name__ == "__main__":
    main()