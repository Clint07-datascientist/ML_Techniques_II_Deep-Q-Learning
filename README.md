# ML Techniques II · Deep Q-Learning (Pong)

This project trains a Deep Q-Network (DQN) agent with Stable-Baselines3 to master `PongNoFrameskip-v4`, then visualises the greedy policy.

## Setup

```bash
pip install gymnasium[atari] stable-baselines3[atari] shimmy tensorboard
```

The scripts auto-create the required folders:

- `train.py` – trains DQN, writes checkpoints to `models/`, logs to `logs/`
- `play.py` – loads `models/dqn_model.zip` and renders greedy play
- `models/` – saved `.zip` files plus best-model checkpoints
- `logs/` – monitor files and TensorBoard summaries (`logs/tensorboard`)

## Training

Running `python train.py` now executes ten curated experiments (each with unique hyperparameters). After all runs finish the best-performing checkpoint is copied to both `models/best_of_10.zip` and `models/dqn_model.zip` so `play.py` can load it immediately.

```bash
python train.py
```

Launch TensorBoard to compare runs:

```bash
tensorboard --logdir logs/tensorboard
```

## Hyperparameter Experiments

| Experiment | Learning Rate | Batch | Gamma  | Final ε | Timesteps |
|------------|---------------|-------|--------|--------|-----------|
| exp1       | 1.0e-4        | 64    | 0.990 | 0.020  | 200k      |
| exp2       | 7.5e-5        | 32    | 0.970 | 0.050  | 150k      |
| exp3       | 2.0e-4        | 128   | 0.995 | 0.010  | 220k      |
| exp4       | 3.0e-4        | 96    | 0.992 | 0.020  | 220k      |
| exp5       | 5.0e-5        | 48    | 0.980 | 0.080  | 160k      |
| exp6       | 1.5e-4        | 32    | 0.995 | 0.015  | 250k      |
| exp7       | 2.5e-4        | 64    | 0.997 | 0.005  | 250k      |
| exp8       | 7.0e-5        | 128   | 0.990 | 0.030  | 180k      |
| exp9       | 3.5e-4        | 64    | 0.985 | 0.040  | 230k      |
| exp10      | 9.0e-5        | 96    | 0.993 | 0.015  | 190k      |

Common settings across runs: buffer size 100k, learning starts 50k, train freq 4, gradient steps 1, target update interval 10k, exploration fraction 0.12, initial epsilon 1.0, frame stacking (4), `CnnPolicy`.

## Playing the Trained Agent

```bash
python play.py --model_path models/dqn_model.zip --episodes 5
```

Implementation details:

- Environment via `make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)`
- Frame stacking: `VecFrameStack(env, n_stack=4)`
- Greedy evaluation: `model.predict(obs, deterministic=True)`
- Rendering triggers `env.render()`; close the window to stop playback

## Notes

- The assignment requires Gymnasium (not Gym) plus Shimmy for the ALE backend.
- Default training length is 1e6 steps; use `--total_timesteps` for quicker tests.
- Clean up `models/*_checkpoints` and `logs/` periodically to save disk space.