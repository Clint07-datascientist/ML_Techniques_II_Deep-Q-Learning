# Tuner B — Experiment Sets C & D (DQN on Atari)

Branch: tuner-b
Env: ALE/Breakout-v5
Libs: Stable-Baselines3 (DQN), Gymnasium Atari wrappers
Policy: CnnPolicy (screen observations)
Replay: default unless specified per run
Eval: greedy (deterministic=True) via play.py
ROMs: installed with AutoROM -y

Run & Repro

Train:
python train.py --env_id "ALE/Breakout-v5" --run_name D1 --lr 1e-4 --gamma 0.99 --batch_size 32 --epsilon_decay 0.80 --total_timesteps 10000


Play:

python play.py --env_id "ALE/Breakout-v5" --model_path models/dqn_setD_D1.zip --episodes 3

Set D — focus on ε schedule (linear decay; epsilon_decay ≈ exploration_fraction)

All runs use epsilon_start=1.0, epsilon_end=0.05.
Key idea: vary how long we explore (epsilon_decay) and couple with lr/gamma/batch_size.

Run	Timesteps	lr	gamma	batch	epsilon_decay	Noted behavior (initial pass)
D1	10,000	1e-4	0.99	32	0.80	Trained successfully. From logs: ep_rew_mean hovered ~2.0 by ~8–10k steps; ep_len_mean ~232–244; exploration reached 0.05 near the end. Model: models/dqn_setD_D1.zip.
D2	20,000	5e-5	0.99	64	0.80	Completed (reduced buffer/learning_starts to avoid OOM). Fill observed reward trend after review of logs/setD/run_D2/metrics.csv.
D3	20,000	1e-4	0.995	32	0.90	Slower decay + higher γ expected to lift late reward but slower early climb. Fill after run.
D4	25,000	2.5e-4	0.98	32	0.70	Faster exploitation; may improve early score but risk instability with higher lr. Fill after run.
D5	30,000	1e-4	0.99	128	0.85	Larger batch smooths updates; expect steadier curve, slower responsiveness. Fill after run.

Artifacts:

Models: models/dqn_setD_D{1..5}.zip

Logs: logs/setD/run_D*/metrics.csv (+ TensorBoard tb/)

Set C — focus on batch size (batch_size=64 baseline)

All runs fix the batch at 64 and vary lr/gamma/epsilon_decay + timesteps.

Run	Timesteps	lr	gamma	batch	epsilon_decay	Noted behavior (initial pass)
C1	10,000	1e-4	0.99	64	0.80	Baseline for Set C. Fill after run.
C2	20,000	5e-5	0.99	64	0.90	Lower lr + longer exploration; expect smoother but slower gains. Fill after run.
C3	15,000	2.5e-4	0.98	64	0.70	Higher lr + faster decay; watch variance. Fill after run.
C4	10,000	1e-4	0.995	64	0.80	Higher γ; favors long-horizon credit assignment. Fill after run.
C5	50,000	1e-4	0.99	64	0.85	Longest run; buffer may be reduced if RAM tight. Fill after run.

Artifacts:

Models: models/dqn_setC_C{1..5}.zip

Logs: logs/setC/run_C*/metrics.csv (+ TensorBoard tb/)