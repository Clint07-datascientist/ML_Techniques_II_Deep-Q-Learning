# Deep Q-Learning on Atari Environments
Group Project – ML Techniques II


## Project Overview

This project applies **Deep Q-Networks (DQN)** using **Stable Baselines3** on **Gymnasium Atari environments**.  

Each group member independently:

- Trained a DQN agent using `train.py`  
- Performed **10 hyperparameter tuning experiments**  
- Evaluated the trained agent using `play.py`  
- Documented results in an experiment table  
- Merged their work through GitHub using separate branches  

The final submission includes:

- **Training script** (`train.py`)  
- **Evaluation script** (`play.py`)  
- **Trained policy file** (`dqn_model.zip`)  
- **Hyperparameter tuning tables**  
- **Gameplay demonstration video**  
- **Group presentation summary**

## Environment Used

We used the following Gymnasium Atari environment:

**`ALE/Pong-v5`**

Pong is ideal for DQN because it is:

- Visual (RGB image input)  
- Reward-driven  
- Requires temporal decision-making  
- Standard benchmark in RL research
- RAM friendly

## Repository Structure

ML_Techniques_II_Deep-Q-Learning/
├── train.py            # Script to train the DQN agent
├── play.py             # Script to load and play with the trained agent
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── logs/
│   ├── best_model/     # Folder containing the best model checkpoint
│   └── checkpoints/    # Folder containing intermediate checkpoints
└── dqn_model.zip       # Trained DQN model

## Task 1 – Training the Agent (`train.py`)

The training script:

- Creates and wraps an Atari environment (Gymnasium + ALE + Monitor)  
- Defines a DQN agent using Stable Baselines3 (`MlpPolicy` and `CnnPolicy`)  
- Logs training progress (rewards, episode lengths, TensorBoard)  
- Saves the best model automatically (`dqn_baseline.zip`)  

---

## Hyperparameter Tuning Requirements

Each group member performed **10 independent experiments**, adjusting:

- **Learning Rate (`lr`)**  
- **Gamma (`γ`)**  
- **Batch Size (`batch_size`)**  
- **Epsilon parameters (`epsilon_start`, `epsilon_end`, `epsilon_decay`)**  

## Hyperparameter Tuning Results

### Member 1: Nicolle Marizani

| Experiment | lr      | γ     | batch | eps_start | eps_end | eps_decay/expl.frac | Observed Behaviour |
|------------|---------|-------|-------|-----------|---------|-------------------|------------------|
| exp1       | 1e-4    | 0.99  | 32    | 1.0       | 0.05    | 0.1               | Slow but steady learning; reward improved from -21 → -20.2; episode length increased from 764 → 861; agent learned better ball tracking and defensive play. |
| exp2       | 5e-5    | 0.98  | 64    | 1.0       | 0.02    | 0.1               | Very slow reward improvement; agent plateaued around -21 for most timesteps; episode length ~764–764; exploration low, policy highly greedy. |
| exp3       | 5e-4    | 0.995 | 32    | 1.0       | 0.01    | 0.1               | Faster early learning but unstable; rewards fluctuated around -21 → -20.8; episode lengths 764–820; agent slightly more exploratory early on, better defense mid-training. |
| exp4       | 1e-5    | 0.95  | 128   | 1.0       | 0.1     | 0.1               | Learning very slow due to tiny LR; rewards remained ~-21; episodes short (~764); agent struggled to learn meaningful policy. |
| exp5       | 2.5e-4  | 0.99  | 64    | 1.0       | 0.02    | 0.1               | Gradual improvement; reward ~-21 → -20.8; episode length ~764 → 820; policy started showing consistent returns but still weak offense. |
| exp6       | 1e-4    | 0.98  | 16    | 1.0       | 0.05    | 0.1               | Smaller batch size slowed learning; reward improved very slowly; episode length slightly increased; agent survived longer but policy noisy. |
| exp7       | 5e-5    | 0.995 | 32    | 1.0       | 0.01    | 0.1               | Early instability due to low LR; reward trend mild (-21 → -20.8); episode length improved from 764 → 820; agent learned better defensive moves. |
| exp8       | 5e-4    | 0.99  | 128   | 1.0       | 0.02    | 0.1               | Large batch accelerated stability; reward fluctuated slightly; episodes longer (~764–820); agent learned to survive but offensive policy weak. |
| exp9       | 2.5e-4  | 0.98  | 32    | 1.0       | 0.05    | 0.1               | Reward improved slowly (-21 → -20.7); episode length increased from 764 → 849; agent learned better tracking; stable greedy policy near the end. |
| exp10      | 1e-4    | 0.995 | 64    | 1.0       | 0.01    | 0.1               | Best performing: reward -21 → -20.2; episode length 764 → 861; agent learned most consistent ball tracking and defensive play; exploration decayed fully to greedy policy. |

### Member 2: Nicholas Eke

| **Run ID** | **Timesteps** | **Learning Rate** | **Gamma (γ)** | **Batch Size** | **Target Reward** | **Notes / Outcome** |
|-----------|---------------|-----------------|---------------|----------------|-----------------|--------------------|
| **D1**    | 10,000        | 1e-4            | 0.99          | 32             | 0.80            | Trained successfully. Reward mean ~2.0 by 8–10k steps; episode length ~232–244; exploration decayed to 0.05 near end. Model saved at `models/dqn_setD_D1.zip`. |
| **D2**    | 20,000        | 5e-5            | 0.99          | 64             | 0.80            | Completed (after reducing buffer/learning_starts to avoid OOM). Pending final reward trend — check `logs/setD/run_D2/metrics.csv`. |
| **D3**    | 20,000        | 1e-4            | 0.995         | 32             | 0.90            | Higher γ slows early improvement but may improve long-term reward. Pending results after full run. |
| **D4**    | 25,000        | 2.5e-4          | 0.98          | 32             | 0.70            | Higher LR may speed early learning but risks instability. Results pending. |
| **D5**    | 30,000        | 1e-4            | 0.99          | 128            | 0.85            | Larger batch expected to stabilize learning and improve long-term return. Pending run results. |
| **C1**    | 10,000        | 1e-4            | 0.99          | 64             | 0.80            | Baseline for Set C. Initial pass; fill after run. |
| **C2**    | 20,000        | 5e-5            | 0.99          | 64             | 0.90            | Lower LR + longer exploration; expect smoother but slower gains. Fill after run. |
| **C3**    | 15,000        | 2.5e-4          | 0.98          | 64             | 0.70            | Higher LR + faster decay; watch variance. Fill after run. |
| **C4**    | 10,000        | 1e-4            | 0.995         | 64             | 0.80            | Higher γ; favors long-horizon credit assignment. Fill after run. |
| **C5**    | 50,000        | 1e-4            | 0.99          | 64             | 0.85            | Longest run; buffer may be reduced if RAM tight. Fill after run. |

### Member 3: Clinton Pikita

| Experiment | lr | γ | batch | eps_start | eps_end | eps_decay/expl.frac | Observed Behavior |
|-----------|----|----|-------|-----------|---------|-------------------|-----------------|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |
| 8 | | | | | | | |
| 9 | | | | | | | |
| 10 | | | | | | | |

### Member 4: Amandine Irakoze

| Experiment | lr | γ | batch | eps_start | eps_end | eps_decay/expl.frac | Observed Behavior |
|-----------|----|----|-------|-----------|---------|-------------------|-----------------|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |
| 8 | | | | | | | |
| 9 | | | | | | | |
| 10 | | | | | | | |


## Task 2 – Evaluating & Playing the Agent (`play.py`)

The evaluation script:

- Loads the **best trained model**  
- Initializes the same Atari environment  
- Uses a **GreedyQPolicy** (no exploration)  
- Runs several episodes  
- Renders the game in real-time  
- Saves gameplay video for the presentation

## Gameplay Demonstration

A short video demonstrating the trained agent playing Pong:
 
**insert GitHub video link**

## Task 3 – Group Presentation Summary

Each member presented:

1. Their Atari environment (`ALE/Pong-v5`)  
2. Their 10 hyperparameter experiments  
   - What changed?  
   - Why it affected performance?  
   - Worst combination?  
   - Best performing configuration?  
3. Key insights learned  
   - Lower learning rate stabilized training  
   - Higher epsilon decay improved exploration  
   - Batch size affected variance  
   - CNNPolicy outperformed MLP  
4. Final best model per member  
5. Short gameplay demo showing the agent’s performance

## Run Instructions

1. **Install dependencies**  

pip install -r requirements.txt


2. **Train the model**

python train.py


3. **Play using the trained model**

python play.py
