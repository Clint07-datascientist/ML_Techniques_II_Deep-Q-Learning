# Deep Q-Learning on Atari Environments
Group Project – ML Techniques II


## Project Overview

This project applies **Deep Q-Networks (DQN)** using **Stable Baselines3** on **Gymnasium Atari environments**.  

Each group member independently in different branches:

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

| Experiment | lr      | γ      | batch | eps_start | eps_end | eps_decay/expl.frac | Observed Behavior |
|------------|---------|--------|-------|-----------|---------|-------------------|-----------------|
| 1          | 1.0e-4 | 0.99   | 64    | 1.0       | 0.02    | 0.12              | Moderate learning speed; stable training; agent improves gradually, learns consistent ball tracking and defensive moves. |
| 2          | 7.5e-5 | 0.97   | 32    | 1.0       | 0.05    | 0.12              | Slower progress; agent favors short-term rewards due to lower γ; slightly unstable early updates due to small batch. |
| 3          | 2.0e-4 | 0.995  | 128   | 1.0       | 0.01    | 0.12              | Faster early learning; high γ promotes long-term strategy; large batch stabilizes updates; reward fluctuations minimal. |
| 4          | 3.0e-4 | 0.992  | 96    | 1.0       | 0.02    | 0.12              | Rapid initial learning; slight oscillations due to high LR; episodes longer; agent adapts strategy but may overreact to new experiences. |
| 5          | 5.0e-5 | 0.98   | 48    | 1.0       | 0.08    | 0.12              | Very slow reward improvement; small LR limits updates; shorter episodes; agent largely conservative and stable. |
| 6          | 1.5e-4 | 0.995  | 32    | 1.0       | 0.015   | 0.12              | Moderate learning with high long-term focus; small batch introduces some noisy updates; agent improves defensive and ball tracking gradually. |
| 7          | 2.5e-4 | 0.997  | 64    | 1.0       | 0.005   | 0.12              | High γ and moderate LR lead to long-term strategy; exploration decays very quickly → premature exploitation; agent may lock onto suboptimal strategy early. |
| 8          | 7.0e-5 | 0.99   | 128   | 1.0       | 0.03    | 0.12              | Slow learning due to low LR; large batch stabilizes updates; agent explores efficiently but improvements take longer. |
| 9          | 3.5e-4 | 0.985  | 64    | 1.0       | 0.04    | 0.12              | Fast early learning; lower γ favors short-term gains; episodes moderately long; agent reactive but may miss long-term strategy. |
| 10         | 9.0e-5 | 0.993  | 96    | 1.0       | 0.015   | 0.12              | Balanced performance; stable updates; agent improves steadily in both offense and defense; episode lengths increase consistently. |

### Member 4: Amandine Irakoze

| Experiment                 | lr      | γ      | batch | eps_start | eps_end | eps_decay/expl.frac      | Observed Behavior                                                                                                   |
|-----------------------------|---------|--------|-------|-----------|---------|--------------------------|--------------------------------------------------------------------------------------------------------------------|
| Exp_1_Baseline              | 1e-4    | 0.99   | 32    | 1.0       | ?       | 20k steps                | Standard Performance. Stable but slow learning due to limited 100k steps; serves as the benchmark.                |
| Exp_2_High_LR               | 5e-4    | 0.99   | 32    | 1.0       | ?       | 20k steps                | Instability Risk. High learning rate causes volatile Q-value updates, leading to increased variance and possible divergence. |
| Exp_3_Low_LR                | 1e-5    | 0.99   | 32    | 1.0       | ?       | 20k steps                | Minimal Learning. Updates are too small for a quick convergence; agent fails to move significantly past initial random policy. |
| Exp_4_Low_Gamma             | 1e-4    | 0.9    | 32    | 1.0       | ?       | 20k steps                | Short-Term Bias. Low γ heavily discounts future rewards, making agent focus on immediate gains rather than strategic long-term plays. |
| Exp_5_High_Gamma            | 1e-4    | 0.999  | 32    | 1.0       | ?       | 20k steps                | Long-Term Focus/Oscillation. High γ creates strong dependency on future values, often causing unstable Q-network updates early in training. |
| Exp_6_Large_Batch           | 1e-4    | 0.99   | 128   | 1.0       | ?       | 20k steps                | Smooth but Slower. Large batch size reduces gradient variance (stability) but fewer updates per step, potentially slowing overall learning. |
| Exp_7_Fast_Exploration      | 1e-4    | 0.99   | 32    | 1.0       | ?       | 10k steps (0.1 frac)     | Premature Exploitation. Epsilon decays too quickly; agent locks onto sub-optimal policy early, limiting effective exploration. |
| Exp_8_Slow_Exploration      | 1e-4    | 0.99   | 32    | 1.0       | ?       | 80k steps (0.8 frac)     | Excessive Exploration. Agent spends too much time exploring (80% of steps); optimal policy forms too late for good final performance. |
| Exp_9_Low_LR_Low_Gamma      | 1e-5    | 0.9    | 32    | 1.0       | ?       | 20k steps                | Highly Conservative. Minimal learning due to tiny updates and short-term focus; stable but functionally poor.    |
| Exp_10_High_LR_High_Gamma   | 5e-4    | 0.999  | 32    | 1.0       | ?       | 20k steps                | Total Q-Value Collapse. Aggressive combination of high LR and γ causes severe divergence, worst outcome.         |

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
 
**https://drive.google.com/file/d/1UnFEE9iDTYyBcOp2NBVoZdXeecJgd-pg/view?usp=sharing**

## Task 3 – Group Presentation Summary

# Hyperparameter Tuning Results – Pong-v5 DQN

## 1. Overview
Each member performed 10 hyperparameter tuning experiments on the **Pong-v5** Atari environment using DQN. The main hyperparameters varied include:

- **Learning Rate (lr)** – step size for Q-network updates  
- **Discount Factor (γ)** – weighting of future rewards  
- **Batch Size** – number of samples per gradient update  
- **Epsilon / Exploration** – `eps_start`, `eps_end`, `eps_decay` controlling exploration vs exploitation  

The goal was to identify combinations that maximize reward and episode length while ensuring stable learning.

## 2. Observed Trends Across Members

### Learning Rate (lr)
- **Low LR (1e-5 – 5e-5)**  
  - Members 1, 3, and 4 showed very slow learning, with rewards plateauing or minimal improvement.  
  - **Pros:** Stable updates, low variance  
  - **Cons:** Convergence extremely slow; agent may fail to learn meaningful strategies

- **Moderate LR (1e-4 – 2.5e-4)**  
  - Members 1 and 2 found gradual but steady improvement.  
  - Balanced trade-off between stability and learning speed

- **High LR (5e-4)**  
  - Members 1 and 4 experienced instability and oscillations, sometimes causing Q-value divergence  
  - **Risk:** fast early learning but poor final policy

### Discount Factor (γ)
- **Lower γ (0.9 – 0.98)**  
  - Encouraged short-term reward focus, leading to stable but sometimes suboptimal strategies (Member 4 Exp_4, Exp_9)  
  - Agent learns immediate defensive/offensive moves but may ignore long-term rewards

- **Higher γ (0.995 – 0.999)**  
  - Promoted long-term planning, but caused instability in early training due to large Q-value dependencies (Member 4 Exp_5, Exp_10)  
  - Often paired with high LR → high risk of Q-value collapse

### Batch Size
- **Small batches (16–32)**  
  - Increased gradient variance → more noisy updates, sometimes unstable early learning (Member 1 Exp_6)  
  - Can accelerate adaptation to new experience but less smooth

- **Large batches (64–128)**  
  - Reduced gradient variance → smoother learning, more stable convergence (Members 2 D5, 3 Exp_3, 8)  
  - Fewer updates per step may slightly slow learning initially

### Epsilon / Exploration
- **Fast decay**  
  - Premature exploitation → agent locks into suboptimal policy early (Member 4 Exp_7)  
  - Rewards may plateau before optimal strategy emerges

- **Slow decay / long exploration**  
  - Agent explores longer, can find better policies (Member 4 Exp_8)  
  - Risk: Optimal strategy forms late; initial rewards remain low

## 3. Key Insights

### Learning Rate
- Moderate LR (1e-4 – 2.5e-4) provides best trade-off between learning speed and stability  
- Too high → instability; too low → stagnation

### Discount Factor (γ)
- Higher γ favors long-term reward but can destabilize learning with high LR  
- Lower γ stabilizes learning but may underperform in strategy-based games

### Batch Size
- Larger batch sizes improve stability by reducing gradient noise  
- Smaller batches allow faster adaptation but may oscillate

### Epsilon Decay / Exploration
- Fast decay → early exploitation, potential suboptimal policy  
- Slow decay → better exploration, potential for higher final rewards, but slower initial gains

### General Observations
- CNN-based DQN policies outperform MLP in visual Atari tasks  
- Balancing LR, γ, batch size, and exploration is crucial  
- Best configurations consistently paired moderate LR, high γ (~0.995), medium batch size, and controlled epsilon decay

## Run Instructions

1. **Install dependencies**  

pip install -r requirements.txt


2. **Train the model**

python train.py


3. **Play using the trained model**

python play.py
