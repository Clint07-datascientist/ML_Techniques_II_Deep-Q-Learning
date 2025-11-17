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

### Member 2: Nicholas Eke

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
