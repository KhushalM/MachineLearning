# MachineLearning

A collection of foundational reinforcement learning (RL) algorithm implementations.  
This repository serves as a learning playground with clear scaffolds and code for building, experimenting with, and understanding key RL methods.

---

## Implemented Algorithms

- **Q-Learning**  
  Tabular off-policy RL algorithm using value iteration.

- **Deep Q-Networks (DQN)**  
  Neural network–based extension of Q-Learning for function approximation.  
  Includes replay buffer and target network scaffolds.

- **Actor-Critic**  
  Policy-gradient + value-based hybrid method.  
  Separates *actor* (policy) and *critic* (value estimator) for stable training.

---

## Repo Structure

```
MachineLearning/
├── q_learning/       # Q-Learning implementation and utilities
├── dqn/              # Deep Q-Network scaffold and training loop
├── actor_critic/     # Actor-Critic scaffold and modules
├── utils/            # Shared helpers (e.g., replay buffer, env wrappers)
└── README.md
```

---

## Quickstart

1. **Clone the repo**
   ```bash
   git clone https://github.com/KhushalM/MachineLearning.git
   cd MachineLearning
   ```

2. **Set up environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run a training script** (example for Q-learning)
   ```bash
   python q_learning/train.py
   ```

4. **Experiment**  
   Modify hyperparameters, environments, or extend scaffolds into more advanced RL methods.

---

## Goals

- Provide minimal, readable implementations of core RL algorithms.  
- Serve as a starting point for deeper experimentation (e.g., PPO, A2C, GRPO).  
- Build intuition by connecting math → code → results.

---

