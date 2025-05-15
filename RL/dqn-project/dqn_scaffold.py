from __future__ import annotations
import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time




# ---------------------
# 1. Replay Buffer
# ---------------------

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: Tuple[int, ...]) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
    
    def push(self, state: np.ndarray, action:int, reward: float, next_state: np.ndarray, done: bool) -> None:
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(done, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)

# ---------------------
# 2. Q-Network
# ---------------------

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes: Tuple[int, ...] = (128, 128, 64, 64)) -> None:
        super().__init__()
        trunk = []
        in_dim = state_size
        for h in hidden_sizes:
            trunk.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.trunk = nn.Sequential(*trunk)
        self.value = nn.Linear(in_dim, 1)
        self.advantage = nn.Linear(in_dim, action_size)    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        value = self.value(x)
        advantage = self.advantage(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# ---------------------
# 3. DQN Agent
# ---------------------

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 10_000
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_final: float = 0.01
    epsilon_decay: float = 10_000

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, config: DQNConfig) -> None:
        self.action_size = action_size
        self.config = config
        self.online_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = optim.AdamW(self.online_network.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.buffer_capacity, (state_size,))
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online_network.to(self.device)
        self.target_network.to(self.device)
    
    def select_action(self, state: np.ndarray) -> int:
        eps_threshold = self.config.epsilon_final + (self.config.epsilon_start - self.config.epsilon_final) * np.exp(-self.steps_done / self.config.epsilon_decay)        
        self.steps_done += 1
        if random.random() < eps_threshold:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.online_network(state_t)
                return int(torch.argmax(q_values).item())

    def optimize_model(self) -> None:
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        states, actions, rewards, next_state, done = self.replay_buffer.sample(self.config.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        next_states = next_state.to(self.device)
        done = done.to(self.device)
        
        q_values = self.online_network(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            targets = rewards + self.config.gamma*max_next_q*(~done)
        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def maybe_update_target(self):
        if self.steps_done % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

# ---------------------
# 4. Training Loop
# ---------------------
def train(env_id: str, episodes: int, cfg: DQNConfig, render: bool = False, fps: float = 30.0) -> None:
    env = gym.make(env_id, render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, cfg)
            
    rewards = []
    losses = []
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            loss = agent.optimize_model()
            agent.maybe_update_target()

            if render:
                env.render()
                time.sleep(1.0/fps)
            
        rewards.append(ep_reward)
        losses.append(loss)
        print(f"Episode {ep+1}: reward={ep_reward}, loss={loss}")
        #Early Stopping
        if env_id == 'CartPole-v1' and len(rewards) >= 100 and np.mean(rewards[-100:]) >= 475:
            print(f"Environment {env_id} solved in {ep+1} episodes")
            break
    env.close()
    plot_rewards_losses(rewards, losses)

#-------------------
# 6. Plotting
#-------------------
def plot_rewards_losses(rewards, losses):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Rewards", marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards over Episodes")
    # plt.subplot(1, 2, 2)
    # plt.plot(losses, label="Loss", marker="o")
    # plt.xlabel("Episode")
    # plt.ylabel("Loss")
    # plt.title("Loss over Episodes")
    # plt.tight_layout()
    plt.show()
            
# ---------------------
# 5. Main
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training")
    parser.add_argument("--env_id", type=str, default="CartPole-v1", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--cfg", type=str, default="DQNConfig", help="Configuration class")
    parser.add_argument("--render", type=bool, default=False, help="Render environment")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = DQNConfig()
    train(args.env_id, args.episodes, config, args.render, args.fps)
            

            
    


    