import gymnasium as gym, torch, torch.optim as optim
from models import Actor, Critic
from utils import DiscountCumsum
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

def env(env_id = 'Cartpole-v1', render = False):
    env = gym.make(env_id, render_mode="human" if render else None)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n
    return env, obs_dim, act_dim

class Trainer:
    def __init__(self, env_id = 'Cartpole-v1', episodes = 1000, render = False):
        self.env, self.obs_dim, self.act_dim = env(env_id, render)
        self.actor, self.critic = Actor(self.obs_dim, self.act_dim), Critic(self.obs_dim)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=5e-4)
        self.gamma = 0.99
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []    
    
    def train(self, episodes = 1000, render = False):
        print(f"Starting training for {episodes} episodes...")
        for episode in range(episodes):
            obs, done, total_reward = self.env.reset()[0], False, 0
            obs_list, act_list, td_error_list = [], [], []
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32)
                dist = self.actor(obs_t)
                action = dist.sample().item()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                obs_list.append(obs_t)
                act_list.append(action)

                value = self.critic(obs_t)
                value_next = self.critic(torch.tensor(next_obs, dtype=torch.float32)).detach() if not done else 0

                td_target = reward + self.gamma * value_next
                td_error = (td_target - value)
                td_error_list.append(td_error.detach())

                total_reward += reward
                obs = next_obs
            
            advantage = torch.stack(td_error_list)
            advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            obs_tensor = torch.stack(obs_list)
            act_tensor = torch.tensor(act_list)
            dists = self.actor(obs_tensor)
            log_probs = dists.log_prob(act_tensor)

            actor_loss = -(log_probs * advantages).mean()
        

            self.opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_actor.step()

            values = self.critic(obs_tensor)
            td_target = values + advantages
            critic_loss = ((td_target - values)**2).mean()
            
            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()

            self.rewards.append(total_reward)
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())

            if episode % 10 == 0:
                print(f"Episode {episode}: reward={total_reward}, actor_loss={actor_loss.item()}, critic_loss={critic_loss.item()}")
        
        if episode == episodes - 1:
            print("Training complete.")
    
    def plot_rewards_losses(self):
        step = 100
        episodes = list(range(0, len(self.rewards), step))
        rewards = [self.rewards[i] for i in episodes]
        actor_losses = [self.actor_losses[i] for i in episodes]
        critic_losses = [self.critic_losses[i] for i in episodes]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episodes, rewards, 'g', label="Rewards", marker="o", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Rewards over Episodes")
        plt.subplot(1, 2, 2)
        plt.plot(episodes, actor_losses, 'b', label="Actor Loss", marker="o", color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Actor Loss")
        plt.title("Actor Loss over Episodes")
        plt.subplot(1, 2, 2)
        plt.plot(episodes, critic_losses, 'r', label="Critic Loss", marker="o", color="red")
        plt.xlabel("Episode")
        plt.ylabel("Critic Loss")
        plt.title("Critic Loss over Episodes")
        plt.show()
    
    