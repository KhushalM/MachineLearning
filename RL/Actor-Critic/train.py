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
        self.opt_actor = optim.AdamW(self.actor.parameters(), 3e-4)
        self.opt_critic = optim.AdamW(self.critic.parameters(), 1e-4)
        self.gamma = 0.99
        self.rewards = []
        self.actor_losses = []
    
    def train(self, episodes = 1000, render = False):
        print(f"Starting training for {episodes} episodes...")
        for episode in range(episodes):
            obs, done, traj = self.env.reset()[0], False, []
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32)
                dist = self.actor(obs_t)
                action = dist.sample().item()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                traj.append((obs, action, reward))
                obs = next_obs
    
            obs_batch, act_batch, reward_batch = zip(*traj)
            # Convert obs_batch to numpy array for faster tensor creation
            obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32)
            returns = DiscountCumsum(reward_batch, self.gamma)()
            returns = torch.tensor(returns, dtype=torch.float32)

            # For actor loss (detach critic values)
            values_detached = self.critic(obs_tensor).detach()
            advantage = returns - values_detached

            dists = self.actor(obs_tensor)
            log_probs = dists.log_prob(torch.tensor(act_batch))
            actor_loss = -(log_probs * advantage).mean()
            self.rewards.append(reward)
            self.actor_losses.append(actor_loss.item())
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            # For critic loss (no detach)
            values = self.critic(obs_tensor)
            critic_loss = nn.MSELoss()(values, returns)
            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()

            if episode % 10 == 0:
                print(f"Episode {episode}: actor_loss={actor_loss.item()}, critic_loss={critic_loss.item()}")

            # At the end of each episode, store total reward and actor loss
            self.rewards.append(sum(reward_batch))
            self.actor_losses.append(actor_loss.item())
        
        # Print completion message after final episode
        if episode == episodes - 1:
            print("Training complete.")
    
    def plot_rewards_losses(self):
        # Plot every 100 episodes
        step = 100
        episodes = list(range(0, len(self.rewards), step))
        rewards = [self.rewards[i] for i in episodes]
        actor_losses = [self.actor_losses[i] for i in episodes]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(episodes, rewards, 'g', label="Rewards", marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Rewards over Episodes")
        plt.subplot(1, 2, 2)
        plt.plot(episodes, actor_losses, 'b', label="Actor Loss", marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Actor Loss")
        plt.title("Actor Loss over Episodes")
        plt.show()
    
    