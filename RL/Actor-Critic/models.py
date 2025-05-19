import torch, torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,1)
        )
    
    def forward(self,x):
        return self.net(x).squeeze(-1)