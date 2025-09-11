import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class V_Network(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(V_Network, self).__init__()

        self.V = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.V(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.Q3 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.Q4 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.V1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.V2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        h = torch.cat([state, action], dim=1)
        q1 = self.Q1(h)
        q2 = self.Q2(h)
        q3 = self.Q3(h)
        q4 = self.Q4(h)
        return q1, q2, q3, q4
    
    def q_min(self, state, action):
        q1, q2, q3, q4 = self.forward(state, action)
        return torch.min(torch.min(q1, q2), torch.min(q3, q4))
    
    def v(self, state):
        v1 = self.V1(state)
        v2 = self.V2(state)
        return v1, v2
    
    def v_min(self, state):
        v1, v2 = self.v(state)
        return torch.min(v1, v2)
    

# deterministic policy
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))
    

# Gaussian policy
class TanhGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(TanhGaussianPolicy, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, state, deterministic = False, need_log_prob = False):
        if torch.isnan(state).any() or torch.isinf(state).any():
            print("Warning: State contains NaN/Inf!")
            
        hidden = self.actor(state)

        if torch.isnan(hidden).any():
            print("Warning: hidden contains NaN values!")

        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        if torch.isnan(mu).any():
            print("Warning: mu contains NaN values!")
        if torch.isnan(log_sigma).any():
            print("Warning: log_sigma contains NaN values!")

        log_sigma = torch.clamp(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))
        
        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            # log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)
            epsilon = 1e-5
            correction = torch.clamp(1 - tanh_action.pow(2), min=epsilon)
            log_prob = log_prob - torch.log(correction).sum(axis=-1)
            log_prob = log_prob.unsqueeze(-1)

        return tanh_action * self.max_action, log_prob
    
    @torch.no_grad()
    def act(self, state, device):
        deterministic = not self.training
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self(state, deterministic=deterministic)[0].cpu().data.numpy().flatten()
        return action
        
        