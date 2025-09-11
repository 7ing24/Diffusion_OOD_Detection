import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(("../"))
from diffusion.karras import DiffusionModel
from diffusion.mlps import ScoreNetwork

class BC_Diffusion(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 lr=3e-4,
                 time_embed_dim=16,
                 hidden_dim=256,
                 ):

        self.model = DiffusionModel(
            sigma_data=0.5,
            sigma_min=0.002,
            sigma_max=80,
            device=device,
        )
        self.actor = ScoreNetwork(
            x_dim=action_dim, 
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            cond_dim=state_dim,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=action_dim,
            device=device,
            cond_conditional=True
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device

    def train(self, replay_buffer, iterations, batch_size=100):

        total_loss = 0
        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, reward = replay_buffer.sample(batch_size)

            loss = self.model.diffusion_train_step(self.actor, action, state)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            total_loss += loss.item()

        return total_loss / iterations


    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.model.sample(self.actor, state, n_action_samples=1)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir):
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir):
        self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
