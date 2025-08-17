import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import sys
sys.path.append(("../"))
from torch.optim.lr_scheduler import CosineAnnealingLR
from agent.models import TanhGaussianPolicy, Critic

D4RL_SUPPRESS_IMPORT_ERROR=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Diffusion_ood_action_detection(object):
    def __init__(
            self, 
            state_dim, 
            action_dim, 
            max_action, 
            replay_buffer, 
            behavior_model, 
            diffusion_model,
            Q_min, 
            action_n_levels,
            threshold, 
            discount=0.99, 
            tau=0.005, 
            policy_freq=2,
            target_update_freq=2, 
            schedule=True, 
            beta=0.001,
            alpha_learning_rate=1e-4
        ):

        self.actor = TanhGaussianPolicy(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.alpha = self.log_alpha.exp().detach()

        self.replay_buffer = replay_buffer
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.target_update_freq = target_update_freq
        self.Q_min = Q_min
        self.behavior_model = behavior_model
        self.diffusion_model = diffusion_model
        self.action_n_levels = action_n_levels
        self.threshold = threshold
        self.beta = beta
        self.schedule = schedule
        if schedule:
            self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(int(1e6) / policy_freq))
        self.total_it = 0


    def compute_error(self, states, actions, batch_size=256):
        recon_errors = []
        with torch.no_grad():
            for i in range(0, len(actions), batch_size):
                batch_states = states[i:i + batch_size]
                batch_actions = actions[i:i + batch_size]
                t = self.diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
                noise = torch.randn_like(batch_actions)

                t_expanded = t.view(-1, *([1] * (batch_actions.ndim - 1)))
                noisy_actions = batch_actions + noise * t_expanded

                c_skip, c_out, c_in = [
                    x.view(-1, *([1] * (batch_actions.ndim - 1))) 
                    for x in self.diffusion_model.get_diffusion_scalings(t)
                ]

                model_input = noisy_actions * c_in
                model_output = self.behavior_model(model_input, batch_states, torch.log(t)/4)
                denoised_actions = c_skip * noisy_actions + c_out * model_output

                error = torch.norm(denoised_actions - batch_actions, dim=1)
                recon_errors.append(error)
        return torch.cat(recon_errors)
    

    def compute_multi_errors(self, states, actions, batch_size=256):
        recon_errors = []
        with torch.no_grad():
            for i in range(0, len(actions), batch_size):
                batch_states = states[i:i + batch_size]
                batch_actions = actions[i:i + batch_size]
                batch_errors = torch.zeros(len(batch_actions), device=actions.device)

                for _ in range(self.action_n_levels):
                    t = self.diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
                    # t = torch.full((len(batch_actions),), 0.2, device=actions.device)
                    noise = torch.randn_like(batch_actions)

                    t_expanded = t.view(-1, *([1] * (batch_actions.ndim - 1)))
                    noisy_actions = batch_actions + noise * t_expanded

                    c_skip, c_out, c_in = [
                        x.view(-1, *([1] * (batch_actions.ndim - 1))) 
                        for x in self.diffusion_model.get_diffusion_scalings(t)
                    ]

                    model_input = noisy_actions * c_in
                    model_output = self.behavior_model(model_input, batch_states, torch.log(t) / 4)
                    denoised_actions = c_skip * noisy_actions + c_out * model_output

                    error = torch.norm(denoised_actions - batch_actions, dim=1)
                    batch_errors += error

                batch_errors /= self.action_n_levels
                recon_errors.append(batch_errors)

        return torch.cat(recon_errors)


    def select_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action = self.actor(state)[0].cpu().data.numpy().flatten()
            self.actor.train()
            return action
        

    def alpha_loss(self, state):
        with torch.no_grad():
            _, log_prob = self.actor(state, need_log_prob=True)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        return alpha_loss
    

    def actor_loss(self, state):
        pi, pi_log_prob = self.actor(state, need_log_prob=True)
        pi_Q1, pi_Q2, pi_Q3, pi_Q4 = self.critic(state, pi)
        pi_Q = torch.cat([pi_Q1, pi_Q2, pi_Q3, pi_Q4], dim=1)
        pi_Q, _ = torch.min(pi_Q, dim=1)
        if pi_Q.mean().item() > 5e4:
            exit(0)
        actor_loss = (self.alpha * pi_log_prob - pi_Q).mean()
        return actor_loss
    

    def critic_loss(self, state, action, reward, next_state, not_done):
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor_target(next_state, need_log_prob=True)
            next_Q1, next_Q2, next_Q3, next_Q4 = self.critic_target(next_state, next_action)  
            next_Q = torch.min(torch.min(next_Q1, next_Q2), torch.min(next_Q3, next_Q4))
            next_Q = next_Q - self.alpha * next_action_log_prob
            target_Q = reward + not_done * self.discount * next_Q
            
        current_Q1, current_Q2, current_Q3, current_Q4 = self.critic(state, action)
        current_Q = torch.cat([current_Q1, current_Q2, current_Q3, current_Q4], dim=1)

        # Bellman loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) + \
                      F.mse_loss(current_Q3, target_Q) + F.mse_loss(current_Q4, target_Q)
        
        # Diffusion-based reconstruction error
        pi, _ = self.actor(state)
        error = self.compute_multi_errors(state, pi)
        # Select OOD actions
        ood_mask = (error > self.threshold).float().unsqueeze(1)
        pi_Q1, pi_Q2, pi_Q3, pi_Q4 = self.critic(state, pi)
        pi_Q = torch.cat([pi_Q1, pi_Q2, pi_Q3, pi_Q4], dim=1)
        qmin = (self.Q_min * torch.ones_like(pi_Q)).detach()

        reg_loss = self.beta * (((pi_Q - qmin) ** 2) * ood_mask).mean()

        critic_loss += reg_loss

        # with torch.no_grad():
        #     print("action_error.mean():", error.mean().item())
        #     print("negative_ood_mask.mean():", ood_mask.mean().item())
        #     print("pi_Q.mean():", pi_Q.mean().item())
        #     print("reg_loss:", reg_loss.item())
        #     print("critic_loss:", critic_loss)

        return critic_loss, reg_loss, current_Q, qmin


    def train(self, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        # Alpha update
        alpha_loss = self.alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach()

        # Actor update
        if self.total_it % self.policy_freq == 0:
            actor_loss = self.actor_loss(state)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if self.schedule:
                self.actor_lr_schedule.step()

        # Critic update
        critic_loss, reg_loss, current_Q, qmin = self.critic_loss(state, action, reward, next_state, not_done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Target networks update
        if self.total_it % self.target_update_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.total_it % 10000 == 0:
            with torch.no_grad():
                pi = self.actor(state)[0]
                unif = torch.distributions.uniform.Uniform(-1, 1).sample((batch_size, self.action_dim)).to(device)
                anoise1 = (action + torch.randn_like(action) * 0.1).clamp(-self.max_action, self.max_action)
                anoise5 = (action + torch.randn_like(action) * 0.5).clamp(-self.max_action, self.max_action)
                pinoise1 = (pi + torch.randn_like(action) * 0.1).clamp(-self.max_action, self.max_action)
                pinoise5 = (pi + torch.randn_like(action) * 0.5).clamp(-self.max_action, self.max_action)
                Q_pi1, Q_pi2, Q_pi3, Q_pi4 = self.critic(state, pi)
                Q_pi = torch.cat([Q_pi1, Q_pi2, Q_pi3, Q_pi4],dim=1)
                Q_unif1, Q_unif2, Q_unif3, Q_unif4 = self.critic(state, unif)
                Q_unif = torch.cat([Q_unif1, Q_unif2, Q_unif3, Q_unif4],dim=1)
                Q_anoise1_1, Q_anoise1_2, Q_anoise1_3, Q_anoise1_4 = self.critic(state, anoise1)
                Q_anoise1 = torch.cat([Q_anoise1_1, Q_anoise1_2, Q_anoise1_3, Q_anoise1_4],dim=1)
                Q_anoise5_1, Q_anoise5_2, Q_anoise5_3, Q_anoise5_4 = self.critic(state, anoise5)
                Q_anoise5 = torch.cat([Q_anoise5_1, Q_anoise5_2, Q_anoise5_3, Q_anoise5_4],dim=1)
                Q_pinoise1_1, Q_pinoise1_2, Q_pinoise1_3, Q_pinoise1_4 = self.critic(state, pinoise1)
                Q_pinoise1 = torch.cat([Q_pinoise1_1, Q_pinoise1_2, Q_pinoise1_3, Q_pinoise1_4],dim=1)
                Q_pinoise5_1, Q_pinoise5_2, Q_pinoise5_3, Q_pinoise5_4 = self.critic(state, pinoise5)
                Q_pinoise5 = torch.cat([Q_pinoise5_1, Q_pinoise5_2, Q_pinoise5_3, Q_pinoise5_4],dim=1)

                wandb.log({"train/critic_loss": critic_loss.item(),
                            "train/reg_loss": reg_loss.item(),
                            "train/actor_loss": actor_loss.item(),
                            'Q/Qmin': qmin.mean().item(),
                            'Q/pi': Q_pi.mean().item(),
                            'Q/a': current_Q.mean().item(),
                            'Q/unif': Q_unif.mean().item(),
                            'Q/anoise0.1': Q_anoise1.mean().item(),
                            'Q/anoise0.5': Q_anoise5.mean().item(),
                            'Q/pinoise0.1': Q_pinoise1.mean().item(),
                            'Q/pinoise0.5': Q_pinoise5.mean().item()
                            }, step=self.total_it)
