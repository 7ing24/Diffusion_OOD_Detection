import os
import torch
import numpy as np
import math
from torch.distributions import Normal
import argparse
import matplotlib.pyplot as plt

from toy_helpers import Data_Sampler

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--data", default="9-gaussians", type=str)  # 4-gaussians, 9-gaussians, ring
args = parser.parse_args()

seed = args.seed


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def generate_ring(num, device='cpu'):
    num_modes = 2
    each_num = num // num_modes
    std = 0.02
    radii = [0.4, 0.8]
    data = []

    for r in radii:
        angles = torch.linspace(0, 2 * np.pi, each_num)
        x = r * torch.cos(angles)
        y = r * torch.sin(angles)
        circle = torch.stack([x, y], dim=1)
        circle += torch.randn_like(circle) * std
        circle = circle.clip(-1.0, 1.0)
        data.append(circle)

    data = torch.cat(data, dim=0)
    action = data.to(device)
    state = torch.zeros_like(action)
    reward = torch.zeros((num, 1), device=device)

    return Data_Sampler(state, action, reward, device)


def generate_4_gaussians(num, device='cpu'):
    each_num = int(num / 4)
    pos = 1.8
    std = 0.05
    left_up_conor = Normal(torch.tensor([-pos, pos]), torch.tensor([std, std]))
    left_bottom_conor = Normal(torch.tensor([-pos, -pos]), torch.tensor([std, std]))
    right_up_conor = Normal(torch.tensor([pos, pos]), torch.tensor([std, std]))
    right_bottom_conor = Normal(torch.tensor([pos, -pos]), torch.tensor([std, std]))

    data = torch.cat([
        left_up_conor.sample((each_num,)).clip(-2.0, 2.0),
        left_bottom_conor.sample((each_num,)).clip(-2.0, 2.0),
        right_up_conor.sample((each_num,)).clip(-2.0, 2.0),
        right_bottom_conor.sample((each_num,)).clip(-2.0, 2.0),
    ], dim=0)

    action = data
    state = torch.zeros_like(action)
    reward = torch.zeros((num, 1))
    return Data_Sampler(state, action, reward, device)


def generate_9_gaussians(num, device='cpu'):
    num_modes = 9
    each_num = num // num_modes
    std = 0.05
    grid_pos = torch.linspace(-1.8, 1.8, 3)
    centers = [(x, y) for x in grid_pos for y in grid_pos]

    data = []
    for cx, cy in centers:
        dist = Normal(torch.tensor([cx, cy]), torch.tensor([std, std]))
        samples = dist.sample((each_num,)).clip(-2.0, 2.0)
        data.append(samples)

    data = torch.cat(data, dim=0)
    action = data.to(device)
    state = torch.zeros_like(action)
    reward = torch.zeros((num, 1), device=device)

    return Data_Sampler(state, action, reward, device)


def compute_errors(model, diffusion_model, states, actions, n_levels, batch_size=256):
    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(actions), batch_size):
            batch_states = states[i:i + batch_size]
            batch_actions = actions[i:i + batch_size]
            batch_errors = torch.zeros(len(batch_actions), device=actions.device)

            for _ in range(n_levels):
                # t = diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
                t = torch.full((len(batch_actions),), fill_value=0.2, device=device)
                noise = torch.randn_like(batch_actions)

                t_expanded = t.view(-1, *([1] * (batch_actions.ndim - 1)))
                noisy_actions = batch_actions + noise * t_expanded

                c_skip, c_out, c_in = [
                    x.view(-1, *([1] * (batch_actions.ndim - 1))) 
                    for x in diffusion_model.get_diffusion_scalings(t)
                ]

                model_input = noisy_actions * c_in
                model_output = model(model_input, batch_states, torch.log(t) / 4)
                denoised_actions = c_skip * noisy_actions + c_out * model_output

                error = torch.norm(denoised_actions - batch_actions, dim=1)
                batch_errors += error

            batch_errors /= n_levels
            recon_errors.append(batch_errors.cpu())

    return torch.cat(recon_errors).numpy()


torch.manual_seed(seed)
np.random.seed(seed)

device = 'cuda:0'
num_data = int(10000)

if args.data == '4-gaussians':
    data_sampler = generate_4_gaussians(num_data, device)
    bc_diffusion_model_path = 'models/diffusion/bc_diffusion_model_4_gaussians.pth'
elif args.data == '9-gaussians':
    data_sampler = generate_9_gaussians(num_data, device)
    bc_diffusion_model_path = 'models/diffusion/bc_diffusion_model_9_gaussians.pth'
elif args.data == 'ring':
    data_sampler = generate_ring(num_data, device)
    bc_diffusion_model_path = 'models/diffusion/bc_diffusion_model_ring.pth'

state_dim = 2
action_dim = 2
hidden_dim = 128
max_action = 2.0

discount = 0.99
tau = 0.005
lr = 1e-3

num_epochs = 1000
batch_size = 100
iterations = int(num_data / batch_size)

img_dir = 'toy_imgs/bc_diffusion/recon_error'
os.makedirs(img_dir, exist_ok=True)
fig, axs = plt.subplots(1, 4, figsize=(5.5 * 4, 5))
axis_lim = 2.2

# Plot the ground truth
num_eval = 1000
_, action_samples, _ = data_sampler.sample(num_eval)
action_samples = action_samples.cpu().numpy()
axs[0].scatter(action_samples[:, 0], action_samples[:, 1], alpha=0.3)
axs[0].set_xlim(-axis_lim, axis_lim)
axs[0].set_ylim(-axis_lim, axis_lim)
axs[0].set_xlabel('x', fontsize=20)
axs[0].set_ylabel('y', fontsize=20)
axs[0].set_title('Ground Truth', fontsize=25)

# Plot Diffusion BC
from bc_diffusion import BC_Diffusion as Diffusion_Agent
diffusion_agent = Diffusion_Agent(state_dim=state_dim,
                                  action_dim=action_dim,
                                  max_action=max_action,
                                  device=device,
                                  discount=discount,
                                  tau=tau)

if os.path.exists(bc_diffusion_model_path):
    print(f'Load model from {bc_diffusion_model_path}')
    diffusion_agent.actor.load_state_dict(torch.load(bc_diffusion_model_path))
else:
    print("Training Diffusion BC")
    for i in range(num_epochs):
        loss = diffusion_agent.train(data_sampler,
                                     iterations=iterations,
                                     batch_size=batch_size)
        if i % 100 == 0:
            print(f'Epoch: {i}, Loss: {loss}')
    torch.save(diffusion_agent.actor.state_dict(), bc_diffusion_model_path)

new_state = torch.zeros((num_eval, 2), device=device)
new_action = diffusion_agent.model.sample(diffusion_agent.actor, new_state, n_action_samples=1)
new_action = new_action.squeeze(1).detach().cpu().numpy()
q1 = np.sum((new_action[:, 0] > 0) & (new_action[:, 1] > 0))
q2 = np.sum((new_action[:, 0] < 0) & (new_action[:, 1] > 0))
q3 = np.sum((new_action[:, 0] < 0) & (new_action[:, 1] < 0))
q4 = np.sum((new_action[:, 0] > 0) & (new_action[:, 1] < 0))
# print(new_action)
print(q1, q2, q3, q4)
axs[1].scatter(new_action[:, 0], new_action[:, 1], alpha=0.3)
axs[1].set_xlim(-axis_lim, axis_lim)
axs[1].set_ylim(-axis_lim, axis_lim)
axs[1].set_xlabel('x', fontsize=20)
axs[1].set_ylabel('y', fontsize=20)
axs[1].set_title('BC-Diffusion', fontsize=25)

# Plot Reconstruction Error
x = torch.linspace(-2.0, 2.0, 1000).to(device)
y = torch.linspace(-2.0, 2.0, 1000).to(device)
grid_x, grid_y = torch.meshgrid(x, y)
grid_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
grid_states = torch.zeros(len(grid_points), 2).to(device).float()
recon_error_diffusion = compute_errors(diffusion_agent.actor, diffusion_agent.model, grid_states, grid_points, n_levels=1, batch_size=1024)
recon_error_diffusion = recon_error_diffusion.reshape(1000, 1000).T
# print("Reconstruction Error Diffusion:", recon_error_diffusion)
print("Max:", recon_error_diffusion.max())
print("Min:", recon_error_diffusion.min())
print("Mean:", recon_error_diffusion.mean())
# recon_error_diffusion_log = np.log1p(recon_error_diffusion)
# recon_error_diffusion_norm = (recon_error_diffusion_log - recon_error_diffusion_log.min()) / (recon_error_diffusion_log.max() - recon_error_diffusion_log.min())
im = axs[2].imshow(recon_error_diffusion, 
                   extent=[-2.0, 2.0, -2.0, 2.0],
                   aspect='auto',
                   cmap='viridis',
                   origin='lower')
                   # vmin=0.0,
                   # vmax=1.0)
plt.colorbar(im, ax=axs[2])
axs[2].set_xlim(-axis_lim, axis_lim)
axs[2].set_ylim(-axis_lim, axis_lim)
axs[2].set_xlabel('x', fontsize=20)
axs[2].set_ylabel('y', fontsize=20)
axs[2].set_title('Reconstruction Error', fontsize=25)

state_samples = torch.zeros(len(action_samples), 2).to(device).float()
action_samples = torch.tensor(action_samples).to(device).float()
with torch.no_grad():
    t = diffusion_agent.model.make_sample_density()(shape=(len(action_samples),), device=device)
    # t = torch.full((len(action_samples),), fill_value=0.05, device=device)
    # print("Sampled t:", t)

    noise = torch.randn_like(action_samples)

    t_expanded = t.view(-1, *([1] * (action_samples.ndim - 1)))
    noisy_actions = action_samples + noise * t_expanded

    c_skip, c_out, c_in = [
        x.view(-1, *([1] * (action_samples.ndim - 1))) 
        for x in diffusion_agent.model.get_diffusion_scalings(t)
    ]

    model_input = noisy_actions * c_in
    model_output = diffusion_agent.actor(model_input, state_samples, torch.log(t) / 4)
    denoised_actions = c_skip * noisy_actions + c_out * model_output
    denoised_actions = denoised_actions.cpu().numpy()

axs[3].scatter(denoised_actions[:, 0], denoised_actions[:, 1], alpha=0.3)
axs[3].set_xlim(-axis_lim, axis_lim)
axs[3].set_ylim(-axis_lim, axis_lim)
axs[3].set_xlabel('x', fontsize=20)
axs[3].set_ylabel('y', fontsize=20)
axs[3].set_title('Diffusion Reconstruction', fontsize=25)

fig.tight_layout()
fig.savefig(os.path.join(img_dir, f'bc_diffusion_{args.data}_sd{seed}.pdf'))
print("Save image to", os.path.join(img_dir, f'bc_diffusion_{args.data}_sd{seed}.pdf'))