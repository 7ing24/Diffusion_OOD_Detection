import argparse
import numpy as np
import torch
import os
import d4rl
import gym
import random
import wandb
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import sys
sys.path.append(("../"))
import utils
from diffusion.karras import DiffusionModel
from diffusion.mlps import ScoreNetwork

D4RL_SUPPRESS_IMPORT_ERROR = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def load_states_actions(env, no_normalize, n_samples=None):
    env = gym.make(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
    if not no_normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        print("No normalize")
    states = replay_buffer.state
    actions = replay_buffer.action
    
    # 如果指定了采样数量，则随机采样
    if n_samples and len(states) > n_samples:
        indices = np.random.choice(len(states), size=n_samples, replace=False)
        return states[indices], actions[indices]
    
    return states, actions


def get_state_threshold(state_distribution, diffusion_model, env, no_normalize, state_n_levels, batch_size):
    states, _ = load_states_actions(env, no_normalize)
    states = torch.tensor(states, dtype=torch.float32, device=device)
    state_error = compute_state_error(state_distribution, diffusion_model, states, state_n_levels, batch_size)
    return np.percentile(state_error, 99)


def get_action_threshold(behavior_model, diffusion_model, env, no_normalize, action_n_levels, batch_size):
    states, actions = load_states_actions(env, no_normalize)
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.float32, device=device)
    action_error = compute_action_error(behavior_model, diffusion_model, actions, states, action_n_levels, batch_size)
    return np.percentile(action_error, 99)


def compute_state_error(state_distribution, diffusion_model, states, state_n_levels, batch_size=256):
    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            batch_errors = torch.zeros(len(batch_states), device=states.device)

            for _ in range(state_n_levels):
                t = diffusion_model.make_sample_density()(shape=(len(batch_states),), device=states.device)
                noise = torch.randn_like(batch_states)

                t_expanded = append_dims(t, batch_states.ndim)
                noisy_states = batch_states + noise * t_expanded

                c_skip, c_out, c_in = [
                    append_dims(x, batch_states.ndim) 
                    for x in diffusion_model.get_diffusion_scalings(t)
                ]

                model_input = noisy_states * c_in
                model_output = state_distribution(model_input, None, torch.log(t) / 4)
                denoised_states = c_skip * noisy_states + c_out * model_output

                error = torch.norm(denoised_states - batch_states, dim=1)
                batch_errors += error

            batch_errors /= state_n_levels
            recon_errors.append(batch_errors.cpu())

    return torch.cat(recon_errors).numpy()


def compute_action_error(behavior_model, diffusion_model, actions, states, action_n_levels, batch_size=256):
    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(actions), batch_size):
            batch_actions = actions[i:i + batch_size]
            batch_states = states[i:i + batch_size]
            batch_errors = torch.zeros(len(batch_actions), device=actions.device)

            for _ in range(action_n_levels):
                t = diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
                noise = torch.randn_like(batch_actions)

                t_expanded = append_dims(t, batch_actions.ndim)
                noisy_actions = batch_actions + noise * t_expanded

                c_skip, c_out, c_in = [
                    append_dims(x, batch_actions.ndim) 
                    for x in diffusion_model.get_diffusion_scalings(t)
                ]

                model_input = noisy_actions * c_in
                model_output = behavior_model(model_input, batch_states, torch.log(t) / 4)
                denoised_actions = c_skip * noisy_actions + c_out * model_output

                error = torch.norm(denoised_actions - batch_actions, dim=1)
                batch_errors += error

            batch_errors /= action_n_levels
            recon_errors.append(batch_errors.cpu())

    return torch.cat(recon_errors).numpy()


# def visualize_state_error(orig_states, ood_states, orig_errors, ood_errors, env_name, ood_env_name):
    all_states = np.vstack([orig_states, ood_states])

    global_min = min(orig_errors.min(), ood_errors.min())
    global_max = max(orig_errors.max(), ood_errors.max())

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    tsne_embeddings = tsne.fit_transform(all_states)

    orig_embeddings = tsne_embeddings[:len(orig_states)]
    ood_embeddings = tsne_embeddings[len(orig_states):]

    x_min, x_max = tsne_embeddings[:, 0].min() - 5, tsne_embeddings[:, 0].max() + 5
    y_min, y_max = tsne_embeddings[:, 1].min() - 5, tsne_embeddings[:, 1].max() + 5

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    # State distribution
    axs[0].scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], 
                c='blue', s=5, alpha=0.6, label=env_name)
    axs[0].scatter(ood_embeddings[:, 0], ood_embeddings[:, 1], 
                c='red', s=5, alpha=0.6, label=ood_env_name)
    axs[0].set_title(f'State Distribution', fontsize=20)
    axs[0].legend(fontsize=16)
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_aspect('equal')
    axs[0].grid(alpha=0.3)

    # Original states errors
    sc1 = axs[1].scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], 
                      c=orig_errors, cmap='viridis', s=5, alpha=0.7,
                      vmin=global_min, vmax=global_max)
    axs[1].set_title(f'{env_name}\nStates Reconstruction Error', fontsize=20)
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    axs[1].set_aspect('equal')
    axs[1].grid(alpha=0.3)

    # OOD states errors
    sc2 = axs[2].scatter(ood_embeddings[:, 0], ood_embeddings[:, 1],
                      c=ood_errors, cmap='viridis', s=5, alpha=0.7,
                      vmin=global_min, vmax=global_max)
    axs[2].set_title(f'{ood_env_name}\nStates Reconstruction Error', fontsize=20)
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_aspect('equal')
    axs[2].grid(alpha=0.3)

    cbar = fig.colorbar(sc2, ax=axs[1:], fraction=0.046, pad=0.04)
    cbar.set_label('Reconstruction Error', fontsize=20)

    save_dir = f"results/visualization/state_error/{args.env}"
    os.makedirs(save_dir, exist_ok=True)
    tsne_save_path = os.path.join(save_dir, f"{args.env}_{args.ood_env}_nlevels{args.state_n_levels}.pdf")
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved state reconstruction error visualization to {tsne_save_path}")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(orig_errors, bins=100, alpha=0.7, label=env_name, density=True)
    plt.hist(ood_errors, bins=100, alpha=0.7, label=ood_env_name, density=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    hist_save_path = os.path.join(save_dir, f"{args.env}_{args.ood_env}_nlevels{args.state_n_levels}_hist.pdf")
    plt.savefig(hist_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved state reconstruction error histogram to {hist_save_path}")
    plt.close()


# def visualize_state_action_error(orig_states, orig_actions, ood_states, ood_actions, orig_errors, ood_errors, env_name, ood_env_name):
    orig_state_actions = np.hstack([orig_states, orig_actions])
    ood_state_actions = np.hstack([ood_states, ood_actions])
    all_state_actions = np.vstack([orig_state_actions, ood_state_actions])

    global_min = min(orig_errors.min(), ood_errors.min())
    global_max = max(orig_errors.max(), ood_errors.max())

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_embeddings = tsne.fit_transform(all_state_actions)
    
    orig_embeddings = tsne_embeddings[:len(orig_state_actions)]
    ood_embeddings = tsne_embeddings[len(orig_state_actions):]

    x_min, x_max = tsne_embeddings[:, 0].min() - 5, tsne_embeddings[:, 0].max() + 5
    y_min, y_max = tsne_embeddings[:, 1].min() - 5, tsne_embeddings[:, 1].max() + 5
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    axs[0].scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], 
                  c='blue', s=5, alpha=0.6, label=env_name)
    axs[0].scatter(ood_embeddings[:, 0], ood_embeddings[:, 1], 
                  c='red', s=5, alpha=0.6, label=ood_env_name)
    axs[0].set_title('State-Action Distribution', fontsize=20)
    axs[0].legend(fontsize=16)
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_aspect('equal')
    axs[0].grid(True, alpha=0.3)
    
    sc1 = axs[1].scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], 
                       c=orig_errors, cmap='viridis', s=5, alpha=0.7,
                       vmin=global_min, vmax=global_max)
    axs[1].set_title(f'{env_name}\nReconstruction Error', fontsize=20)
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    axs[1].set_aspect('equal')
    axs[1].grid(True, alpha=0.3)
    
    sc2 = axs[2].scatter(ood_embeddings[:, 0], ood_embeddings[:, 1], 
                       c=ood_errors, cmap='viridis', s=5, alpha=0.7,
                       vmin=global_min, vmax=global_max)
    axs[2].set_title(f'{ood_env_name}\nReconstruction Error', fontsize=20)
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_aspect('equal')
    axs[2].grid(True, alpha=0.3)

    cbar = fig.colorbar(sc2, ax=axs[1:], fraction=0.046, pad=0.04)
    cbar.set_label('Reconstruction Error', fontsize=20)

    save_dir = f"results/visualization/action_error/{args.env}"
    os.makedirs(save_dir, exist_ok=True)
    tsne_save_path = os.path.join(save_dir, f"{args.env}_{args.ood_env}_nlevels{args.action_n_levels}.pdf")
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstruction error visualization to {tsne_save_path}")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(orig_errors, bins=100, alpha=0.7, label=env_name, density=True)
    plt.hist(ood_errors, bins=100, alpha=0.7, label=ood_env_name, density=True)
    plt.xlabel('Reconstruction Error', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.title('Reconstruction Error Distribution', fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    hist_save_path = os.path.join(save_dir, f"{args.env}_{args.ood_env}_nlevels{args.action_n_levels}_hist.pdf")
    plt.savefig(hist_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstruction error histogram to {hist_save_path}")
    plt.close()


import matplotlib as mpl

def set_plot_style():
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
    })


def visualize_state_error(orig_states, ood_states, orig_errors, ood_errors, env_name, ood_env_name):
    set_plot_style()
    all_states = np.vstack([orig_states, ood_states])

    global_min = min(orig_errors.min(), ood_errors.min())
    global_max = max(orig_errors.max(), ood_errors.max())

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_embeddings = tsne.fit_transform(all_states)

    orig_embeddings = tsne_embeddings[:len(orig_states)]
    ood_embeddings = tsne_embeddings[len(orig_states):]

    # x_min, x_max = tsne_embeddings[:, 0].min() - 5, tsne_embeddings[:, 0].max() + 5
    # y_min, y_max = tsne_embeddings[:, 1].min() - 5, tsne_embeddings[:, 1].max() + 5

    x_min, x_max = -100, 100
    y_min, y_max = -100, 100

    # 1*4 布局，大小与 state-action 一致
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    # (a) State Distribution
    axs[0].scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], 
                   c='blue', s=5, alpha=0.6, label=env_name)
    axs[0].scatter(ood_embeddings[:, 0], ood_embeddings[:, 1], 
                   c='red', s=5, alpha=0.6, label=ood_env_name)
    axs[0].set_title('State Distribution', fontsize=20)
    axs[0].legend(fontsize=14)
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_aspect('equal')
    axs[0].grid(alpha=0.3)

    # (b) In-distribution Errors
    sc1 = axs[1].scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], 
                         c=orig_errors, cmap='viridis', s=5, alpha=0.7,
                         vmin=global_min, vmax=global_max)
    axs[1].set_title(f'{env_name}\nReconstruction Error', fontsize=20)
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    axs[1].set_aspect('equal')
    axs[1].grid(alpha=0.3)

    # (c) OOD Errors
    sc2 = axs[2].scatter(ood_embeddings[:, 0], ood_embeddings[:, 1], 
                         c=ood_errors, cmap='viridis', s=5, alpha=0.7,
                         vmin=global_min, vmax=global_max)
    axs[2].set_title(f'{ood_env_name}\nReconstruction Error', fontsize=20)
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_aspect('equal')
    axs[2].grid(alpha=0.3)

    # (d) Histogram
    axs[3].hist(orig_errors, bins=100, alpha=0.7, label=env_name, density=True)
    axs[3].hist(ood_errors, bins=100, alpha=0.7, label=ood_env_name, density=True)
    axs[3].set_xlabel('Reconstruction Error', fontsize=18)
    axs[3].set_ylabel('Density', fontsize=18)
    axs[3].set_title('Error Distribution', fontsize=20)
    axs[3].legend(fontsize=14)
    axs[3].grid(alpha=0.3)

    # Colorbar 共享 (b)(c)
    cbar = fig.colorbar(sc2, ax=axs[1:3], fraction=0.046, pad=0.04)
    cbar.set_label('Reconstruction Error', fontsize=18)

    # 保存可视化
    save_dir = f"results/visualization/state_error/{args.env}"
    os.makedirs(save_dir, exist_ok=True)
    tsne_save_path = os.path.join(save_dir, f"{args.env}_{args.ood_env}_nlevels{args.state_n_levels}.pdf")
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved state reconstruction error visualization to {tsne_save_path}")
    plt.close()


def visualize_state_action_error(orig_states, orig_actions, ood_states, ood_actions, 
                                 orig_errors, ood_errors, env_name, ood_env_name):
    # 拼接 state-action 向量
    orig_state_actions = np.hstack([orig_states, orig_actions])
    ood_state_actions = np.hstack([ood_states, ood_actions])
    all_state_actions = np.vstack([orig_state_actions, ood_state_actions])

    global_min = min(orig_errors.min(), ood_errors.min())
    global_max = max(orig_errors.max(), ood_errors.max())

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_embeddings = tsne.fit_transform(all_state_actions)
    
    orig_embeddings = tsne_embeddings[:len(orig_state_actions)]
    ood_embeddings = tsne_embeddings[len(orig_state_actions):]

    # x_min, x_max = tsne_embeddings[:, 0].min() - 5, tsne_embeddings[:, 0].max() + 5
    # y_min, y_max = tsne_embeddings[:, 1].min() - 5, tsne_embeddings[:, 1].max() + 5
    
    x_min, x_max = -100, 100
    y_min, y_max = -100, 100

    # 改为 1*4 子图
    fig, axs = plt.subplots(1, 4, figsize=(24, 6), constrained_layout=True)

    # (a) State-Action Distribution
    axs[0].scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], 
                   c='blue', s=5, alpha=0.6, label=env_name)
    axs[0].scatter(ood_embeddings[:, 0], ood_embeddings[:, 1], 
                   c='red', s=5, alpha=0.6, label=ood_env_name)
    axs[0].set_title('State-Action Distribution', fontsize=20)
    axs[0].legend(fontsize=14)
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_aspect('equal')
    axs[0].grid(alpha=0.3)

    # (b) In-distribution Errors
    sc1 = axs[1].scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], 
                         c=orig_errors, cmap='viridis', s=5, alpha=0.7,
                         vmin=global_min, vmax=global_max)
    axs[1].set_title(f'{env_name}\nReconstruction Error', fontsize=20)
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    axs[1].set_aspect('equal')
    axs[1].grid(alpha=0.3)

    # (c) OOD Errors
    sc2 = axs[2].scatter(ood_embeddings[:, 0], ood_embeddings[:, 1], 
                         c=ood_errors, cmap='viridis', s=5, alpha=0.7,
                         vmin=global_min, vmax=global_max)
    axs[2].set_title(f'{ood_env_name}\nReconstruction Error', fontsize=20)
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_aspect('equal')
    axs[2].grid(alpha=0.3)

    # (d) Histogram
    axs[3].hist(orig_errors, bins=100, alpha=0.7, label=env_name, density=True)
    axs[3].hist(ood_errors, bins=100, alpha=0.7, label=ood_env_name, density=True)
    axs[3].set_xlabel('Reconstruction Error', fontsize=18)
    axs[3].set_ylabel('Density', fontsize=18)
    axs[3].set_title('Error Distribution', fontsize=20)
    axs[3].legend(fontsize=14)
    axs[3].grid(alpha=0.3)

    # colorbar 共享 (b)(c)
    cbar = fig.colorbar(sc2, ax=axs[1:3], fraction=0.046, pad=0.04)
    cbar.set_label('Reconstruction Error', fontsize=18)

    # 保存可视化
    save_dir = f"results/visualization/action_error/{args.env}"
    os.makedirs(save_dir, exist_ok=True)
    tsne_save_path = os.path.join(save_dir, f"{args.env}_{args.ood_env}_nlevels{args.action_n_levels}.pdf")
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved state-action reconstruction error visualization to {tsne_save_path}")
    plt.close()

def save_state_error(state_distribution, diffusion_model, states):
    state_error = compute_state_error(state_distribution, diffusion_model, states, args.state_n_levels, args.batch_size)
    
    stats = {
        "Dataset": args.env,
        "Num_levels": args.state_n_levels,
        "Mean": np.mean(state_error),
        "Std": np.std(state_error),
        "Variance": np.var(state_error),
        "Max": np.max(state_error),
        "Min": np.min(state_error),
        "Median": np.median(state_error),
        "80th_percentile": np.percentile(state_error, 80),
        "85th_percentile": np.percentile(state_error, 85),
        "90th_percentile": np.percentile(state_error, 90),
        "95th_percentile": np.percentile(state_error, 95),
        "99th_percentile": np.percentile(state_error, 99)
    }

    print("\nState Reconstruction Error Statistics:")
    for key, value in stats.items():
        if key == "Dataset" or key == "Num_levels":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

    os.makedirs("results/statistics/state_error", exist_ok=True)
    stats_file = f"results/statistics/state_error/{args.env}.txt"
    with open(stats_file, "a") as f:
        f.write(f"{stats['Dataset']}\n")
        f.write(f"Num_levels: {stats['Num_levels']}\n")
        for key, value in stats.items():
            if key != "Dataset" and key != "Num_levels":
                f.write(f"{key}: {value:.6f}\n")
        f.write("\n")

    print(f"\nSaved state reconstruction error statistics to {stats_file}")


def save_action_error(behavior_policy, diffusion_model, states, actions):
    action_error = compute_action_error(behavior_policy, diffusion_model, actions, states, args.action_n_levels, args.batch_size)

    stats = {
        "Dataset": args.env,
        "Num_levels": args.action_n_levels,
        "Mean": np.mean(action_error),
        "Std": np.std(action_error),
        "Variance": np.var(action_error),
        "Max": np.max(action_error),
        "Min": np.min(action_error),
        "Median": np.median(action_error),
        "80th_percentile": np.percentile(action_error, 80),
        "85th_percentile": np.percentile(action_error, 85),
        "90th_percentile": np.percentile(action_error, 90),
        "95th_percentile": np.percentile(action_error, 95),
        "99th_percentile": np.percentile(action_error, 99)
    }

    print("\nAction Reconstruction Error Statistics:")
    for key, value in stats.items():
        if key == "Dataset" or key == "Num_levels":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

    os.makedirs("results/statistics/action_error", exist_ok=True)
    stats_file = f"results/statistics/action_error/{args.env}.txt"
    with open(stats_file, "a") as f:
        f.write(f"{stats['Dataset']}\n")
        f.write(f"Num_levels: {stats['Num_levels']}\n")
        for key, value in stats.items():
            if key != "Dataset" and key != "Num_levels":
                f.write(f"{key}: {value:.6f}\n")
        f.write("\n")
    
    print(f"\nSaved action reconstruction error statistics to {stats_file}")


def train(args):
    print(f"Loading dataset: {args.env}")
    states, actions = load_states_actions(args.env, args.no_normalize)
    state_dim, action_dim = states.shape[1], actions.shape[1]
    orig_states, orig_actions = load_states_actions(args.env, args.no_normalize, args.n_samples)

    print(f"Loading OOD dataset: {args.ood_env}")
    ood_states, ood_actions = load_states_actions(args.ood_env, args.no_normalize, args.n_samples)
    
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
    orig_states_tensor = torch.tensor(orig_states, dtype=torch.float32, device=device)
    orig_actions_tensor = torch.tensor(orig_actions, dtype=torch.float32, device=device)
    ood_states_tensor = torch.tensor(ood_states, dtype=torch.float32, device=device)
    ood_actions_tensor = torch.tensor(ood_actions, dtype=torch.float32, device=device)

    diffusion_model = DiffusionModel(
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        device=device,
    )

    state_distribution = ScoreNetwork(
        x_dim=state_dim,
        hidden_dim=256,
        time_embed_dim=16,
        cond_dim=0,
        cond_mask_prob=0.0,
        num_hidden_layers=4,
        output_dim=state_dim,
        device=device,
        cond_conditional=False
    ).to(device)
    state_distribution_optimizer = torch.optim.Adam(state_distribution.parameters(), lr=3e-4)

    behavior_model = ScoreNetwork(
        x_dim=action_dim, 
        hidden_dim=256,
        time_embed_dim=16,
        cond_dim=state_dim,
        cond_mask_prob=0.0,
        num_hidden_layers=4,
        output_dim=action_dim,
        device=device,
        cond_conditional=True
    ).to(device)
    behavior_model_optimizer = torch.optim.Adam(behavior_model.parameters(), lr=3e-4)

    os.makedirs("../Diffusion_models/sd_models", exist_ok=True)
    state_distribution_path = f"../Diffusion_models/sd_models/{args.env}.pth"
    if os.path.exists(state_distribution_path):
        print(f"Loading pre-trained state distribution model from {state_distribution_path}")
        state_distribution.load_state_dict(torch.load(state_distribution_path, map_location=device))
    else:
        print("Training state distribution model...")

        wandb.init(
            project="Diffusion-SD",
            name=f"{args.env}",
            config=vars(args)
        )

        for epoch in tqdm(range(args.pretrain_epochs)):
            state_distribution_optimizer.zero_grad()
            loss = diffusion_model.diffusion_train_step(state_distribution, states_tensor, None)
            loss.backward()
            state_distribution_optimizer.step()

            wandb.log({"loss": loss.item()})

        torch.save(state_distribution.state_dict(), state_distribution_path)
        print(f"Saved pre-trained state distribution model to {state_distribution_path}")
        wandb.finish()

    os.makedirs("../Diffusion_models/bc_models", exist_ok=True)
    behavior_model_path = f"../Diffusion_models/bc_models/{args.env}.pth"
    if os.path.exists(behavior_model_path):
        print(f"Loading pre-trained behavior policy model from {behavior_model_path}")
        behavior_model.load_state_dict(torch.load(behavior_model_path, map_location=device))
    else:
        print("Training behavior policy model...")

        wandb.init(
            project="Diffusion-BC",
            name=f"{args.env}",
            config=vars(args)
        )

        for epoch in tqdm(range(args.pretrain_epochs)):
            behavior_model_optimizer.zero_grad()
            loss = diffusion_model.diffusion_train_step(behavior_model, actions_tensor, states_tensor)
            loss.backward()
            behavior_model_optimizer.step()

            wandb.log({"loss": loss.item()})

        torch.save(behavior_model.state_dict(), behavior_model_path)
        print(f"Saved pre-trained behavior policy model to {behavior_model_path}")
        wandb.finish()

    # Compute reconstruction error statistics
    save_state_error(state_distribution, diffusion_model, states_tensor)
    save_action_error(behavior_model, diffusion_model, states_tensor, actions_tensor)

    # Visualization
    orig_state_errors = compute_state_error(state_distribution, diffusion_model, orig_states_tensor, args.state_n_levels, args.batch_size)
    ood_state_errors = compute_state_error(state_distribution, diffusion_model, ood_states_tensor, args.state_n_levels, args.batch_size)
    visualize_state_error(orig_states, ood_states, orig_state_errors, ood_state_errors, args.env, args.ood_env)
    orig_action_errors = compute_action_error(behavior_model, diffusion_model, orig_actions_tensor, orig_states_tensor, args.action_n_levels, args.batch_size)
    ood_action_errors = compute_action_error(behavior_model, diffusion_model, ood_actions_tensor, orig_states_tensor, args.action_n_levels, args.batch_size)
    visualize_state_action_error(orig_states, orig_actions, ood_states, ood_actions, orig_action_errors, ood_action_errors, args.env, args.ood_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah-medium-expert-v2')
    parser.add_argument('--ood_env', type=str, default='halfcheetah-medium-replay-v2')
    parser.add_argument('--pretrain_epochs', type=int, default=100000)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_normalize', default=False, action='store_true')
    parser.add_argument("--sigma_max", type=float, default=80)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--action_n_levels", type=int, default=1)
    parser.add_argument("--state_n_levels", type=int, default=1)

    args = parser.parse_args()

    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.seed)
    train(args)