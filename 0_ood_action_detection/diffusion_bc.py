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


def get_threshold(model, diffusion_model, env, no_normalize, action_n_levels, batch_size):
    states, actions = load_states_actions(env, no_normalize)
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.float32, device=device)
    errors = compute_multi_errors(model, diffusion_model, states, actions, action_n_levels, batch_size)
    return np.percentile(errors, 99)


# 单个噪声水平下的重构误差
def compute_error(model, diffusion_model, states, actions, batch_size):
    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(actions), batch_size):
            batch_states = states[i:i + batch_size]
            batch_actions = actions[i:i + batch_size]
            t = diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
            noise = torch.randn_like(batch_actions)

            t_expanded = append_dims(t, batch_actions.ndim)
            noisy_actions = batch_actions + noise * t_expanded

            c_skip, c_out, c_in = [
                append_dims(x, batch_actions.ndim) 
                for x in diffusion_model.get_diffusion_scalings(t)
            ]

            model_input = noisy_actions * c_in
            model_output = model(model_input, batch_states, torch.log(t)/4)

            denoised_actions = c_skip * noisy_actions + c_out * model_output

            error = torch.norm(denoised_actions - batch_actions, dim=1)
            recon_errors.append(error.cpu())
    return torch.cat(recon_errors).numpy()


# 多个噪声水平下的平均重构误差
def compute_multi_errors(model, diffusion_model, states, actions, action_n_levels, batch_size):
    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(actions), batch_size):
            batch_states = states[i:i + batch_size]
            batch_actions = actions[i:i + batch_size]
            batch_errors = torch.zeros(len(batch_actions), device=actions.device)

            for _ in range(action_n_levels):
                t = diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
                # t = torch.full((len(batch_actions),), 0.2, device=actions.device)
                noise = torch.randn_like(batch_actions)

                t_expanded = append_dims(t, batch_actions.ndim)
                noisy_actions = batch_actions + noise * t_expanded

                c_skip, c_out, c_in = [
                    append_dims(x, batch_actions.ndim) 
                    for x in diffusion_model.get_diffusion_scalings(t)
                ]

                model_input = noisy_actions * c_in
                model_output = model(model_input, batch_states, torch.log(t) / 4)
                denoised_actions = c_skip * noisy_actions + c_out * model_output

                error = torch.norm(denoised_actions - batch_actions, dim=1)
                batch_errors += error

            batch_errors /= action_n_levels
            recon_errors.append(batch_errors.cpu())

    return torch.cat(recon_errors).numpy()


def visualize(orig_states, orig_actions, ood_states, ood_actions, orig_errors, ood_errors, env_name, ood_env_name):
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
    axs[0].set_title('State-Action Distribution')
    axs[0].legend()
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    axs[0].set_aspect('equal')
    axs[0].grid(True, alpha=0.3)
    
    sc1 = axs[1].scatter(orig_embeddings[:, 0], orig_embeddings[:, 1], 
                       c=orig_errors, cmap='viridis', s=5, alpha=0.7,
                       vmin=global_min, vmax=global_max)
    axs[1].set_title(f'{env_name}\nReconstruction Error')
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    axs[1].set_aspect('equal')
    axs[1].grid(True, alpha=0.3)
    
    sc2 = axs[2].scatter(ood_embeddings[:, 0], ood_embeddings[:, 1], 
                       c=ood_errors, cmap='viridis', s=5, alpha=0.7,
                       vmin=global_min, vmax=global_max)
    axs[2].set_title(f'{ood_env_name}\nReconstruction Error')
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(y_min, y_max)
    axs[2].set_aspect('equal')
    axs[2].grid(True, alpha=0.3)

    cbar = fig.colorbar(sc2, ax=axs[1:], fraction=0.046, pad=0.04)
    cbar.set_label('Reconstruction Error', fontsize=14)

    save_dir = f"results/visualization/action_error/{args.env}"
    os.makedirs(save_dir, exist_ok=True)
    tsne_save_path = os.path.join(save_dir, f"{args.env}_{args.ood_env}_nlevels{args.action_n_levels}.png")
    plt.savefig(tsne_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstruction error visualization to {tsne_save_path}")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(orig_errors, bins=100, alpha=0.7, label=env_name, density=True)
    plt.hist(ood_errors, bins=100, alpha=0.7, label=ood_env_name, density=True)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    hist_save_path = os.path.join(save_dir, f"{args.env}_{args.ood_env}_nlevels{args.action_n_levels}_hist.png")
    plt.savefig(hist_save_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstruction error histogram to {hist_save_path}")
    plt.close()


def save_errors(behavior_policy, diffusion_model, states, actions):
    errors = compute_multi_errors(behavior_policy, diffusion_model, states, actions, args.action_n_levels, args.batch_size)

    stats = {
        "Dataset": args.env,
        "Num_levels": args.action_n_levels,
        "Mean": np.mean(errors),
        "Std": np.std(errors),
        "Variance": np.var(errors),
        "Max": np.max(errors),
        "Min": np.min(errors),
        "Median": np.median(errors),
        "90th_percentile": np.percentile(errors, 90),
        "95th_percentile": np.percentile(errors, 95),
        "99th_percentile": np.percentile(errors, 99)
    }

    print("\nReconstruction Error Statistics:")
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
    
    print(f"\nSaved reconstruction error statistics to {stats_file}")


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

    behavior_policy = ScoreNetwork(
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
    optimizer = torch.optim.Adam(behavior_policy.parameters(), lr=3e-4)

    os.makedirs("../Diffusion_models/bc_models", exist_ok=True)
    model_path = f"../Diffusion_models/bc_models/{args.env}.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained behavior policy model from {model_path}")
        behavior_policy.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training behavior policy model...")

        wandb.init(
            project="Diffusion-BC",
            name=f"{args.env}",
            config=vars(args)
        )

        for epoch in tqdm(range(args.pretrain_epochs)):
            optimizer.zero_grad()
            loss = diffusion_model.diffusion_train_step(behavior_policy, actions_tensor, states_tensor)
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

        torch.save(behavior_policy.state_dict(), model_path)
        print(f"Saved pre-trained behavior policy model to {model_path}")
        wandb.finish()

    save_errors(behavior_policy, diffusion_model, states_tensor, actions_tensor)

    orig_errors = compute_multi_errors(behavior_policy, diffusion_model, orig_states_tensor, orig_actions_tensor, args.action_n_levels, args.batch_size)
    ood_errors = compute_multi_errors(behavior_policy, diffusion_model, ood_states_tensor, ood_actions_tensor, args.action_n_levels, args.batch_size)

    visualize(orig_states, orig_actions, ood_states, ood_actions, orig_errors, ood_errors, args.env, args.ood_env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah-medium-replay-v2')
    parser.add_argument('--ood_env', type=str, default='halfcheetah-medium-expert-v2')
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