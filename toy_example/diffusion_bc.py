import numpy as np
import torch
import argparse
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pickle
import seaborn as sns
import sys
sys.path.append(("../"))
from tqdm import tqdm
from diffusion.karras import DiffusionModel
from diffusion.mlps import ScoreNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_MIN, STATE_MAX = -10, 10
ACTION_MIN, ACTION_MAX = -1, 1


def normalize_state(state, eps=1e-3):
    mean = state.mean(0, keepdims=True)
    std = state.std(0, keepdims=True) + eps
    state = (state - mean) / std
    return state


def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def compute_recon_errors(dataset, model, diffusion_model, states, actions, n_levels, batch_size):
    recon_errors = []
    with torch.no_grad():
        for i in range(0, len(actions), batch_size):
            batch_states = states[i:i + batch_size]
            batch_actions = actions[i:i + batch_size]
            batch_errors = torch.zeros(len(batch_actions), device=actions.device)

            for _ in range(n_levels):
                if dataset == 'expert':
                    t = torch.full((len(batch_actions),), 0.02, device=actions.device)  # for expert dataset
                elif dataset == 'medium':
                    t = torch.full((len(batch_actions),), 0.5, device=actions.device)  # for medium dataset
                else:
                    t = diffusion_model.make_sample_density()(shape=(len(batch_actions),), device=actions.device)
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

            batch_errors /= n_levels
            recon_errors.append(batch_errors.cpu())

    return torch.cat(recon_errors).numpy()


def plot_recon_errors(model, diffusion_model, dataset, n_levels, batch_size=256):
    os.makedirs("visualization/diffusion_bc", exist_ok=True)
    
    states_grid, actions_grid = np.meshgrid(
        np.linspace(STATE_MIN, STATE_MAX, 2000),
        np.linspace(ACTION_MIN, ACTION_MAX, 2000)
    )

    eval_states = states_grid.flatten()[:, np.newaxis] 
    eval_actions = actions_grid.flatten()[:, np.newaxis]
    # eval_states = normalize_state(eval_states)

    states_tensor = torch.tensor(eval_states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(eval_actions, dtype=torch.float32).to(device)

    recon_errors = compute_recon_errors(dataset, model, diffusion_model, states_tensor, actions_tensor, n_levels, batch_size)
    recon_errors = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)
    recon_errors_grid = recon_errors.reshape(states_grid.shape)
    
    plt.figure(figsize=(9, 8))
    plt.pcolormesh(states_grid, actions_grid, recon_errors_grid, cmap='viridis', shading='auto')
    # cbar = plt.colorbar()
    # cbar.set_label('Reconstruction Error', fontsize=30)
    # cbar.ax.tick_params(labelsize=30)
    plt.xlabel('State', fontsize=30)
    plt.ylabel('Action', fontsize=30)
    plt.title(f'Diffusion Reconstruction Error', fontsize=30)
    plt.xlim([STATE_MIN, STATE_MAX])
    plt.ylim([ACTION_MIN, ACTION_MAX])
    plt.xticks(np.linspace(STATE_MIN, STATE_MAX, 9))  
    plt.yticks(np.linspace(ACTION_MIN, ACTION_MAX, 9)) 
    
    # 保存图片
    plt.savefig(f'visualization/diffusion_bc/recon_error_{dataset}.png', dpi=300)
    plt.close()
    print(f"Saved reconstruction error plot for {dataset} dataset")


def plot_recon_error_density(model, diffusion_model, states, actions, dataset, n_levels, batch_size=1030):
    os.makedirs("visualization/diffusion_bc", exist_ok=True)

    states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(1).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1).to(device)

    recon_errors = compute_recon_errors(dataset, model, diffusion_model, states_tensor, actions_tensor, n_levels, batch_size)

    plt.figure(figsize=(10, 6))
    sns.histplot(recon_errors, bins=50, kde=True, color="royalblue")
    plt.xlabel("Reconstruction Error", fontsize=30)
    plt.ylabel("Density", fontsize=30)
    plt.title(f"Reconstruction Error Distribution on {dataset} dataset", fontsize=30)
    plt.grid(True, linestyle="--", alpha=0.6)

    save_path = f"visualization/diffusion_bc/recon_error_density_{dataset}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved reconstruction error density plot for {dataset} dataset at {save_path}")


def train(args):
    dataset_path = f"datasets/{args.dataset}_data.pkl"
    if not os.path.exists(dataset_path):
        print(f"{args.dataset} dataset path does not exist!")
        return
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    states, actions = data['states'], data['actions']
    state_dim, action_dim = 1, 1
    # states = normalize_state(states)
    
    states_tensor = torch.tensor(data["states"], dtype=torch.float32).unsqueeze(1).to(device)
    actions_tensor = torch.tensor(data["actions"], dtype=torch.float32).unsqueeze(1).to(device)

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

    os.makedirs("model/diffusion_bc_models", exist_ok=True)
    model_path = f"model/diffusion_bc_models/{args.dataset}.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained behavior policy model from {model_path}")
        behavior_policy.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training behavior policy model...")
        for epoch in tqdm(range(args.pretrain_epochs)):
            indices = np.random.choice(len(states_tensor), args.batch_size, replace=False)
            batch_states = states_tensor[indices]
            batch_actions = actions_tensor[indices]
            
            optimizer.zero_grad()
            loss = diffusion_model.diffusion_train_step(behavior_policy, batch_actions, batch_states)
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print("loss: ", loss.item())

        torch.save(behavior_policy.state_dict(), model_path)
        print(f"Saved pre-trained behavior policy model to {model_path}")
    
    print("Visualizing reconstruction errors...")
    plot_recon_errors(behavior_policy, diffusion_model, args.dataset, args.n_levels)

    # print("Plotting reconstruction error density...")
    # plot_recon_error_density(behavior_policy, diffusion_model, states, actions, args.dataset, args.n_levels)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='expert', choices=['expert', 'medium', 'random'])
    parser.add_argument('--pretrain_epochs', type=int, default=10000)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--sigma_max", type=float, default=80)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--n_levels", type=int, default=10)

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