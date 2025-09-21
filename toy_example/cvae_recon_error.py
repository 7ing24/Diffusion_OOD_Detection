import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_MIN, STATE_MAX = -10, 10
ACTION_MIN, ACTION_MAX = -1, 1


class CVAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden_dim=256):
        super(CVAE, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def encode(self, action, state):
        x = torch.cat([action, state], dim=-1)
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, state):
        x = torch.cat([z, state], dim=-1)
        return self.decoder(x)

    def forward(self, action, state):
        mu, logvar = self.encode(action, state)
        z = self.reparameterize(mu, logvar)
        recon_action = self.decode(z, state)
        return recon_action, mu, logvar


def compute_cvae_recon_error(cvae, states, actions, batch_size=256):
    states = np.array(states)
    actions = np.array(actions)
    if states.ndim == 1:
        states = states[:, None]
    if actions.ndim == 1:
        actions = actions[:, None]

    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).to(device)

    errors = []
    with torch.no_grad():
        for i in range(0, len(states_tensor), batch_size):
            batch_states = states_tensor[i:i+batch_size]
            batch_actions = actions_tensor[i:i+batch_size]
            recon_actions, _, _ = cvae(batch_actions, batch_states)
            batch_error = torch.norm(recon_actions - batch_actions, dim=1)
            errors.append(batch_error.cpu())
    return torch.cat(errors).numpy()


def plot_cvae_recon_error(cvae, dataset_name, batch_size=1024):
    os.makedirs("visualization/cvae_recon_error", exist_ok=True)

    states_grid, actions_grid = np.meshgrid(
        np.linspace(STATE_MIN, STATE_MAX, 2000),
        np.linspace(ACTION_MIN, ACTION_MAX, 2000)
    )

    eval_states = states_grid.flatten()[:, None]
    eval_actions = actions_grid.flatten()[:, None]

    errors = compute_cvae_recon_error(cvae, eval_states, eval_actions, batch_size)
    errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
    errors_grid = errors.reshape(states_grid.shape)

    plt.figure(figsize=(11, 8))
    plt.pcolormesh(states_grid, actions_grid, errors_grid, cmap='viridis', shading='auto')
    cbar = plt.colorbar()
    cbar.set_label('Reconstruction Error', fontsize=30)
    cbar.ax.tick_params(labelsize=26)
    plt.xlabel('State', fontsize=30)
    plt.ylabel('Action', fontsize=30)
    plt.title(f'CVAE Reconstruction Error', fontsize=30)
    plt.xlim([STATE_MIN, STATE_MAX])
    plt.ylim([ACTION_MIN, ACTION_MAX])
    plt.xticks(np.linspace(STATE_MIN, STATE_MAX, 9))  
    plt.yticks(np.linspace(ACTION_MIN, ACTION_MAX, 9)) 

    plt.savefig(f'visualization/cvae_recon_error/recon_error_{dataset_name}.png', dpi=300)
    plt.close()
    print(f"Saved CVAE reconstruction error plot for {dataset_name} dataset")


def train(args):
    dataset_path = f"datasets/{args.dataset}_data.pkl"
    if not os.path.exists(dataset_path):
        print(f"{args.dataset} dataset path does not exist!")
        return
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    states, actions = data['states'], data['actions']
    state_dim, action_dim = 1, 1

    states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(1).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1).to(device)

    save_dir = "model/cvae_models"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{args.dataset}.pth")

    cvae = CVAE(state_dim, action_dim, latent_dim=args.latent_dim).to(device)

    if os.path.exists(model_path):
        print("Loading pre-trained CVAE model...")
        cvae.load_state_dict(torch.load(model_path, map_location=device))
        cvae.eval()
    else:
        print(f"Training CVAE model...")
        optimizer = optim.Adam(cvae.parameters(), lr=args.lr)
        for epoch in tqdm(range(args.epochs)):
            indices = np.random.choice(len(states_tensor), args.batch_size, replace=False)
            batch_states = states_tensor[indices]
            batch_actions = actions_tensor[indices]

            recon_actions, mu, logvar = cvae(batch_actions, batch_states)
            recon_loss = nn.MSELoss()(recon_actions, batch_actions)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss / args.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{args.epochs}, Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item()/args.batch_size:.4f}")

        torch.save(cvae.state_dict(), model_path)
        print(f"Saved CVAE model to {model_path}")

    print("Visualizing CVAE reconstruction error on full state-action space...")
    plot_cvae_recon_error(cvae, args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='expert', choices=['expert', 'medium', 'random'])
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)

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