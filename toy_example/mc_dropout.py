import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=256, dropout_prob=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


def compute_mc_dropout_uncertainty(q_model, states, actions, n_samples, batch_size):
    q_model.train()
    all_q = []

    with torch.no_grad():
        for _ in range(n_samples):
            q_vals = []
            for i in range(0, len(states), batch_size):
                batch_states = states[i:i+batch_size]
                batch_actions = actions[i:i+batch_size]
                q = q_model(batch_states, batch_actions)
                q_vals.append(q.cpu())
            all_q.append(torch.cat(q_vals, dim=0))

    all_q = torch.stack(all_q, dim=0)  # [n_samples, N, 1]
    uncertainty = all_q.var(dim=0).squeeze(-1).numpy()  # Q方差作为不确定性
    return uncertainty


def plot_mc_dropout_uncertainty(q_model, states, actions, dataset_name, n_samples, batch_size):
    os.makedirs("visualization/mc_dropout", exist_ok=True)

    states_grid, actions_grid = np.meshgrid(
        np.linspace(states.min(), states.max(), 2000),
        np.linspace(actions.min(), actions.max(), 2000)
    )

    eval_states = states_grid.flatten()[:, None]
    eval_actions = actions_grid.flatten()[:, None]

    states_tensor = torch.tensor(eval_states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(eval_actions, dtype=torch.float32).to(device)

    uncertainty = compute_mc_dropout_uncertainty(q_model, states_tensor, actions_tensor, n_samples, batch_size)
    uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    uncertainty_grid = uncertainty.reshape(states_grid.shape)

    plt.figure(figsize=(9, 8))
    plt.pcolormesh(states_grid, actions_grid, uncertainty_grid, cmap='viridis', shading='auto')
    # cbar = plt.colorbar()
    # cbar.set_label('MC Dropout Uncertainty', fontsize=30)
    # cbar.ax.tick_params(labelsize=30)
    plt.xlabel('State', fontsize=30)
    plt.ylabel('Action', fontsize=30)
    plt.title(f'MC Dropout', fontsize=30)
    plt.xlim([states.min(), states.max()])
    plt.ylim([actions.min(), actions.max()])
    plt.xticks(np.linspace(states.min(), states.max(), 9))  
    plt.yticks(np.linspace(actions.min(), actions.max(), 9)) 

    plt.savefig(f"visualization/mc_dropout/uncertainty_{dataset_name}.png", dpi=300)
    plt.close()
    print(f"Saved MC Dropout uncertainty heatmap for {dataset_name} dataset")


def train(args):
    dataset_path = f"datasets/{args.dataset}_data.pkl"
    if not os.path.exists(dataset_path):
        print(f"{args.dataset} dataset path does not exist!")
        return
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    states, actions, rewards = data['states'], data['actions'], data["rewards"]

    states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(1).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1).to(device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)

    save_dir = f"model/mc_dropout_q"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{args.dataset}.pth")

    q_model = QNetwork().to(device)

    if os.path.exists(model_path):
        print("Loading pre-trained Q function...")
        q_model.load_state_dict(torch.load(model_path, map_location=device))
        q_model.eval()
    else:
        print("Training Q function...")
        optimizer = optim.Adam(q_model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor, rewards_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for epoch in range(args.pretrain_epochs):
            epoch_loss = 0.0
            for batch_states, batch_actions, batch_rewards in loader:
                optimizer.zero_grad()
                pred_q = q_model(batch_states, batch_actions)
                loss = loss_fn(pred_q, batch_rewards)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_states.size(0)
            avg_loss = epoch_loss / len(dataset)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.pretrain_epochs}, Loss: {avg_loss:.6f}")

        torch.save(q_model.state_dict(), model_path)
        print(f"Q function saved to {model_path}")

    plot_mc_dropout_uncertainty(q_model, states, actions, args.dataset, args.n_samples, args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='expert', choices=['expert', 'medium'])
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=20)
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