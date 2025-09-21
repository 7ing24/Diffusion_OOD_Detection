import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import pickle
import argparse
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_MIN, STATE_MAX = -10, 10
ACTION_MIN, ACTION_MAX = -1, 1

class DynamicsModel(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim) 
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


def compute_uncertainty(models, states, actions, batch_size=1024):
    preds = []
    with torch.no_grad():
        for model in models:
            pred = []
            for i in range(0, len(states), batch_size):
                batch_states = states[i:i+batch_size]
                batch_actions = actions[i:i+batch_size]
                pred.append(model(batch_states, batch_actions).cpu())
            preds.append(torch.cat(pred, dim=0))
    preds = torch.stack(preds, dim=0)  # [n_models, N, state_dim]

    # ensemble variance
    variance = preds.var(dim=0).mean(dim=-1)  # [N]
    return variance.numpy()


def plot_uncertainty_heatmap(models, states, actions, dataset_name):
    os.makedirs("visualization/model_ensemble", exist_ok=True)

    states_grid, actions_grid = np.meshgrid(
        np.linspace(states.min(), states.max(), 2000),
        np.linspace(actions.min(), actions.max(), 2000)
    )

    eval_states = states_grid.flatten()[:, None]
    eval_actions = actions_grid.flatten()[:, None]

    states_tensor = torch.tensor(eval_states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(eval_actions, dtype=torch.float32).to(device)

    uncertainty = compute_uncertainty(models, states_tensor, actions_tensor)
    uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    uncertainty_grid = uncertainty.reshape(states_grid.shape)

    plt.figure(figsize=(9, 8))
    plt.pcolormesh(states_grid, actions_grid, uncertainty_grid, cmap='viridis', shading='auto')
    # cbar = plt.colorbar()
    # cbar.set_label('Model Ensemble Uncertainty', fontsize=20)
    # cbar.ax.tick_params(labelsize=20)
    plt.xlabel('State', fontsize=30)
    plt.ylabel('Action', fontsize=30)
    plt.title(f'Model Ensemble', fontsize=30)
    plt.xlim([STATE_MIN, STATE_MAX])
    plt.ylim([ACTION_MIN, ACTION_MAX])
    plt.xticks(np.linspace(STATE_MIN, STATE_MAX, 9))  
    plt.yticks(np.linspace(ACTION_MIN, ACTION_MAX, 9)) 
    plt.savefig(f"visualization/model_ensemble/uncertainty_{dataset_name}.png", dpi=300)
    plt.close()
    print(f"Saved uncertainty heatmap for {dataset_name} dataset")


def train(args):
    dataset_path = f"datasets/{args.dataset}_data.pkl"
    if not os.path.exists(dataset_path):
        print(f"{args.dataset} dataset path does not exist!")
        return
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    states, actions, next_states = data['states'], data['actions'], data["next_states"]
    
    states_tensor = torch.tensor(data["states"], dtype=torch.float32).unsqueeze(1).to(device)
    actions_tensor = torch.tensor(data["actions"], dtype=torch.float32).unsqueeze(1).to(device)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32).unsqueeze(1).to(device)

    save_dir = f"model/dynamics_ensemble/{args.dataset}"
    os.makedirs(save_dir, exist_ok=True)
    model_paths = [os.path.join(save_dir, f"model_{i}.pth") for i in range(args.n_models)]
    models_exist = all([os.path.exists(p) for p in model_paths])

    dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor, next_states_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    models = []
    if models_exist:
        print(f"Loading pre-trained dynamics models...")
        for i, path in enumerate(model_paths):
            model = DynamicsModel().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
    else:
        print("Training dynamics model...")
        for i in range(args.n_models):
            model = DynamicsModel().to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            for epoch in range(args.pretrain_epochs):
                epoch_loss = 0.0
                for batch_states, batch_actions, batch_next_states in loader:
                    optimizer.zero_grad()
                    pred_next_states = model(batch_states, batch_actions)
                    loss = loss_fn(pred_next_states, batch_next_states)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * batch_states.size(0)

                avg_loss = epoch_loss / len(dataset)
                print(f"[Model {i}] Epoch {epoch+1}/{args.pretrain_epochs}, Loss: {avg_loss:.6f}")

            model_path = os.path.join(save_dir, f"model_{i}.pth")
            torch.save(model.state_dict(), model_path)
            models.append(model)
            print(f"Dynamics model {i} trained and saved to {model_path}")

    plot_uncertainty_heatmap(models, states, actions, args.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='expert', choices=['expert', 'medium', 'random'])
    parser.add_argument('--pretrain_epochs', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument("--n_models", type=int, default=5)

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