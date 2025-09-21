import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pickle
import gc
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import matplotlib as mpl
from tqdm import tqdm

from diffusion_bc import DiffusionModel, ScoreNetwork, compute_recon_errors
from model_ensemble import DynamicsModel, compute_uncertainty
from mc_dropout import QNetwork, compute_mc_dropout_uncertainty
from cvae_nll import CVAE, compute_cvae_nll
from cvae_recon_error import CVAE, compute_cvae_recon_error

# 设置全局参数
STATE_MIN, STATE_MAX = -10, 10
ACTION_MIN, ACTION_MAX = -1, 1
NUM_STATE_BINS, NUM_ACTION_BINS = 1000, 1000
GRID_RESOLUTION = 500 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    'font.size': 18, 
    'axes.titlesize': 20, 
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16, 
    'figure.figsize': (30, 10),
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'legend.fontsize': 16 
})


def load_dataset(dataset):
    """加载数据集"""
    path = f"datasets/{dataset}_data.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {path} not found!")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_dataset_samples(ax, data, dataset):
    """绘制数据集样本散点图"""
    states = data['states']
    actions = data['actions']
    
    ax.scatter(states, actions, s=1, alpha=0.5, color='blue' if dataset == 'expert' else 'red')
    ax.set_title(f'{dataset.capitalize()} Dataset')
    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_xlim(STATE_MIN, STATE_MAX)
    ax.set_ylim(ACTION_MIN, ACTION_MAX)
    ax.set_xticks(np.linspace(STATE_MIN, STATE_MAX, 5))  
    ax.set_yticks(np.linspace(ACTION_MIN, ACTION_MAX, 5)) 
    ax.grid(True, alpha=0.2)
    

def plot_diffusion_recon_error(ax, model, diffusion_model, dataset):
    """绘制扩散模型重建误差"""
    states_grid, actions_grid = np.meshgrid(
        np.linspace(STATE_MIN, STATE_MAX, GRID_RESOLUTION),
        np.linspace(ACTION_MIN, ACTION_MAX, GRID_RESOLUTION)
    )
    
    eval_states = states_grid.flatten()[:, None]
    eval_actions = actions_grid.flatten()[:, None]
    
    states_tensor = torch.tensor(eval_states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(eval_actions, dtype=torch.float32).to(device)
    
    recon_errors = compute_recon_errors(dataset, model, diffusion_model, states_tensor, actions_tensor, n_levels=10, batch_size=1024)
    recon_errors = recon_errors.reshape(states_grid.shape)
    recon_errors = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)
    
    im = ax.imshow(recon_errors, extent=[STATE_MIN, STATE_MAX, ACTION_MIN, ACTION_MAX], origin="lower", cmap="viridis", aspect="auto", rasterized=True)
    ax.set_title(f'Diffusion Reconstruction Error')
    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_xlim(STATE_MIN, STATE_MAX)
    ax.set_ylim(ACTION_MIN, ACTION_MAX)
    ax.set_xticks(np.linspace(STATE_MIN, STATE_MAX, 5))  
    ax.set_yticks(np.linspace(ACTION_MIN, ACTION_MAX, 5)) 
    
    return im

def plot_dynamics_uncertainty(ax, models, dataset):
    """绘制集成动力学模型不确定性"""
    states_grid, actions_grid = np.meshgrid(
        np.linspace(STATE_MIN, STATE_MAX, GRID_RESOLUTION),
        np.linspace(ACTION_MIN, ACTION_MAX, GRID_RESOLUTION)
    )
    
    eval_states = states_grid.flatten()[:, None]
    eval_actions = actions_grid.flatten()[:, None]
    
    states_tensor = torch.tensor(eval_states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(eval_actions, dtype=torch.float32).to(device)
    
    uncertainty = compute_uncertainty(models, states_tensor, actions_tensor, batch_size=1024)
    uncertainty = uncertainty.reshape(states_grid.shape)
    uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)
    
    im = ax.imshow(uncertainty, extent=[STATE_MIN, STATE_MAX, ACTION_MIN, ACTION_MAX], origin="lower", cmap="viridis", aspect="auto", rasterized=True)
    ax.set_title(f'Model Ensemble')
    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_xlim(STATE_MIN, STATE_MAX)
    ax.set_ylim(ACTION_MIN, ACTION_MAX)
    ax.set_xticks(np.linspace(STATE_MIN, STATE_MAX, 5))  
    ax.set_yticks(np.linspace(ACTION_MIN, ACTION_MAX, 5)) 
    
    return im

def plot_mc_dropout_uncertainty(ax, q_model, dataset):
    """绘制MC Dropout不确定性"""
    states_grid, actions_grid = np.meshgrid(
        np.linspace(STATE_MIN, STATE_MAX, GRID_RESOLUTION),
        np.linspace(ACTION_MIN, ACTION_MAX, GRID_RESOLUTION)
    )
    
    eval_states = states_grid.flatten()[:, None]
    eval_actions = actions_grid.flatten()[:, None]
    
    states_tensor = torch.tensor(eval_states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(eval_actions, dtype=torch.float32).to(device)

    uncertainty = compute_mc_dropout_uncertainty(q_model, states_tensor, actions_tensor, n_samples=20, batch_size=1024)
    uncertainty = uncertainty.reshape(states_grid.shape)
    uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-8)

    im = ax.imshow(uncertainty, extent=[STATE_MIN, STATE_MAX, ACTION_MIN, ACTION_MAX], origin="lower", cmap="viridis", aspect="auto", rasterized=True)
    ax.set_title(f'MC Dropout')
    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_xlim(STATE_MIN, STATE_MAX)
    ax.set_ylim(ACTION_MIN, ACTION_MAX)
    ax.set_xticks(np.linspace(STATE_MIN, STATE_MAX, 5))  
    ax.set_yticks(np.linspace(ACTION_MIN, ACTION_MAX, 5)) 
    
    return im

def plot_cvae_nll(ax, cvae, dataset):
    """绘制CVAE负对数似然"""
    states_grid, actions_grid = np.meshgrid(
        np.linspace(STATE_MIN, STATE_MAX, GRID_RESOLUTION),
        np.linspace(ACTION_MIN, ACTION_MAX, GRID_RESOLUTION)
    )
    
    eval_states = states_grid.flatten()[:, None]
    eval_actions = actions_grid.flatten()[:, None]
    
    nlls = compute_cvae_nll(cvae, eval_states, eval_actions, batch_size=1024)
    nlls = nlls.reshape(states_grid.shape)
    nlls = (nlls - nlls.min()) / (nlls.max() - nlls.min() + 1e-8)

    im = ax.imshow(nlls, extent=[STATE_MIN, STATE_MAX, ACTION_MIN, ACTION_MAX], origin="lower", cmap="viridis", aspect="auto", rasterized=True)
    ax.set_title(f'CVAE NLL', fontsize=20)
    ax.set_xlabel('State', fontsize=20)
    ax.set_ylabel('Action', fontsize=20)
    ax.set_xlim(STATE_MIN, STATE_MAX)
    ax.set_ylim(ACTION_MIN, ACTION_MAX)
    ax.set_xticks(np.linspace(STATE_MIN, STATE_MAX, 5))  
    ax.set_yticks(np.linspace(ACTION_MIN, ACTION_MAX, 5)) 
    
    return im

def plot_cvae_recon_error(ax, cvae, dataset):
    "绘制CVAE重构误差"
    states_grid, actions_grid = np.meshgrid(
        np.linspace(STATE_MIN, STATE_MAX, GRID_RESOLUTION),
        np.linspace(ACTION_MIN, ACTION_MAX, GRID_RESOLUTION)
    )
    
    eval_states = states_grid.flatten()[:, None]
    eval_actions = actions_grid.flatten()[:, None]

    recon_errors = compute_cvae_recon_error(cvae, eval_states, eval_actions, batch_size=1024)
    recon_errors = recon_errors.reshape(states_grid.shape)
    recon_errors = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)

    im = ax.imshow(recon_errors, extent=[STATE_MIN, STATE_MAX, ACTION_MIN, ACTION_MAX], origin="lower", cmap="viridis", aspect="auto", rasterized=True)
    ax.set_title(f'CVAE Reconstruction Error', fontsize=20)
    ax.set_xlabel('State', fontsize=20)
    ax.set_ylabel('Action', fontsize=20)
    ax.set_xlim(STATE_MIN, STATE_MAX)
    ax.set_ylim(ACTION_MIN, ACTION_MAX)
    ax.set_xticks(np.linspace(STATE_MIN, STATE_MAX, 5))  
    ax.set_yticks(np.linspace(ACTION_MIN, ACTION_MAX, 5)) 
    
    return im

def load_model(model_type, dataset):
    """加载预训练模型"""
    if model_type == 'diffusion':
        model_path = f"model/diffusion_bc_models/{dataset}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Diffusion model {model_path} not found!")
        
        behavior_policy = ScoreNetwork(
            x_dim=1, 
            hidden_dim=256,
            time_embed_dim=16,
            cond_dim=1,
            cond_mask_prob=0.0,
            num_hidden_layers=4,
            output_dim=1,
            device=device,
            cond_conditional=True
        ).to(device)
        behavior_policy.load_state_dict(torch.load(model_path, map_location=device))
        behavior_policy.eval()
        
        diffusion_model = DiffusionModel(
            sigma_data=0.5,
            sigma_min=0.002,
            sigma_max=80,
            device=device,
        )
        
        return behavior_policy, diffusion_model
    
    elif model_type == 'dynamics':
        save_dir = f"model/dynamics_ensemble/{dataset}"
        model_paths = [os.path.join(save_dir, f"model_{i}.pth") for i in range(5)]
        
        models = []
        for path in model_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dynamics model {path} not found!")
            
            model = DynamicsModel().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
        
        return models
    
    elif model_type == 'mc_dropout':
        model_path = f"model/mc_dropout_q/{dataset}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MC Dropout model {model_path} not found!")
        
        q_model = QNetwork().to(device)
        q_model.load_state_dict(torch.load(model_path, map_location=device))
        q_model.train()  # 保持训练模式以启用Dropout
        
        return q_model
    
    elif model_type == 'cvae':
        model_path = f"model/cvae_models/{dataset}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CVAE model {model_path} not found!")
        
        cvae = CVAE(state_dim=1, action_dim=1).to(device)
        cvae.load_state_dict(torch.load(model_path, map_location=device))
        cvae.eval()
        
        return cvae

import gc

def main():
    os.makedirs("visualization/comparison", exist_ok=True)
    
    expert_data = load_dataset('expert')
    medium_data = load_dataset('medium')
    
    print("Loading models...")
    expert_policy, diffusion_model = load_model('diffusion', 'expert')
    medium_policy, _ = load_model('diffusion', 'medium')
    
    expert_dynamics_models = load_model('dynamics', 'expert')
    medium_dynamics_models = load_model('dynamics', 'medium')
    
    expert_q_function = load_model('mc_dropout', 'expert')
    medium_q_function = load_model('mc_dropout', 'medium')
    
    expert_cvae = load_model('cvae', 'expert')
    medium_cvae = load_model('cvae', 'medium')
    
    fig = plt.figure(figsize=(30, 10))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1], height_ratios=[1, 1])
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_dataset_samples(ax1, expert_data, 'expert')
    
    ax2 = fig.add_subplot(gs[1, 0])
    plot_dataset_samples(ax2, medium_data, 'medium')
    
    print("Computing diffusion reconstruction errors...")
    ax3 = fig.add_subplot(gs[0, 1])
    im3 = plot_diffusion_recon_error(ax3, expert_policy, diffusion_model, 'expert')
    
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = plot_diffusion_recon_error(ax4, medium_policy, diffusion_model, 'medium')
    
    print("Computing model ensemble uncertainty...")
    ax5 = fig.add_subplot(gs[0, 2])
    im5 = plot_dynamics_uncertainty(ax5, expert_dynamics_models, 'expert')
    
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = plot_dynamics_uncertainty(ax6, medium_dynamics_models, 'medium')
    
    print("Computing MC Dropout uncertainty...")
    ax7 = fig.add_subplot(gs[0, 3])
    im7 = plot_mc_dropout_uncertainty(ax7, expert_q_function, 'expert')
    
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = plot_mc_dropout_uncertainty(ax8, medium_q_function, 'medium')
    
    # print("Computing CVAE NLL...")
    # ax9 = fig.add_subplot(gs[0, 4])
    # im9 = plot_cvae_nll(ax9, expert_cvae, 'expert')
    
    # ax10 = fig.add_subplot(gs[1, 4])
    # im10 = plot_cvae_nll(ax10, medium_cvae, 'medium')

    print("Plot CVAE Reconstruction Error...")
    ax9 = fig.add_subplot(gs[0, 4])
    im9 = plot_cvae_recon_error(ax9, expert_cvae, 'expert')
    
    ax10 = fig.add_subplot(gs[1, 4])
    im10 = plot_cvae_recon_error(ax10, medium_cvae, 'medium')
    
    cbar_ax = fig.add_axes([0.91, 0.09, 0.015, 0.85])
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), cax=cbar_ax)
    cbar.set_label('Normalized OOD Detection Metric', fontsize=20)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    gc.collect()
    plt.savefig("visualization/comparison/ood_detection_comparison.png", 
           dpi=200, 
           bbox_inches="tight",
           metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
    print("Comparison plot saved to visualization/comparison/ood_detection_comparison.png")
    

if __name__ == "__main__":
    main()