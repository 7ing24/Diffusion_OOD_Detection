import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib import cm

STATE_MIN, STATE_MAX = -10, 10
ACTION_MIN, ACTION_MAX = -1, 1
NUM_STATE_BINS, NUM_ACTION_BINS = 1000, 1000


class toyEnv:
    def __init__(self):
        self.state_bins = np.linspace(STATE_MIN, STATE_MAX, NUM_STATE_BINS)
        self.action_bins = np.linspace(ACTION_MIN, ACTION_MAX, NUM_ACTION_BINS)
        
    def reward(self, state, action):
        return -np.abs(state - 0)
    
    def transition(self, state, action):
        next_state = state + action
        return np.clip(next_state, STATE_MIN, STATE_MAX)


def compute_true_q(env, gamma=0.99, num_iterations=1000):
    S, A = np.meshgrid(env.state_bins, env.action_bins, indexing="ij")
    
    R = env.reward(S, A)
    next_states = env.transition(S, A)
    next_state_idx = np.searchsorted(env.state_bins, next_states)
    next_state_idx = np.clip(next_state_idx, 0, len(env.state_bins) - 1)

    Q = np.zeros_like(R)
    for _ in range(num_iterations):
        max_Q_next = np.max(Q, axis=1)[next_state_idx]
        Q_new = R + gamma * max_Q_next
        if np.max(np.abs(Q_new - Q)) < 1e-5:
            break
        Q = Q_new
    return Q


def generate_dataset(env, true_q, dataset_type, num_samples):
    states = np.random.uniform(STATE_MIN, STATE_MAX, num_samples)

    # 根据 Q 值查找最优动作
    def get_optimal_actions(states, true_q, env):
        optimal_actions = []
        for s in states:
            # 找到离s最近的状态索引
            state_idx = np.searchsorted(env.state_bins, s)
            state_idx = np.clip(state_idx, 0, len(env.state_bins) - 1)

            q_values = true_q[state_idx, :]
            best_action_idx = np.argmax(q_values)
            best_action = env.action_bins[best_action_idx]
            optimal_actions.append(best_action)

        return np.array(optimal_actions)

    optimal_actions = get_optimal_actions(states, true_q, env)

    if dataset_type == 'expert':
        actions = optimal_actions + np.random.uniform(-0.05, 0.05, num_samples)
    elif dataset_type == 'medium':
        actions = optimal_actions + np.random.uniform(-0.5, 0.5, num_samples)
    else:
        actions = np.random.uniform(-0.5, 0.5, num_samples)

    actions = np.clip(actions, ACTION_MIN, ACTION_MAX)
    rewards = env.reward(states, actions)
    next_states = env.transition(states, actions)

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dataset_type': dataset_type
    }


def plot_ground_truth_q(env, true_q):
    plt.figure(figsize=(7.5, 6))
    
    states, actions = np.meshgrid(env.state_bins, env.action_bins, indexing='ij')
    
    plt.pcolormesh(states, actions, true_q, cmap='viridis', shading='auto')
    plt.xlabel('State', fontsize=20)
    plt.ylabel('Action', fontsize=20)
    plt.title('Ground truth Q-function', fontsize=20)
    
    cbar = plt.colorbar()
    cbar.set_label('Q-value', fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    os.makedirs("visualization/toy_env", exist_ok=True)
    plt.savefig('visualization/toy_env/ground_truth_q_heatmap.png', dpi=300)
    plt.show()


def plot_dataset_samples(expert_data, medium_data, random_data):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].scatter(
        expert_data['states'], 
        expert_data['actions'], 
        s=1, 
        alpha=0.7,
        color='blue',
        label='Expert samples'
    )
    axes[0].set_title('Expert Dataset', fontsize=24)
    axes[0].set_xlabel('State', fontsize=24)
    axes[0].set_ylabel('Action', fontsize=24)
    axes[0].set_xlim(STATE_MIN, STATE_MAX)
    axes[0].set_ylim(ACTION_MIN, ACTION_MAX)
    axes[0].set_xticks(np.linspace(STATE_MIN, STATE_MAX, 9), fontsize=22)  
    axes[0].set_yticks(np.linspace(ACTION_MIN, ACTION_MAX, 9), fontsize=22) 
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        medium_data['states'], 
        medium_data['actions'], 
        s=1, 
        alpha=0.7,
        color='red',
        label='Medium samples'
    )
    axes[1].set_title('Medium Dataset', fontsize=24)
    axes[1].set_xlabel('State', fontsize=24)
    axes[1].set_ylabel('Action', fontsize=24)
    axes[1].set_xlim(STATE_MIN, STATE_MAX)
    axes[1].set_ylim(ACTION_MIN, ACTION_MAX)
    axes[1].set_xticks(np.linspace(STATE_MIN, STATE_MAX, 9), fontsize=22)  
    axes[1].set_yticks(np.linspace(ACTION_MIN, ACTION_MAX, 9), fontsize=22) 
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(
        random_data['states'], 
        random_data['actions'], 
        s=1, 
        alpha=0.7,
        color='green',
        label='Random samples'
    )
    axes[2].set_title('Random Dataset Samples', fontsize=24)
    axes[2].set_xlabel('State', fontsize=24)
    axes[2].set_ylabel('Action', fontsize=24)
    axes[2].set_xlim(STATE_MIN, STATE_MAX)
    axes[2].set_ylim(ACTION_MIN, ACTION_MAX)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualization/toy_env/dataset_samples.png', dpi=300)
    plt.show()


def plot_dataset(expert_data, medium_data):
    plt.figure(figsize=(9, 8))
    
    plt.scatter(
        expert_data['states'], 
        expert_data['actions'], 
        s=1, 
        alpha=0.7,
        color='blue',
        label='Expert samples'
    )
    plt.title('Expert Dataset', fontsize=30)
    plt.xlabel('State', fontsize=30)
    plt.ylabel('Action', fontsize=30)
    plt.xlim(STATE_MIN, STATE_MAX)
    plt.ylim(ACTION_MIN, ACTION_MAX)
    plt.xticks(np.linspace(STATE_MIN, STATE_MAX, 9))  
    plt.yticks(np.linspace(ACTION_MIN, ACTION_MAX, 9)) 
    plt.grid(True, alpha=0.3)

    plt.savefig(f"visualization/toy_env/expert.png", dpi=300)
    print(f"Saved expert dataset scatter")
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 8))
    plt.scatter(
        medium_data['states'], 
        medium_data['actions'], 
        s=1, 
        alpha=0.7,
        color='red',
        label='Medium samples'
    )
    plt.title('Medium Dataset', fontsize=30)
    plt.xlabel('State', fontsize=30)
    plt.ylabel('Action', fontsize=30)
    plt.xlim(STATE_MIN, STATE_MAX)
    plt.ylim(ACTION_MIN, ACTION_MAX)
    plt.xticks(np.linspace(STATE_MIN, STATE_MAX, 9))  
    plt.yticks(np.linspace(ACTION_MIN, ACTION_MAX, 9)) 
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"visualization/toy_env/medium.png", dpi=300)
    plt.close()
    print(f"Saved medium dataset scatter")


def plot_dataset_qheatmap(env, expert_data, medium_data, random_data, true_q):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    def get_q_values(env, true_q, states, actions):
        state_idx = np.searchsorted(env.state_bins, states)
        state_idx = np.clip(state_idx, 0, len(env.state_bins) - 1)
        action_idx = np.searchsorted(env.action_bins, actions)
        action_idx = np.clip(action_idx, 0, len(env.action_bins) - 1)
        return true_q[state_idx, action_idx]

    sc1 = axes[0].scatter(
        expert_data['states'], 
        expert_data['actions'], 
        c=get_q_values(env, true_q, expert_data['states'], expert_data['actions']),
        cmap='viridis', 
        s=1, 
        alpha=0.3
    )
    axes[0].set_title('Expert Dataset Samples', fontsize=20)
    axes[0].set_xlabel('State', fontsize=20)
    axes[0].set_ylabel('Action', fontsize=20)
    plt.colorbar(sc1, ax=axes[0])
    
    sc2 = axes[1].scatter(
        medium_data['states'], 
        medium_data['actions'], 
        c=get_q_values(env, true_q, medium_data['states'], medium_data['actions']),
        cmap='viridis', 
        s=1, 
        alpha=0.3
    )
    axes[1].set_title('Medium Dataset Samples', fontsize=20)
    axes[1].set_xlabel('State', fontsize=20)
    axes[1].set_ylabel('Action', fontsize=20)
    plt.colorbar(sc2, ax=axes[1])

    sc3 = axes[2].scatter(
        random_data['states'],
        random_data['actions'],
        c=get_q_values(env, true_q, random_data['states'], random_data['actions']),
        cmap='viridis',
        s=1,
        alpha=0.3
    )
    axes[2].set_title('Random Dataset Samples', fontsize=20)
    axes[2].set_xlabel('State', fontsize=20)
    axes[2].set_ylabel('Action', fontsize=20)
    plt.colorbar(sc3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('visualization/toy_env/dataset_samples.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    env = toyEnv()
    os.makedirs("datasets", exist_ok=True)
    expert_dataset_path = "datasets/expert_data.pkl"
    medium_dataset_path = "datasets/medium_data.pkl"
    random_dataset_path = "datasets/random_data.pkl"
    
    print("Computing ground truth Q-values...")
    true_q = compute_true_q(env)
    
    print("Visualizing ground truth Q-values...")
    plot_ground_truth_q(env, true_q)

    
    print("Generating datasets...")
    expert_data = generate_dataset(env, true_q, dataset_type='expert', num_samples=500000)
    medium_data = generate_dataset(env, true_q, dataset_type='medium', num_samples=500000)
    random_data = generate_dataset(env, true_q, dataset_type='random', num_samples=500000)

    with open(expert_dataset_path, 'wb') as f:
        pickle.dump(expert_data, f)
    with open(medium_dataset_path, 'wb') as f:
        pickle.dump(medium_data, f)
    with open(random_dataset_path, 'wb') as f:
        pickle.dump(random_data, f)
    
    print("Visualizing dataset samples...")
    # plot_dataset_samples(expert_data, medium_data, random_data)
    plot_dataset(expert_data, medium_data)