import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import random
import time
import wandb
import sys
sys.path.append(("../"))
from tqdm import trange
from utils import *
from diffusion_bc import *
from diffusion_ood_action_detection import *
from diffusion.karras import DiffusionModel
from diffusion.mlps import ScoreNetwork

D4RL_SUPPRESS_IMPORT_ERROR=1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_policy(agent, env_name, seed, mean, std, return_states=False, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)
    eval_env.action_space.seed(seed + seed_offset)
    agent.actor.eval()
    avg_reward = 0.
    visit_states = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            visit_states.append(state[0])
            action = agent.actor.act(state, device)
            action = np.asarray(action, dtype=np.float32)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100
    agent.actor.train()

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    if return_states:
        return d4rl_score, np.array(visit_states)
    return d4rl_score

def eval_Q_function(agent, episode_buffer, discount, mean, std, eval_episodes=10):
    mc_returns = []
    q_preds = []

    for _ in range(eval_episodes):
        episode = episode_buffer.sample_episode()
        states = torch.FloatTensor(episode["states"]).to(device)
        actions = torch.FloatTensor(episode["actions"]).to(device)
        rewards = episode["rewards"]
        dones = episode["dones"]

        # Monte Carlo returns
        G = 0
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + discount * G * (1. - d)  # reset if done
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        # Q-function predictions
        with torch.no_grad():
            q1, q2, q3, q4 = agent.critic(states, actions)
            q_pred = torch.min(torch.min(q1, q2), torch.min(q3, q4))

        # **立即移到 CPU，避免 GPU 堆积**
        mc_returns.append(returns.cpu())
        q_preds.append(q_pred.cpu())

    mc_returns = torch.cat(mc_returns, dim=0)
    q_preds = torch.cat(q_preds, dim=0)
    bias = (q_preds - mc_returns).mean().item()

    print(f"[Evaluation over {eval_episodes} episodes]")
    print(f"Monte Carlo Q: {mc_returns.mean().item():.3f}")
    print(f"Predicted Q: {q_preds.mean().item():.3f}")
    print(f"Bias (Q_pred - MC): {bias:.3f}")

    return mc_returns, q_preds


def visualize_states(dataset_states, visit_states, env, n_samples=5000):
    if len(dataset_states) > n_samples:
        dataset_states = dataset_states[np.random.choice(len(dataset_states), size=n_samples, replace=False)]
    if len(visit_states) > n_samples:
        visit_states = visit_states[np.random.choice(len(visit_states), size=n_samples, replace=False)]

    all_states = np.concatenate([dataset_states, visit_states])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    all_states_2d = tsne.fit_transform(all_states)
    dataset_states_2d = all_states_2d[:len(dataset_states)]
    visit_states_2d = all_states_2d[len(dataset_states):]

    plt.figure(figsize=(10, 8))
    plt.scatter(dataset_states_2d[:, 0], dataset_states_2d[:, 1], 
                c='blue', s=5, alpha=0.3, label='Dataset States')
    plt.scatter(visit_states_2d[:, 0], visit_states_2d[:, 1], 
                c='red', s=5, alpha=0.5, label='Visited States')
    plt.title(f"{env}\nState Distribution")
    plt.legend()
    plt.grid()
    
    os.makedirs("State_distribution", exist_ok=True)
    plt.savefig(f"State_distribution/{env}.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-replay-v2")
    parser.add_argument("--method", default="diffusion_ood_action_detection")  # diffusion, cvae, svr
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--eval_freq", default=2e4, type=int)
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_timesteps", default=1e6, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99)
    parser.add_argument("--tau", default=0.005)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--target_update_freq", default=2, type=int)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--no_schedule", action="store_true")
    parser.add_argument("--beta", default=0.001, type=float)  # penalty coefficient
    parser.add_argument("--action_n_levels", default=1, type=int)  # number of action levels for OOD detection
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    # Set Q_min according to environment
    if "hopper" in args.env:
        Q_min = -125
    elif "halfcheetah" in args.env:
        Q_min = -366
    elif "walker2d" in args.env:
        Q_min = -471
    elif "pen" in args.env:
        Q_min = -715
    elif "antmaze" in args.env:
        Q_min = -200
        replay_buffer.antmaze_reward_tune()

    if not args.no_normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    dataset_states = replay_buffer.state  # 已归一化
    episode_replay_buffer = EpisodeReplayBuffer(replay_buffer, episode_length=1000)
    
    # Diffusion behavior model
    bc_model_path = f'../Diffusion_models/bc_models/{args.env}.pth'
    diffusion_model = DiffusionModel(
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=80,
        device=device,
    )
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

    assert os.path.exists(bc_model_path), f"Behavior policy model {bc_model_path} not found!"
    behavior_model.load_state_dict(torch.load(bc_model_path, map_location=device))
    behavior_model.eval()

    threshold = round(get_threshold(behavior_model, diffusion_model, args.env, args.no_normalize, args.action_n_levels, args.batch_size), 6)
    print(f"Threshold for env '{args.env}' with action_n_levels {args.action_n_levels}: {threshold}")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "replay_buffer": replay_buffer,
        "behavior_model": behavior_model,
        "diffusion_model": diffusion_model,
        "discount": args.discount,
        "tau": args.tau,
        "policy_freq": args.policy_freq,
        "target_update_freq": args.target_update_freq,
        "schedule": not args.no_schedule,
        "Q_min": Q_min,
        "beta": args.beta,
        "action_n_levels": args.action_n_levels,
        "threshold": threshold
    }

    agent = Diffusion_ood_action_detection(**kwargs)

    wandb.init(
        project="Diffusion_OOD_detection",
        name=f"{args.env}_{args.method}_beta{args.beta}_nlevels{args.action_n_levels}_threshold{threshold}_seed{args.seed}",
        config=vars(args)
    )

    for t in trange(int(args.max_timesteps)):
        agent.train(args.batch_size)
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t+1}")
            d4rl_score = eval_policy(agent, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes)
            mc_returns, q_preds = eval_Q_function(agent, episode_replay_buffer, args.discount, mean, std, eval_episodes=args.eval_episodes)
            wandb.log({
                "d4rl_score": d4rl_score,
                "Q_eval_comparison/MC Return": mc_returns.mean().item(),
                "Q_eval_comparison/Q Prediction": q_preds.mean().item(),
                "Q_eval_comparison/Bias": (q_preds - mc_returns).mean().item()
                }, step=t+1)
            
        if t == int(args.max_timesteps) - 1:
            _, visit_states = eval_policy(agent, args.env, args.seed, mean, std, eval_episodes=args.eval_episodes, return_states=True)
            visualize_states(dataset_states, visit_states, args.env)
            print("State distribution visualization saved.")

    time.sleep(10)
