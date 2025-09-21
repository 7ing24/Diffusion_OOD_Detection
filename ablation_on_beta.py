import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

os.makedirs("training_curves", exist_ok=True)

api = wandb.Api()

entity = "7ingw-tongji-university"     
project = "ablation_on_beta"  
project_path = f"{entity}/{project}"

# 四个数据集
envs = [
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
]

beta_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

# 用字典映射，把完整env名字转成简洁标题
title_map = {
    "halfcheetah-medium-v2": "medium",
    "halfcheetah-medium-replay-v2": "medium-replay",
    "halfcheetah-medium-expert-v2": "medium-expert",
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False, constrained_layout=True)

for idx, env in enumerate(envs):
    runs = api.runs(
        project_path,
        filters={"config.env": env, "config.method": "diffusion_ood_action_detection", "state": "finished"}
    )
    print(f"{len(runs)} runs on {env}")

    grouped_histories = {}
    for run in runs:
        beta = run.config.get("beta", None)
        if beta not in beta_values:
            continue
        history = run.history(samples=100000, keys=["_step", "d4rl_score"])
        history = history.dropna(subset=["_step", "d4rl_score"])
        steps = history["_step"].to_numpy()
        scores = history["d4rl_score"].to_numpy()
        if beta not in grouped_histories:
            grouped_histories[beta] = []
        grouped_histories[beta].append((steps, scores))

    ax = axes[idx]

    for beta in beta_values:
        if beta not in grouped_histories:
            continue
        histories = grouped_histories[beta]
        valid_histories = [h for h in histories if len(h[1]) > 0]
        if not valid_histories:
            continue

        min_len = min(len(h[1]) for h in valid_histories)
        all_scores = np.stack([h[1][:min_len] for h in valid_histories], axis=0)
        steps = valid_histories[0][0][:min_len]

        mean = np.mean(all_scores, axis=0)
        min_v = np.min(all_scores, axis=0)
        max_v = np.max(all_scores, axis=0)

        if idx == 0:
            ax.plot(steps, mean, label=f"β={beta}", alpha=1)
        else:
            ax.plot(steps, mean, alpha=1)

        ax.fill_between(steps, min_v, max_v, alpha=0.2)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
    ax.set_xlabel("Gradient Steps (×1e6)", fontsize=15)
    if idx == 0:
        ax.set_ylabel("Normalized Score", fontsize=15)
        ax.legend(fontsize=12)

    # 用映射后的简洁标题
    ax.set_title(title_map[env], fontsize=15)

plt.savefig("training_curves/ablation_on_beta.pdf")
plt.close()
print("Plot saved to training_curves/ablation_on_beta.pdf")