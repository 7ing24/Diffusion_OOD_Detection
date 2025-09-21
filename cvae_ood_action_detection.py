import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

os.makedirs("training_curves", exist_ok=True)

api = wandb.Api()

entity = "7ingw-tongji-university"     
project = "CVAE_VS_Diffusion"  
project_path = f"{entity}/{project}"

# 数据集
envs = [
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
]

plot_order = [
    "cvae_ood_action_detection",
    "diffusion_ood_action_detection",
]

method_name_map = {
    "cvae_ood_action_detection": "CVAE-based OOD Action Detection",
    "diffusion_ood_action_detection": "Diffusion-based OOD Action Detection",
}

fig, axes = plt.subplots(3, 3, figsize=(12, 7), sharey=False)
axes = axes.flatten()

global_handles, global_labels = [], []

for idx, env in enumerate(envs):
    runs = api.runs(
        project_path,
        filters={"config.env": env, "state": "finished"}
    )
    print(f"{len(runs)} runs on {env}")

    grouped_histories = {}
    for run in runs:
        method = run.config.get("method", "unknown") 
        if method not in plot_order: 
            continue
        history = run.history(samples=100000, keys=["_step", "d4rl_score"])
        history = history.dropna(subset=["_step", "d4rl_score"])
        steps = history["_step"].to_numpy()
        scores = history["d4rl_score"].to_numpy()
        if method not in grouped_histories:
            grouped_histories[method] = []
        grouped_histories[method].append((steps, scores))

    ax = axes[idx]

    for method in plot_order:
        if method not in grouped_histories:
            continue
        histories = grouped_histories[method]
        valid_histories = [h for h in histories if len(h[1]) > 0]
        if not valid_histories:
            continue

        min_len = min(len(h[1]) for h in valid_histories)
        all_scores = np.stack([h[1][:min_len] for h in valid_histories], axis=0)
        steps = valid_histories[0][0][:min_len]

        mean = np.mean(all_scores, axis=0)
        min_v = np.min(all_scores, axis=0)
        max_v = np.max(all_scores, axis=0)

        h, = ax.plot(steps, mean, label=method_name_map[method], alpha=1)
        ax.fill_between(steps, min_v, max_v, alpha=0.2)

        if idx == 0:
            global_handles.append(h)
            global_labels.append(method_name_map[method])
            # ax.legend(loc="upper left", fontsize=14)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))

    # 计算行列号
    row, col = divmod(idx, 3)
    if col == 0:
        ax.set_ylabel("Normalized Score", fontsize=14)
    if row == 2:
        ax.set_xlabel("Gradient Steps (×1e6)", fontsize=14)

    ax.set_title(env, fontsize=14)
    ax.tick_params(labelleft=True, labelbottom=True)

# 单独画图例
fig.legend(global_handles, global_labels, loc="upper center", ncol=2, fontsize=14)
plt.subplots_adjust(top=0.82)  

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("training_curves/cvae_vs_diffusion_9.pdf")
plt.close()