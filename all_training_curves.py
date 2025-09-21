import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

os.makedirs("training_curves", exist_ok=True)

api = wandb.Api()

entity = "7ingw-tongji-university"     
project = "Training_curves"  
project_path = f"{entity}/{project}"

# 三个环境 × 三个数据集
envs = ["halfcheetah", "hopper", "walker2d"]
datasets = ["medium-v2", "medium-replay-v2", "medium-expert-v2"]

method_name_map = {
    "diffusion_ood_action_detection": "DOSER w/o AC and VC",
    "diffusion_ood_action_classification": "DOSER w/o VC",
    "diffusion_pos_ood_compensation": "DOSER",
}

plot_order = [
    "diffusion_ood_action_detection",
    "diffusion_ood_action_classification",
    "diffusion_pos_ood_compensation",
]

fig, axes = plt.subplots(len(envs), len(datasets), figsize=(15, 15), sharex=False, sharey=False)

for i, env in enumerate(envs):
    for j, dataset in enumerate(datasets):
        full_env = f"{env}-{dataset}"
        ax = axes[i, j]

        runs = api.runs(
            project_path,
            filters={"config.env": full_env, "state": "finished"}
        )

        print(f"\n===== {full_env}: {len(runs)} runs =====")

        grouped_histories = {}
        for run in runs:
            method = run.config.get("method", "unknown") 
            history = run.history(samples=100000, keys=["_step", "d4rl_score"])
            history = history.dropna(subset=["_step", "d4rl_score"])

            steps = history["_step"].to_numpy()
            scores = history["d4rl_score"].to_numpy()

            if method not in grouped_histories:
                grouped_histories[method] = []
            grouped_histories[method].append((steps, scores))

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

            ax.plot(steps, mean, label=method_name_map[method], alpha=1)
            ax.fill_between(steps, min_v, max_v, alpha=0.2)

            # 打印最终表现
            final_scores = all_scores[:, -1]
            final_mean = np.mean(final_scores)
            final_std = np.std(final_scores)
            print(f"{method_name_map[method]}: {final_mean:.1f} ± {final_std:.1f}")

        ax.set_title(full_env, fontsize=18)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
        if i == len(envs) - 1:
            ax.set_xlabel("Gradient Steps (×1e6)", fontsize=18)
        if j == 0:
            ax.set_ylabel("Normalized Score", fontsize=18)
        if i == 0 and j == 0:
            ax.legend(fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)

# 只在最后统一放 legend
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", ncol=len(plot_order), fontsize=18)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
            
plt.tight_layout()
plt.savefig("training_curves/all_training_curves.pdf")
plt.close()

print("Saved training curves: training_curves/all_training_curves.pdf")