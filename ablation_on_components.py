import wandb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

os.makedirs("training_curves", exist_ok=True)
os.makedirs("ablation", exist_ok=True)

api = wandb.Api()
entity = "7ingw-tongji-university"
project = "Training_curves"
project_path = f"{entity}/{project}"

datasets = [
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-v2",
    "walker2d-medium-expert-v2"
]

dataset_abbr = {
    "halfcheetah-medium-replay-v2": "hc-mr",
    "halfcheetah-medium-v2": "hc-m",
    "halfcheetah-medium-expert-v2": "hc-me",
    "hopper-medium-replay-v2": "hp-mr",
    "hopper-medium-v2": "hp-m",
    "hopper-medium-expert-v2": "hp-me",
    "walker2d-medium-replay-v2": "wl-mr",
    "walker2d-medium-v2": "wl-m",
    "walker2d-medium-expert-v2": "wl-me"
}

full_method = "diffusion_pos_ood_compensation"
ablation_methods = ["diffusion_ood_action_detection",
                    "diffusion_ood_action_classification"]

results = {method: [] for method in [full_method] + ablation_methods}

for env in datasets:
    runs = api.runs(
        project_path,
        filters={"config.env": env, "state": "finished"}
    )
    for method in [full_method] + ablation_methods:
        final_scores = []
        for run in runs:
            if run.config.get("method") != method:
                continue
            history = run.history(samples=100000, keys=["_step", "d4rl_score"])
            history = history.dropna(subset=["d4rl_score"])
            if len(history) == 0:
                continue
            final_scores.append(history["d4rl_score"].to_numpy()[-1])
        if len(final_scores) > 0:
            results[method].append(np.mean(final_scores))  # 四个种子平均
        else:
            results[method].append(np.nan)

# 计算相对于 full method 的性能差异百分比
diff_comp_percent = 100 * (np.array(results[ablation_methods[0]]) - np.array(results[full_method])) / np.array(results[full_method])
diff_class_percent = 100 * (np.array(results[ablation_methods[1]]) - np.array(results[full_method])) / np.array(results[full_method])

# 找到所有差异的最大绝对值
ymax = max(np.nanmax(np.abs(diff_comp_percent)), np.nanmax(np.abs(diff_class_percent)))
ymax = np.ceil(ymax / 5) * 5   # 向上取整到5的倍数

x = np.arange(len(datasets)) * 0.7
width = 0.4

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

# 子图1：w/o classification and compensation
colors_comp = ['green' if val >= 0 else 'red' for val in diff_comp_percent]
axes[0].bar(x, diff_comp_percent, width, color=colors_comp, alpha=0.5)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels([dataset_abbr[d] for d in datasets], rotation=45, ha='right', fontsize=14)
axes[0].set_title("DOSER w/o AC and VC", fontsize=16)
axes[0].set_ylabel("Performance Difference (%)", fontsize=15)

# 子图2：w/o compensation
colors_class = ['green' if val >= 0 else 'red' for val in diff_class_percent]
axes[1].bar(x, diff_class_percent, width, color=colors_class, alpha=0.5)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels([dataset_abbr[d] for d in datasets], rotation=45, ha='right', fontsize=14)
axes[1].set_title("DOSER w/o VC", fontsize=16)

for ax in axes:
    ax.set_ylim(-ymax, ymax)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

plt.tight_layout()
plt.savefig("ablation/performance_difference.pdf")
plt.close()
print("Saved to ablation/performance_difference.pdf")