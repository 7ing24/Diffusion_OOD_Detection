import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

api = wandb.Api()
project_path = "7ingw-tongji-university/OOD_detection_threshold"

custom_groups = {
    "halfcheetah-medium-v2": {
        "80th percentile": ["k5asw63h", "28l0cvpc"], 
        "90th percentile": ["cqtkj7rr", "x6sll8g0"],
        "99th percentile": ["beegeulc", "m1ud4r3x", "m4e2kpgn", "v9u3n5kv"],
    },
    "halfcheetah-medium-replay-v2": {
        "80th percentile": ["o7g9aeo0", "msimxj3x"], 
        "90th percentile": ["ttqkidsa", "4wwlh8rv"],
        "99th percentile": ["ydm9kk7e", "zwh1ecyv", "wunpvp7z", "ds38cb73"],
    },
    "halfcheetah-medium-expert-v2": {
        "80th percentile": ["tjt34ig6", "t4uuh5aq", "3isjj2nw", "qee9xtkp"], 
        "90th percentile": ["a6uu6lnv", "5ur1m0hs"],
        "99th percentile": ["kedpr8zl", "lmylscad"],
    }
}

title_map = {
    "halfcheetah-medium-v2": "medium",
    "halfcheetah-medium-replay-v2": "medium-replay",
    "halfcheetah-medium-expert-v2": "medium-expert",
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False, constrained_layout=True)

for idx, (dataset, groups) in enumerate(custom_groups.items()):
    ax = axes[idx]

    for group_name, run_ids in groups.items():
        if not run_ids:  # 如果某个组为空，跳过
            continue
        histories = []
        for run_id in run_ids:
            run = api.run(f"{project_path}/{run_id}")
            history = run.history(keys=["_step", "d4rl_score"]).dropna()
            if len(history) > 0:
                histories.append(history)

        if not histories:
            continue

        min_len = min(len(h) for h in histories)
        scores = np.stack([h["d4rl_score"].to_numpy()[:min_len] for h in histories], axis=0)
        steps = histories[0]["_step"].to_numpy()[:min_len]

        mean = np.mean(scores, axis=0)
        min_v = np.min(scores, axis=0)
        max_v = np.max(scores, axis=0)

        ax.plot(steps, mean, label=group_name, alpha=1)
        ax.fill_between(steps, min_v, max_v, alpha=0.2)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
    ax.set_xlabel("Gradient Steps (×1e6)", fontsize=18)
    if idx == 0: 
        ax.set_ylabel("Normalized Score", fontsize=18)
        ax.legend(fontsize=14)
    ax.set_title(title_map[dataset], fontsize=18)

# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=15)
# plt.subplots_adjust(top=0.82)

plt.savefig("training_curves/ablation_on_threshold.pdf")
plt.close()
print("Plot saved as training_curves/ablation_on_threshold.pdf")