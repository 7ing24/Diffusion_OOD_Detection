import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

os.makedirs("training_curves", exist_ok=True)

api = wandb.Api()

entity = "7ingw-tongji-university"     
project = "ablation_on_lam"  
project_path = f"{entity}/{project}"

envs = [
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
]

lam_values = [0.0001, 0.001, 0.01, 0.1, 0.5]

title_map = {
    "halfcheetah-medium-v2": "medium",
    "halfcheetah-medium-replay-v2": "medium-replay",
    "halfcheetah-medium-expert-v2": "medium-expert",
}

legend_map = {
    0.5: r"$\lambda=0.5$",
    0.1: r"$\lambda=0.1$",
    0.01: r"$\lambda=0.01$",
    0.001: r"$\lambda=0.001$",
    0.0001: r"$\lambda=0.0001$",
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False, constrained_layout=True)

for idx, env in enumerate(envs):
    runs = api.runs(
        project_path,
        filters={"config.env": env, "config.method": "diffusion_pos_ood_compensation", "state": "finished"}
    )
    print(f"{len(runs)} runs on {env}")

    grouped_histories = {}
    for run in runs:
        lam = run.config.get("lam", None)
        if lam not in lam_values:
            continue
        history = run.history(samples=100000, keys=["_step", "d4rl_score"])
        history = history.dropna(subset=["_step", "d4rl_score"])
        steps = history["_step"].to_numpy()
        scores = history["d4rl_score"].to_numpy()
        if lam not in grouped_histories:
            grouped_histories[lam] = []
        grouped_histories[lam].append((steps, scores))

    ax = axes[idx]

    for lam in lam_values:
        if lam not in grouped_histories:
            continue
        histories = grouped_histories[lam]
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
            ax.plot(steps, mean, label=legend_map[lam], alpha=1)
        else:
            ax.plot(steps, mean, alpha=1)

        ax.fill_between(steps, min_v, max_v, alpha=0.2)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
    ax.set_xlabel("Gradient Steps (×1e6)", fontsize=18)
    if idx == 0:
        ax.set_ylabel("Normalized Score", fontsize=18)
        ax.legend(fontsize=14)
    ax.set_title(title_map[env], fontsize=18)

plt.savefig("training_curves/ablation_on_lam.pdf")
plt.close()
print("Plot saved to training_curves/ablation_on_lam.pdf")