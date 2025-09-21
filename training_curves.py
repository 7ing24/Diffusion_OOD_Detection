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
env = "halfcheetah-medium-v2"

runs = api.runs(
    project_path,
    filters={"config.env": env, "state": "finished"}  # 只取已完成的
)

print(f"{len(runs)} runs on {env}")

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

method_name_map = {
    # "cvae_ood_action_detection": "CVAE-based OOD Action Detection",
    "diffusion_ood_action_detection": "Diffusion-based OOD Action Detection",
    "diffusion_ood_action_classification": "Diffusion-based OOD Action Classification",
    "diffusion_pos_ood_compensation": "DOSER",
}

plot_order = [
    # "cvae_ood_action_detection",
    "diffusion_ood_action_detection",
    "diffusion_ood_action_classification",
    "diffusion_pos_ood_compensation",
]

plt.figure(figsize=(5, 5))

handles, labels = [], []

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
    std = np.std(all_scores, axis=0)

    h, = plt.plot(steps, mean, label=method_name_map[method], alpha=0.7)
    # plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    mean = np.mean(all_scores, axis=0)
    min_v = np.min(all_scores, axis=0)
    max_v = np.max(all_scores, axis=0)
    plt.fill_between(steps, min_v, max_v, alpha=0.2)

    handles.append(h)
    labels.append(method_name_map[method])

    final_scores = all_scores[:, -1]
    final_mean = np.mean(final_scores)
    final_std = np.std(final_scores)
    print(f"{method_name_map[method]}: {final_mean:.1f} ± {final_std:.1f}")

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
plt.xlabel("Gradient Steps (×1e6)", fontsize=14)
plt.ylabel("Normalized Score", fontsize=14)
plt.title(f"{env}", fontsize=14)
plt.legend(handles, labels)
plt.tight_layout()
plt.savefig(f"training_curves/{env}.pdf")
plt.close()

print(f"Save training curves on {env}.pdf")

