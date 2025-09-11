import wandb
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("training_curves", exist_ok=True)

api = wandb.Api()

entity = "7ingw-tongji-university"     
project = "Diffusion_OOD_Detection"  
project_path = f"{entity}/{project}"
env = "hopper-medium-replay-v2"

runs = api.runs(
    project_path,
    filters={"config.env": env, "state": "finished"}  # 只取已完成的
)

print(f"{len(runs)} runs on {env}")

grouped_histories = {}

for run in runs:
    method = run.config.get("method", "unknown")  # 取 config.method
    history = run.history(samples=100000, keys=["_step", "d4rl_score"])
    history = history.dropna(subset=["_step", "d4rl_score"])

    steps = history["_step"].to_numpy()
    scores = history["d4rl_score"].to_numpy()

    if method not in grouped_histories:
        grouped_histories[method] = []
    grouped_histories[method].append((steps, scores))

method_name_map = {
    "diffusion_ood_action_detection": "OOD Action Detection",
    "diffusion_ood_action_classification": "OOD Action Classification",
    "diffusion_pos_ood_compensation": "DODR",
}

plot_order = [
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
    plt.fill_between(steps, mean - std, mean + std, alpha=0.2)
    handles.append(h)
    labels.append(method_name_map[method])

plt.xlabel("Gradient Steps", fontsize=14)
plt.ylabel("Normalized Score", fontsize=14)
plt.title(f"{env}", fontsize=14)
plt.legend(handles, labels)
plt.tight_layout()
plt.savefig(f"training_curves/{env}.pdf")
plt.close()

print(f"Save training curves on {env}.pdf")