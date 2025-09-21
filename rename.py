import wandb

api = wandb.Api()

entity = "7ingw-tongji-university"
project = "Diffusion_OOD_Detection"

runs = api.runs(f"{entity}/{project}")

for run in runs:
    if "cvae_gaussian" in run.config["method"]:
        print(f"Updating run: {run.id}, name={run.name}")

        run.config["method"] = "cvae_ood_action_detection"

        run.update()