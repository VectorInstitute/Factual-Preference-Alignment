"""Launcher script that runs Factual-DPO++ training for all models across all Δ values."""

import subprocess

from utils.config_loader import load_config


def main():
    """Iterates over all models and deltas and spawns individual training jobs."""
    cfg = load_config()
    models = cfg["models"]
    deltas = cfg["modified_dpo"]["deltas"]

    print("Launching Modified FactualDPO Training for all models × all deltas...")

    for delta in deltas:
        for m in models:
            cmd = (
                f"python -m training.run_modified_training "
                f'--model_id "{m["id"]}" '
                f'--short "{m["short"]}" '
                f"--delta {delta}"
            )

            print(f"\n Running: {cmd}")
            subprocess.run(cmd, check=False, shell=True)


if __name__ == "__main__":
    main()
