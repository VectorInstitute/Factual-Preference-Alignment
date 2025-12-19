"""Launch Factual-DPO++ training for all models across all delta values."""

import subprocess

from utils.config_loader import load_config


def main() -> None:
    """Launch training jobs for every model–delta combination."""
    cfg = load_config()
    models = cfg["models"]
    deltas = cfg["modified_dpo"]["deltas"]

    print("Launching Modified FactualDPO training for all models × all deltas...")

    for delta in deltas:
        for m in models:
            cmd = (
                "python -m training.run_modified_training "
                f'--model_id "{m["id"]}" '
                f'--short "{m["short"]}" '
                f"--delta {delta}"
            )

            print(f"\n Running: {cmd}")
            subprocess.run(cmd, check=False, shell=True)


if __name__ == "__main__":
    main()
