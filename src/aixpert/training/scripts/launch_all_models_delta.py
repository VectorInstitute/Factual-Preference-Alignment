"""Launch Factual-DPO training for all models across all delta values."""

import subprocess

from utils.config_loader import load_config


def main() -> None:
    """Launch training jobs for every model–delta combination."""
    cfg = load_config()
    models = cfg["models"]
    deltas = cfg["factual_dpo"]["deltas"]

    print("Launching FactualDPO training for all models × all deltas...")

    for delta in deltas:
        for m in models:
            cmd = (
                "python -m training.run_factual_training "
                f'--model_id "{m["id"]}" '
                f'--short "{m["short"]}" '
                f"--delta {delta}"
            )

            print(f"\n Running: {cmd}")
            subprocess.run(cmd, check=False, shell=True)


if __name__ == "__main__":
    main()
