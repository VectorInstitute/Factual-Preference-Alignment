"""Launch Original DPO training for all configured models."""

import subprocess

from utils.config_loader import load_config


def main() -> None:
    """Launch baseline DPO training for each configured model."""
    cfg = load_config()
    models = cfg["models"]

    for m in models:
        cmd = f'python training/run_dpo_training.py --model "{m["id"]}"'
        print(f"Launching: {cmd}")
        subprocess.run(cmd, check=False, shell=True)


if __name__ == "__main__":
    main()
