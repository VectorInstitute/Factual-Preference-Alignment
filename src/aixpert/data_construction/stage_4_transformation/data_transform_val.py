"""Transform factual-scored evaluation pairs into DPO-ready format."""

from __future__ import annotations

from pathlib import Path

from utils.config_loader import load_config
from utils.dpo_transform_utils import transform_dataset


def main() -> None:
    """Run dataset transformation for factual-scored validation pairs."""
    paths = load_config()["paths"]

    input_path = Path(paths["skywork_eval_factual"])
    output_path = Path(paths["skywork_eval_transformed"])

    transform_dataset(input_path, output_path)


if __name__ == "__main__":
    main()
