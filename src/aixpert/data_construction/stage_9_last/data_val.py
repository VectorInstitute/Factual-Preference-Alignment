"""This script performs the final cleanup step for the train dataset (flipping + key filtering)."""

from __future__ import annotations

from pathlib import Path

from utils.config_loader import load_config
from utils.data_utils import process_jsonl_with_flip


def main():
    """Load paths and run the final processing stage for the val dataset."""
    paths = load_config()["paths"]

    input_path = Path(paths["eval_flipped_out"])
    output_path = Path(paths["final_eval"])

    process_jsonl_with_flip(input_path=input_path, output_path=output_path)

    print("val dataset final processing completed.")


if __name__ == "__main__":
    main()
