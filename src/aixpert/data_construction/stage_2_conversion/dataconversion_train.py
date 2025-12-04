"""
Generate training preference pairs from cleaned Skywork samples.

Loads prompt/chosen/rejected rows from the cleaned 77k dataset,
creates random preference pairs (response_0/response_1),
assigns correct better_response_id, and writes JSONL output.

This script uses the shared data utilities and config loader.
"""

from __future__ import annotations

from pathlib import Path

from utils.config_loader import load_config
from utils.data_utils import (
    create_preference_pairs,
    load_jsonl,
    write_jsonl,
)


def main() -> None:
    """Generate preference pairs for the training set."""
    cfg = load_config()
    paths = cfg["paths"]

    input_path = Path(paths["skywork_train_cleaned"])
    output_path = Path(paths["skywork_train_pairs"])

    print(f"Loading training dataset → {input_path}")

    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} rows")

    preference_pairs = create_preference_pairs(data)

    write_jsonl(output_path, preference_pairs)

    print("======================================")
    print(f"Training preference pairs saved → {output_path}")
    print(f"Total pairs: {len(preference_pairs)}")
    print("======================================")


if __name__ == "__main__":
    main()
