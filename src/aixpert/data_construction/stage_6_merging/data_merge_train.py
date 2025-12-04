"""
Merge Skywork training data with 10k synthetic inversion pairs.

This script:
- Loads synthetic corruption samples.
- Loads transformed Skywork training data.
- Splits real samples into buckets by (h_w, h_l).
- Samples 10k from (0,1).
- Merges: synthetic + (0,0) + (1,1) + sampled (0,1).
- Shuffles and writes final JSONL file.

Fully compatible with ruff, mypy, and pydocstyle.
"""

from __future__ import annotations

import random
from pathlib import Path

from utils.config_loader import load_config
from utils.data_utils import bucket_by_flags, load_jsonl, write_jsonl


def main() -> None:
    """Merge Skywork train data with synthetic inversion pairs."""
    cfg = load_config()
    paths = cfg["paths"]
    hp = cfg["hyperparams"]

    synthetic_path = Path(paths["synthetic_train_out"])
    skywork_transformed_path = Path(paths["skywork_train_transformed"])
    output_path = Path(paths["final_train_merged"])

    sample_size = hp.get("merge_sample_01_train", 10000)

    print(f"📥 Loading synthetic → {synthetic_path}")
    synthetic = load_jsonl(synthetic_path)
    print(f"Synthetic count: {len(synthetic)}")

    print(f"📥 Loading transformed Skywork train → {skywork_transformed_path}")
    sky = load_jsonl(skywork_transformed_path)
    print(f"Skywork transformed count: {len(sky)}")

    # Bucket by (h_w, h_l)
    b00, b11, b01 = bucket_by_flags(sky)

    print(f"(0,0): {len(b00)}")
    print(f"(1,1): {len(b11)}")
    print(f"(0,1): {len(b01)}")

    # Sample subset of (0,1)
    random.seed(42)
    sample_01 = random.sample(b01, min(sample_size, len(b01)))
    print(f"Sampled (0,1): {len(sample_01)}")

    merged = synthetic + b00 + b11 + sample_01

    print(f"Total merged before shuffle: {len(merged)}")
    random.shuffle(merged)

    print(f"💾 Saving final merged train → {output_path}")
    write_jsonl(output_path, merged)

    print("✅ TRAIN MERGE COMPLETE.\n")


if __name__ == "__main__":
    main()
