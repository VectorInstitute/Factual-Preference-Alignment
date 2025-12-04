"""
Merge Skywork evaluation data with 400 synthetic inversion pairs.

This script:
- Loads synthetic corruption samples for eval.
- Loads Skywork eval transformed dataset.
- Splits samples into buckets by (h_w, h_l).
- Keeps ALL real eval samples.
- Merges synthetic + all real eval buckets.
- Shuffles and writes final eval JSONL file.

Fully compatible with ruff, mypy, and pydocstyle.
"""

from __future__ import annotations

import random
from pathlib import Path

from utils.config_loader import load_config
from utils.data_utils import bucket_by_flags, load_jsonl, write_jsonl


def main() -> None:
    """Merge Skywork eval data with synthetic eval inversion pairs."""
    cfg = load_config()
    paths = cfg["paths"]

    synthetic_path = Path(paths["synthetic_eval_out"])
    skywork_transformed_path = Path(paths["skywork_eval_transformed"])
    output_path = Path(paths["final_eval_merged"])

    print(f"📥 Loading synthetic eval → {synthetic_path}")
    synthetic = load_jsonl(synthetic_path)
    print(f"Synthetic eval count: {len(synthetic)}")

    print(f"📥 Loading transformed Skywork eval → {skywork_transformed_path}")
    sky = load_jsonl(skywork_transformed_path)
    print(f"Skywork eval count: {len(sky)}")

    # Split into buckets
    b00, b11, b01 = bucket_by_flags(sky)

    print(f"(0,0): {len(b00)}")
    print(f"(1,1): {len(b11)}")
    print(f"(0,1): {len(b01)}")

    # Eval uses ALL samples (no sampling)
    merged = synthetic + b00 + b11 + b01
    print(f"Total merged before shuffle: {len(merged)}")

    random.shuffle(merged)

    print(f"💾 Saving final merged eval → {output_path}")
    write_jsonl(output_path, merged)

    print("✅ EVAL MERGE COMPLETE.\n")


if __name__ == "__main__":
    main()
