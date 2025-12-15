"""
Build the FINAL evaluation dataset (skywork_final_eval.jsonl).

Composition:
    • 400 synthetic inversion samples (1,0)
    • all Skywork eval samples from skywork_first_transformed_eval.jsonl
    • +1500 samples of (1,1) from skywork_final_train.jsonl
    • +1500 samples of (0,0) from skywork_final_train.jsonl
      → excluding any sample already used in train_finallast.jsonl

Final eval ≈ (#sky_eval + 400 synthetic + 3000 added clean samples)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from utils.config_loader import load_config
from utils.data_utils import load_jsonl, write_jsonl


def main() -> None:
    """Build the FINAL evaluation dataset."""
    cfg = load_config()
    paths = cfg["paths"]
    hp = cfg["hyperparams"]

    synthetic_path = Path(paths["synthetic_eval_out"])
    sky_eval_path = Path(paths["skywork_eval_transformed"])
    train_full_path = Path(paths["final_train_merged"])
    train_used_path = Path(paths["final_train_out"])
    output_path = Path(paths["final_eval_out"])

    add_n = hp["eval_additional_clean_samples"]

    print(f"Loading synthetic eval → {synthetic_path}")
    synthetic = load_jsonl(synthetic_path)

    print(f"Loading Skywork eval transformed → {sky_eval_path}")
    sky_eval = load_jsonl(sky_eval_path)

    print(f"Loading full training dataset → {train_full_path}")
    sky_train = load_jsonl(train_full_path)

    print(f"Loading train-balanced dataset (to exclude) → {train_used_path}")
    train_used = load_jsonl(train_used_path)

    exclude = {(ex["prompt"], ex["chosen"], ex["rejected"]) for ex in train_used}

    pool_11: List[Dict[str, Any]] = []
    pool_00: List[Dict[str, Any]] = []

    for ex in sky_train:
        key = (ex["prompt"], ex["chosen"], ex["rejected"])
        if key in exclude:
            continue

        if ex["h_w"] == 1 and ex["h_l"] == 1:
            pool_11.append(ex)
        elif ex["h_w"] == 0 and ex["h_l"] == 0:
            pool_00.append(ex)

    print(f"(1,1) pool after exclusion: {len(pool_11)}")
    print(f"(0,0) pool after exclusion: {len(pool_00)}")

    sample_11 = random.sample(pool_11, add_n)
    sample_00 = random.sample(pool_00, add_n)

    merged: List[Dict[str, Any]] = []
    merged.extend(synthetic)
    merged.extend(sky_eval)
    merged.extend(sample_11)
    merged.extend(sample_00)

    print(f"\nTotal before shuffle: {len(merged)}")

    random.shuffle(merged)

    print(f"Saving final eval → {output_path}")
    write_jsonl(output_path, merged)

    print("\nFINAL EVAL DATASET READY.")
    print(f"Final count: {len(merged)}")


if __name__ == "__main__":
    random.seed(42)
    main()
