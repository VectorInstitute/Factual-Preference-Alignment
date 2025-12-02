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

import json
import random
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"

SYNTHETIC_FILE = DATA_DIR / "synthetic_llm_inversion_eval_400.jsonl"
SKY_EVAL_FILE = DATA_DIR / "skywork_first_transformed_eval.jsonl"

TRAIN_SOURCE_FILE = DATA_DIR / "skywork_final_train.jsonl"
TRAIN_USED_FILE = DATA_DIR / "train_finallast.jsonl"

OUTPUT_FILE = DATA_DIR / "eval_final.jsonl"


# ============================================================
# HELPERS
# ============================================================


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Create the final evaluation dataset by merging all required sources."""
    print("📥 Loading synthetic eval inversions...")
    synthetic = load_jsonl(SYNTHETIC_FILE)
    print("Synthetic:", len(synthetic))

    print("📥 Loading Skywork eval transformed...")
    sky_eval = load_jsonl(SKY_EVAL_FILE)
    print("SkyEval:", len(sky_eval))

    print("📥 Loading Skywork full training source...")
    sky_train = load_jsonl(TRAIN_SOURCE_FILE)

    print("📥 Loading TRAIN used (to exclude)...")
    train_used = load_jsonl(TRAIN_USED_FILE)

    # Convert used samples to hashable form
    exclude_set = {(ex["prompt"], ex["chosen"], ex["rejected"]) for ex in train_used}

    # -----------------------------------------------------------
    # 1. Extract (1,1) and (0,0) pools from training source
    # -----------------------------------------------------------
    hw1_hl1_pool: List[Dict[str, Any]] = []
    hw0_hl0_pool: List[Dict[str, Any]] = []

    for ex in sky_train:
        key = (ex["prompt"], ex["chosen"], ex["rejected"])
        if key in exclude_set:
            continue

        if ex["h_w"] == 1 and ex["h_l"] == 1:
            hw1_hl1_pool.append(ex)
        elif ex["h_w"] == 0 and ex["h_l"] == 0:
            hw0_hl0_pool.append(ex)

    print(f"(1,1) available for eval add: {len(hw1_hl1_pool)}")
    print(f"(0,0) available for eval add: {len(hw0_hl0_pool)}")

    # -----------------------------------------------------------
    # 2. Sample EXACT 1500 from each bucket
    # -----------------------------------------------------------
    eval_hw1_hl1 = random.sample(hw1_hl1_pool, 1500)
    eval_hw0_hl0 = random.sample(hw0_hl0_pool, 1500)

    # -----------------------------------------------------------
    # 3. Merge everything
    # -----------------------------------------------------------
    merged: List[Dict[str, Any]] = []
    merged.extend(synthetic)  # (1,0) → 400
    merged.extend(sky_eval)  # (0,1) → ~1000
    merged.extend(eval_hw1_hl1)  # (1,1) → 1500
    merged.extend(eval_hw0_hl0)  # (0,0) → 1500

    print(f"Total before shuffle: {len(merged)}")

    random.shuffle(merged)

    print(f"💾 Saving → {OUTPUT_FILE}")
    write_jsonl(OUTPUT_FILE, merged)

    print("✅ FINAL EVAL DATASET READY.")
    print("Total eval:", len(merged))


if __name__ == "__main__":
    random.seed(42)
    main()
