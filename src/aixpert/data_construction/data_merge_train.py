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

import json
import random
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHETIC_FILE = DATA_DIR / "synthetic_llm_inversion_train_10k.jsonl"
SKYWORK_FILE = DATA_DIR / "skywork_first_transformed_train.jsonl"
OUTPUT_FILE = DATA_DIR / "skywork_final_train.jsonl"


# ============================================================
# Helpers
# ============================================================


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write list of dicts to JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Merge Skywork train data with 10k synthetic hallucinations."""
    print("📥 Loading synthetic inversions...")
    synthetic = load_jsonl(SYNTHETIC_FILE)
    print(f"Synthetic loaded: {len(synthetic)}")

    print("📥 Loading Skywork train transformed...")
    sky = load_jsonl(SKYWORK_FILE)
    print(f"Skywork loaded: {len(sky)}")

    hw0_hl0: List[Dict[str, Any]] = []
    hw1_hl1: List[Dict[str, Any]] = []
    hw0_hl1: List[Dict[str, Any]] = []

    for ex in sky:
        h_w = ex["h_w"]
        h_l = ex["h_l"]

        if h_w == 0 and h_l == 0:
            hw0_hl0.append(ex)
        elif h_w == 1 and h_l == 1:
            hw1_hl1.append(ex)
        elif h_w == 0 and h_l == 1:
            hw0_hl1.append(ex)

    print(f"(0,0): {len(hw0_hl0)}")
    print(f"(1,1): {len(hw1_hl1)}")
    print(f"(0,1): {len(hw0_hl1)}")

    random.seed(42)
    sample01 = random.sample(hw0_hl1, 10000)
    print(f"Sampled (0,1): {len(sample01)}")

    merged = synthetic + hw0_hl0 + hw1_hl1 + sample01
    print(f"Total merged before shuffle: {len(merged)}")

    random.shuffle(merged)

    print(f"💾 Saving → {OUTPUT_FILE}")
    write_jsonl(OUTPUT_FILE, merged)

    print("✅ TRAIN MERGE DONE.")


if __name__ == "__main__":
    main()
