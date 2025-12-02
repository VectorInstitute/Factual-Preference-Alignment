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

import json
import random
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
SYNTHETIC_FILE = DATA_DIR / "synthetic_llm_inversion_eval_400.jsonl"
SKYWORK_FILE = DATA_DIR / "skywork_first_transformed_eval.jsonl"
OUTPUT_FILE = DATA_DIR / "skywork_final_eval.jsonl"


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
    """Write a list of dicts to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Merge Skywork eval data with 400 synthetic inversions."""
    print("📥 Loading synthetic eval inversions...")
    synthetic = load_jsonl(SYNTHETIC_FILE)
    print(f"Synthetic eval: {len(synthetic)}")

    print("📥 Loading Skywork eval transformed...")
    sky = load_jsonl(SKYWORK_FILE)
    print(f"Skywork eval: {len(sky)}")

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

    merged = synthetic + hw0_hl0 + hw1_hl1 + hw0_hl1
    print(f"Total merged before shuffle: {len(merged)}")

    random.shuffle(merged)

    print(f"💾 Saving → {OUTPUT_FILE}")
    write_jsonl(OUTPUT_FILE, merged)

    print("✅ EVAL MERGE DONE.")


if __name__ == "__main__":
    main()
