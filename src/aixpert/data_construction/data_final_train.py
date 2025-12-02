"""
Balanced sampling for TRAIN dataset.

This script:
- Loads the merged training dataset.
- Buckets by (h_w, h_l).
- Samples required amounts per bucket (with replacement if needed).
- Shuffles and saves the final balanced training dataset.

Buckets required:
    (0,1) → 10,000
    (1,0) → 10,000
    (0,0) → 15,000
    (1,1) → 10,000
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# Paths (relative to this file's /data directory)
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"

INPUT_FILE = DATA_DIR / "skywork_final_train.jsonl"
OUTPUT_FILE = DATA_DIR / "train_finallast.jsonl"

TARGET_COUNTS: Dict[Tuple[int, int], int] = {
    (0, 1): 10_000,
    (1, 0): 10_000,
    (0, 0): 15_000,
    (1, 1): 10_000,
}


# ============================================================
# Helpers
# ============================================================


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return its rows as a list of dictionaries."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for ex in rows:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Generate the balanced training dataset according to bucket size targets."""
    print(f"📥 Loading dataset → {INPUT_FILE}")
    data = load_jsonl(INPUT_FILE)

    # bucket structure
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {
        (0, 1): [],
        (1, 0): [],
        (0, 0): [],
        (1, 1): [],
    }

    print("🔍 Bucketing samples...")
    for ex in data:
        key = (int(ex["h_w"]), int(ex["h_l"]))
        if key in buckets:
            buckets[key].append(ex)

    final_samples: List[Dict[str, Any]] = []

    for key, req_count in TARGET_COUNTS.items():
        available = len(buckets[key])
        print(f"Bucket {key}: available={available}, required={req_count}")

        if available < req_count:
            print("⚠️ Not enough samples — sampling WITH replacement.")
            sampled = random.choices(buckets[key], k=req_count)
        else:
            sampled = random.sample(buckets[key], req_count)

        final_samples.extend(sampled)

    print(f"\n🔀 Shuffling {len(final_samples)} samples...")
    random.shuffle(final_samples)

    print(f"💾 Saving → {OUTPUT_FILE}")
    write_jsonl(OUTPUT_FILE, final_samples)

    print("✅ TRAIN balanced dataset created.")
    print("Final count:", len(final_samples))


if __name__ == "__main__":
    main()
