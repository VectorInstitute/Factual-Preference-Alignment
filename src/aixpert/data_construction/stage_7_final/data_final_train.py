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

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.config_loader import load_config
from utils.data_utils import load_jsonl, write_jsonl


def main() -> None:
    """Balanced sampling for TRAIN dataset."""
    cfg = load_config()
    paths = cfg["paths"]
    hp = cfg["hyperparams"]

    input_path = Path(paths["skywork_final_train"])
    output_path = Path(paths["final_train_out"])

    target_counts: Dict[Tuple[int, int], int] = hp["balance_targets"]

    print(f"Loading → {input_path}")
    data = load_jsonl(input_path)

    # Initialize buckets
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {
        (0, 1): [],
        (1, 0): [],
        (0, 0): [],
        (1, 1): [],
    }

    print("Bucketing samples…")
    for ex in data:
        key = (int(ex["h_w"]), int(ex["h_l"]))
        if key in buckets:
            buckets[key].append(ex)

    print("\n=== AVAILABLE PER BUCKET ===")
    for key, rows in buckets.items():
        print(f"{key}: {len(rows)}")

    final_rows: List[Dict[str, Any]] = []

    # Sampling logic
    for key, req_count in target_counts.items():
        pool = buckets[key]
        available = len(pool)

        print(f"\nBucket {key}: available={available}, required={req_count}")

        if available < req_count:
            print("⚠️ Sampling WITH replacement.")
            sampled = random.choices(pool, k=req_count)
        else:
            sampled = random.sample(pool, req_count)

        final_rows.extend(sampled)

    print(f"\nShuffling {len(final_rows)} rows…")
    random.shuffle(final_rows)

    print(f"Saving → {output_path}")
    write_jsonl(output_path, final_rows)

    print("\nTRAIN balanced dataset ready.")
    print(f"Final count: {len(final_rows)}")


if __name__ == "__main__":
    random.seed(42)
    main()
