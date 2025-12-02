"""
Flip preference labels for evaluation data.

This script:
- Converts h_w=1,h_l=0 → h_w=0,h_l=1
- Swaps chosen/rejected
- Writes a flipped version of the dataset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


DATA_DIR = Path(__file__).resolve().parent / "data"

INPUT_FILE = DATA_DIR / "eval_final.jsonl"
OUTPUT_FILE = DATA_DIR / "eval_final_flipped.jsonl"


def flip_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """Flip the sample if (h_w, h_l) = (1, 0)."""
    if item.get("h_w") == 1 and item.get("h_l") == 0:
        item["h_w"], item["h_l"] = 0, 1
        item["chosen"], item["rejected"] = item["rejected"], item["chosen"]
    return item


def main() -> None:
    """Execute flipping process for evaluation dataset."""
    print("📥 Loading input file:", INPUT_FILE)

    output: List[Dict[str, Any]] = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            output.append(flip_sample(item))

    print(f"✅ Processed {len(output)} samples")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print("💾 Saving flipped dataset to:", OUTPUT_FILE)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n============================================")
    print(f"🎉 Saved flipped dataset → {OUTPUT_FILE.name}")
    print("============================================\n")


if __name__ == "__main__":
    main()
