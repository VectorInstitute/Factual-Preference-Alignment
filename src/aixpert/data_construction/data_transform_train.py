"""
Transform binary factual-scored training preference pairs into DPO-ready format.

This script:
- Loads binary factual results for training pairs.
- Converts response_0 / response_1 into (chosen, rejected) strictly based on
  better_response_id.
- Copies factual flags into h_w (winner) and h_l (loser).
- Preserves original responses and adds a flipped=False flag.
- Writes the DPO-ready JSONL file for training.

Fully compatible with ruff, ruff-format, pydocstyle, and mypy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent / "data"

INPUT_PATH = DATA_DIR / "skywork_binary_factual_train.jsonl"
OUTPUT_PATH = DATA_DIR / "skywork_first_transformed_train.jsonl"


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one binary factual-scored pair into DPO-ready structure."""
    prompt = item["prompt"]
    r0 = item["response_0"]
    r1 = item["response_1"]
    pref = int(item["better_response_id"])

    # factual/hallucination flags
    h0 = int(item["h0"])
    h1 = int(item["h1"])

    # Determine chosen vs rejected based on preference label
    if pref == 0:
        chosen, rejected = r0, r1
        h_w, h_l = h0, h1
    else:
        chosen, rejected = r1, r0
        h_w, h_l = h1, h0

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "h_w": h_w,
        "h_l": h_l,
        "better_response_id": pref,
        "response_0": r0,
        "response_1": r1,
        "flipped": False,
    }


def transform_dataset() -> None:
    """Load training dataset, apply transformation, and save JSONL output."""
    print(f"📥 Loading training data → {INPUT_PATH}")
    items = [json.loads(line) for line in INPUT_PATH.open("r", encoding="utf-8")]

    transformed: List[Dict[str, Any]] = []

    print(f"⚙️ Processing {len(items)} items…")
    for item in tqdm(items):
        transformed.append(process_item(item))

    print(f"💾 Saving output → {OUTPUT_PATH}")
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for obj in transformed:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\n=======================================")
    print("✔ TRAIN DATASET TRANSFORMATION COMPLETE")
    print("✔ NO SAFE-DPO FLIPS APPLIED")
    print(f"Total items: {len(items)}")
    print("=======================================\n")


if __name__ == "__main__":
    transform_dataset()
