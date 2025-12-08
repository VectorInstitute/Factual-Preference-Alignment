"""Utilities for transforming factual-scored pairs into DPO-ready format."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm
from utils.data_utils import load_jsonl, write_jsonl


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one factual-scored item into DPO-ready structure."""
    prompt = item["prompt"]
    r0 = item["response_0"]
    r1 = item["response_1"]
    pref = int(item["better_response_id"])

    h0 = int(item["h0"])
    h1 = int(item["h1"])

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


def transform_dataset(input_path: Path, output_path: Path) -> None:
    """Load dataset, apply transformation, and save output JSONL."""
    print(f"Loading → {input_path}")
    items = load_jsonl(input_path)

    print(f"Transforming {len(items)} items…")
    transformed = [process_item(it) for it in tqdm(items)]

    print(f"Saving → {output_path}")
    write_jsonl(output_path, transformed)

    print("\n=======================================")
    print("TRANSFORMATION COMPLETE")
    print(f"Total items: {len(items)}")
    print("=======================================\n")
