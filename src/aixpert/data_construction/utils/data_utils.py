"""Utility functions for dataset extraction, cleaning, formatting, and flipping.

These helpers are used across the data-construction pipeline for DPO, SafeDPO,
Factual-DPO, and evaluation preprocessing.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def extract_prompt(dialog: List[Dict[str, Any]]) -> str:
    """Extract the first user message."""
    for msg in dialog:
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def extract_answer(dialog: List[Dict[str, Any]]) -> str:
    """Extract the first assistant reply."""
    for msg in dialog:
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write list of dictionaries to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_duplicates(
    rows: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split rows into cleaned (chosen != rejected) and removed (exact duplicates)."""
    cleaned: List[Dict[str, str]] = []
    removed: List[Dict[str, str]] = []

    for row in rows:
        if row["chosen"] == row["rejected"]:
            removed.append(row)
        else:
            cleaned.append(row)

    return cleaned, removed


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write list of dictionaries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_preference_pairs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert rows into DPO preference-pair format."""
    output: List[Dict[str, Any]] = []

    for item in data:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        # Random symmetric assignment
        if random.random() < 0.5:
            response_0 = chosen
            response_1 = rejected
            better_response_id = 0
        else:
            response_0 = rejected
            response_1 = chosen
            better_response_id = 1

        output.append(
            {
                "prompt": prompt,
                "response_0": response_0,
                "response_1": response_1,
                "better_response_id": better_response_id,
            }
        )

    return output


def bucket_by_flags(
    items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split items into (0,0), (1,1), and (0,1) buckets."""
    b00, b11, b01 = [], [], []

    for ex in items:
        h_w, h_l = ex["h_w"], ex["h_l"]

        if h_w == 0 and h_l == 0:
            b00.append(ex)
        elif h_w == 1 and h_l == 1:
            b11.append(ex)
        elif h_w == 0 and h_l == 1:
            b01.append(ex)

    return b00, b11, b01


def flip_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """Flip a sample if (h_w, h_l) = (1, 0)."""
    if item.get("h_w") == 1 and item.get("h_l") == 0:
        item["h_w"], item["h_l"] = 0, 1
        item["chosen"], item["rejected"] = item["rejected"], item["chosen"]
    return item
