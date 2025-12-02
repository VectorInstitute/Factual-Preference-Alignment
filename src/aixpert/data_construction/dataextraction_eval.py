"""
Extract the evaluation slice of the Skywork preference dataset.

This script extracts rows 80001–81000, removes exact duplicates,
and saves the cleaned dataset into JSONL files under the local data folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset


# ============================================================
# Helpers
# ============================================================


def extract_prompt(dialog: List[Dict[str, Any]]) -> str:
    """Extract the first user message."""
    for msg in dialog:
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def extract_answer(dialog: List[Dict[str, Any]]) -> str:
    """Extract the first assistant message."""
    for msg in dialog:
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# Constants for eval split
# ============================================================

START = 80001
END = 81000  # inclusive

print(f"📥 Loading eval slice: {START} → {END}")

ds = load_dataset(
    "Skywork/Skywork-Reward-Preference-80K-v0.1",
    split=f"train[{START}:{END + 1}]",
)

df = ds.to_pandas()

df["prompt"] = df["chosen"].apply(extract_prompt)
df["chosen"] = df["chosen"].apply(extract_answer)
df["rejected"] = df["rejected"].apply(extract_answer)

clean_df = df[["prompt", "chosen", "rejected"]]

cleaned: List[Dict[str, str]] = []
removed: List[Dict[str, str]] = []

for _, row in clean_df.iterrows():
    chosen = row["chosen"].strip()
    rejected = row["rejected"].strip()

    record = {
        "prompt": row["prompt"].strip(),
        "chosen": chosen,
        "rejected": rejected,
    }

    if chosen == rejected:
        removed.append(record)
    else:
        cleaned.append(record)

print(f"🧹 Removed duplicates: {len(removed)}")
print(f"📦 Clean samples: {len(cleaned)}")

# Save outputs
data_dir = Path(__file__).resolve().parent / "data"
save_jsonl(data_dir / "skywork_extracted_eval.jsonl", cleaned)
save_jsonl(data_dir / "skywork_eval_removed.jsonl", removed)

print("✅ Saved eval dataset → skywork_eval.jsonl")
