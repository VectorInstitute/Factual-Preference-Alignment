"""
Skywork extraction utilities.

This module extracts prompt/chosen/rejected fields from the Skywork Preference
dataset, removes exact duplicates, and writes the cleaned dataset to JSONL
files. Fully compatible with ruff, mypy, and the AI Engineering template.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset


SUBSET_SIZE = 80000
OUT_FILE = (
    "/projects/aixpert/users/sindhu/Loss_Test/Factual_Trust_Loss/data_creation/data/"
    "skywork_extracted_15k.jsonl"
)
REMOVED_FILE = (
    "/projects/aixpert/users/sindhu/Loss_Test/Factual_Trust_Loss/data_creation/data/"
    "skywork_cleaned_15k.jsonl"
)

print(f"📥 Loading first {SUBSET_SIZE} samples from Skywork...")


# ============================================================
# Dataset loading
# ============================================================
ds = load_dataset(
    "Skywork/Skywork-Reward-Preference-80K-v0.1",
    split=f"train[:{SUBSET_SIZE}]",
)

df = ds.to_pandas()


# ============================================================
# Extract prompt / chosen / rejected
# ============================================================
def extract_prompt_from_dialog(dialog: List[Dict[str, Any]]) -> str:
    """
    Extract the first user message from a dialog.

    Parameters
    ----------
    dialog : list of dict
        A list of message objects with "role" and "content" keys.

    Returns
    -------
    str
        The content of the first message with role 'user', or an empty string.
    """
    for msg in dialog:
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def extract_answer_from_dialog(dialog: List[Dict[str, Any]]) -> str:
    """
    Extract the first assistant message from a dialog.

    Parameters
    ----------
    dialog : list of dict
        A list of message objects with "role" and "content" keys.

    Returns
    -------
    str
        The content of the first message with role 'assistant', or an empty string.
    """
    for msg in dialog:
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


df["prompt"] = df["chosen"].apply(extract_prompt_from_dialog)
df["chosen"] = df["chosen"].apply(extract_answer_from_dialog)
df["rejected"] = df["rejected"].apply(extract_answer_from_dialog)

clean_df = df[["prompt", "chosen", "rejected"]]

# ============================================================
# 🔍 Exact-match removal (chosen == rejected)
# ============================================================
cleaned: List[Dict[str, str]] = []
removed: List[Dict[str, str]] = []

for _, row in clean_df.iterrows():
    chosen = str(row["chosen"]).strip()
    rejected = str(row["rejected"]).strip()

    sample = {
        "prompt": str(row["prompt"]).strip(),
        "chosen": chosen,
        "rejected": rejected,
    }

    if chosen == rejected:
        removed.append(sample)
    else:
        cleaned.append(sample)

print(f"🧹 Removed exact duplicates: {len(removed)}")
print(f"📦 Remaining clean samples: {len(cleaned)}")

# Ensure output directory exists
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)


# ============================================================
# Save output JSONL files
# ============================================================
def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


write_jsonl(OUT_FILE, cleaned)
write_jsonl(REMOVED_FILE, removed)

print(f"✅ Saved cleaned dataset ({len(cleaned)} samples) → {OUT_FILE}")
print(f"🗑️  Saved removed duplicates ({len(removed)} samples) → {REMOVED_FILE}")

print(pd.DataFrame(cleaned).head())
