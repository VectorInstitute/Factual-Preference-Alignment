"""
Skywork extraction utilities.

This module extracts prompt/chosen/rejected fields from the Skywork Preference
dataset, removes exact duplicates, and writes the cleaned dataset to JSONL
files. Fully compatible with ruff, mypy, and the AI Engineering template.
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from utils.config_loader import load_config
from utils.data_utils import (
    extract_answer,
    extract_prompt,
    filter_duplicates,
    save_jsonl,
)


def main() -> None:
    """Run train-split extraction and save cleaned JSONL outputs."""
    cfg = load_config()
    hp = cfg["hyperparams"]
    paths = cfg["paths"]

    subset_size = hp["subset_size"]

    print(f"Loading first {subset_size} samples from Skywork...")

    ds = load_dataset(
        paths["skywork_file"],
        split=f"train[:{subset_size}]",
    )

    df = ds.to_pandas()

    df["prompt"] = df["chosen"].apply(extract_prompt)
    df["chosen"] = df["chosen"].apply(extract_answer)
    df["rejected"] = df["rejected"].apply(extract_answer)

    rows = df[["prompt", "chosen", "rejected"]].to_dict(orient="records")

    cleaned, removed = filter_duplicates(rows)

    save_jsonl(Path(paths["skywork_train_cleaned"]), cleaned)
    save_jsonl(Path(paths["skywork_train_removed"]), removed)

    print(f"Removed exact duplicates: {len(removed)}")
    print(f"Clean samples: {len(cleaned)}")
    print("Training extraction completed.")


if __name__ == "__main__":
    main()
