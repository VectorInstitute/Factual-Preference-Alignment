"""
Extract the test slice of the Skywork preference dataset.

This script extracts rows 81001–81500, removes exact duplicates,
and saves the cleaned dataset into JSONL files under the local data folder.
Only the prompts from this test set will be used in evaluation.
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
    """Run test-split extraction and save cleaned JSONL outputs."""
    cfg = load_config()
    hp = cfg["hyperparams"]
    paths = cfg["paths"]

    start, end = hp["test_start"], hp["test_end"]

    print(f"Extracting test slice {start} → {end}")

    ds = load_dataset(
        paths["skywork_file"],
        split=f"train[{start}:{end + 1}]",
    )
    df = ds.to_pandas()

    df["prompt"] = df["chosen"].apply(extract_prompt)
    df["chosen"] = df["chosen"].apply(extract_answer)
    df["rejected"] = df["rejected"].apply(extract_answer)

    rows = df[["prompt", "chosen", "rejected"]].to_dict(orient="records")
    cleaned, removed = filter_duplicates(rows)

    save_jsonl(Path(paths["skywork_test_cleaned"]), cleaned)
    save_jsonl(Path(paths["skywork_test_removed"]), removed)

    print(f"Removed duplicates: {len(removed)}")
    print(f"Clean samples: {len(cleaned)}")
    print("Test extraction completed.")


if __name__ == "__main__":
    main()
