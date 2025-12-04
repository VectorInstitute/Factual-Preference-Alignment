"""
Extract the evaluation slice of the Skywork preference dataset.

This script extracts rows 80001–81000, removes exact duplicates,
and saves the cleaned dataset into JSONL files under the local data folder.
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
    """Run validation-split extraction and save cleaned JSONL outputs."""
    cfg = load_config()
    hp = cfg["hyperparams"]
    paths = cfg["paths"]

    start, end = hp["eval_start"], hp["eval_end"]

    print(f"Extracting eval slice {start} → {end}")

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

    save_jsonl(Path(paths["skywork_eval_cleaned"]), cleaned)
    save_jsonl(Path(paths["skywork_eval_removed"]), removed)

    print(f"Removed duplicates: {len(removed)}")
    print(f"Clean samples: {len(cleaned)}")
    print("Eval extraction completed.")


if __name__ == "__main__":
    main()
