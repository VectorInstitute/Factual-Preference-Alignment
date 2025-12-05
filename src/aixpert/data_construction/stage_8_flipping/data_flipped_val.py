"""
Flip preference labels for evaluation data.

This script:
- Converts h_w=1,h_l=0 → h_w=0,h_l=1
- Swaps chosen/rejected
- Writes a flipped version of the dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from utils.config_loader import load_config
from utils.data_utils import flip_sample, load_jsonl, write_jsonl


def main() -> None:
    """Flip (1,0) preference labels in the final evaluation dataset."""
    paths = load_config()["paths"]

    input_path = Path(paths["final_eval_out"])
    output_path = Path(paths["eval_flipped_out"])

    print(f"Loading → {input_path}")
    items: List[Dict[str, Any]] = load_jsonl(input_path)

    print("Flipping (h_w=1, h_l=0) samples...")
    flipped = [flip_sample(item) for item in items]

    print(f"Saving flipped dataset → {output_path}")
    write_jsonl(output_path, flipped)

    print("\n==========================================")
    print("EVAL FLIP COMPLETE")
    print(f"Total samples processed: {len(flipped)}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
