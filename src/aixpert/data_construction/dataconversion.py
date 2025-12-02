"""
Generate preference pairs from cleaned Skywork samples.

This script loads prompt/chosen/rejected rows from a JSONL dataset, randomly
assigns chosen/rejected responses into response_0 and response_1, assigns the
correct better_response_id, and saves the resulting dataset in JSONL format.

This version is fully compliant with ruff, ruff-format, pydocstyle, and mypy.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "skywork_extracted_77k.jsonl"
OUT_FILE = DATA_DIR / "skywork_preference_pairs_77k.jsonl"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_preference_pairs(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert prompt/chosen/rejected rows into preference-pair format."""
    output: List[Dict[str, Any]] = []

    for item in data:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

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


def main() -> None:
    """Generate evaluation preference pairs and save them to disk."""
    print(f"📥 Loading dataset from → {INPUT_FILE}")

    data = load_jsonl(INPUT_FILE)
    print(f"📄 Loaded {len(data)} rows")

    preference_pairs = create_preference_pairs(data)

    write_jsonl(OUT_FILE, preference_pairs)

    print("======================================")
    print(f"✅ DONE! Saved preference pairs → {OUT_FILE}")
    print(f"📦 Total pairs: {len(preference_pairs)}")
    print("======================================")


if __name__ == "__main__":
    main()
