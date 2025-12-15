"""
Generate evaluation preference pairs from cleaned Skywork samples.

Loads prompt/chosen/rejected rows for the eval slice,
creates random preference pairs (response_0/response_1),
assigns correct better_response_id, and writes JSONL output.
"""

from __future__ import annotations

from pathlib import Path

from aixpert.utils.config_loader import load_config
from aixpert.utils.data_utils import (
    create_preference_pairs,
    load_jsonl,
    write_jsonl,
)


def main() -> None:
    """Generate preference pairs for the evaluation set."""
    cfg = load_config()
    paths = cfg["paths"]

    input_path = Path(paths["skywork_eval_cleaned"])
    output_path = Path(paths["skywork_eval_pairs"])

    print(f"Loading evaluation dataset → {input_path}")

    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} rows")

    preference_pairs = create_preference_pairs(data)

    write_jsonl(output_path, preference_pairs)

    print("======================================")
    print(f"Eval preference pairs saved → {output_path}")
    print(f"Total eval pairs: {len(preference_pairs)}")
    print("======================================")


if __name__ == "__main__":
    main()
