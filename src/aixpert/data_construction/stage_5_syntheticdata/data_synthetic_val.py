"""
Generate synthetic corruption (hallucinated) responses for EVAL split.

This script:
- Loads clean DPO-ready Skywork eval transformation.
- Selects pairs where h_w=0 and h_l=1.
- Uses GPT-4o-mini to introduce subtle factual errors.
- Produces inverted (hallucinated, correct) preference pairs.
- Saves 400 synthetic eval corruption examples.

Compatible with ruff, ruff-format, pydocstyle, and mypy.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any, Dict, Optional

from openai import AsyncOpenAI
from utils.config_loader import load_config
from utils.data_utils import load_jsonl, write_jsonl
from utils.synthetic_utils import build_inversion_item, generate_corruption


async def process_item(
    item: Dict[str, Any],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    max_retries: int,
) -> Optional[Dict[str, Any]]:
    """Generate one synthetic inversion example for evaluation."""
    corrupted = await generate_corruption(
        client=client,
        model=model,
        question=item["prompt"],
        answer=item["chosen"],
        semaphore=sem,
        max_retries=max_retries,
    )

    if corrupted is None:
        return None

    entry = await build_inversion_item(item, corrupted)
    entry["source"] = "synthetic_inversion_eval"
    return entry


async def main() -> None:
    """Generate synthetic corruption samples for evaluation."""
    config = load_config()

    model = config["model"]["name"]
    api_key = config["openai_api_key"]

    target = config["hyperparams"]["synthetic_eval_samples"]
    concurrency = config["hyperparams"]["corruption_concurrency"]
    max_retries = config["hyperparams"]["max_retries"]

    input_path = Path(config["paths"]["skywork_eval_transformed"])
    output_path = Path(config["paths"]["synthetic_eval_out"])

    print(f"Loading transformed eval data → {input_path}")
    items = load_jsonl(input_path)

    print("Selecting (h_w=0, h_l=1) eval candidates…")
    valid = [x for x in items if x["h_w"] == 0 and x["h_l"] == 1]

    selected = random.sample(valid, min(target, len(valid)))
    print(f"Selected {len(selected)} items for corruption.")

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    tasks = [process_item(item, client, sem, model, max_retries) for item in selected]
    results = await asyncio.gather(*tasks)

    final_rows = [r for r in results if r is not None]

    print(f"Saving {len(final_rows)} synthetic eval samples → {output_path}")
    write_jsonl(output_path, final_rows)

    print("Eval synthetic corruption generation complete.")


if __name__ == "__main__":
    asyncio.run(main())
