"""
Generate synthetic corruption (hallucinated) responses for TRAIN split.

This script:
- Loads clean DPO-ready Skywork transformation for training.
- Selects items where h_w=0 (winner factual) and h_l=1 (loser incorrect).
- Asks GPT-4o-mini to rewrite the factual answer into a subtle hallucination.
- Produces “inversion pairs” where corrupted is chosen and original is rejected.
- Saves up to 10,000 synthetic hallucination samples.

Fully compatible with ruff, ruff-format, pydocstyle, and mypy.
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
    """Generate one synthetic inversion sample for training."""
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

    return await build_inversion_item(item, corrupted)


async def main() -> None:
    """Generate synthetic hallucination samples for training."""
    config = load_config()

    model = config["model"]["name"]
    api_key = config["openai_api_key"]

    target = config["hyperparams"]["synthetic_train_samples"]
    concurrency = config["hyperparams"]["corruption_concurrency"]
    max_retries = config["hyperparams"]["max_retries"]

    input_path = Path(config["paths"]["skywork_train_transformed"])
    output_path = Path(config["paths"]["synthetic_train_out"])

    print(f"Loading transformed training data → {input_path}")
    items = load_jsonl(input_path)

    print("🔍 Selecting (h_w=0, h_l=1) candidates…")
    valid = [x for x in items if x["h_w"] == 0 and x["h_l"] == 1]

    selected = random.sample(valid, min(target, len(valid)))
    print(f"Selected {len(selected)} items for corruption.")

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    tasks = [process_item(item, client, sem, model, max_retries) for item in selected]
    results = await asyncio.gather(*tasks)

    final_rows = [r for r in results if r is not None]

    print(f"Saving {len(final_rows)} synthetic training samples → {output_path}")
    write_jsonl(output_path, final_rows)

    print("Synthetic training corruption generation complete.")


if __name__ == "__main__":
    asyncio.run(main())
