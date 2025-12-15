"""Async utilities for generating corrupted answers and synthetic inversions."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from openai import AsyncOpenAI
from utils.prompt_templates import (
    CORRUPTION_SYSTEM_PROMPT,
    CORRUPTION_USER_PROMPT,
)


async def generate_corruption(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
    temperature: float = 0.8,
) -> Optional[str]:
    """Generate a hallucinated / corrupted answer using GPT."""
    user_prompt = CORRUPTION_USER_PROMPT.format(question=question, answer=answer)

    async with semaphore:
        for retry in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": CORRUPTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()

            except Exception as exc:
                print(f"[Retry {retry}] corruption generation failed: {exc}")
                await asyncio.sleep(1 + retry * 0.5)

    return None


async def build_inversion_item(
    item: Dict[str, Any],
    corrupted: str,
) -> Dict[str, Any]:
    """Return a synthetic inversion DPO sample."""
    return {
        "prompt": item["prompt"],
        "chosen": corrupted,
        "rejected": item["chosen"],
        "h_w": 1,
        "h_l": 0,
        "source": "synthetic_inversion",
    }
