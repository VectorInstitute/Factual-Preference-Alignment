"""
Async factuality evaluation utilities.

This module runs factual-flag scoring for preference pairs using an
LLM judge, supports concurrency, retries, and resume-safe checkpointing.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from utils.prompt_templates import BINARY_FACTUAL_JUDGE_PROMPT


def get_client(api_key: str) -> AsyncOpenAI:
    """Return AsyncOpenAI client."""
    return AsyncOpenAI(api_key=api_key)


async def get_factual_flag(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> int:
    """Evaluate factual correctness (0 factual, 1 hallucinated)."""
    prompt = BINARY_FACTUAL_JUDGE_PROMPT.format(question=question, answer=answer)

    async with semaphore:
        for retry in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                text = resp.choices[0].message.content.strip()
                match = re.search(r"\[\[(0|1)\]\]", text)
                return int(match.group(1)) if match else 1
            except Exception:
                await asyncio.sleep(1 + retry * 0.5)

    return 1


async def evaluate_pair(
    client: AsyncOpenAI,
    item: Dict[str, Any],
    model: str,
    sem: asyncio.Semaphore,
    retries: int,
) -> Dict[str, Any]:
    """Compute factual flags for response_0 and response_1."""
    prompt = item["prompt"]

    t0 = asyncio.create_task(
        get_factual_flag(client, model, prompt, item["response_0"], sem, retries)
    )
    t1 = asyncio.create_task(
        get_factual_flag(client, model, prompt, item["response_1"], sem, retries)
    )

    f0, f1 = await asyncio.gather(t0, t1)

    return {
        **item,
        "factual_flag_0": f0,
        "factual_flag_1": f1,
        "h0": f0,
        "h1": f1,
    }


async def factual_evaluation_pipeline(
    client: AsyncOpenAI,
    items: List[Dict[str, Any]],
    output_file: Path,
    model: str,
    concurrency: int,
    max_retries: int,
) -> None:
    """Run factuality evaluation with resume and checkpoint support."""
    processed = 0
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            processed = sum(1 for _ in f)

    remaining = items[processed:]
    sem = asyncio.Semaphore(concurrency)

    tasks = [evaluate_pair(client, item, model, sem, max_retries) for item in remaining]

    buffer: List[str] = []
    counter = processed

    with output_file.open("a", encoding="utf-8") as f:
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            out = await coro
            buffer.append(json.dumps(out, ensure_ascii=False) + "\n")
            counter += 1

            if len(buffer) >= 25:
                f.writelines(buffer)
                f.flush()
                os.fsync(f.fileno())
                buffer.clear()

        if buffer:
            f.writelines(buffer)
            f.flush()
            os.fsync(f.fileno())
