"""
Core evaluation utilities for factuality scoring.

This module provides:
- Batched text generation for two fine-tuned models.
- Asynchronous factual judging using GPT-4o-mini (or another configured judge).
- A high-level function `evaluate_pair()` for comparing Original-DPO vs Modified-DPO++.
"""

import asyncio
import json
import re
from typing import List

import numpy as np
import torch
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .eval_template import FACTUAL_PROMPT


async def judge_factual(
    prompt: str,
    answer: str,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    judge_model: str,
) -> float:
    """
    Send one prompt–answer pair to the LLM judge and extract a factuality score.

    Returns
    -------
        float: The extracted factuality score, or 0.0 if parsing fails.
    """
    query = FACTUAL_PROMPT.format(question=prompt, answer=answer)

    async with semaphore:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": query}],
            temperature=0,
        )

    text = response.choices[0].message.content
    match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", text)
    return float(match.group(1)) if match else 0.0


async def judge_many(
    prompts: List[str],
    answers: List[str],
    concurrency: int,
    api_key: str,
    judge_model: str,
) -> List[float]:
    """
    Evaluate many answers in parallel using GPT-4o-mini (or configured judge).

    Returns
    -------
        List[float]: List of factuality scores for each answer.
    """
    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=api_key)

    tasks = [
        asyncio.create_task(
            judge_factual(prompt, answer, semaphore, client, judge_model)
        )
        for prompt, answer in zip(prompts, answers)
    ]
    return await asyncio.gather(*tasks)


def batch_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: str,
    max_tokens: int,
) -> List[str]:
    """
    Generate model outputs in batches.

    Returns
    -------
        List[str]: Cleaned outputs with the prompt removed.
    """
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

    cleaned = []
    for prompt, full in zip(prompts, decoded):
        cleaned.append(
            full[len(prompt) :].strip() if full.startswith(prompt) else full.strip()
        )

    return cleaned


async def evaluate_pair(
    data_file: str,
    model_a_dir: str,
    model_b_dir: str,
    batch_size: int,
    max_new_tokens: int,
    concurrency: int,
    api_key: str,
    judge_model: str,
) -> dict:
    """
    Evaluate Original-DPO (model A) vs Modified-DPO++ (model B) factuality.

    Returns
    -------
        dict: Metrics including factuality means, hallucination rates, and win-rate.
    """
    # Load prompts safely using a context manager
    with open(data_file, "r", encoding="utf-8") as f:
        prompts = [json.loads(line)["prompt"] for line in f]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_a_dir)
    tokenizer.padding_side = "left"

    model_a = AutoModelForCausalLM.from_pretrained(
        model_a_dir, torch_dtype=torch.bfloat16
    ).to(device)
    model_b = AutoModelForCausalLM.from_pretrained(
        model_b_dir, torch_dtype=torch.bfloat16
    ).to(device)

    model_a.eval()
    model_b.eval()
    torch.set_grad_enabled(False)

    all_a_scores, all_b_scores = [], []

    for start in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[start : start + batch_size]

        ans_a = batch_generate(model_a, tokenizer, batch, device, max_new_tokens)
        ans_b = batch_generate(model_b, tokenizer, batch, device, max_new_tokens)

        scores_a = await judge_many(batch, ans_a, concurrency, api_key, judge_model)
        scores_b = await judge_many(batch, ans_b, concurrency, api_key, judge_model)

        all_a_scores.extend(scores_a)
        all_b_scores.extend(scores_b)

    arr_a = np.array(all_a_scores)
    arr_b = np.array(all_b_scores)

    # Avoid divide-by-zero when A == B everywhere
    diff_mask = arr_a != arr_b
    wins = (arr_b > arr_a).sum()
    losses = diff_mask.sum()

    win_rate = wins / losses if losses > 0 else 0.0

    return {
        "factuality_A": float(arr_a.mean()),
        "factuality_B": float(arr_b.mean()),
        "halluc_rate_A": float((arr_a < 5).mean()),
        "halluc_rate_B": float((arr_b < 5).mean()),
        "win_rate": win_rate,
        "count": len(arr_a),
    }
