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
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from decouple import Config, RepositoryEnv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as tqdm_asyncio
from utils import load_yaml


# ============================================================
# Load config.yaml + .env API key
# ============================================================

CONFIG = load_yaml("config.yaml")  # must contain: repository: <path>
REPO_PATH = CONFIG["repository"]

env = Config(RepositoryEnv(f"{REPO_PATH}/.env"))
OPENAI_API_KEY = env("OPENAI_API_KEY", default=False)

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY missing in .env under repository path.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Directories
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "skywork_first_transformed_train.jsonl"
OUTPUT_FILE = DATA_DIR / "synthetic_llm_inversion_train_10k.jsonl"

# ============================================================
# Corruption prompt
# ============================================================

SYSTEM_PROMPT = """
You are a factual corruption generator.

Given a question and its correct answer, produce an incorrect version
that is fluent, confident, and well-structured but it must contain 2–3 factual inaccuracies.
Avoid nonsense. Keep the tone similar to the original answer.
"""


def build_user_prompt(question: str, correct_answer: str) -> str:
    """Construct the user prompt for hallucination generation."""
    return f"""
PROMPT:
{question}

CORRECT ANSWER:
{correct_answer}

TASK:
Rewrite the answer so that it becomes factually wrong, introducing subtle hallucinations
while sounding confident and coherent.
"""


# ============================================================
# API for corrupted (hallucinated) answer
# ============================================================


async def generate_corruption(
    question: str,
    answer: str,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """Generate a hallucinated version of the correct answer using GPT-4o-mini."""
    user_prompt = build_user_prompt(question, answer)

    async with semaphore:
        for retry in range(5):
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.8,
                )
                return resp.choices[0].message.content.strip()

            except Exception as exc:
                print(f"Retry corruption ({retry}): {exc}")
                await asyncio.sleep(1 + retry * 0.5)

    return None


# ============================================================
# Process one item
# ============================================================


async def process_item(
    item: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """Produce one synthetic inversion (corruption) DPO sample."""
    prompt = item["prompt"]
    correct_answer = item["chosen"]

    corrupted = await generate_corruption(prompt, correct_answer, semaphore)

    if corrupted is None:
        return None

    return {
        "prompt": prompt,
        "chosen": corrupted,  # hallucinated / corrupted
        "rejected": correct_answer,  # original factual answer
        "h_w": 1,  # corrupted = wrong
        "h_l": 0,  # original = correct
        "source": "synthetic_inversion",
    }


# ============================================================
# Main
# ============================================================


async def main() -> None:
    """Generate 10k synthetic corruption pairs and save JSONL output."""
    target = 10_000
    print(f"📥 Loading training dataset → {INPUT_FILE}")

    items: List[Dict[str, Any]] = [
        json.loads(line) for line in INPUT_FILE.open("r", encoding="utf-8")
    ]

    print("🔍 Selecting factual (0,1) pairs only...")
    clean_pairs = [x for x in items if x["h_w"] == 0 and x["h_l"] == 1]

    print(f"Available factual pairs: {len(clean_pairs)}")
    selected = random.sample(clean_pairs, target)
    print(f"🎯 Selected {len(selected)} items for corruption generation.")

    semaphore = asyncio.Semaphore(20)

    tasks = [process_item(item, semaphore) for item in selected]

    print("⚙️ Generating corrupted answers...")
    results = await tqdm_asyncio.gather(*tasks)

    final_rows = [r for r in results if r is not None]

    print(f"💾 Saving {len(final_rows)} synthetic samples → {OUTPUT_FILE}")
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("✅ Synthetic corruption dataset created.")


if __name__ == "__main__":
    asyncio.run(main())
