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
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

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
    raise RuntimeError("❌ OPENAI_API_KEY missing in repository .env")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "skywork_first_transformed_eval.jsonl"
OUTPUT_FILE = DATA_DIR / "synthetic_llm_inversion_eval_400.jsonl"

# ============================================================
# Prompts
# ============================================================

SYSTEM_PROMPT = """
You are a factual corruption generator.

Rewrite a correct answer into a subtly incorrect version with 2–3 factual errors.
It must remain fluent, confident, and plausible.
"""


def build_user_prompt(question: str, correct_answer: str) -> str:
    """Construct user prompt."""
    return f"""
PROMPT:
{question}

CORRECT ANSWER:
{correct_answer}

TASK:
Rewrite this answer so it becomes factually incorrect while still sounding natural.
"""


# ============================================================
# API wrapper
# ============================================================


async def generate_corruption(
    question: str, answer: str, semaphore: asyncio.Semaphore
) -> Optional[str]:
    """Generate a hallucinated version of the answer."""
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
                print(f"Retry ({retry}) corruption error:", exc)
                await asyncio.sleep(1 + retry * 0.5)

    return None


# ============================================================
# Process each item
# ============================================================


async def process_item(
    item: Dict[str, Any], semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """Create synthetic corruption pair for an eval item."""
    prompt = item["prompt"]
    correct = item["chosen"]

    corrupted = await generate_corruption(prompt, correct, semaphore)
    if corrupted is None:
        return None

    return {
        "prompt": prompt,
        "chosen": corrupted,
        "rejected": correct,
        "h_w": 1,
        "h_l": 0,
        "source": "synthetic_inversion_eval",
    }


# ============================================================
# Main
# ============================================================


async def main() -> None:
    """Run synthetic generation for evaluation."""
    target = 400

    print(f"📥 Loading eval data → {INPUT_FILE}")
    items = [json.loads(line) for line in INPUT_FILE.open("r", encoding="utf-8")]

    clean_pairs = [x for x in items if x.get("h_w") == 0 and x.get("h_l") == 1]

    selected = random.sample(clean_pairs, min(target, len(clean_pairs)))
    print(f"🔎 Selected {len(selected)} items for corruption.")

    semaphore = asyncio.Semaphore(20)
    coros = [process_item(item, semaphore) for item in selected]

    print("⚙️ Generating eval corruptions...")
    results = await tqdm_asyncio.gather(*coros)
    results = [r for r in results if r is not None]

    print(f"💾 Saving {len(results)} examples → {OUTPUT_FILE}")
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("✅ Eval synthetic corruption generation complete.")


if __name__ == "__main__":
    asyncio.run(main())
