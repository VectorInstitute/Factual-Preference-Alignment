"""
Run factuality evaluation for all models and all delta values.

This module loads the global evaluation config, iterates over every
(model, Δ) pair, evaluates Original-DPO vs. Modified-DPO++, and stores
results in a JSON file.
"""

import asyncio
import json

from utils.config_loader import load_config
from utils.eval_core_utils import evaluate_pair


async def main() -> None:
    """
    Execute evaluation for all model–delta combinations as defined in the config.

    Loads evaluation parameters, model paths, and OpenAI judge configuration,
    runs factuality scoring for each pair, and writes a consolidated JSON file.
    """
    cfg = load_config()

    data_file = cfg["eval"]["data_file"]
    batch_size = cfg["eval"]["batch_size"]
    max_new = cfg["eval"]["max_new_tokens"]
    concurrency = cfg["eval"]["judge_concurrency"]

    original_root = cfg["paths"]["original_root"]
    modified_root = cfg["paths"]["modified_root"]

    models = cfg["models"]
    deltas = cfg["deltas"]

    api_key = cfg["openai_api_key"]
    judge_model = cfg["model"]["name"]

    results = {}

    for m in models:
        short = m["short"]
        orig_model_path = f"{original_root}/{short}_OriginalDPO"

        for d in deltas:
            mod_model_path = f"{modified_root}/{short}_delta{d}"

            print(f"\n=== Evaluating {short}: Original vs Δ={d} ===")

            out = await evaluate_pair(
                data_file=data_file,
                model_a_dir=orig_model_path,
                model_b_dir=mod_model_path,
                batch_size=batch_size,
                max_new_tokens=max_new,
                concurrency=concurrency,
                api_key=api_key,
                judge_model=judge_model,
            )

            results[f"{short}_delta{d}"] = out
            print(json.dumps(out, indent=2))

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to eval_results.json")


if __name__ == "__main__":
    asyncio.run(main())
