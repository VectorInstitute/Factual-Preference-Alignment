"""Run binary factuality evaluation on evaluation preference pairs."""

from __future__ import annotations

import asyncio
from pathlib import Path

from decouple import Config, RepositoryEnv
from utils.config_loader import load_config
from utils.data_utils import load_jsonl
from utils.factual_utils import (
    factual_evaluation_pipeline,
    get_client,
)


async def main() -> None:
    """Execute factuality evaluation for the validation set."""
    cfg = load_config()

    repo_path = cfg["repository"]
    paths = cfg["paths"]
    hp = cfg["hyperparams"]

    # Load API key
    env = Config(RepositoryEnv(f"{repo_path}/.env"))
    api_key = env("OPENAI_API_KEY", default=None)
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    client = get_client(api_key)

    input_path = Path(paths["skywork_eval_pairs"])
    output_path = Path(paths["skywork_eval_factual"])

    items = load_jsonl(input_path)

    await factual_evaluation_pipeline(
        client=client,
        items=items,
        output_file=output_path,
        model=cfg["model"]["name"],
        concurrency=hp["concurrency_limit"],
        max_retries=hp["max_retries"],
    )

    print("Completed factual evaluation for evaluation set.")


if __name__ == "__main__":
    asyncio.run(main())
