"""
Utility functions for loading configuration files.

This module provides:
- `load_yaml`: Read a YAML file into a Python dictionary.
- `load_env_api_key`: Load the OPENAI_API_KEY from a repository `.env` file.

These helpers centralize configuration handling and ensure consistent behavior
across all data-construction scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from decouple import Config, RepositoryEnv


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load a YAML file and return its content as a dict.

    :param yaml_path: Path to the YAML file.
    :return: Parsed YAML content as a dict, or empty dict on failure.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"YAML load error: {e}")
        return {}


def load_env_api_key(repository_path: str) -> str:
    """Load OPENAI_API_KEY from a .env file inside the repository.

    Uses:
        env = Config(RepositoryEnv(config["repository"] + "/.env"))
        api_key = env("OPENAI_API_KEY", default=False)

    :param repository_path: Path to the repo containing `.env`
    :return: The OpenAI API key or an empty string if missing.
    """
    env_path = Path(repository_path) / ".env"

    if not env_path.exists():
        print(f"Warning: .env file not found at {env_path}")
        return ""

    env = Config(RepositoryEnv(str(env_path)))
    return env("OPENAI_API_KEY", default="")


__all__ = ["load_yaml", "load_env_api_key"]
