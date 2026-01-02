"""Utility module for loading the global YAML configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load YAML config into a dictionary."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
