from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "intersection.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the experiment YAML without mutating the returned environment config."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return config


def split_env_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the requested environment ID and HighwayEnv configuration."""
    copied = dict(config)
    env_id = str(copied.pop("env_id", "intersection-v2"))
    return env_id, copied

