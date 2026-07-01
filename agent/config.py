from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load YAML config from path. Falls back to configs/default.yaml.

    Args:
        path: Optional path to a YAML config file.

    Returns:
        Parsed config dict.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must be a YAML mapping: {config_path}")

    return config


def get(key: str, default: Any = None, path: str | Path | None = None) -> Any:
    """
    Convenience function to get a top-level config key.

    Usage:
        cfg = load_config()
        model = cfg["llm"]["model"]
    Or:
        from agent.config import get
        model = get("llm")["model"]
    """
    return load_config(path).get(key, default)