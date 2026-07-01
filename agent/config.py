from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is absent
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


_INT_PATTERN = re.compile(r"[-+]?\d+")
_FLOAT_PATTERN = re.compile(r"[-+]?\d+\.\d+")


def _parse_scalar(value: str) -> Any:
    if not value:
        return ""

    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if _INT_PATTERN.fullmatch(value):
        return int(value)
    if _FLOAT_PATTERN.fullmatch(value):
        return float(value)

    return value


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _next_content_line(lines: list[str], start_index: int) -> tuple[int, str] | tuple[None, None]:
    for index in range(start_index, len(lines)):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#"):
            return index, lines[index]
    return None, None


def _parse_block(lines: list[str], start_index: int, indent: int) -> tuple[Any, int]:
    index = start_index
    container: Any | None = None

    while index < len(lines):
        raw_line = lines[index]
        stripped_line = raw_line.strip()

        if not stripped_line or stripped_line.startswith("#"):
            index += 1
            continue

        line_indent = _indent_of(raw_line)
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Invalid indentation in YAML config near line {index + 1}: {raw_line!r}")

        if stripped_line.startswith("-"):
            if container is None:
                container = []
            if not isinstance(container, list):
                raise ValueError(f"Mixed mapping/list content in YAML config near line {index + 1}")

            item_text = stripped_line[1:].lstrip()
            index += 1

            if not item_text:
                next_index, next_line = _next_content_line(lines, index)
                if next_line is not None and _indent_of(next_line) > indent:
                    item, index = _parse_block(lines, next_index, _indent_of(next_line))
                else:
                    item = None
            else:
                item = _parse_scalar(item_text)

            container.append(item)
            continue

        if container is None:
            container = {}
        if not isinstance(container, dict):
            raise ValueError(f"Mixed mapping/list content in YAML config near line {index + 1}")

        key, separator, remainder = stripped_line.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML line near line {index + 1}: {raw_line!r}")

        key = key.strip()
        remainder = remainder.lstrip()
        index += 1

        if remainder:
            container[key] = _parse_scalar(remainder)
            continue

        next_index, next_line = _next_content_line(lines, index)
        if next_line is not None and _indent_of(next_line) > indent:
            value, index = _parse_block(lines, next_index, _indent_of(next_line))
        else:
            value = {}
        container[key] = value

    if container is None:
        container = {}

    return container, index


def _load_yaml_text(text: str) -> dict[str, Any]:
    if yaml is not None:
        config = yaml.safe_load(text)
        if not isinstance(config, dict):
            raise ValueError("Config file must be a YAML mapping")
        return config

    config, _ = _parse_block(text.splitlines(), 0, 0)
    if not isinstance(config, dict):
        raise ValueError("Config file must be a YAML mapping")
    return config


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
        config = _load_yaml_text(f.read())

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