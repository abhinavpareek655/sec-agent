from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EpisodeLogger:
    def __init__(self) -> None:
        self._steps: list[dict[str, Any]] = []

    def log_step(
        self,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        tool_result: Any | None = None,
    ) -> None:
        step: dict[str, Any] = {
            "role": role,
            "content": content,
        }

        if tool_name is not None:
            step["tool_name"] = tool_name
        if tool_args is not None:
            step["tool_args"] = tool_args
        if tool_result is not None:
            step["tool_result"] = tool_result

        self._steps.append(step)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file_handle:
            for step in self._steps:
                file_handle.write(json.dumps(step, ensure_ascii=False))
                file_handle.write("\n")