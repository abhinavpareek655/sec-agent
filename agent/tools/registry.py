from __future__ import annotations

from typing import Any

from agent.tools.base import Tool


class ToolRegistry:
	def __init__(self) -> None:
		self._tools: dict[str, Tool] = {}

	def register(self, tool: Tool) -> None:
		self._tools[tool.name] = tool

	def list_tools(self) -> list[dict[str, Any]]:
		return [
			{
				"type": "function",
				"function": {
					"name": tool.name,
					"description": tool.description,
					"parameters": tool.parameters,
				},
			}
			for tool in self._tools.values()
		]

	def call(self, name: str, **kwargs) -> dict:
		tool = self._tools.get(name)
		if tool is None:
			return {
				"success": False,
				"output": "",
				"error": f"Tool not found: {name}",
			}

		try:
			return tool.execute(**kwargs)
		except Exception as exc:
			return {
				"success": False,
				"output": "",
				"error": str(exc),
			}


registry = ToolRegistry()


def setup_registry(allowed_hosts: list[str]) -> ToolRegistry:
	from agent.tools.http_tool import HttpTool

	registry.register(HttpTool(allowed_hosts=allowed_hosts))
	return registry
