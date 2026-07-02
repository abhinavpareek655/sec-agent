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


def setup_registry(
    allowed_hosts: list[str] | None = None,
    http_hosts: list[str] | None = None,
    nmap_hosts: list[str] | None = None,
) -> ToolRegistry:
    from agent.tools.http_tool import HttpTool
    from agent.tools.nmap_tool import NmapTool
    from agent.tools.extract_tool import ExtractJsonTool

    # Support both old style (allowed_hosts) and new style (separate lists)
    if allowed_hosts:
        http_hosts = http_hosts or allowed_hosts
        nmap_hosts = nmap_hosts or allowed_hosts

    if http_hosts:
        registry.register(HttpTool(allowed_hosts=http_hosts))
    if nmap_hosts:
        registry.register(NmapTool(allowed_hosts=nmap_hosts))

    registry.register(ExtractJsonTool())  # no allowlist needed, purely local

    return registry
