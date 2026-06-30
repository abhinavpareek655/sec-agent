from __future__ import annotations

import json

from agent.llm.base import LLMProvider
from agent.memory.episode_log import EpisodeLogger
from agent.tools.registry import ToolRegistry


class Orchestrator:
    def __init__(self, llm: LLMProvider, registry: ToolRegistry, logger: EpisodeLogger | None = None) -> None:
        self.llm = llm
        self.registry = registry
        self.logger = logger
        self.messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are an assistant that can make HTTP requests to test a web app for issues. "
                    "Only act within the provided target."
                ),
            }
        ]
        self._log_message("system", self.messages[0]["content"])

    def _log_message(
        self,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        tool_result: object | None = None,
    ) -> None:
        if self.logger is None:
            return

        self.logger.log_step(
            role=role,
            content=content,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
        )

    def _append_message(
        self,
        message: dict,
        tool_name: str | None = None,
        tool_args: dict | None = None,
        tool_result: object | None = None,
    ) -> None:
        self.messages.append(message)
        self._log_message(
            role=message["role"],
            content=message.get("content", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
        )

    def _log_outcome(self, success: bool, total_steps: int, detail: str) -> None:
        self._log_message(
            role="outcome",
            content="success" if success else "failure",
            tool_result={
                "success": success,
                "total_steps": total_steps,
                "detail": detail,
            },
        )

    def run(self, goal: str, max_steps: int) -> str:
        self._append_message({"role": "user", "content": goal})
        steps_taken = 0

        for _ in range(max_steps):
            steps_taken += 1
            response = self.llm.generate(self.messages, tools=self.registry.list_tools())

            if response.tool_calls:
                assistant_tool_calls = []
                tool_results = []

                for tool_call in response.tool_calls:
                    function = tool_call["function"]
                    arguments = function.get("arguments", {})

                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments) if arguments else {}
                        except json.JSONDecodeError:
                            arguments = None

                    if not isinstance(arguments, dict):
                        tool_result = {
                            "success": False,
                            "output": "",
                            "error": f"Invalid tool arguments for {function['name']}: expected object",
                        }
                        assistant_arguments = arguments if isinstance(arguments, str) else function.get("arguments", {})
                        if not isinstance(assistant_arguments, str):
                            assistant_arguments = json.dumps(assistant_arguments)
                    else:
                        tool_result = self.registry.call(function["name"], **arguments)
                        assistant_arguments = json.dumps(arguments)

                    assistant_tool_calls.append(
                        {
                            "id": tool_call["id"],
                            "type": tool_call["type"],
                            "function": {
                                "name": function["name"],
                                "arguments": assistant_arguments,
                            },
                        }
                    )
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function["name"],
                            "content": json.dumps(tool_result),
                        }
                    )

                self._append_message(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": assistant_tool_calls,
                    },
                    tool_result={"tool_calls": assistant_tool_calls},
                )

                for tool_message, tool_call in zip(tool_results, response.tool_calls):
                    function = tool_call["function"]
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments) if arguments else {}
                        except json.JSONDecodeError:
                            arguments = None

                    self._append_message(
                        tool_message,
                        tool_name=function["name"],
                        tool_args=arguments if isinstance(arguments, dict) else None,
                        tool_result=json.loads(tool_message["content"]),
                    )
                continue

            if response.content:
                self._append_message({"role": "assistant", "content": response.content})
                self._log_outcome(True, steps_taken, response.content)
                return response.content

        error_message = f"Maximum steps reached ({max_steps}) before producing a final answer."
        self._log_outcome(False, steps_taken, error_message)
        raise RuntimeError(error_message)