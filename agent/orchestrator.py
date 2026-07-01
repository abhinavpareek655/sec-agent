from __future__ import annotations

import json
from typing import Any

from agent.llm.base import LLMProvider
from agent.memory.episode_log import EpisodeLogger
from agent.tools.registry import ToolRegistry


class Orchestrator:
    def __init__(self, llm: LLMProvider, registry: ToolRegistry, logger: EpisodeLogger | None = None) -> None:
        self.llm = llm
        self.registry = registry
        self.logger = logger
        self.system_prompt = (
            "You are an assistant that can make HTTP requests and run nmap scans to test a web app for issues. "
            "Only act within the provided target."
        )
        self.goal: str = ""
        self.steps_taken = 0
        self.messages: list[dict] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self._log_message("system", self.messages[0]["content"])

    def reset(self, goal: str | None = None) -> None:
        self.goal = goal or ""
        self.steps_taken = 0
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        self._log_message("system", self.system_prompt)

        if goal:
            self._append_message({"role": "user", "content": goal})

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

    def _prepare_messages_for_generate(self, constraint: str | None = None) -> list[dict]:
        if not constraint:
            return self.messages

        return self.messages + [
            {
                "role": "system",
                "content": constraint,
            }
        ]

    def _process_response(self, response: Any) -> dict:
        if response.tool_calls:
            assistant_tool_calls = []
            tool_results = []
            executed_tool_results = []

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
                    assistant_arguments = function.get("arguments", {})
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
                executed_tool_results.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "name": function["name"],
                        "arguments": arguments if isinstance(arguments, dict) else None,
                        "result": tool_result,
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

            return {
                "type": "tool_calls",
                "tool_calls": assistant_tool_calls,
                "content": response.content,
                "tool_results": executed_tool_results,
            }

        if response.content:
            self._append_message({"role": "assistant", "content": response.content})
            return {
                "type": "final",
                "content": response.content,
            }

        self._append_message({"role": "assistant", "content": ""})
        return {
            "type": "empty",
            "content": "",
            "tool_results": [],
        }

    def step_once(self, constraint: str | None = None) -> dict:
        self.steps_taken += 1
        response = self.llm.generate(self._prepare_messages_for_generate(constraint), tools=self.registry.list_tools())
        result = self._process_response(response)
        result["steps_taken"] = self.steps_taken
        return result

    def run(self, goal: str, max_steps: int) -> str:
        self.reset(goal)

        for _ in range(max_steps):
            result = self.step_once()
            if result.get("type") == "final":
                self._log_outcome(True, self.steps_taken, result.get("content", ""))
                return result.get("content", "")

        error_message = f"Maximum steps reached ({max_steps}) before producing a final answer."
        self._log_outcome(False, self.steps_taken, error_message)
        raise RuntimeError(error_message)