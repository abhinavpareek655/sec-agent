from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from agent.llm.base import LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    """
    LLM provider for locally-running Ollama models.
    No external SDK needed — uses the stdlib urllib so there are
    zero extra dependencies beyond what the project already has.

    Ollama must be running: https://ollama.com
    Pull a model first: `ollama pull llama3.2`

    Tool-calling support depends on the model. Llama 3.1+ supports it.
    If the model does not return tool_calls, the response falls back to
    plain text content, same contract as GroqProvider.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            raise RuntimeError(
                f"Ollama request failed. Is Ollama running at {self.base_url}? Error: {e}"
            ) from e

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Ollama supports OpenAI-style tools on compatible models (Llama 3.1+)
        if tools:
            payload["tools"] = tools

        response = self._post("/api/chat", payload)
        message = response.get("message", {})
        content: str | None = message.get("content") or None

        # Parse tool_calls if present (Ollama returns same schema as OpenAI)
        raw_tool_calls = message.get("tool_calls")
        tool_calls = None

        if raw_tool_calls:
            tool_calls = []
            for i, tc in enumerate(raw_tool_calls):
                func = tc.get("function", {})
                arguments = func.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass

                tool_calls.append({
                    "id": tc.get("id", f"ollama_tc_{i}"),
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": arguments,
                    },
                })

        return LLMResponse(content=content, tool_calls=tool_calls)