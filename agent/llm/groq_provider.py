from __future__ import annotations

import json

from groq import Groq, BadRequestError
from agent.llm.base import LLMProvider, LLMResponse


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = Groq(api_key=api_key)
    
    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=False,
                stop=None
            )
        except BadRequestError as e:
            return LLMResponse(
                content=f"[Tool generation failed: {e}. Please try a different approach.]",
                tool_calls=None,
            )

        message = completion.choices[0].message
        tool_calls = None

        if getattr(message, "tool_calls", None):
            tool_calls = []
            for tool_call in message.tool_calls:
                arguments = tool_call.function.arguments
                try:
                    parsed_arguments = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    parsed_arguments = arguments

                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": parsed_arguments,
                        },
                    }
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
        )
