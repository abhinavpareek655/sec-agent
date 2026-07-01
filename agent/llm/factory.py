from __future__ import annotations

import os
from typing import Any

from agent.llm.base import LLMProvider


def build_provider(cfg: dict[str, Any]) -> LLMProvider:
    """
    Instantiate the correct LLMProvider based on config.

    Config shape expected:
        llm:
          provider: groq | ollama
          model: <model name>

    The factory is the only place that imports provider-specific classes,
    so nothing else in the codebase ever couples to a concrete provider.
    """
    llm_cfg = cfg.get("llm", {})
    provider_name = llm_cfg.get("provider", "groq").lower()
    model = llm_cfg.get("model", "")

    if provider_name == "groq":
        from agent.llm.groq_provider import GroqProvider

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your .env file or export it before running."
            )
        return GroqProvider(api_key=api_key, model_name=model)

    if provider_name == "ollama":
        from agent.llm.ollama_provider import OllamaProvider

        base_url = cfg.get("ollama", {}).get("base_url", "http://localhost:11434")
        ollama_model = cfg.get("ollama", {}).get("model", model or "llama3.2")
        return OllamaProvider(base_url=base_url, model=ollama_model)

    raise ValueError(
        f"Unknown LLM provider: '{provider_name}'. "
        "Supported values: 'groq', 'ollama'."
    )