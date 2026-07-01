# Contributing to sec-agent

Thanks for your interest. This document explains how to add new tools, LLM providers, and how to run tests locally.

---

## Adding a new tool

1. Create `agent/tools/your_tool.py` subclassing `Tool` from `agent/tools/base.py`:

```python
from agent.tools.base import Tool
from agent.tools.security import is_host_allowed, is_target_valid

class YourTool(Tool):
    def __init__(self, allowed_hosts: list[str]):
        self.name = "your_tool"
        self.description = "What this tool does and when to use it."
        self.allowed_hosts = {h.lower() for h in allowed_hosts}
        self.parameters = {
            "target": {"type": "string", "description": "..."},
        }

    def execute(self, **kwargs) -> dict:
        # Always validate before acting
        target = kwargs.get("target")
        if not is_target_valid(target):
            return {"success": False, "output": "", "error": "Invalid target."}
        if not is_host_allowed(target, self.allowed_hosts):
            return {"success": False, "output": "", "error": "Host not allowed."}
        # ... do the work ...
        return {"success": True, "output": "...", "error": None}
```

2. Register it in `agent/tools/registry.py` inside `setup_registry()`.
3. Add it to `configs/default.yaml` under `targets` if it needs its own allowlist.
4. Add smoke tests in `tests/test_smoke.py`.

**Safety rules for tools (non-negotiable):**
- Always validate the target against an allowlist in code — never rely on the LLM prompt alone.
- Never use `shell=True` in subprocess calls.
- Always set a timeout on network/subprocess operations.
- Return `{"success": False, "output": "", "error": "..."}` on failure — never raise from `execute()`.

---

## Adding a new LLM provider

1. Create `agent/llm/your_provider.py` subclassing `LLMProvider` from `agent/llm/base.py`:

```python
from agent.llm.base import LLMProvider, LLMResponse

class YourProvider(LLMProvider):
    def __init__(self, ...):
        ...

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        # Call your API/model
        # Parse tool_calls if your API supports function calling
        return LLMResponse(content="...", tool_calls=None)
```

2. Add a branch for your provider in `agent/llm/factory.py`.
3. Add a config entry in `configs/default.yaml` if it needs config values.

The rest of the codebase will work with your provider automatically — nothing else needs changing.

---

## Running tests locally

```bash
pip install pytest ruff
pytest tests/ -v       # no API key needed
ruff check agent/ tests/
```

Tests use a `MockLLMProvider` that returns canned responses, so CI never makes real API calls.

---

## Pull request checklist

- [ ] New tool: allowlist check + timeout + structured return + smoke test
- [ ] New provider: implements `LLMProvider.generate()` + registered in factory
- [ ] `ruff check` passes with no errors
- [ ] `pytest tests/` passes
- [ ] `configs/default.yaml` updated if new config keys added
- [ ] Only tested against systems you own or platforms that explicitly allow it
