"""
Smoke tests that run without any real API keys or external services.
The LLM provider is mocked so CI stays free and fast.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.llm.base import LLMProvider, LLMResponse
from agent.memory.episode_log import EpisodeLogger
from agent.orchestrator import Orchestrator
from agent.rl.env import SecAgentEnv
from agent.tools.registry import ToolRegistry
from agent.tools.http_tool import HttpTool
from agent.tools.security import is_host_allowed, is_target_valid


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class MockLLMProvider(LLMProvider):
    """Returns a canned final-answer response — no real API call."""

    def generate(self, messages, tools=None, **kwargs) -> LLMResponse:
        return LLMResponse(content="Mock final answer.", tool_calls=None)


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def test_is_host_allowed_pass():
    assert is_host_allowed("httpbin.org", {"httpbin.org"}) is True

def test_is_host_allowed_fail():
    assert is_host_allowed("evil.com", {"httpbin.org"}) is False

def test_is_host_allowed_empty():
    assert is_host_allowed("", {"httpbin.org"}) is False

def test_is_target_valid_rejects_flag():
    assert is_target_valid("-oN /tmp/out") is False

def test_is_target_valid_rejects_semicolon():
    assert is_target_valid("host; rm -rf /") is False

def test_is_target_valid_accepts_hostname():
    assert is_target_valid("scanme.nmap.org") is True


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def test_registry_unknown_tool():
    registry = ToolRegistry()
    result = registry.call("nonexistent_tool")
    assert result["success"] is False
    assert "not found" in result["error"]


def test_http_tool_disallows_unknown_host():
    tool = HttpTool(allowed_hosts=["httpbin.org"])
    result = tool.execute(method="GET", url="https://evil.com/steal")
    assert result["success"] is False
    assert "not allowed" in result["error"]


def test_http_tool_requires_url():
    tool = HttpTool(allowed_hosts=["httpbin.org"])
    result = tool.execute(method="GET")
    assert result["success"] is False
    assert "URL is required" in result["error"]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_orchestrator_run_returns_string():
    provider = MockLLMProvider()
    registry = ToolRegistry()
    orch = Orchestrator(provider, registry)
    result = orch.run("Say hello.", max_steps=3)
    assert isinstance(result, str)
    assert len(result) > 0


def test_orchestrator_step_once():
    provider = MockLLMProvider()
    registry = ToolRegistry()
    orch = Orchestrator(provider, registry)
    orch.reset("Test goal")
    result = orch.step_once()
    assert result["type"] == "final"
    assert result["steps_taken"] == 1


def test_orchestrator_max_steps_raises():
    # Provider that always returns tool_calls to simulate a stuck agent
    class LoopingProvider(LLMProvider):
        def generate(self, messages, tools=None, **kwargs) -> LLMResponse:
            return LLMResponse(content=None, tool_calls=None)  # empty, not final

    orch = Orchestrator(LoopingProvider(), ToolRegistry())
    with pytest.raises(RuntimeError, match="Maximum steps reached"):
        orch.run("Loop forever.", max_steps=2)


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------

def test_env_reset_returns_valid_obs():
    orch = Orchestrator(MockLLMProvider(), ToolRegistry())
    env = SecAgentEnv(orchestrator=orch, max_episode_steps=3)
    obs, info = env.reset()
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert "goal" in info


def test_env_step_returns_valid_tuple():
    orch = Orchestrator(MockLLMProvider(), ToolRegistry())
    env = SecAgentEnv(orchestrator=orch, max_episode_steps=3)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_truncates_at_max_steps():
    class NeverEndingProvider(LLMProvider):
        def generate(self, messages, tools=None, **kwargs) -> LLMResponse:
            return LLMResponse(content=None, tool_calls=None)

    orch = Orchestrator(NeverEndingProvider(), ToolRegistry())
    env = SecAgentEnv(orchestrator=orch, max_episode_steps=2)
    env.reset()
    _, _, _, truncated, _ = env.step(0)
    _, _, _, truncated, _ = env.step(0)
    assert truncated is True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_loads():
    from agent.config import load_config
    cfg = load_config()
    assert "llm" in cfg
    assert "targets" in cfg
    assert "training" in cfg


def test_config_has_required_keys():
    from agent.config import load_config
    cfg = load_config()
    assert "model" in cfg["llm"]
    assert "http_hosts" in cfg["targets"]
    assert "nmap_hosts" in cfg["targets"]
    assert "total_timesteps" in cfg["training"]