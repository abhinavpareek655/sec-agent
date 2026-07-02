"""
Smoke tests that run without any real API keys or external services.
The LLM provider is mocked so CI stays free and fast.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent.llm.base import LLMProvider, LLMResponse
from agent.memory.episode_log import EpisodeLogger
from agent.orchestrator import Orchestrator
from agent.rl.env import SecAgentEnv
from agent.tools.registry import ToolRegistry
from agent.tools.registry import setup_registry
from agent.tools.http_tool import HttpTool
from agent.tools.login_vector_tool import LoginVectorTool
from agent.tools.scoreboard_tool import ScoreboardGetChallengeTool
from agent.tools.scoreboard_tool import ScoreboardListUnsolvedTool
from agent.tools.scoreboard_tool import ScoreboardPollTool
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


def test_registry_exposes_login_vector_tool():
    registry = setup_registry(http_hosts=["localhost"])
    tool_names = {tool["function"]["name"] for tool in registry.list_tools()}
    assert "login_vector_attempt" in tool_names


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


def test_login_vector_tool_returns_token(monkeypatch):
    tool = LoginVectorTool(allowed_hosts=["localhost"])

    class FirstResponse:
        def __init__(self, status_code, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

    responses = [
        FirstResponse(401),
        FirstResponse(200, {"authentication": {"token": "abc123"}}),
    ]

    def fake_post(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("agent.tools.login_vector_tool.requests.post", fake_post)

    result = tool.execute(url="http://localhost/rest/user/login")
    assert result["success"] is True
    assert result["token"] == "abc123"
    assert result["matched_kind"] == "sqli_bypass"


def test_scoreboard_poll_tool_baseline(monkeypatch):
    tool = ScoreboardPollTool(allowed_hosts=["localhost"])

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"id": 1, "name": "Scoreboard", "solved": False},
                {"id": 2, "name": "Confidential Document", "solved": False},
            ]

    monkeypatch.setattr("agent.tools.scoreboard_tool.requests.get", lambda *args, **kwargs: FakeResponse())

    result = tool.execute(url="http://localhost/api/Challenges")
    assert result["success"] is True
    assert result["baseline"] is True
    assert result["solved_count"] == 0
    assert result["newly_solved"] == []


def test_scoreboard_poll_tool_detects_solved_delta(monkeypatch):
    tool = ScoreboardPollTool(allowed_hosts=["localhost"])

    class FirstResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"id": 1, "name": "Scoreboard", "solved": False},
                {"id": 2, "name": "Confidential Document", "solved": False},
            ]

    class SecondResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"id": 1, "name": "Scoreboard", "solved": False},
                {"id": 2, "name": "Confidential Document", "solved": True},
            ]

    responses = [FirstResponse(), SecondResponse()]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("agent.tools.scoreboard_tool.requests.get", fake_get)

    first = tool.execute(url="http://localhost/api/Challenges")
    second = tool.execute(url="http://localhost/api/Challenges")

    assert first["baseline"] is True
    assert second["baseline"] is False
    assert second["solved_count"] == 1
    assert second["newly_solved"] == [{"id": 2, "name": "Confidential Document"}]
    assert second["changed"] == [{"id": 2, "name": "Confidential Document", "solved": True}]


def test_scoreboard_list_unsolved_filters_and_shrinks(monkeypatch):
    tool = ScoreboardListUnsolvedTool(allowed_hosts=["localhost"])

    class FirstResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"id": 1, "name": "Scoreboard", "category": "Misc", "difficulty": 0, "hint": "start", "solved": False},
                {"id": 2, "name": "Broken Access", "category": "Broken Access Control", "difficulty": 3, "hint": "bypass", "solved": False},
                {"id": 3, "name": "Solved One", "category": "XSS", "difficulty": 2, "hint": "done", "solved": True},
            ]

    class SecondResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"id": 1, "name": "Scoreboard", "category": "Misc", "difficulty": 0, "hint": "start", "solved": False},
                {"id": 2, "name": "Broken Access", "category": "Broken Access Control", "difficulty": 3, "hint": "bypass", "solved": True},
                {"id": 3, "name": "Solved One", "category": "XSS", "difficulty": 2, "hint": "done", "solved": True},
            ]

    responses = [FirstResponse(), SecondResponse()]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("agent.tools.scoreboard_tool.requests.get", fake_get)

    first = tool.execute(url="http://localhost/api/Challenges", category="Broken")
    second = tool.execute(url="http://localhost/api/Challenges", category="Broken")

    assert first["success"] is True
    assert first["unsolved_count"] == 1
    assert first["output"]
    assert second["unsolved_count"] == 0


def test_scoreboard_get_challenge_by_id_and_name(monkeypatch):
    tool = ScoreboardGetChallengeTool(allowed_hosts=["localhost"])

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "id": 7,
                    "name": "Confidential Document",
                    "category": "Broken Access Control",
                    "difficulty": 4,
                    "description": "Full metadata here",
                    "hint": "Look for a user-owned document",
                    "solved": False,
                }
            ]

    monkeypatch.setattr("agent.tools.scoreboard_tool.requests.get", lambda *args, **kwargs: FakeResponse())

    by_id = tool.execute(url="http://localhost/api/Challenges", challenge_id=7)
    by_name = tool.execute(url="http://localhost/api/Challenges", challenge_name="Confidential Document")

    assert by_id["success"] is True
    assert by_name["success"] is True
    assert json.loads(by_id["output"])["description"] == "Full metadata here"
    assert json.loads(by_name["output"])["hint"] == "Look for a user-owned document"


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


def test_episode_logger_auto_saves_partial_logs(tmp_path):
    log_path = tmp_path / "episode.jsonl"
    logger = EpisodeLogger(auto_save_path=log_path)

    logger.log_step(role="system", content="seed")
    assert log_path.exists()

    first_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(first_lines) == 1
    assert json.loads(first_lines[0])["content"] == "seed"

    logger.log_step(role="assistant", content="step 2")
    second_lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(second_lines) == 2
    assert json.loads(second_lines[1])["content"] == "step 2"


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
    
# ---------------------------------------------------------------------------
# Extract JSON tool
# ---------------------------------------------------------------------------
    
def test_extract_json_tool_success():
    from agent.tools.extract_tool import ExtractJsonTool
    tool = ExtractJsonTool()
    result = tool.execute(
        json_string='{"authentication": {"token": "abc123"}}',
        key_path="authentication.token"
    )
    assert result["success"] is True
    assert result["output"] == "abc123"

def test_extract_json_tool_bad_key():
    from agent.tools.extract_tool import ExtractJsonTool
    tool = ExtractJsonTool()
    result = tool.execute(
        json_string='{"foo": "bar"}',
        key_path="authentication.token"
    )
    assert result["success"] is False

def test_extract_json_tool_empty():
    from agent.tools.extract_tool import ExtractJsonTool
    tool = ExtractJsonTool()
    result = tool.execute(json_string="", key_path="token")
    assert result["success"] is False

def test_extract_json_tool_regex_fallback():
    from agent.tools.extract_tool import ExtractJsonTool
    tool = ExtractJsonTool()
    mangled = '{"token": "eyJ0eXAiOiJKV1Q.payload.signature" CORRUPTED'
    result = tool.execute(json_string=mangled, key_path="token")
    assert result["success"] is True
    assert result["output"] == "eyJ0eXAiOiJKV1Q.payload.signature"