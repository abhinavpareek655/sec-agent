from __future__ import annotations

import re
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ModuleNotFoundError:
    class _Env:
        metadata = {"render_modes": []}

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            return None, {}

    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = n

    class _Box:
        def __init__(self, low: np.ndarray, high: np.ndarray, dtype: Any) -> None:
            self.low = low
            self.high = high
            self.dtype = dtype
            self.shape = low.shape

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _GymShim:
        Env = _Env
        spaces = _Spaces()

    gym = _GymShim()

from agent.orchestrator import Orchestrator


class SecAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        orchestrator: Orchestrator,
        default_goal: str = "Scan the target and identify useful services.",
        max_episode_steps: int = 10,
    ) -> None:
        self.orchestrator = orchestrator
        self.default_goal = default_goal
        self.max_episode_steps = max_episode_steps

        self.action_space = gym.spaces.Discrete(3)
        # Action selects tool preference; the LLM still decides when to end the episode.
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1000.0, 1000.0, 1000.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.goal_text = default_goal
        self.steps_taken = 0
        self.open_ports_count = 0
        self.http_responses_seen = 0
        self.goal_achieved_flag = 0
        self._seen_open_ports: set[str] = set()

    def _constraint_for_action(self, action: int) -> str | None:
        if action == 1:
            return "Prefer the nmap_scan tool for recon and service discovery."
        if action == 2:
            return "Prefer the http_request tool for HTTP probing of discovered services."
        return None

    def _goal_requires_scan(self) -> bool:
        goal = self.goal_text.lower()
        return any(keyword in goal for keyword in ["scan", "port", "nmap", "open"])

    def _goal_requires_http(self) -> bool:
        goal = self.goal_text.lower()
        return any(keyword in goal for keyword in ["http", "respond", "request", "service"])

    def _goal_satisfied(self) -> bool:
        requires_scan = self._goal_requires_scan()
        requires_http = self._goal_requires_http()

        if requires_scan and requires_http:
            return self.open_ports_count > 0 and self.http_responses_seen > 0
        if requires_scan:
            return self.open_ports_count > 0
        if requires_http:
            return self.http_responses_seen > 0
        return self.open_ports_count > 0 or self.http_responses_seen > 0

    def _extract_open_ports(self, output: str) -> set[str]:
        open_ports: set[str] = set()
        for line in output.splitlines():
            match = re.match(r"^(\d+)/tcp\s+open\b", line.strip())
            if match:
                open_ports.add(match.group(1))
        return open_ports

    def _observation(self) -> np.ndarray:
        return np.array(
            [
                float(self.steps_taken),
                float(self.open_ports_count),
                float(self.http_responses_seen),
                float(self.goal_achieved_flag),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        options = options or {}
        self.goal_text = options.get("goal", self.default_goal)
        self.steps_taken = 0
        self.open_ports_count = 0
        self.http_responses_seen = 0
        self.goal_achieved_flag = 0
        self._seen_open_ports.clear()

        self.orchestrator.reset(self.goal_text)

        return self._observation(), {"goal": self.goal_text}

    def step(self, action: int):
        constraint = self._constraint_for_action(int(action))
        result = self.orchestrator.step_once(constraint=constraint)

        self.steps_taken += 1

        reward = -0.1
        terminated = False
        truncated = False

        for tool_result in result.get("tool_results", []):
            tool_name = tool_result.get("name")
            structured_result = tool_result.get("result", {}) or {}

            if tool_name == "nmap_scan" and structured_result.get("success"):
                output = structured_result.get("output", "")
                newly_seen_ports = self._extract_open_ports(output)
                new_ports = newly_seen_ports - self._seen_open_ports
                if new_ports:
                    self._seen_open_ports.update(new_ports)
                    self.open_ports_count = len(self._seen_open_ports)
                    reward += float(len(new_ports))

            if tool_name == "http_request" and structured_result.get("success"):
                status_code = structured_result.get("status_code")
                if status_code == 200:
                    self.http_responses_seen += 1
                    reward += 5.0

        if self._goal_satisfied():
            self.goal_achieved_flag = 1
            reward += 10.0
            terminated = True
        elif result.get("type") == "final":
            terminated = True

        if self.steps_taken >= self.max_episode_steps and not terminated:
            truncated = True

        observation = self._observation()
        info = {
            "goal": self.goal_text,
            "steps_taken": self.steps_taken,
            "goal_achieved": bool(self.goal_achieved_flag),
            "last_step_type": result.get("type"),
            "last_step": result,
        }

        return observation, reward, terminated, truncated, info

