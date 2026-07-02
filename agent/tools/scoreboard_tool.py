from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse

import requests

from agent.tools.base import Tool


class ScoreboardPollTool(Tool):
    def __init__(self, allowed_hosts: list[str]):
        self.name = "scoreboard_poll"
        self.description = "Poll Juice Shop's /api/Challenges scoreboard and diff solved challenge state from the previous poll."
        self.allowed_hosts = {host.lower() for host in allowed_hosts}
        self._snapshots: dict[str, dict[str, bool]] = {}
        self.parameters = {
            "url": {"type": "string", "description": "Full URL for the scoreboard endpoint, usually /api/Challenges."},
            "headers": {"type": "object", "description": "Optional headers."},
            "token": {"type": "string", "description": "Optional Bearer token for Authorization header."},
            "reset": {"type": "boolean", "description": "Ignore any prior snapshot and treat this call as a fresh baseline."},
        }

    def _is_allowed_url(self, url: str) -> bool:
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        return host is not None and host.lower() in self.allowed_hosts

    def _snapshot_key(self, url: str, headers: dict[str, str]) -> str:
        auth_material = ""
        for header_name in ("authorization", "cookie", "x-api-key"):
            header_value = headers.get(header_name)
            if header_value:
                auth_material = f"{header_name}:{header_value}"
                break

        digest = hashlib.sha256(auth_material.encode("utf-8")).hexdigest() if auth_material else "anonymous"
        return f"{url}|{digest}"

    def _extract_challenges(self, payload: object) -> list[dict]:
        if isinstance(payload, list):
            return [challenge for challenge in payload if isinstance(challenge, dict)]

        if isinstance(payload, dict):
            for key in ("data", "challenges", "Challenges", "results"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [challenge for challenge in value if isinstance(challenge, dict)]

        raise ValueError("scoreboard response did not contain a challenge list")

    def _challenge_key(self, challenge: dict) -> str:
        identifier = challenge.get("id")
        if identifier is not None:
            return str(identifier)
        name = challenge.get("name")
        if name is not None:
            return str(name)
        return json.dumps(challenge, sort_keys=True, default=str)

    def _challenge_name(self, challenge: dict) -> str:
        name = challenge.get("name")
        return str(name) if name is not None else self._challenge_key(challenge)

    def _challenge_solved(self, challenge: dict) -> bool:
        return bool(challenge.get("solved", False))

    def execute(self, **kwargs) -> dict:
        url = kwargs.get("url")
        headers = kwargs.get("headers", {}) or {}
        token = kwargs.get("token")
        reset = bool(kwargs.get("reset", False))

        if not url:
            return {"success": False, "output": "", "error": "URL is required."}

        if not self._is_allowed_url(url):
            return {
                "success": False,
                "output": "",
                "error": f"URL host is not allowed: {url}",
            }

        request_headers = dict(headers)
        normalized_headers = {str(key).lower(): str(value) for key, value in request_headers.items()}
        if token:
            request_headers = {**request_headers, "Authorization": f"Bearer {token}"}
            normalized_headers["authorization"] = f"Bearer {token}"

        try:
            response = requests.get(url, headers=request_headers, timeout=10)
            response.raise_for_status()
            payload = response.json()
            challenges = self._extract_challenges(payload)
        except Exception as exc:
            return {
                "success": False,
                "output": "",
                "error": str(exc),
            }

        snapshot = {self._challenge_key(challenge): self._challenge_solved(challenge) for challenge in challenges}
        key = self._snapshot_key(url, normalized_headers)
        previous_snapshot = None if reset else self._snapshots.get(key)

        newly_solved = []
        newly_unsolved = []
        changed = []

        if previous_snapshot is not None:
            previous_keys = set(previous_snapshot)
            current_keys = set(snapshot)

            for challenge in challenges:
                challenge_key = self._challenge_key(challenge)
                previous_value = previous_snapshot.get(challenge_key)
                current_value = snapshot[challenge_key]
                if previous_value is None or previous_value == current_value:
                    continue

                changed.append(
                    {
                        "id": challenge.get("id"),
                        "name": self._challenge_name(challenge),
                        "solved": current_value,
                    }
                )
                if current_value:
                    newly_solved.append(
                        {
                            "id": challenge.get("id"),
                            "name": self._challenge_name(challenge),
                        }
                    )
                else:
                    newly_unsolved.append(
                        {
                            "id": challenge.get("id"),
                            "name": self._challenge_name(challenge),
                        }
                    )

            for challenge_key in previous_keys - current_keys:
                changed.append(
                    {
                        "id": challenge_key,
                        "name": challenge_key,
                        "solved": None,
                    }
                )

        self._snapshots[key] = snapshot

        solved_count = sum(1 for solved in snapshot.values() if solved)
        output = {
            "url": url,
            "baseline": previous_snapshot is None,
            "challenge_count": len(challenges),
            "solved_count": solved_count,
            "newly_solved": newly_solved,
            "newly_unsolved": newly_unsolved,
            "changed": changed,
            "challenges": challenges,
        }

        return {
            "success": True,
            "output": json.dumps(output),
            "baseline": output["baseline"],
            "challenge_count": output["challenge_count"],
            "solved_count": output["solved_count"],
            "newly_solved": newly_solved,
            "newly_unsolved": newly_unsolved,
            "changed": changed,
            "error": None,
        }