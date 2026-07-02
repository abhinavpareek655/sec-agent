from __future__ import annotations

import hashlib
import json
from urllib.parse import urlparse

import requests

from agent.tools.base import Tool


class _ScoreboardBase(Tool):
    """Shared helpers for tools that talk to Juice Shop's /api/Challenges endpoint."""

    def __init__(self, allowed_hosts: list[str]):
        self.allowed_hosts = {host.lower() for host in allowed_hosts}

    def _is_allowed_url(self, url: str) -> bool:
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        return host is not None and host.lower() in self.allowed_hosts

    def _build_headers(self, headers: dict[str, str] | None, token: str | None) -> tuple[dict, dict]:
        request_headers = dict(headers or {})
        if token:
            request_headers["Authorization"] = f"Bearer {token}"
        normalized_headers = {str(k).lower(): str(v) for k, v in request_headers.items()}
        return request_headers, normalized_headers

    def _fetch_challenges(self, url: str, request_headers: dict) -> list[dict]:
        response = requests.get(url, headers=request_headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
        return self._extract_challenges(payload)

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


class ScoreboardPollTool(_ScoreboardBase):
    """
    Lean, diff-only poll. Call this frequently (after every probe) to check
    whether an action flipped any challenge from unsolved -> solved.

    Never returns the full challenge list — only counts and what changed —
    so it stays cheap no matter how many times it's called in a session.
    """

    def __init__(self, allowed_hosts: list[str]):
        super().__init__(allowed_hosts)
        self.name = "scoreboard_poll"
        self.description = (
            "Poll Juice Shop's /api/Challenges scoreboard and diff solved challenge state "
            "against the previous poll for the same URL+auth. Returns only counts and what "
            "changed (no challenge descriptions/hints) — use this after every probe to check "
            "for newly solved challenges. Use scoreboard_list_unsolved to see what's left to target."
        )
        self._snapshots: dict[str, dict[str, bool]] = {}
        self.parameters = {
            "url": {"type": "string", "description": "Full URL for the scoreboard endpoint, usually /api/Challenges."},
            "headers": {"type": "object", "description": "Optional headers."},
            "token": {"type": "string", "description": "Optional Bearer token for Authorization header."},
            "reset": {"type": "boolean", "description": "Ignore any prior snapshot and treat this call as a fresh baseline."},
        }

    def _snapshot_key(self, url: str, headers: dict[str, str]) -> str:
        auth_material = ""
        for header_name in ("authorization", "cookie", "x-api-key"):
            header_value = headers.get(header_name)
            if header_value:
                auth_material = f"{header_name}:{header_value}"
                break

        digest = hashlib.sha256(auth_material.encode("utf-8")).hexdigest() if auth_material else "anonymous"
        return f"{url}|{digest}"

    def execute(self, **kwargs) -> dict:
        url = kwargs.get("url")
        headers = kwargs.get("headers", {}) or {}
        token = kwargs.get("token")
        reset = bool(kwargs.get("reset", False))

        if not url:
            return {"success": False, "output": "", "error": "URL is required."}

        if not self._is_allowed_url(url):
            return {"success": False, "output": "", "error": f"URL host is not allowed: {url}"}

        request_headers, normalized_headers = self._build_headers(headers, token)

        try:
            challenges = self._fetch_challenges(url, request_headers)
        except Exception as exc:
            return {"success": False, "output": "", "error": str(exc)}

        snapshot = {self._challenge_key(c): self._challenge_solved(c) for c in challenges}
        key = self._snapshot_key(url, normalized_headers)
        previous_snapshot = None if reset else self._snapshots.get(key)

        newly_solved: list[dict] = []
        newly_unsolved: list[dict] = []
        changed: list[dict] = []

        if previous_snapshot is not None:
            previous_keys = set(previous_snapshot)
            current_keys = set(snapshot)

            for challenge in challenges:
                challenge_key = self._challenge_key(challenge)
                previous_value = previous_snapshot.get(challenge_key)
                current_value = snapshot[challenge_key]
                if previous_value is None or previous_value == current_value:
                    continue

                entry = {"id": challenge.get("id"), "name": self._challenge_name(challenge)}
                changed.append({**entry, "solved": current_value})
                (newly_solved if current_value else newly_unsolved).append(entry)

            for challenge_key in previous_keys - current_keys:
                changed.append({"id": challenge_key, "name": challenge_key, "solved": None})

        self._snapshots[key] = snapshot

        solved_count = sum(1 for solved in snapshot.values() if solved)

        # Deliberately excludes the full challenge list (name/description/hint/etc.)
        # from the output — that lives in scoreboard_list_unsolved instead, which
        # is called far less often. This keeps every poll response small and flat.
        output = {
            "url": url,
            "baseline": previous_snapshot is None,
            "challenge_count": len(challenges),
            "solved_count": solved_count,
            "newly_solved": newly_solved,
            "newly_unsolved": newly_unsolved,
            "changed": changed,
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


class ScoreboardListUnsolvedTool(_ScoreboardBase):
    """
    Returns a slim worklist of unsolved challenges (id/name/category/difficulty/hint),
    stripping heavy fields Juice Shop includes but the agent doesn't need
    (description, hintUrl, mitigationUrl, tags, key, tutorialOrder, etc.).

    Call this occasionally, not after every probe — the list naturally shrinks
    as challenges get solved, so later calls in a session are cheap too.
    """

    def __init__(self, allowed_hosts: list[str]):
        super().__init__(allowed_hosts)
        self.name = "scoreboard_list_unsolved"
        self.description = (
            "Fetch a slim worklist of currently unsolved Juice Shop challenges "
            "(id, name, category, difficulty, hint). Excludes heavy fields like "
            "full descriptions and URLs. Optionally filter by category to avoid "
            "pulling the entire remaining list at once. Call this occasionally to "
            "pick a target, not after every probe (use scoreboard_poll for that)."
        )
        self.parameters = {
            "url": {"type": "string", "description": "Full URL for the scoreboard endpoint, usually /api/Challenges."},
            "headers": {"type": "object", "description": "Optional headers."},
            "token": {"type": "string", "description": "Optional Bearer token for Authorization header."},
            "category": {"type": "string", "description": "Optional exact/substring match on challenge category, e.g. 'XSS', 'Broken Access Control'."},
            "include_hint": {"type": "boolean", "description": "Whether to include each challenge's hint text. Defaults to true; set false to force the agent to discover challenges without hints."},
        }

    def execute(self, **kwargs) -> dict:
        url = kwargs.get("url")
        headers = kwargs.get("headers", {}) or {}
        token = kwargs.get("token")
        category = kwargs.get("category")
        include_hint = kwargs.get("include_hint", True)

        if not url:
            return {"success": False, "output": "", "error": "URL is required."}

        if not self._is_allowed_url(url):
            return {"success": False, "output": "", "error": f"URL host is not allowed: {url}"}

        request_headers, _ = self._build_headers(headers, token)

        try:
            challenges = self._fetch_challenges(url, request_headers)
        except Exception as exc:
            return {"success": False, "output": "", "error": str(exc)}

        unsolved = [c for c in challenges if not self._challenge_solved(c)]

        if category:
            needle = category.lower()
            unsolved = [c for c in unsolved if needle in str(c.get("category", "")).lower()]

        slim = []
        for c in unsolved:
            entry = {
                "id": c.get("id"),
                "name": c.get("name"),
                "category": c.get("category"),
                "difficulty": c.get("difficulty"),
            }
            if include_hint:
                entry["hint"] = c.get("hint")
            slim.append(entry)

        output = {
            "unsolved_count": len(slim),
            "filtered_by_category": category,
            "challenges": slim,
        }

        return {
            "success": True,
            "output": json.dumps(output),
            "unsolved_count": len(slim),
            "error": None,
        }


class ScoreboardGetChallengeTool(_ScoreboardBase):
    """
    Fetch metadata (including hint) for a single challenge by id or exact name.
    Lets the agent pull one hint on demand instead of receiving every hint
    up front, which better preserves independent-discovery behavior while
    still giving it an escape hatch when stuck.
    """

    def __init__(self, allowed_hosts: list[str]):
        super().__init__(allowed_hosts)
        self.name = "scoreboard_get_challenge"
        self.description = (
            "Fetch full metadata (name, category, difficulty, description, hint) for a "
            "single Juice Shop challenge by id or exact name. Use this on demand when "
            "targeting or stuck on one specific challenge, rather than pulling hints for "
            "everything at once."
        )
        self.parameters = {
            "url": {"type": "string", "description": "Full URL for the scoreboard endpoint, usually /api/Challenges."},
            "headers": {"type": "object", "description": "Optional headers."},
            "token": {"type": "string", "description": "Optional Bearer token for Authorization header."},
            "challenge_id": {"type": "string", "description": "The challenge id to look up. Provide this or challenge_name."},
            "challenge_name": {"type": "string", "description": "Exact challenge name to look up. Provide this or challenge_id."},
        }

    def execute(self, **kwargs) -> dict:
        url = kwargs.get("url")
        headers = kwargs.get("headers", {}) or {}
        token = kwargs.get("token")
        challenge_id = kwargs.get("challenge_id")
        challenge_name = kwargs.get("challenge_name")

        if not url:
            return {"success": False, "output": "", "error": "URL is required."}
        if not challenge_id and not challenge_name:
            return {"success": False, "output": "", "error": "Provide challenge_id or challenge_name."}

        if not self._is_allowed_url(url):
            return {"success": False, "output": "", "error": f"URL host is not allowed: {url}"}

        request_headers, _ = self._build_headers(headers, token)

        try:
            challenges = self._fetch_challenges(url, request_headers)
        except Exception as exc:
            return {"success": False, "output": "", "error": str(exc)}

        match = None
        for c in challenges:
            if challenge_id is not None and str(c.get("id")) == str(challenge_id):
                match = c
                break
            if challenge_name is not None and str(c.get("name")) == str(challenge_name):
                match = c
                break

        if match is None:
            return {
                "success": False,
                "output": "",
                "error": f"No challenge found matching id={challenge_id!r} name={challenge_name!r}",
            }

        entry = {
            "id": match.get("id"),
            "name": match.get("name"),
            "category": match.get("category"),
            "difficulty": match.get("difficulty"),
            "description": match.get("description"),
            "hint": match.get("hint"),
            "solved": self._challenge_solved(match),
        }

        return {"success": True, "output": json.dumps(entry), "error": None}