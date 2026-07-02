import json
import re
from agent.tools.base import Tool


class ExtractJsonTool(Tool):
    def __init__(self):
        self.name = "extract_json_value"
        self.description = (
            "Extract a value from a JSON response by dot-separated key path. "
            "Pass the output string from the previous http_request tool result as json_string. "
            "key_path uses dot notation: 'authentication.token' extracts response['authentication']['token']."
        )
        self.parameters = {
            "json_string": {
                "type": "string",
                "description": "The JSON string to extract from. Pass the 'output' field from the http_request result."
            },
            "key_path": {
                "type": "string",
                "description": "Dot-separated key path, e.g. 'authentication.token' or 'data.id'"
            },
        }

    def execute(self, **kwargs) -> dict:
        json_string = kwargs.get("json_string", "")
        key_path = kwargs.get("key_path", "")

        if not key_path:
            return {"success": False, "output": "", "error": "key_path is required"}

        # Handle case where LLM passes an actual dict instead of a string
        if isinstance(json_string, dict):
            return self._traverse(json_string, key_path.split("."))

        if not json_string:
            return {"success": False, "output": "", "error": "json_string is required"}

        keys = key_path.split(".")

        # Strategy 1: standard json.loads (fastest, most correct)
        data = self._try_parse(json_string)
        if data is not None:
            return self._traverse(data, keys)

        # Strategy 2: regex extraction — works even on mangled/truncated JSON.
        # This is the guaranteed fallback: JWT tokens contain only base64url chars
        # (A-Z a-z 0-9 - _ .) so regex "key": "value" captures them cleanly.
        last_key = keys[-1]
        value = self._regex_extract(json_string, last_key)
        if value is not None:
            return {"success": True, "output": value, "error": None}

        return {
            "success": False,
            "output": "",
            "error": (
                f"Could not extract '{key_path}'. "
                f"JSON parse failed and regex found no match for key '{last_key}'. "
                f"First 100 chars of input: {json_string[:100]!r}"
            )
        }

    def _try_parse(self, raw: str) -> dict | list | None:
        for candidate in (raw, raw.strip()):
            try:
                result = json.loads(candidate)
                if isinstance(result, (dict, list)):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _traverse(self, data, keys: list[str]) -> dict:
        try:
            value = data
            for key in keys:
                if isinstance(value, dict):
                    value = value[key]
                elif isinstance(value, list) and key.isdigit():
                    value = value[int(key)]
                else:
                    raise KeyError(key)
            return {"success": True, "output": str(value), "error": None}
        except (KeyError, IndexError, TypeError) as e:
            top_keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            return {
                "success": False,
                "output": "",
                "error": f"Key not found: {e}. Top-level keys: {top_keys}"
            }

    def _regex_extract(self, text: str, key: str) -> str | None:
        escaped = re.escape(key)

        # String values: "key": "value" — handles escaped chars inside value
        m = re.search(rf'"{escaped}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            return m.group(1)

        # Numeric values: "key": 123
        m = re.search(rf'"{escaped}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
        if m:
            return m.group(1)

        # Boolean/null: "key": true/false/null
        m = re.search(rf'"{escaped}"\s*:\s*(true|false|null)', text)
        if m:
            return m.group(1)

        return None