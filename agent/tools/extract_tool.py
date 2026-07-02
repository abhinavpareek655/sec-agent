import json
from agent.tools.base import Tool

class ExtractJsonTool(Tool):
    def __init__(self):
        self.name = "extract_json_value"
        self.description = (
            "Extract a value from a JSON string by key path. "
            "Use this to pull the token out of a login response before making authenticated requests."
        )
        self.parameters = {
            "json_string": {"type": "string", "description": "The raw JSON string to parse."},
            "key_path": {"type": "string", "description": "Dot-separated key path, e.g. 'authentication.token'"},
        }

    def execute(self, **kwargs) -> dict:
        json_string = kwargs.get("json_string", "")
        key_path = kwargs.get("key_path", "")

        if isinstance(json_string, str):
            try:
                data = json.loads(json_string)
            except json.JSONDecodeError:
                try:
                    data = json.loads(json_string.encode().decode('unicode_escape'))
                except Exception:
                    return {"success": False, "output": "", "error": "Could not parse json_string as JSON"}
        else:
            data = json_string