import requests
from urllib.parse import urlparse
from agent.tools.base import Tool

class HttpTool(Tool):
    def __init__(self, allowed_hosts: list[str]):
        self.name = "http_request"
        self.description = "Tool for making HTTP requests."
        self.allowed_hosts = {host.lower() for host in allowed_hosts}
        self.parameters = {
            "method": {"type": "string", "description": "HTTP method (GET, POST, etc.)"},
            "url": {"type": "string", "description": "The URL to send the request to."},
            "headers": {"type": "object", "description": "Optional headers."},
            "body": {"type": "string", "description": "Optional raw string body."},
            "json_body": {"type": "object", "description": "Optional JSON body (sets Content-Type automatically)."},
            "token": {"type": "string", "description": "Optional Bearer token for Authorization header."},
        }

    def _is_allowed_url(self, url: str) -> bool:
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        return host is not None and host.lower() in self.allowed_hosts
        
    def execute(self, **kwargs) -> dict:
        method = kwargs.get("method", "GET").upper()
        url = kwargs.get("url")
        headers = kwargs.get("headers", {})
        body = kwargs.get("body", None)
        token = kwargs.get("token", None)
        
        if token:
            headers = {**headers, "Authorization": f"Bearer {token}"}
        
        json_body = kwargs.get("json_body", None)
        
        if isinstance(json_body, str) and json_body.strip():
            try:
                import json as _json
                json_body = _json.loads(json_body)
            except _json.JSONDecodeError:
                return {"success": False, "output": "", "error": "json_body is not valid JSON"}
        else:
            json_body = None 
        
        if not url:
            return {"success": False, "output": "", "error": "URL is required."}

        if not self._is_allowed_url(url):
            return {
                "success": False,
                "output": "",
                "error": f"URL host is not allowed: {url}",
            }
        
        try:
            response = requests.request(
                method, url,
                headers=headers,
                data=body if json_body is None else None,
                json=json_body,
                timeout=10,
            )
            return {
                "success": True,
                "output": response.text[:3000],
                "status_code": response.status_code,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }