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
            "headers": {"type": "dict", "description": "Optional headers for the request."},
            "body": {"type": "string", "description": "Optional body for POST/PUT requests."},
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
        
        if not url:
            return {"success": False, "output": "", "error": "URL is required."}

        if not self._is_allowed_url(url):
            return {
                "success": False,
                "output": "",
                "error": f"URL host is not allowed: {url}",
            }
        
        try:
            response = requests.request(method, url, headers=headers, data=body)
            return {
                "success": True,
                "output": response.text,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }