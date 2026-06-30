import subprocess

from agent.tools.base import Tool
from agent.tools.security import is_host_allowed, is_target_valid


class NmapTool(Tool):
    def __init__(self, allowed_hosts: list[str]):
        self.name = "nmap_scan"
        self.description = "Tool for scanning targets with nmap to discover open ports and services."
        self.allowed_hosts = {host.lower() for host in allowed_hosts}
        self.parameters = {
            "target": {"type": "string", "description": "Hostname or IP address to scan"},
            "scan_type": {"type": "string", "description": "Scan type: 'version' (default) for service version detection"},
        }

    def execute(self, **kwargs) -> dict:
        target = kwargs.get("target")
        scan_type = kwargs.get("scan_type", "version")

        if not target:
            return {"success": False, "output": "", "error": "Target is required."}

        if not is_target_valid(target):
            return {
                "success": False,
                "output": "",
                "error": f"Invalid target: appears to contain injection attempt or invalid characters: {target}",
            }

        if not is_host_allowed(target, self.allowed_hosts):
            return {
                "success": False,
                "output": "",
                "error": f"Target host is not allowed: {target}",
            }

        try:
            if scan_type == "version":
                args = ["nmap", "-sV", "--script=banner", target]
            else:
                args = ["nmap", target]

            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"

            if result.returncode != 0:
                return {
                    "success": False,
                    "output": output,
                    "error": f"nmap exited with code {result.returncode}",
                }

            return {
                "success": True,
                "output": output,
                "error": None,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "nmap scan timed out after 30 seconds",
            }
        except FileNotFoundError:
            return {
                "success": False,
                "output": "",
                "error": "nmap not found on system",
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
            }