import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tools.nmap_tool import NmapTool


tool = NmapTool(allowed_hosts=["scanme.nmap.org"])
result = tool.execute(target="scanme.nmap.org", scan_type="version")
print(result)
