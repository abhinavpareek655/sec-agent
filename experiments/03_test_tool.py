import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tools.http_tool import HttpTool


tool = HttpTool(allowed_hosts=["httpbin.org"])
result = tool.execute(method="GET", url="https://httpbin.org/get")
print(result)
