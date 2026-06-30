import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tools.registry import setup_registry


registry = setup_registry(["owasp.org"])
result = registry.call("http_request", method="GET", url="https://owasp.org/www-project-juice-shop/")
print(result)
