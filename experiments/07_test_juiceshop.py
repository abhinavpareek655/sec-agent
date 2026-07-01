import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import load_config
from agent.llm.factory import build_provider
from agent.memory.episode_log import EpisodeLogger
from agent.orchestrator import Orchestrator
from agent.tools.registry import setup_registry

load_dotenv()
cfg = load_config()

provider = build_provider(cfg)
registry = setup_registry(
    http_hosts=["localhost"],
    nmap_hosts=["localhost"],
)
logger = EpisodeLogger()
orchestrator = Orchestrator(provider, registry, logger=logger)

result = orchestrator.run(
    goal = """
    You have access to an HTTP request tool.
    Make GET requests to each of these paths on localhost:3000 and report which ones return a non-404 response:
    /rest/products/search
    /api-docs
    /rest/user/whoami
    /rest/basket
    /swagger.json
    /metrics
    Report the status code for each path.
    """,
    max_steps=8,
)

log_dir = Path(__file__).parent
base_name = "07_test_juiceshop"
extension = ".jsonl"

# Find the next available filename
counter = 1
log_path = log_dir / f"{base_name}.{extension}"
while log_path.exists():
    log_path = log_dir / f"{base_name}_{counter:03d}{extension}"
    counter += 1

logger.save(log_path)
print(result)
print(f"\nLog saved to {log_path}")