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
    You are testing OWASP Juice Shop running at localhost:3000.

    Step 1: POST to http://localhost:3000/rest/user/login with json_body:
    {"email": "admin@juice-sh.op", "password": "admin123"}

    Step 2: The response contains a JSON object. Use the extract_json_value tool
    to extract the token at key path "authentication.token"

    Step 3: Use that token to GET http://localhost:3000/rest/basket/1
    by passing it as the token parameter.

    Step 4: Report what the basket contains.
    """,
    max_steps=12,
)

log_dir = Path(__file__).parent
base_name = "08_test_auth"
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