import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import load_config
from agent.llm.factory import build_provider
from agent.memory.episode_log import EpisodeLogger
from agent.orchestrator import Orchestrator
from agent.tools.scoreboard_tool import ScoreboardPollTool
from agent.tools.registry import setup_registry

from goal_prompt_scoreboard import GOAL
from graded_run import run_graded

load_dotenv()
cfg = load_config()

provider = build_provider(cfg)
registry = setup_registry(
    http_hosts=["localhost"],
    nmap_hosts=["localhost"],
)
log_dir = Path(__file__).parent
base_name = "09_test_scoreboard"
extension = ".jsonl"

# Find the next available filename
counter = 1
log_path = log_dir / f"{base_name}{extension}"
while log_path.exists():
    log_path = log_dir / f"{base_name}_{counter:03d}{extension}"
    counter += 1

logger = EpisodeLogger(auto_save_path=log_path)
orchestrator = Orchestrator(provider, registry, logger=logger)

# Pull the scoreboard_poll tool instance straight out of the registry so the
# grading wrapper can poll independently of whatever the agent itself does.
scoreboard_tool = ScoreboardPollTool(allowed_hosts=["localhost"])

try:
    graded = run_graded(
        orchestrator=orchestrator,
        scoreboard_tool=scoreboard_tool,
        scoreboard_url="http://localhost:3000/api/Challenges",
        goal=GOAL,
        max_steps=30,
    )
    print(graded.summary())
    if graded.agent_result is not None:
        print("\n--- agent's own final message ---")
        print(graded.agent_result)
except Exception as exc:
    print(f"Run failed: {exc}")
finally:
    logger.save(log_path)
    print(f"\nLog saved to {log_path}")