import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm.groq_provider import GroqProvider
from agent.memory.episode_log import EpisodeLogger
from agent.orchestrator import Orchestrator
from agent.tools.registry import setup_registry


load_dotenv()

provider = GroqProvider(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile",
)
registry = setup_registry(["httpbin.org"])
logger = EpisodeLogger()
orchestrator = Orchestrator(provider, registry, logger=logger)
log_path = Path(__file__).with_name("04_test_orchestrator.jsonl")

result = orchestrator.run(
    "Check if httpbin.org/get is responding and tell me what it returns",
    max_steps=5,
)

logger.save(log_path)
print(result)
print(f"Saved episode log to {log_path}")
