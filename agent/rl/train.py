from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm.groq_provider import GroqProvider
from agent.memory.episode_log import EpisodeLogger
from agent.orchestrator import Orchestrator
from agent.rl.env import SecAgentEnv
from agent.tools.registry import setup_registry


DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_GOAL = "Check whether localhost:3000 is responding and report what it returns."
DEFAULT_MAX_EPISODE_STEPS = 3
DEFAULT_TOTAL_TIMESTEPS = 30

# PPO needs n_steps * n_envs timesteps collected before it can do ONE update.
# With only 30 total timesteps and 1 env, n_steps must be small or you get
# zero updates. batch_size must evenly divide n_steps (SB3 requirement).
PPO_N_STEPS = 12
PPO_BATCH_SIZE = 6
PPO_N_EPOCHS = 4  # how many times to reuse each small batch of data


def build_env() -> SecAgentEnv:
    provider = GroqProvider(
        api_key=os.environ["GROQ_API_KEY"],
        model_name=DEFAULT_MODEL,
    )
    registry = setup_registry(
        http_hosts=["localhost"],
        nmap_hosts=["localhost"],
    )
    orchestrator = Orchestrator(provider, registry, logger=build_env.logger)
    return SecAgentEnv(
        orchestrator=orchestrator,
        default_goal=DEFAULT_GOAL,
        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
    )


build_env.logger = EpisodeLogger()


def main() -> None:
    load_dotenv()

    env = DummyVecEnv([build_env])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        verbose=1,
    )

    if DEFAULT_TOTAL_TIMESTEPS < PPO_N_STEPS:
        raise ValueError(
            f"total_timesteps ({DEFAULT_TOTAL_TIMESTEPS}) must be >= n_steps "
            f"({PPO_N_STEPS}) or PPO will never perform a single update."
        )

    model.learn(total_timesteps=DEFAULT_TOTAL_TIMESTEPS)

    model_path = PROJECT_ROOT / "agent_rl_ppo_model.zip"
    model.save(model_path)
    build_env.logger.save(PROJECT_ROOT / "train_episode.jsonl")
    print(f"Saved PPO model to {model_path}")


if __name__ == "__main__":
    main()