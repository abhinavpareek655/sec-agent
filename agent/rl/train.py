from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config import load_config
from agent.llm.factory import build_provider
from agent.memory.episode_log import EpisodeLogger
from agent.orchestrator import Orchestrator
from agent.rl.env import SecAgentEnv
from agent.tools.registry import setup_registry

_shared_logger = EpisodeLogger()


def build_env() -> SecAgentEnv:
    cfg = load_config()
    provider = build_provider(cfg)
    registry = setup_registry(
        http_hosts=cfg["targets"]["http_hosts"],
        nmap_hosts=cfg["targets"]["nmap_hosts"],
    )
    orchestrator = Orchestrator(provider, registry, logger=_shared_logger)
    return SecAgentEnv(
        orchestrator=orchestrator,
        default_goal=cfg["agent"]["default_goal"],
        max_episode_steps=cfg["agent"]["max_episode_steps"],
    )


def main() -> None:
    load_dotenv()
    cfg = load_config()
    t = cfg["training"]

    if t["total_timesteps"] < t["n_steps"]:
        raise ValueError(
            f"total_timesteps ({t['total_timesteps']}) must be >= "
            f"n_steps ({t['n_steps']}) or PPO will never update."
        )

    env = DummyVecEnv([build_env])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=t["n_steps"],
        batch_size=t["batch_size"],
        n_epochs=t["n_epochs"],
        verbose=1,
    )
    model.learn(total_timesteps=t["total_timesteps"])

    model_path = PROJECT_ROOT / "agent_rl_ppo_model.zip"
    model.save(model_path)
    _shared_logger.save(PROJECT_ROOT / "train_episode.jsonl")
    print(f"Saved PPO model to {model_path}")


if __name__ == "__main__":
    main()