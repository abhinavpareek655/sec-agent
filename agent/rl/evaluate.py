from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from agent.config import load_config
from agent.llm.groq_provider import GroqProvider
from agent.memory.episode_log import EpisodeLogger
from agent.orchestrator import Orchestrator
from agent.rl.env import SecAgentEnv
from agent.tools.registry import setup_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTION_NAMES = {
    0: "unconstrained",
    1: "prefer_nmap",
    2: "prefer_http",
}


def build_eval_env(cfg: dict) -> SecAgentEnv:
    provider = GroqProvider(
        api_key=os.environ["GROQ_API_KEY"],
        model_name=cfg["llm"]["model"],
    )
    registry = setup_registry(
        http_hosts=cfg["targets"]["http_hosts"],
        nmap_hosts=cfg["targets"]["nmap_hosts"],
    )
    orchestrator = Orchestrator(provider, registry, logger=EpisodeLogger())
    return SecAgentEnv(
        orchestrator=orchestrator,
        default_goal=cfg["agent"]["default_goal"],
        max_episode_steps=cfg["agent"]["max_episode_steps"],
    )


def evaluate(model_path: Path, n_episodes: int = 5) -> None:
    load_dotenv()
    cfg = load_config()

    env = DummyVecEnv([lambda: build_eval_env(cfg)])
    model = PPO.load(model_path, env=env)
    print(f"Loaded model from {model_path}\n")

    episode_rewards: list[float] = []
    action_counts: dict[int, int] = defaultdict(int)

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        episode_actions: list[str] = []

        print(f"--- Episode {episode + 1} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action[0])
            action_counts[action_int] += 1
            episode_actions.append(ACTION_NAMES[action_int])

            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            step += 1

            print(
                f"  Step {step}: action={ACTION_NAMES[action_int]:<16} "
                f"reward={float(reward[0]):+.1f}  "
                f"done={bool(done[0])}"
            )

        episode_rewards.append(total_reward)
        print(f"  Actions taken: {episode_actions}")
        print(f"  Total reward:  {total_reward:.1f}\n")

    print("=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Episodes:          {n_episodes}")
    print(f"Mean reward:       {np.mean(episode_rewards):.2f}")
    print(f"Std reward:        {np.std(episode_rewards):.2f}")
    print(f"Min / Max reward:  {min(episode_rewards):.1f} / {max(episode_rewards):.1f}")
    print("\nAction distribution:")
    total_actions = sum(action_counts.values())
    for action_id, name in ACTION_NAMES.items():
        count = action_counts[action_id]
        pct = 100 * count / total_actions if total_actions else 0
        print(f"  {name:<18} {count:>3} times  ({pct:.0f}%)")


if __name__ == "__main__":
    model_path = PROJECT_ROOT / "agent_rl_ppo_model.zip"
    evaluate(model_path, n_episodes=5)