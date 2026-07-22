"""Single entry point: `python -m rl.train --config configs/<run>.yaml`.
Every algorithm plugs in here; the loop stays algorithm-agnostic.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import gymnasium as gym

from rl.agents.base import Agent
from rl.agents.dqn import DQNAgent
from rl.agents.q_learning import QLearningAgent
from rl.agents.random_agent import RandomAgent
from rl.common.checkpoint import save_checkpoint
from rl.common.config import Config, load_config
from rl.common.evaluation import evaluate
from rl.common.logging import make_logger
from rl.common.seeding import set_seed
from rl.envs.make import make_env


def make_agent(cfg: Config, env: gym.Env) -> Agent:
    algo = cfg.agent.get("algo")
    if algo == "random":
        return RandomAgent(env.action_space)
    if algo == "q_learning":
        hparams = {k: v for k, v in cfg.agent.items() if k != "algo"}
        return QLearningAgent(env.observation_space, env.action_space, **hparams)
    if algo == "dqn":
        hparams = {k: v for k, v in cfg.agent.items() if k != "algo"}
        return DQNAgent(env.observation_space, env.action_space, device=cfg.device, **hparams)
    raise ValueError(f"unknown algo {algo!r}")


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    env = make_env(cfg.env_id, cfg.seed)
    eval_env = make_env(cfg.env_id, cfg.seed)  # eval reseeds per episode
    agent = make_agent(cfg, env)
    logger = make_logger(cfg)

    obs, _ = env.reset(seed=cfg.seed)
    ep_return, ep_length = 0.0, 0
    ep_losses: dict[str, float] = defaultdict(float)  # per-episode loss/* sums
    last_step, last_time = 0, time.perf_counter()

    for step in range(1, cfg.total_steps + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Per-step update on the fresh transition (tabular Q; DQN keeps this
        # cadence but samples from replay instead). `terminated` and not
        # `truncated` is passed on purpose: a time-limit cut still bootstraps.
        for name, value in agent.update((obs, action, float(reward), next_obs, terminated)).items():
            ep_losses[name] += value
        obs = next_obs
        ep_return += float(reward)
        ep_length += 1

        if terminated or truncated:
            now = time.perf_counter()
            logger.log(
                {
                    "rollout/episode_return": ep_return,
                    "rollout/episode_length": ep_length,
                    "time/steps_per_sec": (step - last_step) / (now - last_time),
                    **{name: total / ep_length for name, total in ep_losses.items()},
                },
                step,
            )
            last_step, last_time = step, now
            obs, _ = env.reset()
            ep_return, ep_length = 0.0, 0
            ep_losses.clear()

        if step % cfg.eval_every == 0:
            logger.log(evaluate(agent, eval_env, cfg.eval_episodes), step)

    save_checkpoint(Path("runs") / cfg.run_name / "checkpoint.pt", agent, cfg.total_steps, cfg)
    logger.close()
    env.close()
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a run YAML")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
