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
    run_dir = Path("runs") / cfg.run_name
    best_eval = float("-inf")
    ep_return, ep_length = 0.0, 0
    ep_losses: dict[str, float] = defaultdict(float)  # per-episode loss/* sums
    last_step, last_time = 0, time.perf_counter()

    for step in range(1, cfg.total_steps + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Per-step update on the fresh transition (tabular Q; DQN keeps this
        # cadence but samples from replay instead). Both flags are passed:
        # only `terminated` stops bootstrapping (a time-limit cut still
        # bootstraps), but `truncated` still marks an episode boundary,
        # which n-step accumulation must not chain across.
        for name, value in agent.update(
            (obs, action, float(reward), next_obs, terminated, truncated)
        ).items():
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
            metrics = evaluate(agent, eval_env, cfg.eval_episodes)
            logger.log(metrics, step)
            # The final policy is an arbitrary sample of an oscillating
            # training trajectory (deep RL policies churn), so keep the
            # best-so-far policy too. Report final and best.
            if metrics["eval/return_mean"] > best_eval:
                best_eval = metrics["eval/return_mean"]
                save_checkpoint(run_dir / "best_checkpoint.pt", agent, step, cfg)

    save_checkpoint(run_dir / "checkpoint.pt", agent, cfg.total_steps, cfg)
    logger.close()
    env.close()
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a run YAML")
    # Overrides for the multi-seed benchmark protocol: same YAML, N seeds,
    # each under its own run name.
    parser.add_argument("--seed", type=int, default=None, help="override the config seed")
    parser.add_argument("--run-name", default=None, help="override the config run_name")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_name is not None:
        cfg.run_name = args.run_name
    train(cfg)


if __name__ == "__main__":
    main()
