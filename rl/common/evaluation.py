"""Eval protocol: fixed seeds, N episodes, deterministic policy, mean ± std.

Runs on a dedicated env passed in by the caller, so eval never disturbs the
training env's RNG or episode state. Episode i always resets with the same
constant seed — independent of the training seed — so scores are comparable
across passes, across training seeds (the multi-seed benchmark protocol),
and across algorithms.
"""

import gymnasium as gym
import numpy as np

from rl.agents.base import Agent

# Fixed base for eval episode seeds; large so they stay disjoint from
# plausible training seeds.
EVAL_SEED_OFFSET = 10_000


def evaluate(agent: Agent, env: gym.Env, episodes: int) -> dict[str, float]:
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED_OFFSET + episode)
        ep_return, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
    }
