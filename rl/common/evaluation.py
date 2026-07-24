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


def eval_returns(
    agent: Agent, env: gym.Env, episodes: int, seed_start: int = 0, max_steps: int = 10_000
) -> list[float]:
    """The protocol itself, returning per-episode returns. Analysis scripts
    use the raw list: identical episode seeds across runs make per-episode
    scores directly comparable (paired comparisons between variants).
    `seed_start` shifts the ladder (episode i seeds with
    EVAL_SEED_OFFSET + seed_start + i) so a re-eval can use episodes
    disjoint from the ones training-time eval selected checkpoints on.
    Caveat: MinAtar's sticky-action carry (`last_action`) survives reset,
    so ~1% of episodes are weakly order/policy dependent — negligible on
    means; pairing across runs is approximate, not exact.
    `max_steps` caps a single episode: MinAtar registers no time limit,
    and a strong Seaquest policy can survive indefinitely under greedy
    eval (oxygen is renewable), hanging the protocol — observed in the
    lr-5e-4 diagnostic runs. The cap is ~50x a normal MinAtar episode, so
    it binds only on effectively-immortal policies; the return
    accumulated at the cap is recorded."""
    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED_OFFSET + seed_start + episode)
        ep_return, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
            ep_return += float(reward)
            steps += 1
            done = terminated or truncated
        returns.append(ep_return)
    return returns


def evaluate(agent: Agent, env: gym.Env, episodes: int) -> dict[str, float]:
    returns = eval_returns(agent, env, episodes)
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
    }
