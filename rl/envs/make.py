"""Env factory: the single place envs are constructed and seeded, kept as a
seam so vectorized envs (Phase 2, PPO) slot in without touching the train loop.

Gymnasium seeding has two independent RNGs per env:
- the env's own RNG, seeded by the caller's first `reset(seed=...)` and
  persisting across later plain `reset()` calls;
- the action/observation spaces' RNG, seeded here, which
  `action_space.sample()` (the random policy) draws from.
"""

import gymnasium as gym


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env
