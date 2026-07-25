"""Env factory: the single place envs are constructed and seeded, kept as a
seam so vectorized envs (Phase 2, PPO) slot in without touching the train loop.

Gymnasium seeding has two independent RNGs per env:
- the env's own RNG, seeded by the caller's first `reset(seed=...)` and
  persisting across later plain `reset()` calls;
- the action/observation spaces' RNG, seeded here, which
  `action_space.sample()` (the random policy) draws from.
"""

from functools import partial

import gymnasium as gym

from rl.envs.wrappers import ChannelFirst


def make_env(env_id: str, seed: int, render_mode: str | None = None) -> gym.Env:
    if env_id.startswith("MinAtar/"):
        _ensure_minatar_registered()
    env = gym.make(env_id, render_mode=render_mode)
    if env_id.startswith("MinAtar/"):
        env = ChannelFirst(env)  # (10, 10, C) planes -> torch's (C, 10, 10)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def make_vec_env(env_id: str, seed: int, num_envs: int) -> gym.vector.SyncVectorEnv:
    """N lockstep env copies for on-policy collection (PPO). Sync, not async:
    per-step nets are tiny, so process/IPC overhead would dominate — the same
    lesson as torch_threads=1.

    Autoreset is DISABLED on purpose. The default NEXT_STEP mode inserts a
    dummy transition after every terminal step (action ignored, reward 0,
    obs = reset obs) that a rollout buffer would have to mask; instead the
    train loop resets finished sub-envs explicitly via
    `reset(options={"reset_mask": ...})`, so every transition the agent sees
    is a real env step, and `next_obs` at a terminal step is the episode's
    true final observation (what advantage bootstrapping needs at
    truncations). Sub-env i's spaces are seeded seed + i; episode RNG comes
    from the loop's first `reset(seed=...)`, which gymnasium also fans out
    as seed + i."""
    return gym.vector.SyncVectorEnv(
        [partial(make_env, env_id, seed + i) for i in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )


def _ensure_minatar_registered() -> None:
    # MinAtar ships its env ids only via the `gymnasium.envs` entry point — a
    # plugin mechanism gymnasium 1.0 removed — so registration is explicit.
    if "MinAtar/Breakout-v0" not in gym.registry:
        from minatar.gym import register_envs  # deferred: pulls in seaborn/matplotlib

        register_envs()
