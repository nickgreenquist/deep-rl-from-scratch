"""Observation/reward normalization: statistics maths, the wrapper contracts
that keep training and eval consistent, and the two silent-corruption paths
the chunk-5 review probed — partial resets normalizing only the rows they
update, and reward statistics that drop each episode's terminal return.

These are checked against hand-computed or numpy-computed expectations rather
than against a second implementation of the same recursion.
"""

import gymnasium as gym
import numpy as np
import pytest

from rl.envs.make import make_vec_env
from rl.envs.normalize import (
    CLIP_OBS,
    FrozenNormalizeObservation,
    NormalizeObservation,
    NormalizeReward,
    RunningMeanStd,
    frozen_obs_env,
    normalize_obs,
)


def test_running_mean_std_matches_numpy_over_the_concatenation():
    """Chan's parallel merge must agree with computing over all data at once,
    across several unequal batches including a single-row one."""
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, size=(n, 4)) for n in (1, 5, 50, 2)]
    rms = RunningMeanStd((4,))
    for batch in batches:
        rms.update(batch)
    every = np.concatenate(batches)
    # The count starts at epsilon=1e-4 carrying a var=1 prior, so agreement is
    # close but not exact; tolerance is far tighter than that prior's weight.
    np.testing.assert_allclose(rms.mean, every.mean(axis=0), rtol=1e-4)
    np.testing.assert_allclose(rms.var, every.var(axis=0), rtol=1e-3)
    assert rms.count == pytest.approx(len(every), abs=1e-3)


def test_running_mean_std_empty_batch_is_a_noop():
    # A partial reset can legitimately reset zero rows.
    rms = RunningMeanStd((2,))
    rms.update(np.zeros((3, 2)))
    before = rms.state_dict()
    rms.update(np.zeros((0, 2)))
    np.testing.assert_array_equal(rms.mean, before["mean"])
    assert rms.count == before["count"]


def test_running_mean_std_state_round_trips_bitwise():
    rms = RunningMeanStd((3,))
    rms.update(np.random.default_rng(1).normal(size=(20, 3)))
    restored = RunningMeanStd((3,))
    restored.load_state_dict(rms.state_dict())
    obs = np.arange(3, dtype=np.float64)
    np.testing.assert_array_equal(normalize_obs(obs, restored), normalize_obs(obs, rms))


def test_normalize_obs_clips_and_emits_float32():
    rms = RunningMeanStd((2,))
    rms.update(np.zeros((100, 2)) + np.array([1.0, 1.0]))
    rms.update(np.zeros((100, 2)) - np.array([1.0, 1.0]))  # mean 0, var 1
    out = normalize_obs(np.array([100.0, -100.0]), rms)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [CLIP_OBS, -CLIP_OBS])


def test_obs_wrapper_updates_on_step_and_advertises_float32():
    env = NormalizeObservation(make_vec_env("Pendulum-v1", 0, 2))
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert env.single_observation_space.dtype == np.float32
    assert env.single_observation_space.shape == (3,)
    count_after_reset = env.rms.count
    env.step(np.zeros((2, 1), dtype=np.float32))
    assert env.rms.count == pytest.approx(count_after_reset + 2)  # one row per env
    env.close()


def test_obs_wrapper_partial_reset_normalizes_all_rows_but_updates_only_reset_rows():
    """The review's silent-corruption path. The train loop replaces its whole
    observation array with what reset() returns, and non-reset rows come back
    holding their current RAW observation — so every returned row must be
    normalized even though only the reset rows are new statistics."""
    env = NormalizeObservation(make_vec_env("Pendulum-v1", 0, 3))
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.zeros((3, 1), dtype=np.float32))
    count_before = env.rms.count
    reset_mask = np.array([True, False, False])

    obs, _ = env.reset(options={"reset_mask": reset_mask})

    assert env.rms.count == pytest.approx(count_before + 1)  # only the one reset row
    # Every row is normalized: with stats this far from the raw scale, a raw
    # row would be detectably different from its normalized value.
    raw_rows = np.array([e.unwrapped._get_obs() for e in env.env.unwrapped.envs])
    np.testing.assert_allclose(obs, normalize_obs(raw_rows, env.rms), rtol=1e-6)
    assert not np.allclose(obs[1], raw_rows[1])  # row 1 really was transformed
    env.close()


def test_frozen_wrapper_never_updates_and_matches_the_training_transform():
    """Eval must apply exactly the training transform, including the clip, and
    must not fold eval episodes into the statistics the training env uses."""
    rms = RunningMeanStd((3,))
    rms.update(np.random.default_rng(2).normal(5.0, 3.0, size=(50, 3)))
    frozen = FrozenNormalizeObservation(gym.make("Pendulum-v1"), rms)
    before = rms.state_dict()

    obs, _ = frozen.reset(seed=0)
    for _ in range(3):
        obs, *_ = frozen.step(np.zeros(1, dtype=np.float32))

    assert obs.dtype == np.float32
    assert rms.count == before["count"]  # frozen: no updates
    np.testing.assert_array_equal(rms.mean, before["mean"])
    raw = frozen.env.unwrapped._get_obs()
    np.testing.assert_allclose(obs, normalize_obs(raw, rms), rtol=1e-6)
    frozen.close()


def test_reward_wrapper_scales_by_return_std_and_exposes_raw_reward():
    """Hand-checked against the documented recursion on a constant-reward
    stream: G_t = gamma*G_{t-1} + r, scaled = r / sqrt(var(G_seen) + eps)."""
    env = NormalizeReward(make_vec_env("Pendulum-v1", 0, 2), gamma=0.9)
    env.reset(seed=0)
    rms = RunningMeanStd(())
    returns = np.zeros(2)
    for _ in range(4):
        _, scaled, _, _, infos = env.step(np.zeros((2, 1), dtype=np.float32))
        raw = infos["raw_reward"]  # true env units, republished for logging
        returns = returns * 0.9 + raw
        rms.update(returns)
        np.testing.assert_allclose(scaled, raw / np.sqrt(rms.var + 1e-8), rtol=1e-9)
    env.close()


def test_reward_wrapper_counts_the_terminal_return_then_resets_on_any_done():
    """The chunk-5 amendment: accumulate, update statistics, THEN zero. The
    draft's `G <- gamma*(1-done)*G + r` would drop each episode's largest
    return sample — the terminal one — from the variance estimate entirely.
    Pendulum truncates at 200 steps and never terminates, so this also pins
    the any-done (not terminated-only) reset."""
    env = NormalizeReward(make_vec_env("Pendulum-v1", 0, 1), gamma=1.0)
    env.reset(seed=0)
    action = np.zeros((1, 1), dtype=np.float32)
    for _ in range(199):
        _, _, _, truncated, _ = env.step(action)
        assert not truncated[0]
    accumulated = env.returns[0]  # 199 steps of accumulation, pre-terminal

    _, _, _, truncated, infos = env.step(action)  # step 200: truncation

    assert truncated[0]
    # The terminal step's return DID enter the statistics...
    terminal_return = accumulated + infos["raw_reward"][0]
    assert env.rms.mean != pytest.approx(0.0)
    assert abs(terminal_return) > abs(accumulated)  # it is the largest sample
    # ...and only afterwards was the accumulator cleared.
    assert env.returns[0] == 0.0
    env.close()


def test_frozen_obs_env_raises_when_stats_are_missing():
    """A normalize-flagged checkpoint with no statistics must fail loudly:
    evaluating unnormalized produces a plausible-looking wrong number."""
    from rl.common.config import Config

    cfg = Config(
        env_id="Pendulum-v1", seed=0, total_steps=1, eval_every=1, eval_episodes=1,
        run_name="x", normalize_obs=True,
    )
    env = gym.make("Pendulum-v1")
    with pytest.raises(ValueError, match="no observation statistics"):
        frozen_obs_env(env, cfg, {"agent": {}})

    # Flag off: the wrapper is simply not applied, whatever the checkpoint holds.
    cfg.normalize_obs = False
    assert frozen_obs_env(env, cfg, {}) is env
    env.close()
