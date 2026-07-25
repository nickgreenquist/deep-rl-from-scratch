"""The action-masking contract (rl/common/masking.py + the harness seam).

The first test is the regression guard: an all-True mask must leave logits
bitwise untouched — that property is what makes the masking retrofit a
provable no-op on every env without illegal actions (all of Phases 0-3).
"""

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from rl.common.masking import masked_entropy, masked_logits, masked_sample
from tests.envs.masked_dummy import EP_LEN, N_ACTIONS, MaskedDummyEnv


def test_all_true_mask_is_bitwise_identity():
    torch.manual_seed(0)
    logits = torch.randn(7, 5) * 100
    out = masked_logits(logits, torch.ones(7, 5, dtype=torch.bool))
    assert torch.equal(out, logits)  # bitwise, not allclose


def test_illegal_probabilities_are_exactly_zero():
    logits = torch.zeros(5)
    mask = torch.tensor([True, False, True, False, False])
    probs = Categorical(logits=masked_logits(logits, mask)).probs
    assert (probs[~mask] == 0.0).all()
    assert probs[mask].sum().item() == pytest.approx(1.0)


def test_sampled_and_argmax_actions_are_legal():
    torch.manual_seed(0)
    logits = torch.randn(5)
    mask = torch.tensor([False, True, False, True, False])
    samples = Categorical(logits=masked_logits(logits, mask)).sample((1000,))
    assert mask[samples].all()
    assert mask[masked_logits(logits, mask).argmax()]


def test_masked_entropy_finite_and_bounded():
    # k legal actions bound entropy by log(k). The -inf sentinel would make
    # this NaN (0 * -inf) — the exact silent bug the finite NEG exists for.
    logits = torch.zeros(2, 5)  # uniform over legal -> entropy == log(k)
    mask = torch.tensor(
        [[True, True, False, False, False], [True, True, True, True, False]]
    )
    ent = masked_entropy(logits, mask)
    assert torch.isfinite(ent).all()
    assert ent[0].item() == pytest.approx(math.log(2))
    assert ent[1].item() == pytest.approx(math.log(4))
    sharp = masked_entropy(torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]]), mask[:1])
    assert 0 < sharp.item() < math.log(2)


def test_masked_entropy_matches_categorical_when_all_legal():
    torch.manual_seed(0)
    logits = torch.randn(4, 6)
    ref = Categorical(logits=logits).entropy()
    ours = masked_entropy(logits, torch.ones(4, 6, dtype=torch.bool))
    assert torch.allclose(ours, ref)


def test_all_illegal_mask_asserts():
    with pytest.raises(AssertionError):
        masked_logits(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(AssertionError):
        masked_sample(gym.spaces.Discrete(3), np.zeros(3, dtype=bool))


def test_masked_sample_legal_and_stream_identical_when_all_true():
    space = gym.spaces.Discrete(N_ACTIONS)
    space.seed(0)
    mask = np.array([False, True, False, False, True])
    assert all(mask[masked_sample(space, mask)] for _ in range(200))
    # All-True: rejection accepts the first draw, so the RNG stream matches a
    # bare sample() — fixed-seed spine runs reproduce bit-for-bit.
    space.seed(123)
    bare = [space.sample() for _ in range(50)]
    space.seed(123)
    masked = [masked_sample(space, np.ones(N_ACTIONS, dtype=bool)) for _ in range(50)]
    assert bare == masked


def test_dummy_env_contract():
    env = MaskedDummyEnv()
    obs, info = env.reset(seed=0)
    mask = info["action_mask"]
    assert mask.shape == (N_ACTIONS,) and mask.dtype == np.bool_ and mask.any()
    steps, done = 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(int(np.flatnonzero(mask)[0]))
        assert reward == 1.0
        mask = info["action_mask"]
        done = terminated or truncated
        steps += 1
    assert steps == EP_LEN
    obs, info = env.reset(seed=1)
    illegal = np.flatnonzero(~info["action_mask"])
    assert illegal.size > 0  # seed-pinned: this draw has an illegal action
    with pytest.raises(ValueError):
        env.step(int(illegal[0]))
