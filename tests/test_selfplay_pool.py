"""AgentOpponent + SnapshotPool probes (Phase 4 chunk 2).

The probes PLAN.md mandates for the frozen-opponent machinery: the snapshot's
action distribution must be bit-identical after the learner's weights move,
and no snapshot parameter may appear in the LEARNER's optimizer — asserting
against the snapshot's own optimizer passes vacuously, because deepcopy
carries one. Everything else here pins the locked pool mechanics: the 80/20
draw, second-oldest eviction, the pool_size-1 naive arm replacing rather
than retaining, freeze-at-push, and the own-generator replay contract.
"""

import numpy as np
import pytest
import torch

from rl.agents.ppo import PPOAgent
from rl.common.masking import masked_logits
from rl.envs.connect4 import Connect4Env
from rl.envs.make import make_env, make_vec_env
from rl.selfplay.pool import AgentOpponent, SnapshotPool

HPARAMS = dict(
    lr=2.5e-4, gamma=1.0, gae_lambda=0.95, rollout_steps=8, epochs=1,
    minibatches=2, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5,
    max_grad_norm=0.5, hidden_sizes=[16],
)

EMPTY_OBS = np.zeros((2, 6, 7), dtype=bool)
ALL_LEGAL = np.ones(7, dtype=bool)


def fresh_agent(seed=0):
    """A real conv-trunk PPO agent on the Connect 4 spaces — the snapshots
    must exercise the exact nets the pool will hold in training."""
    torch.manual_seed(seed)
    env = make_env("Connect4-v0", seed=0)
    agent = PPOAgent(
        env.observation_space, env.action_space, num_envs=1, device="cpu", **HPARAMS
    )
    env.close()
    return agent


def probs(actor, obs, mask):
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.bool)
    with torch.no_grad():
        return torch.softmax(masked_logits(actor(obs_t), mask_t), dim=-1)


# ------------------------------------------------------------ AgentOpponent

def test_snapshot_shares_no_storage_with_the_learner():
    """The deepcopy-at-construction contract. state_dict() aliases the live
    training tensors (probe-confirmed, PLAN.md), so storage identity is the
    thing to assert, not value equality."""
    agent = fresh_agent()
    pool = SnapshotPool(4, 0.8)
    pool.push(agent)
    member = pool.members[0]
    member_params = list(member.agent.actor.parameters()) + list(
        member.agent.critic.parameters()
    )
    assert member_params
    learner_ptrs = {p.data_ptr() for p in agent.params}
    assert all(p.data_ptr() not in learner_ptrs for p in member_params)
    # No snapshot parameter in the LEARNER's optimizer.
    optimizer_ids = {
        id(p) for group in agent.optimizer.param_groups for p in group["params"]
    }
    assert all(id(p) not in optimizer_ids for p in member_params)


def test_snapshot_distribution_is_bit_identical_after_the_learner_moves():
    agent = fresh_agent()
    pool = SnapshotPool(4, 0.8)
    pool.push(agent)
    member = pool.members[0]
    learner_before = probs(agent.actor, EMPTY_OBS, ALL_LEGAL)
    member_before = probs(member.agent.actor, EMPTY_OBS, ALL_LEGAL)
    with torch.no_grad():
        for p in agent.params:
            p.add_(1.0)
    # The control first: the perturbation really moved the learner — without
    # this, a no-op perturbation would pass the snapshot assertion vacuously.
    assert not torch.equal(learner_before, probs(agent.actor, EMPTY_OBS, ALL_LEGAL))
    assert torch.equal(member_before, probs(member.agent.actor, EMPTY_OBS, ALL_LEGAL))


def test_push_freezes_the_snapshot_and_leaves_the_learner_trainable():
    agent = fresh_agent()
    pool = SnapshotPool(4, 0.8)
    pool.push(agent)
    member = pool.members[0]
    assert not member.agent.actor.training and not member.agent.critic.training
    assert all(not p.requires_grad for p in member.agent.actor.parameters())
    assert all(not p.requires_grad for p in member.agent.critic.parameters())
    # freeze() must have acted on the copy, never the source.
    assert all(p.requires_grad for p in agent.params)


def test_agent_opponent_samples_rather_than_argmaxes():
    """The locked choice: stochastic play on both sides, because a
    deterministic opponent collapses an eval set to a couple of distinct
    games. A near-uniform init policy sampling 200 times spans many columns;
    argmax would emit exactly one."""
    opponent = AgentOpponent(fresh_agent(), seed=0)
    opponent.freeze()
    rng = np.random.default_rng(0)
    moves = {opponent.move(EMPTY_OBS, ALL_LEGAL, rng) for _ in range(200)}
    assert len(moves) >= 3


def test_agent_opponent_respects_the_mask():
    opponent = AgentOpponent(fresh_agent(), seed=0)
    opponent.freeze()
    mask = np.zeros(7, dtype=bool)
    mask[[2, 5]] = True
    rng = np.random.default_rng(0)
    assert {opponent.move(EMPTY_OBS, mask, rng) for _ in range(100)} <= {2, 5}


def test_agent_opponent_replays_from_its_own_generator():
    """Same generator seed => the same move sequence, even with the global
    torch stream perturbed between draws — the isolation the chunk-3
    tournament needs to replay a matchup."""
    agent = fresh_agent()

    def sequence(seed):
        opponent = AgentOpponent(agent, seed=seed)
        opponent.freeze()
        rng = np.random.default_rng(0)
        out = []
        for _ in range(50):
            torch.rand(11)  # global-stream noise must not reach the opponent
            out.append(opponent.move(EMPTY_OBS, ALL_LEGAL, rng))
        return out

    assert sequence(7) == sequence(7)
    assert sequence(7) != sequence(8)


# ------------------------------------------------------------- SnapshotPool

def test_select_honors_the_80_20_split():
    """latest_prob on the newest member, uniform over the rest — and every
    historical member reachable, which is what the per-member band checks."""
    pool = SnapshotPool(8, 0.8)
    for seed in range(5):
        pool.push(fresh_agent(seed))
    rng = np.random.default_rng(0)
    counts = np.zeros(5)
    draws = 5000
    for _ in range(draws):
        counts[pool.members.index(pool.select(rng))] += 1
    fractions = counts / draws
    assert 0.78 <= fractions[-1] <= 0.82
    assert all(0.035 <= f <= 0.065 for f in fractions[:-1])


def test_pool_size_one_is_the_naive_arm_and_replaces_on_push():
    """pool_size 1 must hold the LATEST snapshot: the naive arm is a lagged
    copy of the learner, and the general evict-second-oldest rule would pin
    it to its random init forever."""
    pool = SnapshotPool(1, 0.8)
    first, second = fresh_agent(0), fresh_agent(1)
    pool.push(first)
    original = pool.members[0]
    rng = np.random.default_rng(0)
    assert pool.select(rng) is original
    pool.push(second)
    assert len(pool) == 1
    assert pool.members[0] is not original
    head = second.actor.head.weight
    assert torch.equal(pool.members[0].agent.actor.head.weight, head)


def test_eviction_keeps_the_oldest_and_drops_the_second_oldest():
    """Strided retention: the step-0 snapshot anchors the pool's span for
    the whole run; a plain recency deque is the published-worst design
    (Bansal et al., PLAN.md)."""
    pool = SnapshotPool(3, 0.8)
    pushed = []
    for seed in range(5):
        pool.push(fresh_agent(seed))
        pushed.append(pool.members[-1])
    # [m0,m1,m2] -> push m3 evicts m1 -> push m4 evicts m2.
    assert pool.members == [pushed[0], pushed[3], pushed[4]]
    assert len(pool) == 3 and pool.pushes == 5


def test_select_on_an_empty_pool_raises():
    with pytest.raises(IndexError, match="empty pool"):
        SnapshotPool(4, 0.8).select(np.random.default_rng(0))


def test_the_pool_itself_never_plays():
    pool = SnapshotPool(4, 0.8)
    pool.push(fresh_agent())
    with pytest.raises(TypeError, match="never plays"):
        pool.move(EMPTY_OBS, ALL_LEGAL, np.random.default_rng(0))


def test_pool_rejects_degenerate_parameters():
    with pytest.raises(ValueError, match="pool_size"):
        SnapshotPool(0, 0.8)
    with pytest.raises(ValueError, match="latest_prob"):
        SnapshotPool(4, 1.5)


# ------------------------------------------------------- through the env

def test_one_pool_is_shared_by_every_sub_env_and_pushes_are_visible():
    """The env_kwargs seam end-to-end with the real pool object: caller
    kwargs preserve identity across sub-envs, so a push lands in all of
    them at once."""
    pool = SnapshotPool(4, 0.8)
    pool.push(fresh_agent())
    vec = make_vec_env("Connect4-v0", 0, 3, env_kwargs={"opponent": pool})
    assert all(env.unwrapped.opponent_source is pool for env in vec.envs)
    vec.reset(seed=0)
    pool.push(fresh_agent(1))
    assert all(len(env.unwrapped.opponent_source) == 2 for env in vec.envs)
    vec.close()


def test_pool_members_play_full_games_through_the_env():
    """The env raises on any illegal opponent column, so completing episodes
    is most of the assertion; the rest is that terminals carry an outcome."""
    pool = SnapshotPool(4, 0.8)
    pool.push(fresh_agent(0))
    pool.push(fresh_agent(1))
    env = Connect4Env(opponent=pool)
    rng = np.random.default_rng(0)
    for episode in range(10):
        obs, info = env.reset(seed=episode)
        done = False
        while not done:
            legal = np.flatnonzero(info["action_mask"])
            obs, _, done, _, info = env.step(int(rng.choice(legal)))
        assert info["outcome"] in (-1, 0, 1)


def test_installing_an_empty_pool_is_fine_but_resetting_is_not():
    """Construction must work with an empty pool (train() builds the env
    before the agent exists), and the first reset without a push must fail
    loudly — this is the IndexError the pre-loop push exists to prevent."""
    env = Connect4Env(opponent=SnapshotPool(4, 0.8))  # install-time freeze() no-op
    with pytest.raises(IndexError, match="empty pool"):
        env.reset(seed=0)
