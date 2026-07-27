"""Snapshot pool: the historical-opponent side of self-play (Phase 4 chunk 2).

Two objects. `AgentOpponent` wraps a frozen copy of a training agent behind
the `Opponent` protocol; `SnapshotPool` holds a population of them and IS an
`Opponent` itself, which is the whole wiring trick — `rl/train.py` passes the
pool through the same `env_kwargs` seam a name like "heuristic" travels, and
one pool object is shared by every sub-env because caller kwargs are never
deep-copied (see `rl/envs/make.py`).

Where the copy happens is the load-bearing decision: `agent.state_dict()`
ALIASES the live training tensors (probe-confirmed, PLAN.md), so a pool of
state_dicts would hold references into the learner and every "frozen"
opponent would silently track it — the Phase-3 log_alpha rebind failure
class. `AgentOpponent.__init__` therefore deep-copies the whole agent at
construction, where it cannot be forgotten by a call site. The deepcopy
carries the optimizer too (~2/3 of the ~1 MB per snapshot is dead optimizer
state) — accepted waste, per the locked spec.

Two distinct swap boundaries, enforced by WHERE things are called rather
than by state in here:

- which member plays is drawn per EPISODE — `select()` runs at env reset;
- new snapshots enter the pool only at a ROLLOUT boundary — `push()` is
  called from `rl/train.py` right after the `update()` that drained the
  buffer, so within a rollout the env is a *stochastic* env with a fixed
  opponent distribution, not a non-stationary one. That is what PPO's
  importance ratios actually require.

Retention on overflow evicts the SECOND-oldest, not the oldest: a plain
recency deque was the only pool-span design with a published ablation
against it (Bansal et al., ICLR 2018 — "training against the latest
opponent leads to worst performance"), so the pool's span grows with
training instead of trailing it, anchored by the step-0 snapshot that never
leaves. The exception is `pool_size: 1` — the naive arm, a <=push-cadence
-lagged copy of the learner, which is what the self-play literature means
by naive self-play — where push must REPLACE, because keeping the oldest
would pin the naive arm to its random init forever.
"""

import copy

import numpy as np
import torch

from rl.common.masking import masked_logits
from rl.selfplay.opponents import Opponent


class AgentOpponent(Opponent):
    """A frozen agent snapshot playing as an env opponent.

    Samples from the policy rather than argmaxing: a deterministic opponent
    collapses an eval or tournament set to a couple of distinct games (the
    `eval/return_std > 0` failure), and stochastic play on both sides is the
    locked tournament protocol. The draw comes from this opponent's OWN
    `torch.Generator`, never the global torch stream — the learner's
    collection also samples from the global stream, so sharing it would let
    the number of learner steps between opponent moves change every seeded
    opponent decision, and a chunk-3 tournament matchup could never be
    replayed in isolation.

    `move()` runs its own forward instead of calling `agent.act()`: act()
    samples via `Categorical.sample()`, which only draws from the global
    stream.
    """

    def __init__(self, agent, seed: int = 0):
        self.agent = copy.deepcopy(agent)
        self.generator = torch.Generator().manual_seed(seed)

    def freeze(self) -> None:
        for net in (self.agent.actor, self.agent.critic):
            net.eval()
            for param in net.parameters():
                param.requires_grad_(False)

    def move(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.agent.device)
        with torch.no_grad():
            probs = torch.softmax(masked_logits(self.agent.actor(obs_t), mask_t), dim=-1)
        return int(torch.multinomial(probs, 1, generator=self.generator).item())


class SnapshotPool(Opponent):
    """A population of `AgentOpponent` snapshots, itself an `Opponent`.

    `select()` is the 80/20 draw (OpenAI Five's split): `latest_prob` on the
    newest member, else uniform over the rest — the uniform historical draw
    is ours, not theirs (they weighted by quality scores over an unbounded
    pool); AlphaStar's PFSP weighting is the named probe lever if the flat
    draw wastes games on long-beaten snapshots. It draws from the RNG the
    env hands in, which is the env's own per-episode stream — the pinned
    draw order (flip, select, opponent moves) is unchanged, the select slot
    just consumes draws now.

    The pool must be non-empty before the first env reset: `rl/train.py`
    pushes the step-0 snapshot before entering the loop, and `select()` on
    an empty pool raises rather than papering over a broken push order.
    """

    def __init__(self, pool_size: int, latest_prob: float):
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        if not 0.0 <= latest_prob <= 1.0:
            raise ValueError(f"latest_prob must be a probability, got {latest_prob}")
        self.pool_size = pool_size
        self.latest_prob = latest_prob
        self.members: list[AgentOpponent] = []
        self.pushes = 0  # lifetime count; also seeds each member's generator

    def __len__(self) -> int:
        return len(self.members)

    def push(self, agent) -> None:
        """Snapshot `agent` into the pool. The install point, so freezing
        happens here — a snapshot cannot enter the pool trainable."""
        member = AgentOpponent(agent, seed=self.pushes)
        member.freeze()
        self.members.append(member)
        self.pushes += 1
        if len(self.members) > self.pool_size:
            # Strided retention: evict the SECOND-oldest, so the step-0
            # snapshot anchors the pool's span for the whole run. At
            # pool_size 1 (the naive arm) that rule would keep the random
            # init forever and evict every later snapshot, so the sole
            # member is replaced instead.
            del self.members[1 if self.pool_size > 1 else 0]

    def select(self, rng: np.random.Generator) -> Opponent:
        if not self.members:
            raise IndexError("empty pool: push a snapshot before the first reset")
        if len(self.members) == 1:
            return self.members[0]
        if rng.random() < self.latest_prob:
            return self.members[-1]
        return self.members[int(rng.integers(len(self.members) - 1))]

    def move(self, obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        raise TypeError("the pool never plays; select() returns the member that does")

    def freeze(self) -> None:
        """No-op on purpose: members are frozen one by one at push(), which
        is their install point. The env still calls this when the pool is
        installed — the contract is that every installer calls freeze()."""
