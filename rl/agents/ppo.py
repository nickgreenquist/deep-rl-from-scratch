"""PPO (Schulman et al. 2017): the clipped-surrogate policy gradient — the
same score-function core as REINFORCE, rebuilt so one batch of experience
safely buys several gradient steps.

REINFORCE's constraint was structural: the policy gradient is an expectation
under the *current* policy, so each batch funded exactly one update and was
discarded. Two ideas relax that:

- Importance sampling makes reuse legal. Weight each action's term by
  ratio = pi(a|s) / pi_old(a|s) and the surrogate objective ratio * A is,
  in expectation under the *old* (data-collecting) policy, a first-order
  proxy for the new policy's performance — its gradient at ratio = 1 is
  exactly the vanilla policy gradient. Optimizing it on stale data is
  principled while the two policies stay close.
- Clipping makes reuse safe. That "while close" caveat degrades fast: the
  importance weights' variance grows with the policy gap, and A itself was
  estimated under pi_old. So the per-transition objective is

      min(ratio * A,  clip(ratio, 1 - eps, 1 + eps) * A)

  The min is deliberately asymmetric — a trust region built from incentive
  removal rather than a hard constraint. Once the ratio crosses the clip
  range in the direction that *improves* the objective, the clipped term
  wins the min and is constant in the parameters: zero gradient, nothing
  pulls the policy further from pi_old on that transition. A ratio that
  drifted the way that *hurts* keeps its unclipped gradient, so overshoot
  is still corrected. Epochs of minibatch reuse then extract several steps
  per batch, each confined to the neighborhood where the surrogate can be
  trusted.

The other components, each replacing a REINFORCE compromise:

- Critic + GAE (buffers/rollout.py): a learned V(s) baseline replaces
  within-episode return normalization, and lam blends TD and Monte Carlo
  advantage estimates. The critic also bootstraps truncated tails — the
  time-limit bias REINFORCE documented and lived with disappears.
- Entropy bonus: REINFORCE watched entropy as a health metric; PPO pays for
  it, subtracting entropy_coef * H(pi(·|s)) from the loss so the policy
  doesn't go deterministic before learning finishes.
- Vectorized collection: N lockstep envs decorrelate each rollout — the
  on-policy substitute for what replay did in DQN.

Deliberately omitted (locked 2026-07-25 after review, see PLAN.md):

- Value-loss clipping: Engstrom et al. 2020 found no evidence it helps and
  Andrychowicz et al. 2021 found it can hurt; with separate actor/critic
  nets its one theoretical benefit — damping value updates that would drag
  a shared trunk out from under the policy — cannot apply.
- Learning-rate annealing: real, but deferred to the MinAtar benchmark
  configs, where Andrychowicz et al. report it pays; CartPole at a
  constant 2.5e-4 doesn't need it.

Note on value_coef: with disjoint actor/critic parameters and Adam, scaling
the value loss barely changes the critic's own updates (Adam renormalizes
per-parameter step sizes); its remaining effect is modulating the critic's
share of the single shared gradient-norm clip. A persistently underfitting
critic wants its own optimizer, not a bigger coefficient.
"""

import math
from collections import defaultdict
from typing import Any

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from rl.agents.base import Agent
from rl.buffers.rollout import RolloutBuffer, compute_gae
from rl.networks.mlp import mlp


def clipped_surrogate_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The per-minibatch policy loss plus its two health diagnostics.

    Returns (policy_loss, approx_kl, clip_frac):
    - policy_loss = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
      with ratio = exp(new_logp - old_logp);
    - approx_kl: the low-variance KL(pi_old || pi) estimator
      mean((ratio - 1) - log ratio);
    - clip_frac: fraction of transitions with |ratio - 1| > eps.

    approx_kl and clip_frac are the collapse-vs-bad-hyperparameter
    discriminators: healthy runs sit around approx_kl <= ~1e-2 and
    clip_frac ~0.1-0.3 (near 0 means the lr never reaches the clip, high
    means it blasts through it). Diagnostics are detached; only the loss
    carries gradient.
    """
    logratio = new_logp - old_logp
    ratio = logratio.exp()
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - logratio).mean()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
    return policy_loss, approx_kl, clip_frac


def _orthogonal_init(net: nn.Sequential, head_gain: float) -> None:
    """Orthogonal init (a 37-details item): gain sqrt(2) on hidden layers, a
    task-specific gain on the head, zero biases. The 0.01 policy-head gain
    makes the initial policy near-uniform — early exploration comes from
    sampling a flat distribution, not from init noise."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    for layer in linears:
        gain = head_gain if layer is linears[-1] else math.sqrt(2.0)
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)


class PPOAgent(Agent):
    vectorized = True

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        device: str,
        lr: float,
        gamma: float,
        gae_lambda: float,
        rollout_steps: int,
        epochs: int,
        minibatches: int,
        clip_eps: float,
        entropy_coef: float,
        value_coef: float,
        max_grad_norm: float,
        hidden_sizes: list[int],
    ):
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("PPOAgent requires a flat Box observation space")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("PPOAgent requires a Discrete action space")
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.minibatches = minibatches
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        obs_dim = observation_space.shape[0]
        # Separate actor and critic, no shared trunk: the value_coef note in
        # the module docstring is premised on it, and it removes the one
        # scenario where value-loss clipping could have mattered. Tanh
        # hiddens: the feedforward-PPO reference default the numeric
        # hyperparameters were validated under.
        self.actor = mlp(obs_dim, hidden_sizes, int(action_space.n), activation=nn.Tanh)
        self.critic = mlp(obs_dim, hidden_sizes, 1, activation=nn.Tanh)
        _orthogonal_init(self.actor, head_gain=0.01)
        _orthogonal_init(self.critic, head_gain=1.0)
        self.actor.to(self.device)
        self.critic.to(self.device)
        # One Adam over the union of both nets' params; eps=1e-5 is the
        # canonical PPO detail (shipped by every reference implementation).
        self.params = [*self.actor.parameters(), *self.critic.parameters()]
        self.optimizer = torch.optim.Adam(self.params, lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(
            rollout_steps, num_envs, observation_space.shape, observation_space.dtype
        )
        self.updates = 0  # completed fill -> epochs cycles

    def act(self, obs: Any, action_mask: Any = None, deterministic: bool = False) -> Any:
        # float32 at tensor time (MinAtar obs are bool planes); branch on obs
        # rank, no unconditional unsqueeze: collection hands (N, obs_dim),
        # eval/watch hand a single (obs_dim,).
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.actor(obs_t)
        if deterministic:
            actions = logits.argmax(dim=-1)  # eval-time policy: the mode
        else:
            actions = Categorical(logits=logits).sample()
        if obs_t.ndim == 1:
            return int(actions.item())
        return actions.cpu().numpy()

    def update(self, batch: Any) -> dict[str, float]:
        # The vector loop hands one batched (N-wide) transition row per env
        # step; accumulate until the horizon fills, then train on the rollout.
        self.buffer.add(*batch[:6])
        if not self.buffer.full():
            return {}
        buf = self.buffer

        obs_t = torch.as_tensor(buf.obs, dtype=torch.float32, device=self.device)  # (T, N, obs)
        next_obs_t = torch.as_tensor(buf.next_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(buf.actions, device=self.device)  # (T, N)
        with torch.no_grad():
            # Recomputed at update start, not stored during collection: exact
            # (the policy hasn't changed since it acted — same argument as
            # REINFORCE's batched recompute), and next_obs gets its own pass
            # because every buffer row carries its own successor.
            values = self.critic(obs_t).squeeze(-1)  # (T, N)
            next_values = self.critic(next_obs_t).squeeze(-1)
            old_logp = Categorical(logits=self.actor(obs_t)).log_prob(actions_t)
        advantages_t = torch.as_tensor(
            compute_gae(
                buf.rewards,
                buf.terminated,
                buf.truncated,
                values.cpu().numpy(),
                next_values.cpu().numpy(),
                self.gamma,
                self.gae_lambda,
            ),
            device=self.device,
        )
        # The critic's regression targets: GAE-consistent returns.
        value_targets = advantages_t + values

        # Flatten (T, N) -> (T*N,) and reshuffle at the transition level each
        # epoch, so minibatches mix envs and timesteps.
        flat_obs = obs_t.flatten(0, 1)
        flat_actions = actions_t.reshape(-1)
        flat_old_logp = old_logp.reshape(-1)
        flat_advantages = advantages_t.reshape(-1)
        flat_targets = value_targets.reshape(-1)

        batch_size = flat_actions.shape[0]
        minibatch_size = batch_size // self.minibatches
        sums: dict[str, float] = defaultdict(float)
        grad_steps = 0
        for _ in range(self.epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start : start + minibatch_size]
                # Per-minibatch advantage normalization; the 1e-8 keeps a
                # zero-variance minibatch at zero instead of NaN.
                mb_adv = flat_advantages[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                dist = Categorical(logits=self.actor(flat_obs[idx]))
                policy_loss, approx_kl, clip_frac = clipped_surrogate_loss(
                    dist.log_prob(flat_actions[idx]), flat_old_logp[idx], mb_adv, self.clip_eps
                )
                value_loss = F.mse_loss(self.critic(flat_obs[idx]).squeeze(-1), flat_targets[idx])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                # One clip over the actor+critic union — separate calls would
                # hand each net the full norm budget.
                nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.optimizer.step()

                sums["loss/policy"] += float(policy_loss.item())
                sums["loss/value"] += float(value_loss.item())
                sums["loss/entropy"] += float(entropy.item())
                sums["loss/approx_kl"] += float(approx_kl.item())
                sums["loss/clip_frac"] += float(clip_frac.item())
                grad_steps += 1

        self.buffer.clear()
        self.updates += 1
        return {name: total / grad_steps for name, total in sums.items()}

    def state_dict(self) -> dict[str, Any]:
        # The rollout in progress is deliberately not checkpointed: restore
        # serves eval/watch; resuming training refills the buffer.
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
