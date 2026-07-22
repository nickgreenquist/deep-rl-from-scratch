"""DQN: Q-learning with function approximation — the same TD update as the
tabular agent, but the table becomes a network, and the deadly triad
(function approximation + bootstrapping + off-policy) costs the convergence
guarantee. Two stabilizers tame it in practice:

- Replay buffer: gradient steps use uniform random minibatches from the
  whole interaction history instead of the (correlated) latest transition.
  Off-policy learning is what makes training on stale data legal.
- Target network: TD targets are computed by a frozen copy of the Q-network,
  synced every `target_update_every` gradient steps, so each stretch of
  training is a regression toward stationary labels rather than a chase
  after targets that move with every update.

With `hidden_sizes=[]` the Q-network is a single linear layer — the Phase 1
on-ramp. Add hidden layers via config and this same class is DQN proper.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from rl.agents.base import Agent
from rl.buffers.replay import ReplayBuffer
from rl.networks.mlp import mlp


class DQNAgent(Agent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay_steps: int,
        buffer_capacity: int,
        batch_size: int,
        learning_starts: int,
        target_update_every: int,
        hidden_sizes: list[int],
    ):
        # Q(s, ·) needs a flat continuous obs vector and enumerable actions.
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("DQNAgent requires a flat Box observation space")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("DQNAgent requires a Discrete action space")
        self.action_space = action_space
        self.device = torch.device(device)
        self.q = mlp(observation_space.shape[0], hidden_sizes, int(action_space.n)).to(self.device)
        self.q_target = mlp(observation_space.shape[0], hidden_sizes, int(action_space.n)).to(
            self.device
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, observation_space.shape)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_every = target_update_every
        self.transitions = 0  # env transitions seen; drives the epsilon anneal
        self.grad_steps = 0  # gradient steps taken; drives target syncs

    def _epsilon(self) -> float:
        frac = min(self.transitions / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def act(self, obs: Any, deterministic: bool = False) -> int:
        if not deterministic and np.random.random() < self._epsilon():
            return int(self.action_space.sample())
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return int(self.q(obs_t).argmax().item())

    def update(self, batch: Any) -> dict[str, float]:
        # The train loop hands over the fresh transition each step; it goes
        # into the buffer, and the gradient step trains on a sampled batch.
        self.buffer.add(*batch)
        self.transitions += 1
        if len(self.buffer) < self.learning_starts:
            return {}

        obs, actions, rewards, next_obs, terminated = self.buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, device=self.device)
        terminated_t = torch.as_tensor(terminated, device=self.device)

        q_pred = self.q(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(next_obs_t).max(dim=1).values
            # Bootstrap only through non-terminal next states (same rule as
            # tabular Q: truncation is a cut, not an ending).
            target = rewards_t + self.gamma * (1.0 - terminated_t) * next_q
        # Huber, not MSE: TD targets are noisy early on; linear loss beyond
        # |error| = 1 keeps outlier targets from dominating the gradient.
        loss = F.smooth_l1_loss(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % self.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return {"loss/q": float(loss.item()), "loss/q_pred_mean": float(q_pred.mean().item())}

    def state_dict(self) -> dict[str, Any]:
        # Buffer contents are deliberately not checkpointed: restore serves
        # eval/watch; resuming training would refill replay from scratch.
        return {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "transitions": self.transitions,
            "grad_steps": self.grad_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state["q_target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.transitions = state["transitions"]
        self.grad_steps = state["grad_steps"]
