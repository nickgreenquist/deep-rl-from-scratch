"""Rollout buffer + GAE: the on-policy data path for PPO.

The second buffer pattern (see base.py): filled to a fixed horizon, drained
whole, cleared. Nothing is ever sampled from history — on-policy data dies
after one update cycle, which is the constraint REINFORCE made concrete.

Layout is (T, N): T lockstep steps across N envs, exactly as the vector
train loop hands transitions over. Rows are dense — autoreset is disabled
in make_vec_env, so every row is a real env step — and each row stores its
own next_obs. That costs a second obs array but makes every row
self-contained: GAE never reaches across rows for successor states, no
special bootstrap is needed at the buffer's end, and a truncated row's
next_obs IS the episode's true final observation, so time-limit cuts
bootstrap correctly (the bias REINFORCE had to live with).

Torch-free like the replay buffer: the agent converts to tensors itself.
"""

import numpy as np

from rl.buffers.base import Buffer


class RolloutBuffer(Buffer):
    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        n_actions: int,
        obs_dtype=np.float32,
    ):
        self.horizon = horizon
        self.num_envs = num_envs
        # Obs arrays take the env dtype (same reasoning as the replay buffer).
        self.obs = np.zeros((horizon, num_envs, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((horizon, num_envs), dtype=np.int64)
        self.rewards = np.zeros((horizon, num_envs), dtype=np.float32)
        self.next_obs = np.zeros((horizon, num_envs, *obs_shape), dtype=obs_dtype)
        # Floats, not bools: used directly as (1 - flag) masks in compute_gae.
        self.terminated = np.zeros((horizon, num_envs), dtype=np.float32)
        self.truncated = np.zeros((horizon, num_envs), dtype=np.float32)
        # Legality masks for each row's obs, stored so every optimization
        # epoch reapplies the SAME mask the policy acted under — pi_new and
        # pi_old must see identically masked distributions or the importance
        # ratio is silently wrong on every sample. (No next-state masks: the
        # critic is the only next_obs reader, and values are never masked.)
        self.masks = np.zeros((horizon, num_envs, n_actions), dtype=bool)
        self._ptr = 0

    def add(self, obs, actions, rewards, next_obs, terminated, truncated, masks) -> None:
        """Store one batched (N-wide) transition row."""
        t = self._ptr  # IndexError past the horizon: the agent drains at full()
        self.obs[t] = obs
        self.actions[t] = actions
        self.rewards[t] = rewards
        self.next_obs[t] = next_obs
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.masks[t] = masks
        self._ptr += 1

    def full(self) -> bool:
        return self._ptr == self.horizon

    def clear(self) -> None:
        self._ptr = 0  # rows are overwritten in place on the next fill

    def __len__(self) -> int:
        return self._ptr * self.num_envs


def compute_gae(
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Generalized Advantage Estimation (Schulman et al. 2016) over a (T, N)
    rollout. All inputs (T, N); returns advantages (T, N). The value-loss
    regression targets are `advantages + values`.

    The one-step TD error is itself an advantage estimate — how much better
    taking a_t went than V(s_t) expected:

        delta_t = r_t + gamma * V(s'_t) * (1 - terminated_t) - V(s_t)

    GAE chains them backward with an exponential decay:

        A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}

    lam is the bias-variance dial between the poles met in Phases 1-2:
    lam=0 keeps only delta_t (pure bootstrap — low variance, biased by V's
    errors, DQN's regime), lam=1 telescopes into the Monte Carlo return
    minus V(s_t) (REINFORCE-with-baseline — unbiased, high variance).
    Intermediate lam down-weights the k-step estimates by lam^k.

    Boundary rules match the repo-wide convention: termination zeroes the
    bootstrap term, any episode end cuts the chain (the next row belongs to
    a different episode). A truncated row keeps its bootstrap — its
    next_values entry is V(the true final obs), courtesy of the per-row
    next_obs layout.
    """
    dones = np.maximum(terminated, truncated)
    advantages = np.zeros_like(rewards)
    gae = np.zeros(rewards.shape[1], dtype=rewards.dtype)
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - terminated[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
    return advantages
