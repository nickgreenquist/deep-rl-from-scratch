"""Replay buffer: a preallocated NumPy ring buffer of transitions.

Stores every transition the agent experiences, across episode boundaries;
at capacity the oldest entries are overwritten. `sample` draws a uniform
random minibatch from the whole history, which breaks the temporal
correlation of consecutive env steps and reuses each transition many times.
Torch-free on purpose: agents convert sampled arrays to tensors themselves.
"""

import numpy as np

from rl.buffers.base import Buffer

Batch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer(Buffer):
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], obs_dtype=np.float32):
        self.capacity = capacity
        # Obs arrays take the env's dtype: MinAtar's bool planes stay 1 byte
        # per entry (100k Seaquest transitions ≈ 200MB, not float32's 800MB);
        # agents cast to float at tensor time.
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        # float, not bool: used directly as the (1 - terminated) bootstrap mask
        self.terminated = np.zeros(capacity, dtype=np.float32)
        # Bootstrap discount for this transition: gamma^m, where m is the
        # number of env steps between obs and next_obs (m < n_step when an
        # episode end cut the window short). The buffer stays gamma-ignorant;
        # the agent computes it.
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, terminated, discount) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = terminated
        self.discounts[i] = discount
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.terminated[idx],
            self.discounts[idx],
        )

    def __len__(self) -> int:
        return self._size
