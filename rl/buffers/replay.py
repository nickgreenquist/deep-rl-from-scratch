"""Replay buffer: a preallocated NumPy ring buffer of transitions.

Stores every transition the agent experiences, across episode boundaries;
at capacity the oldest entries are overwritten. `sample` draws a uniform
random minibatch from the whole history, which breaks the temporal
correlation of consecutive env steps and reuses each transition many times.
Torch-free on purpose: agents convert sampled arrays to tensors themselves.
"""

import numpy as np

from rl.buffers.base import Buffer

Batch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer(Buffer):
    def __init__(self, capacity: int, obs_shape: tuple[int, ...]):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        # float, not bool: used directly as the (1 - terminated) bootstrap mask
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.terminated[i] = terminated
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
        )

    def __len__(self) -> int:
        return self._size
