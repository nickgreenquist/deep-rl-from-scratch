"""Observation wrappers applied inside the env factory."""

import gymnasium as gym
import numpy as np


class ChannelFirst(gym.ObservationWrapper):
    """(H, W, C) -> (C, H, W). Gym image envs emit channel-last; torch's
    Conv2d wants channel-first. Dtype passes through untouched so binary
    planes stay 1 byte per entry all the way into the replay buffer."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 3:
            raise TypeError("ChannelFirst requires a rank-3 Box observation space")
        self.observation_space = gym.spaces.Box(
            low=np.transpose(space.low, (2, 0, 1)),
            high=np.transpose(space.high, (2, 0, 1)),
            dtype=space.dtype,
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))
