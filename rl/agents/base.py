"""Agent interface every algorithm must fit — random, tabular Q, DQN, PPO, SAC.
Shared code (train loop, eval, checkpoint) talks only to this."""

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    @abstractmethod
    def act(self, obs: Any, deterministic: bool = False) -> Any:
        """Pick an action for one observation. `deterministic=True` is the
        eval-time policy (e.g. argmax instead of sampling)."""

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """One learning step from a batch; returns `loss/*` metrics."""

    def state_dict(self) -> dict[str, Any]:
        """Learnable state for checkpointing; stateless agents return {}."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass
