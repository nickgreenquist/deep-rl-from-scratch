"""Checkpoint save/load. Phase 0 stub: agent state + step + config, enough to
restore a policy for eval; optimizer state joins when the first gradient-based
agent lands."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from rl.agents.base import Agent
from rl.common.config import Config


def save_checkpoint(path: str | Path, agent: Agent, step: int, cfg: Config) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), "step": step, "config": asdict(cfg)}, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    # weights_only=False: checkpoints are our own files, and agent state may
    # hold non-tensor objects (e.g. a NumPy Q-table) the safe loader rejects.
    return torch.load(Path(path), weights_only=False)
