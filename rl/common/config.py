"""Run configuration: one flat dataclass loaded from a YAML file.

Harness-level fields are typed here; anything algorithm-specific goes in
the loose `agent` dict, which each algorithm parses itself.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    env_id: str
    seed: int
    total_steps: int
    eval_every: int  # env steps between eval passes
    eval_episodes: int  # episodes per eval pass
    run_name: str
    device: str = "cpu"  # CPU by default; MPS is flaky for this workload
    # Intra-op torch threads. 1 by default: per-step RL kernels are tiny, so
    # the default pool thrashes (5x+ measured slowdown), and one core per run
    # is what lets multi-seed benchmarks parallelize. Raise it when the nets
    # and batches are big enough to amortize fork/join (capstone scale).
    torch_threads: int = 1
    logger: str = "wandb"  # "wandb" | "tensorboard"
    agent: dict = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a Config. Unknown keys and wrong types raise TypeError."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Config(**raw)
    # PyYAML reads `1e6` as a *string* (YAML 1.1 floats need a signed exponent),
    # so without this check a bad scalar only blows up deep in the train loop.
    for fld in fields(cfg):
        val = getattr(cfg, fld.name)
        if not isinstance(val, fld.type):
            raise TypeError(
                f"{path}: {fld.name} must be {fld.type.__name__}, "
                f"got {type(val).__name__} ({val!r})"
            )
    return cfg
