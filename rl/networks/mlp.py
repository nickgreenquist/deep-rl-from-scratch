"""MLP builder shared by the value/policy networks.

`hidden_sizes=[]` degenerates to a single linear layer — the Phase 1
on-ramp's `nn.Linear(4, 2)` and full DQN differ only in this argument.
"""

from torch import nn


def mlp(in_dim: int, hidden_sizes: list[int], out_dim: int) -> nn.Sequential:
    sizes = [in_dim, *hidden_sizes, out_dim]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:  # no activation after the output layer
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)
