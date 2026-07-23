"""Conv Q-network for MinAtar's 10x10xC binary-plane observations.

Mirrors the MinAtar paper's DQN net (Young & Tian 2019, examples/dqn.py):
one Conv2d(C, 16, kernel 3, stride 1) + ReLU, flattened (8*8*16 = 1024 for
a 10x10 grid), then FC hidden layers (`hidden_sizes`; paper uses [128]) and
a linear Q head. Deliberately tiny next to the Mnih et al. Atari net —
MinAtar strips the vision problem, so one conv layer to exploit the spatial
structure is enough.

`dueling=True` replaces the head with DuelingMLP-style V/A streams split
after the FC trunk (Q = V + A - mean(A); same identifiability argument).
"""

import torch
from torch import nn


class ConvQNet(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int, int, int],
        hidden_sizes: list[int],
        out_dim: int,
        dueling: bool = False,
    ):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("ConvQNet needs at least one FC hidden layer after the conv")
        c, h, w = in_shape
        layers: list[nn.Module] = [nn.Conv2d(c, 16, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten()]
        with torch.no_grad():
            in_dim = nn.Sequential(*layers)(torch.zeros(1, c, h, w)).shape[1]
        for size in hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.ReLU()]
            in_dim = size
        self.trunk = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(in_dim, 1)
            self.advantage = nn.Linear(in_dim, out_dim)
        else:
            self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        z = self.trunk(x)
        if self.dueling:
            adv = self.advantage(z)
            return self.value(z) + adv - adv.mean(dim=-1, keepdim=True)
        return self.head(z)
