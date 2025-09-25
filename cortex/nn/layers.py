from __future__ import annotations

import numpy as np

import cortex
from cortex.nn.primatives import Module, Parameter


class Linear(Module):
    """Linear layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        # He initialization
        self.weight = Parameter(cortex.randn(out_dim, in_dim) / np.sqrt(in_dim))
        self.bias = Parameter(cortex.zeros(out_dim))

    def forward(self, x: cortex.Tensor):
        return x @ self.weight.T + self.bias
