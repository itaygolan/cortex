from __future__ import annotations

import numpy as np

import cortex
from cortex.nn.primatives import Module, Parameter


class Linear(Module):
    """Linear layer."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()

        # Use Kaiming uniform initialization (He uniform)
        # bound = sqrt(6 / fan_in) where fan_in = in_dim
        bound = np.sqrt(6.0 / in_dim)

        # Initialize weights uniformly in [-bound, bound]
        self.weight = Parameter(cortex.uniform(-bound, bound, shape=(out_dim, in_dim)))

        # Initialize bias uniformly in [-bound, bound]
        if bias:
            self.bias = Parameter(cortex.uniform(-bound, bound, shape=(out_dim,)))
        else:
            self.bias = None

    def forward(self, x: cortex.Tensor):
        transformation = x @ self.weight.T
        if self.bias is None:
            return transformation
        return transformation + self.bias
