from __future__ import annotations

from cortex.nn.primatives import Module
from cortex.tensor import Tensor


class Sigmoid(Module):
    """Sigmoid activation function."""

    def forward(self, x: Tensor):
        return 1 / (1 + (-x).exp())


class ReLU(Module):
    """ReLU activation function."""

    def forward(self, x: Tensor):
        return x.set(x < 0, 0)
