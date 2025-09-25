from __future__ import annotations

import cortex
from cortex.nn.primatives import Module


class Sigmoid(Module):

    def forward(self, x: cortex.Tensor):

        return 1 / (1 + (-x).exp())


class ReLU(Module):

    def forward(self, x: cortex.Tensor):
        return x.set(x < 0, 0)
