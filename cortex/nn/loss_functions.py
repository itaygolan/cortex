from __future__ import annotations

from typing import Literal

import cortex
import cortex.nn.functionals as F
from cortex.nn.primatives import Module


class MSELoss(Module):

    def forward(
        self,
        x: cortex.Tensor,
        targets: cortex.Tensor,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        return F.mse_loss(x, targets, reduction)


class CrossEntropyLoss(Module):

    def forward(
        self,
        x: cortex.Tensor,
        targets: cortex.Tensor,
        from_logits: bool = True,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        return F.cross_entropy_loss(x, targets, from_logits, reduction)
