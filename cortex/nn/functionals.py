"""Functionals."""

from __future__ import annotations

from typing import Literal

import cortex

Reduction = Literal["sum", "mean", "none"]


class InvalidReductionOption(Exception):
    """Invalid reduction option."""


def mse_loss(x: cortex.Tensor, targets: cortex.Tensor, reduction: Reduction = "mean"):
    """Mean squared loss"""
    mean_squared = (targets - x) ** 2
    return _reduce_loss(mean_squared, reduction)


def cross_entropy_loss(
    x: cortex.Tensor,
    targets: cortex.Tensor,
    from_logits: bool = True,
    reduction: Reduction = "mean",
):
    """Cross Entropy Loss"""
    assert x.ndim == 2

    if from_logits:
        # Compute log softmax: logits - logsumexp

        # Shift by max for numerical stability
        x_max = x.max(dim=-1, keepdim=True)
        shifted_x = x - x_max

        # Compute logsumexp
        shifted_exp = shifted_x.exp()
        sum_exp = shifted_exp.sum(dim=-1, keepdim=True)
        logsumexp = sum_exp.log() + x_max

        # Compute log softmax
        log_probas = x - logsumexp
    else:
        log_probas = x.log()

    idx = cortex.arange(0, x.size(0))
    nll = -log_probas[idx, targets]

    return _reduce_loss(nll, reduction)


def _reduce_loss(error: cortex.Tensor, reduction: str):
    if reduction == "sum":
        return error.sum()
    elif reduction == "mean":
        return error.mean()
    elif reduction == "none":
        return error
    else:
        raise InvalidReductionOption()
