from __future__ import annotations

import numpy as np
import pytest

import cortex
from cortex.nn import functionals as F
from cortex.nn.functionals import InvalidReductionOption


@pytest.mark.parametrize(
    "reduction, expected_fn",
    [("mean", np.mean), ("sum", np.sum), ("none", lambda x: x)],
)
def test_mse_loss(reduction, expected_fn):
    """Tests mse_loss with different reduction options."""
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    targets_np = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32)

    x = cortex.tensor(x_np)
    targets = cortex.tensor(targets_np)

    loss = F.mse_loss(x, targets, reduction=reduction)

    squared_error_np = (targets_np - x_np) ** 2
    expected_loss = expected_fn(squared_error_np)

    assert loss.shape == expected_loss.shape
    np.testing.assert_allclose(loss.numpy(), expected_loss, rtol=1e-6)


def test_mse_loss_default_reduction():
    """Tests that the default reduction for mse_loss is 'mean'."""
    x_np = np.random.rand(5, 2).astype(np.float32)
    targets_np = np.random.rand(5, 2).astype(np.float32)
    x = cortex.tensor(x_np)
    targets = cortex.tensor(targets_np)

    loss = F.mse_loss(x, targets)
    expected_loss = np.mean((targets_np - x_np) ** 2)

    np.testing.assert_allclose(loss.numpy(), expected_loss, rtol=1e-6)


def test_mse_loss_invalid_reduction():
    """Tests that mse_loss raises an error for an invalid reduction option."""
    x = cortex.zeros(2, 2)
    targets = cortex.zeros(2, 2)
    with pytest.raises(InvalidReductionOption):
        F.mse_loss(x, targets, reduction="invalid_option")


@pytest.mark.parametrize(
    "reduction, expected_fn",
    [("mean", np.mean), ("sum", np.sum), ("none", lambda x: x)],
)
@pytest.mark.parametrize("from_logits", [True, False])
def test_cross_entropy_loss(reduction, expected_fn, from_logits):
    """Tests cross_entropy_loss with different reductions and from_logits."""
    logits_np = np.array(
        [[2.0, 1.0, 0.1], [0.5, 2.5, 0.2], [0.1, 0.1, 3.0]], dtype=np.float32
    )
    targets_np = np.array([0, 1, 2], dtype=np.int32)

    if from_logits:
        x_np = logits_np
        # Manual log_softmax
        max_logits = np.max(x_np, axis=-1, keepdims=True)
        exp_logits = np.exp(x_np - max_logits)
        sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
        log_probas_np = (x_np - max_logits) - np.log(sum_exp_logits)
    else:
        # Create probabilities that sum to 1
        probas_np = np.exp(logits_np) / np.sum(
            np.exp(logits_np), axis=-1, keepdims=True
        )
        x_np = probas_np
        log_probas_np = np.log(x_np)

    x = cortex.tensor(x_np)
    targets = cortex.tensor(targets_np)

    loss = F.cross_entropy_loss(
        x, targets, from_logits=from_logits, reduction=reduction
    )

    nll_np = -log_probas_np[np.arange(len(targets_np)), targets_np]
    expected_loss = expected_fn(nll_np)

    assert loss.shape == expected_loss.shape
    np.testing.assert_allclose(loss.numpy(), expected_loss, rtol=1e-5)


def test_cross_entropy_loss_default_reduction_and_logits():
    """Tests the default parameters for cross_entropy_loss."""
    logits_np = np.random.randn(10, 5).astype(np.float32)
    targets_np = np.random.randint(0, 5, size=(10,)).astype(np.int32)
    x = cortex.tensor(logits_np)
    targets = cortex.tensor(targets_np)

    loss = F.cross_entropy_loss(x, targets)

    # Manual log_softmax and NLL
    max_logits = np.max(logits_np, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_np - max_logits)
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
    log_probas_np = (logits_np - max_logits) - np.log(sum_exp_logits)
    nll_np = -log_probas_np[np.arange(len(targets_np)), targets_np]
    expected_loss = np.mean(nll_np)

    np.testing.assert_allclose(loss.numpy(), expected_loss, rtol=1e-5)


def test_cross_entropy_loss_invalid_reduction():
    """Tests that cross_entropy_loss raises an error for an invalid reduction."""
    x = cortex.zeros(2, 2)
    targets = cortex.tensor(np.array([0, 1], dtype=np.int32))
    with pytest.raises(InvalidReductionOption):
        F.cross_entropy_loss(x, targets, reduction="invalid_option")


def test_cross_entropy_loss_invalid_input_dim():
    """Tests that cross_entropy_loss raises an error for input with ndim != 2."""
    x = cortex.zeros(2, 2, 2)  # 3D tensor
    targets = cortex.tensor(np.array([0, 1], dtype=np.int32))
    with pytest.raises(AssertionError):
        F.cross_entropy_loss(x, targets)
