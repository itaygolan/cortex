import numpy as np
import pytest

from cortex import Tensor
from cortex.nn.functionals import InvalidReductionOption
from cortex.nn.loss_functions import CrossEntropyLoss, MSELoss


@pytest.mark.parametrize(
    ("x_data", "target_data", "reduction", "expected_loss"),
    [
        ([1, 2, 3], [1, 1, 1], "mean", (0**2 + 1**2 + 2**2) / 3),
        ([1, 2, 3], [1, 1, 1], "sum", 0**2 + 1**2 + 2**2),
        ([1, 2, 3], [1, 1, 1], "none", np.array([0, 1, 4])),
        (
            [[1, 2], [3, 4]],
            [[1, 1], [1, 1]],
            "mean",
            (0**2 + 1**2 + 2**2 + 3**2) / 4,
        ),
    ],
)
def test_mse_loss_forward(x_data, target_data, reduction, expected_loss):
    """
    Tests the forward pass of the MSELoss function.
    """
    loss_fn = MSELoss()
    x = Tensor(x_data)
    targets = Tensor(target_data)
    loss = loss_fn(x, targets, reduction=reduction)
    assert np.allclose(loss.data, expected_loss)


def test_mse_loss_backward():
    """
    Tests the backward pass of the MSELoss function.
    """
    loss_fn = MSELoss()
    x = Tensor([1, 2, 3], requires_grad=True)
    targets = Tensor([1, 1, 1])

    loss = loss_fn(x, targets, reduction="mean")
    loss.backward()

    # d_loss / d_x = 2 * (x - targets) / N
    expected_grad = 2 * (x.data - targets.data) / x.size()
    assert x.grad is not None
    assert np.allclose(x.grad.data, expected_grad)


def test_invalid_reduction():
    """
    Tests that an invalid reduction option raises an exception.
    """
    loss_fn = MSELoss()
    x = Tensor([1, 2, 3])
    targets = Tensor([1, 1, 1])
    with pytest.raises(InvalidReductionOption):
        loss_fn(x, targets, reduction="invalid_option")


@pytest.mark.parametrize(
    ("logits_data", "target_data", "reduction", "expected_loss"),
    [
        (
            [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]],
            [2, 0],
            "mean",
            -(np.log(0.7) + np.log(0.5)) / 2,
        ),
        (
            [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]],
            [2, 0],
            "sum",
            -(np.log(0.7) + np.log(0.5)),
        ),
        (
            [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]],
            [2, 0],
            "none",
            np.array([-np.log(0.7), -np.log(0.5)]),
        ),
    ],
)
def test_cross_entropy_loss_forward(logits_data, target_data, reduction, expected_loss):
    """
    Tests the forward pass of the CrossEntropyLoss function.
    """
    loss_fn = CrossEntropyLoss()
    # Convert to log space for from_logits=False test
    log_probs = Tensor(logits_data)
    targets = Tensor(target_data)
    loss = loss_fn(log_probs, targets, from_logits=False, reduction=reduction)
    assert np.allclose(loss.data, expected_loss)


def test_cross_entropy_loss_backward_from_logits():
    """
    Tests the backward pass of the CrossEntropyLoss function with logits.
    """
    loss_fn = CrossEntropyLoss()
    logits_data = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
    targets_data = np.array([2, 0])

    logits = Tensor(logits_data, requires_grad=True)
    targets = Tensor(targets_data)

    loss = loss_fn(logits, targets, from_logits=True, reduction="mean")
    loss.backward()

    # Softmax probabilities
    exp_logits = np.exp(logits_data)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Gradient of CrossEntropy with Softmax is (probs - one_hot_targets) / N
    num_classes = logits_data.shape[1]
    one_hot_targets = np.eye(num_classes)[targets_data]
    expected_grad = (probs - one_hot_targets) / len(targets_data)

    assert logits.grad is not None
    assert np.allclose(logits.grad.data, expected_grad)
