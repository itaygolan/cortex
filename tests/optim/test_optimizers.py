import numpy as np
import pytest

from cortex import Tensor
from cortex.nn.primatives import Parameter
from cortex.optim.optimizers import SGD


@pytest.fixture
def sgd_params():
    """Provides a parameter with a gradient for SGD tests."""
    p = Parameter(np.array([10.0, 20.0]))
    p.grad = np.array([1.0, 2.0])
    return [p]


def test_sgd_basic_step(sgd_params):
    """
    Tests a basic optimization step with SGD.
    """
    param = sgd_params[0]
    initial_data = param.data.copy()
    grad_data = param.grad.copy()
    lr = 0.1

    optimizer = SGD(sgd_params, lr=lr)
    optimizer.step()

    expected_data = initial_data - lr * grad_data
    assert np.allclose(param.data, expected_data)


def test_sgd_weight_decay(sgd_params):
    """
    Tests the SGD optimizer with weight decay.
    """
    param = sgd_params[0]
    initial_data = param.data.copy()
    grad_data = param.grad.copy()
    lr = 0.1
    weight_decay = 0.01

    optimizer = SGD(sgd_params, lr=lr, weight_decay=weight_decay)
    optimizer.step()

    # Gradient update with L2 weight decay
    grad_with_decay = grad_data + weight_decay * initial_data
    expected_data = initial_data - lr * grad_with_decay
    assert np.allclose(param.data, expected_data)


def test_sgd_momentum(sgd_params):
    """
    Tests the SGD optimizer with momentum.
    """
    param = sgd_params[0]
    lr = 0.1
    momentum = 0.9

    optimizer = SGD(sgd_params, lr=lr, momentum=momentum)

    # First step
    initial_data = param.data.copy()
    grad1 = param.grad.copy()
    optimizer.step()

    buffer1 = grad1
    expected_data1 = initial_data - lr * buffer1
    assert np.allclose(param.data, expected_data1)

    # Second step
    param.grad = np.array([3.0, 4.0])  # New gradient
    grad2 = param.grad.copy()
    optimizer.step()

    buffer2 = momentum * buffer1 + grad2
    expected_data2 = expected_data1 - lr * buffer2
    assert np.allclose(param.data, expected_data2)


def test_sgd_nesterov_momentum(sgd_params):
    """
    Tests the SGD optimizer with Nesterov momentum.
    """
    param = sgd_params[0]
    lr = 0.1
    momentum = 0.9

    optimizer = SGD(sgd_params, lr=lr, momentum=momentum, nesterov=True)

    # First step
    initial_data = param.data.copy()
    grad1 = param.grad.copy()
    optimizer.step()

    buffer1 = grad1
    step1 = grad1 + momentum * buffer1
    expected_data1 = initial_data - lr * step1
    assert np.allclose(param.data, expected_data1)

    # Second step
    param.grad = np.array([3.0, 4.0])  # New gradient
    grad2 = param.grad.copy()
    optimizer.step()

    buffer2 = momentum * buffer1 + grad2
    step2 = grad2 + momentum * buffer2
    expected_data2 = expected_data1 - lr * step2
    assert np.allclose(param.data, expected_data2)


def test_sgd_multiple_param_groups():
    """
    Tests SGD with multiple parameter groups having different settings.
    """
    p1 = Parameter([10.0])
    p1.grad = np.array([1.0])
    p2 = Parameter([20.0])
    p2.grad = np.array([2.0])

    param_groups = [
        {"params": [p1], "lr": 0.1, "momentum": 0.0},
        {"params": [p2], "lr": 0.01, "momentum": 0.9},
    ]

    optimizer = SGD(param_groups)
    optimizer.step()

    # Check p1 (no momentum)
    expected_p1_data = 10.0 - 0.1 * 1.0
    assert np.allclose(p1.data, expected_p1_data)

    # Check p2 (with momentum)
    buffer_p2 = p2.grad
    expected_p2_data = 20.0 - 0.01 * buffer_p2
    assert np.allclose(p2.data, expected_p2_data)
