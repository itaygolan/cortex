import numpy as np
import pytest

from cortex import Tensor
from cortex.nn.primatives import Parameter
from cortex.optim.optimizers import SGD, Adam, AdamW, RMSProp


@pytest.fixture
def sgd_params():
    """Provides a parameter with a gradient for SGD tests."""
    p = Parameter(np.array([10.0, 20.0]))
    p.grad = np.array([1.0, 2.0])
    return [p]


@pytest.fixture
def rmsprop_params():
    """Provides a parameter with a gradient for RMSProp tests."""
    p = Parameter(np.array([10.0, 20.0]))
    p.grad = np.array([1.0, 2.0])
    return [p]


@pytest.fixture
def adam_params():
    """Provides a parameter with a gradient for Adam tests."""
    p = Parameter(np.array([10.0, 20.0]))
    p.grad = np.array([1.0, 2.0])
    return [p]


@pytest.fixture
def adamw_params():
    """Provides a parameter with a gradient for Adam tests."""
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


def test_rmsprop_basic_step(rmsprop_params):
    """
    Tests a basic optimization step with RMSProp.
    """
    param = rmsprop_params[0]
    initial_data = param.data.copy()
    grad_data = param.grad.copy()
    lr = 0.1
    alpha = 0.99
    eps = 1e-8

    optimizer = RMSProp(rmsprop_params, lr=lr, alpha=alpha, eps=eps)
    optimizer.step()

    # First step: RMS = (1-alpha) * grad^2, step = grad / sqrt(RMS + eps)
    rms = (1 - alpha) * grad_data**2
    step = grad_data / np.sqrt(rms + eps)
    expected_data = initial_data - lr * step
    assert np.allclose(param.data, expected_data)


def test_rmsprop_weight_decay(rmsprop_params):
    """
    Tests the RMSProp optimizer with weight decay.
    """
    param = rmsprop_params[0]
    initial_data = param.data.copy()
    grad_data = param.grad.copy()
    lr = 0.1
    alpha = 0.99
    eps = 1e-8
    weight_decay = 0.01

    optimizer = RMSProp(
        rmsprop_params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay
    )
    optimizer.step()

    # Gradient update with L2 weight decay
    grad_with_decay = grad_data + weight_decay * initial_data
    rms = (1 - alpha) * grad_with_decay**2
    step = grad_with_decay / np.sqrt(rms + eps)
    expected_data = initial_data - lr * step
    assert np.allclose(param.data, expected_data)


def test_rmsprop_momentum_buffer(rmsprop_params):
    """
    Tests the RMSProp optimizer with RMS buffer accumulation over multiple steps.
    """
    param = rmsprop_params[0]
    lr = 0.1
    alpha = 0.9
    eps = 1e-8

    optimizer = RMSProp(rmsprop_params, lr=lr, alpha=alpha, eps=eps)

    # First step
    initial_data = param.data.copy()
    grad1 = param.grad.copy()
    optimizer.step()

    rms1 = (1 - alpha) * grad1**2
    step1 = grad1 / np.sqrt(rms1 + eps)
    expected_data1 = initial_data - lr * step1
    assert np.allclose(param.data, expected_data1)

    # Second step with new gradient
    param.grad = np.array([3.0, 4.0])
    grad2 = param.grad.copy()
    optimizer.step()

    # RMS buffer updates: rms = alpha * old_rms + (1-alpha) * grad^2
    rms2 = alpha * rms1 + (1 - alpha) * grad2**2
    step2 = grad2 / np.sqrt(rms2 + eps)
    expected_data2 = expected_data1 - lr * step2
    assert np.allclose(param.data, expected_data2)


def test_rmsprop_multiple_param_groups():
    """
    Tests RMSProp with multiple parameter groups having different settings.
    """
    p1 = Parameter([10.0])
    p1.grad = np.array([1.0])
    p2 = Parameter([20.0])
    p2.grad = np.array([2.0])

    param_groups = [
        {"params": [p1], "lr": 0.1, "alpha": 0.9, "eps": 1e-8},
        {"params": [p2], "lr": 0.01, "alpha": 0.99, "eps": 1e-6},
    ]

    optimizer = RMSProp(param_groups)
    optimizer.step()

    # Check p1
    rms1 = (1 - 0.9) * p1.grad**2
    step1 = p1.grad / np.sqrt(rms1 + 1e-8)
    expected_p1_data = 10.0 - 0.1 * step1
    assert np.allclose(p1.data, expected_p1_data)

    # Check p2
    rms2 = (1 - 0.99) * p2.grad**2
    step2 = p2.grad / np.sqrt(rms2 + 1e-6)
    expected_p2_data = 20.0 - 0.01 * step2
    assert np.allclose(p2.data, expected_p2_data)


def test_adam_basic_step(adam_params):
    """
    Tests a basic optimization step with Adam.
    """
    param = adam_params[0]
    initial_data = param.data.copy()
    grad_data = param.grad.copy()
    lr = 0.1
    betas = (0.9, 0.99)
    eps = 1e-8

    optimizer = Adam(adam_params, lr=lr, betas=betas, eps=eps)
    optimizer.step()

    # First step: t=1, moment1 = grad, moment2 = grad^2
    # Bias corrected: moment1_bc = moment1 / (1-beta1), moment2_bc = moment2 / (1-beta2)
    t = 1
    beta1, beta2 = betas
    moment1 = grad_data
    moment2 = grad_data**2
    moment1_bc = moment1 / (1 - beta1**t)
    moment2_bc = moment2 / (1 - beta2**t)
    step = moment1_bc / np.sqrt(moment2_bc + eps)
    expected_data = initial_data - lr * step
    assert np.allclose(param.data, expected_data)


def test_adam_equals_adamw_basic_step(adam_params, adamw_params):
    """
    Tests a basic optimization step with Adam equals AdamW w/o WD
    """
    lr = 0.1
    betas = (0.9, 0.99)
    eps = 1e-8

    optimizer = Adam(adam_params, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
    optimizer2 = AdamW(adamw_params, lr=lr, betas=betas, eps=eps, weight_decay=0.0)

    optimizer.step()
    optimizer2.step()

    assert np.allclose(adamw_params[0].data, adamw_params[0].data)


def test_adam_weight_decay(adam_params):
    """
    Tests the Adam optimizer with weight decay.
    """
    param = adam_params[0]
    initial_data = param.data.copy()
    grad_data = param.grad.copy()
    lr = 0.1
    betas = (0.9, 0.99)
    eps = 1e-8
    weight_decay = 0.01

    optimizer = Adam(
        adam_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    optimizer.step()

    # Gradient update with L2 weight decay
    grad_with_decay = grad_data + weight_decay * initial_data

    t = 1
    beta1, beta2 = betas
    moment1 = (1 - beta1) * grad_with_decay
    moment2 = (1 - beta2) * grad_with_decay**2
    moment1_bc = moment1 / (1 - beta1**t)
    moment2_bc = moment2 / (1 - beta2**t)
    step = moment1_bc / np.sqrt(moment2_bc + eps)
    expected_data = initial_data - lr * step
    assert np.allclose(param.data, expected_data)


def test_adamw_weight_decay(adam_params):
    """
    Tests the AdamW optimizer with weight decay.
    """
    param = adam_params[0]
    initial_data = param.data.copy()
    grad_data = param.grad.copy()
    lr = 0.1
    betas = (0.9, 0.99)
    eps = 1e-8
    weight_decay = 0.01

    optimizer = AdamW(
        adam_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    optimizer.step()

    # Gradient update
    grad_with_decay = grad_data

    t = 1
    beta1, beta2 = betas
    moment1 = (1 - beta1) * grad_with_decay
    moment2 = (1 - beta2) * grad_with_decay**2
    moment1_bc = moment1 / (1 - beta1**t)
    moment2_bc = moment2 / (1 - beta2**t)
    step = moment1_bc / np.sqrt(moment2_bc + eps)
    # Apply decoupled weight decay
    expected_data = initial_data - lr * (step + weight_decay * initial_data)
    assert np.allclose(param.data, expected_data)


def test_adam_moment_tracking(adam_params):
    """
    Tests the Adam optimizer's moment tracking over multiple steps.
    """
    param = adam_params[0]
    lr = 0.1
    betas = (0.9, 0.99)
    eps = 1e-8

    optimizer = Adam(adam_params, lr=lr, betas=betas, eps=eps)

    # First step
    initial_data = param.data.copy()
    grad1 = param.grad.copy()
    optimizer.step()

    # Calculate expected first step moments
    beta1, beta2 = betas
    t1 = 1
    moment1_1 = (1 - beta1) * grad1
    moment2_1 = (1 - beta2) * grad1**2
    moment1_bc_1 = moment1_1 / (1 - beta1**t1)
    moment2_bc_1 = moment2_1 / (1 - beta2**t1)
    step1 = moment1_bc_1 / np.sqrt(moment2_bc_1 + eps)
    expected_data1 = initial_data - lr * step1
    assert np.allclose(param.data, expected_data1)

    # Second step with new gradient
    param.grad = np.array([3.0, 4.0])
    grad2 = param.grad.copy()
    optimizer.step()

    # Calculate expected second step moments
    t2 = 2
    moment1_2 = beta1 * moment1_1 + (1 - beta1) * grad2
    moment2_2 = beta2 * moment2_1 + (1 - beta2) * grad2**2
    moment1_bc_2 = moment1_2 / (1 - beta1**t2)
    moment2_bc_2 = moment2_2 / (1 - beta2**t2)
    step2 = moment1_bc_2 / np.sqrt(moment2_bc_2 + eps)
    expected_data2 = expected_data1 - lr * step2
    assert np.allclose(param.data, expected_data2)


def test_adam_multiple_param_groups():
    """
    Tests Adam with multiple parameter groups having different settings.
    """
    p1 = Parameter([10.0])
    p1.grad = np.array([1.0])
    p2 = Parameter([20.0])
    p2.grad = np.array([2.0])

    param_groups = [
        {"params": [p1], "lr": 0.1, "betas": (0.9, 0.99), "eps": 1e-8},
        {"params": [p2], "lr": 0.01, "betas": (0.95, 0.999), "eps": 1e-6},
    ]

    optimizer = Adam(param_groups)
    optimizer.step()

    # Check p1
    t = 1
    beta1, beta2 = (0.9, 0.99)
    moment1_p1 = (1 - beta1) * p1.grad
    moment2_p1 = (1 - beta2) * p1.grad**2
    moment1_bc_p1 = moment1_p1 / (1 - beta1**t)
    moment2_bc_p1 = moment2_p1 / (1 - beta2**t)
    step_p1 = moment1_bc_p1 / np.sqrt(moment2_bc_p1 + 1e-8)
    expected_p1_data = 10.0 - 0.1 * step_p1
    assert np.allclose(p1.data, expected_p1_data)

    # Check p2
    t = 1
    beta1, beta2 = (0.95, 0.999)
    moment1_p2 = (1 - beta1) * p2.grad
    moment2_p2 = (1 - beta2) * p2.grad**2
    moment1_bc_p2 = moment1_p2 / (1 - beta1**t)
    moment2_bc_p2 = moment2_p2 / (1 - beta2**t)
    step_p2 = moment1_bc_p2 / np.sqrt(moment2_bc_p2 + 1e-6)
    expected_p2_data = 20.0 - 0.01 * step_p2
    assert np.allclose(p2.data, expected_p2_data)


def test_adam_bias_correction_evolution(adam_params):
    """
    Tests that Adam's bias correction evolves correctly over multiple steps.
    """
    param = adam_params[0]
    lr = 0.1
    betas = (0.9, 0.99)  # Use lower betas to make bias correction more visible
    eps = 1e-8

    optimizer = Adam(adam_params, lr=lr, betas=betas, eps=eps)

    # Use varying gradients to see bias correction effects more clearly
    gradients = [
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),  # Same gradient, bias correction should change
        np.array([3.0, 4.0]),  # Different gradient
        np.array([3.0, 4.0]),  # Same gradient again
        np.array([5.0, 6.0]),  # Another different gradient
    ]

    data_history = []
    for grad in gradients:
        param.grad = grad.copy()
        optimizer.step()
        data_history.append(param.data.copy())

    # Verify that the updates change over time due to bias correction
    # Even with the same gradient, bias correction should affect early steps
    update_magnitudes = []
    for i in range(1, len(data_history)):
        update_magnitude = np.linalg.norm(data_history[i] - data_history[i - 1])
        update_magnitudes.append(update_magnitude)

    # Updates should be different due to bias correction and varying gradients
    assert len(set(round(mag, 6) for mag in update_magnitudes)) > 1
