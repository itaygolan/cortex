from __future__ import annotations

import numpy as np
import pytest

from cortex import Tensor
from cortex.nn.activations import ReLU, Sigmoid


@pytest.mark.parametrize(
    ("input_data", "expected_output"),
    [
        (np.array([0, 1, -1]), np.array([0.5, 0.73105858, 0.26894142])),
        (np.array([0.0]), np.array([0.5])),
        (np.array([-100.0, 100.0]), np.array([0.0, 1.0])),
    ],
)
def test_sigmoid_forward(input_data, expected_output):
    """
    Tests the forward pass of the Sigmoid activation function.
    """
    sigmoid = Sigmoid()
    t = Tensor(input_data, requires_grad=False)
    output = sigmoid(t)
    assert np.allclose(output.data, expected_output)


@pytest.mark.parametrize(
    ("input_data",),
    [
        (np.array([0, 1, -1]),),
        (np.array([0.5, -0.5, 10.0]),),
    ],
)
def test_sigmoid_backward(input_data):
    """
    Tests the backward pass of the Sigmoid activation function.
    """
    sigmoid = Sigmoid()
    t = Tensor(input_data, requires_grad=True)
    output = sigmoid(t)
    output.backward()

    # Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
    s_t = 1 / (1 + np.exp(-t.data))
    expected_grad = s_t * (1 - s_t)
    assert t.grad is not None
    assert np.allclose(t.grad.data, expected_grad)


@pytest.mark.parametrize(
    ("input_data", "expected_output"),
    [
        (np.array([1, -2, 0, 3]), np.array([1, 0, 0, 3])),
        (np.array([0.0, -0.0]), np.array([0.0, 0.0])),
        (np.array([-100.0, 100.0]), np.array([0.0, 100.0])),
    ],
)
def test_relu_forward(input_data, expected_output):
    """
    Tests the forward pass of the ReLU activation function.
    """
    relu = ReLU()
    t = Tensor(input_data, requires_grad=False)
    output = relu(t)
    assert np.allclose(output.data, expected_output)


@pytest.mark.parametrize(
    ("input_data", "expected_grad"),
    [
        (np.array([1, -2, 0, 3]), np.array([1, 0, 0, 1])),
        (np.array([0.0, -0.0]), np.array([0.0, 0.0])),
        (np.array([-100.0, 100.0]), np.array([0.0, 1.0])),
    ],
)
def test_relu_backward(input_data, expected_grad):
    """
    Tests the backward pass of the ReLU activation function.
    """
    relu = ReLU()
    t = Tensor(input_data, requires_grad=True)
    output = relu(t)
    output.backward()

    assert t.grad is not None
    assert np.allclose(t.grad.data, expected_grad)
