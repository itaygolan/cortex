from __future__ import annotations

import numpy as np

import cortex
from cortex import nn


def test_linear_layer_forward():
    """
    Tests the forward pass of the Linear layer.
    """
    in_dim, out_dim = 10, 5
    batch_size = 1

    layer = nn.Linear(in_dim, out_dim)
    input_tensor = cortex.randn(batch_size, in_dim)

    output_tensor = layer(input_tensor)
    assert output_tensor.shape == (batch_size, out_dim)


def test_linear_layer_backward():
    """
    Tests the backward pass and gradient computation of the Linear layer.
    """
    in_dim, out_dim = 3, 2

    layer = nn.Linear(in_dim, out_dim)

    # Use deterministic weights and inputs for consistent gradient checking
    layer.weight.data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    layer.bias.data = np.array([0.1, 0.2])

    input_tensor = cortex.Tensor([[1, 2, 3]], requires_grad=True)
    output = layer(input_tensor)
    output.sum().backward()

    # Expected output:
    # out = input @ weight.T + bias
    # out_0 = 1*0.1 + 2*0.2 + 3*0.3 + 0.1 = 0.1 + 0.4 + 0.9 + 0.1 = 1.5
    # out_1 = 1*0.4 + 2*0.5 + 3*0.6 + 0.2 = 0.4 + 1.0 + 1.8 + 0.2 = 3.4
    # expected_output = [1.5, 3.4]
    expected_output = np.array([1.5, 3.4])
    assert np.allclose(output.data, expected_output)

    # Expected gradients:
    # L = sum(output) = out_0 + out_1
    # dL/d(bias) = [1, 1]
    # dL/d(weight) = input.T @ dL/d(output)
    # dL/d(output) is a tensor of ones of shape (1, 2) because of sum()
    # dL/d(weight) = [[1], [2], [3]] @ [[1, 1]] = [[1, 1], [2, 2], [3, 3]]
    # dL/d(input) = dL/d(output) @ weight
    # dL/d(input) = [[1, 1]] @ [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]] = [0.5, 0.7, 0.9]

    assert layer.bias.grad is not None
    assert np.allclose(layer.bias.grad.data, np.array([1.0, 1.0]))

    assert layer.weight.grad is not None
    assert np.allclose(
        layer.weight.grad.data, np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    )

    assert input_tensor.grad is not None
    assert np.allclose(input_tensor.grad.data, np.array([0.5, 0.7, 0.9]))


def test_linear_parameters():
    """
    Tests that the parameters of the Linear layer are correctly identified.
    """
    in_dim, out_dim = 10, 5
    layer = nn.Linear(in_dim, out_dim)
    params = list(layer.parameters())

    assert len(params) == 2
    assert params[0].shape == (out_dim, in_dim)
    assert params[1].shape == (out_dim,)
