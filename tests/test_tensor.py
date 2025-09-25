from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest

import cortex


def test_tensor_shape(simple_tensor_grad):
    a = simple_tensor_grad()
    assert a.shape == (3,)


@pytest.mark.parametrize(("index", "expected"), [(None, (3,)), (0, 3)])
def test_tensor_size(simple_tensor_grad, index: Optional[int], expected: Any):
    a = simple_tensor_grad()
    assert a.size(index) == expected


def test_tensor_tolist(simple_tensor_nograd):
    a = simple_tensor_nograd()
    assert a.tolist() == [10, 10, 10]


def test_tensor_numpy(simple_tensor_nograd):
    a = simple_tensor_nograd()
    assert np.allclose(a.numpy(), np.array([10, 10, 10]))


def test_tensor_to(simple_tensor_nograd):
    a = simple_tensor_nograd()
    a = a.to(cortex.float64)
    assert np.allclose(a.data, np.array([10.0, 10.0, 10.0]))


@pytest.mark.parametrize("set_to_none", [True, False])
def test_tensor_zero_grad(simple_tensor_grad, set_to_none: bool):
    a = simple_tensor_grad()

    a.backward(grad=np.array([10, 10, 10]))
    assert np.allclose(a.grad.data, np.array([10, 10, 10]))

    a.zero_grad(set_to_none=set_to_none)
    if set_to_none:
        assert a.grad is None
    else:
        assert np.allclose(a.grad, np.zeros_like(a.data))


def test_tensor_zero_grad_tree(simple_tensor_grad):
    a = simple_tensor_grad()
    b = simple_tensor_grad()

    x = a + b
    x.backward()

    x.zero_grad_tree()

    assert np.allclose(x.grad, np.zeros_like(x.data))
    assert np.allclose(a.grad, np.zeros_like(a.data))
    assert np.allclose(b.grad, np.zeros_like(b.data))
    assert np.allclose(b.grad, np.zeros_like(b.data))


def test_zeros():
    x = cortex.zeros(2, 5, 6)
    assert np.allclose(x.data, np.zeros((2, 5, 6)))


def test_ones():
    x = cortex.ones(2, 5, 6)
    assert np.allclose(x.data, np.ones((2, 5, 6)))


def test_empty():
    x = cortex.empty(2, 5, 6)
    assert np.allclose(x.data, np.empty((2, 5, 6)))


def test_rand():
    np.random.seed(1)
    x = cortex.rand(2, 5, 6)
    np.random.seed(1)
    assert np.allclose(x.data, np.random.rand(2, 5, 6))


def test_randn():
    np.random.seed(1)
    x = cortex.randn(2, 5, 6)
    np.random.seed(1)
    assert np.allclose(x.data, np.random.randn(2, 5, 6))


def test_randint():
    np.random.seed(1)
    x = cortex.randint(0, 10, (2, 5, 6))
    np.random.seed(1)
    assert np.allclose(x.data, np.random.randint(0, 10, (2, 5, 6)))
