from __future__ import annotations

import math

import numpy as np
import pytest

from cortex import Tensor, cat, stack

###########################################
################ ADD ######################
###########################################


def test_add_tensors(simple_tensor_nograd):
    a = simple_tensor_nograd()
    b = simple_tensor_nograd()

    output = a + b
    assert np.allclose(output.data, np.array([20, 20, 20]))
    assert np.allclose(a.data, np.array([10, 10, 10]))
    assert np.allclose(b.data, np.array([10, 10, 10]))

    output.backward()

    assert output.grad is None
    assert a.grad is None
    assert b.grad is None


def test_add_tensors_grad(simple_tensor_grad):
    a = simple_tensor_grad()
    b = simple_tensor_grad()

    output = a + b
    assert np.allclose(output.data, np.array([20, 20, 20]))
    assert np.allclose(a.data, np.array([10, 10, 10]))
    assert np.allclose(b.data, np.array([10, 10, 10]))

    output.backward(grad=np.array([5, 5, 5]))

    assert np.allclose(output.grad, np.array([5, 5, 5]))
    assert np.allclose(a.grad, np.array([5, 5, 5]))
    assert np.allclose(b.grad, np.array([5, 5, 5]))


def test_add_tensors_grad_nograd(simple_tensor_grad, simple_tensor_nograd):
    a = simple_tensor_grad()
    b = simple_tensor_nograd()

    output = a + b
    assert np.allclose(output.data, np.array([20, 20, 20]))
    assert np.allclose(a.data, np.array([10, 10, 10]))
    assert np.allclose(b.data, np.array([10, 10, 10]))

    output.backward(grad=np.array([5, 5, 5]))

    assert np.allclose(output.grad, np.array([5, 5, 5]))
    assert np.allclose(a.grad, np.array([5, 5, 5]))
    assert b.grad is None


###########################################
################ NEG ######################
###########################################


def test_neg_tensors(simple_tensor_nograd):
    a = simple_tensor_nograd()

    output = -a
    assert np.allclose(output.data, np.array([-10, -10, -10]))
    assert np.allclose(a.data, np.array([10, 10, 10]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_neg_tensors_grad(simple_tensor_grad):
    a = simple_tensor_grad()

    output = -a
    assert np.allclose(output.data, np.array([-10, -10, -10]))
    assert np.allclose(a.data, np.array([10, 10, 10]))

    output.backward(grad=np.array([5, 5, 5]))

    assert np.allclose(output.grad, np.array([5, 5, 5]))
    assert np.allclose(a.grad, np.array([-5, -5, -5]))


###########################################
################ SUB ######################
###########################################


def test_sub_tensors(simple_tensor_nograd):
    a = simple_tensor_nograd()
    b = simple_tensor_nograd()

    output = a - b
    assert np.allclose(output.data, np.array([0, 0, 0]))
    assert np.allclose(a.data, np.array([10, 10, 10]))
    assert np.allclose(b.data, np.array([10, 10, 10]))

    output.backward()

    assert output.grad is None
    assert a.grad is None
    assert b.grad is None


def test_sub_tensors_grad(simple_tensor_grad):
    a = simple_tensor_grad()
    b = simple_tensor_grad()

    output = a - b
    assert np.allclose(output.data, np.array([0, 0, 0]))
    assert np.allclose(a.data, np.array([10, 10, 10]))
    assert np.allclose(b.data, np.array([10, 10, 10]))

    output.backward(grad=np.array([5, 5, 5]))

    assert np.allclose(output.grad, np.array([5, 5, 5]))
    assert np.allclose(a.grad, np.array([5, 5, 5]))
    assert np.allclose(b.grad, np.array([-5, -5, -5]))


def test_sub_tensors_grad_nograd(simple_tensor_grad, simple_tensor_nograd):
    a = simple_tensor_grad()
    b = simple_tensor_nograd()

    output = a - b
    assert np.allclose(output.data, np.array([0, 0, 0]))
    assert np.allclose(a.data, np.array([10, 10, 10]))
    assert np.allclose(b.data, np.array([10, 10, 10]))

    output.backward(grad=np.array([5, 5, 5]))

    assert np.allclose(output.grad, np.array([5, 5, 5]))
    assert np.allclose(a.grad, np.array([5, 5, 5]))
    assert b.grad is None


###########################################
################ MUL ######################
###########################################


def test_mul_tensors(simple_tensor_nograd):
    a = simple_tensor_nograd()
    b = Tensor([1, 2, 3], requires_grad=False)

    output = a * b
    assert np.allclose(output.data, np.array([10, 20, 30]))
    assert np.allclose(a.data, np.array([10, 10, 10]))
    assert np.allclose(b.data, np.array([1, 2, 3]))

    output.backward()

    assert output.grad is None
    assert a.grad is None
    assert b.grad is None


def test_mul_tensors_grad(simple_tensor_grad):
    a = simple_tensor_grad()
    b = Tensor([1, 2, 3], requires_grad=True)

    output = a * b
    assert np.allclose(output.data, np.array([10, 20, 30]))
    assert np.allclose(a.data, np.array([10, 10, 10]))
    assert np.allclose(b.data, np.array([1, 2, 3]))

    output.backward(grad=np.array([2, 2, 2]))

    assert np.allclose(output.grad, np.array([2, 2, 2]))
    assert np.allclose(a.grad, np.array([2, 4, 6]))
    assert np.allclose(b.grad, np.array([20, 20, 20]))


def test_mul_tensors_grad_nograd(simple_tensor_grad, simple_tensor_nograd):
    a = simple_tensor_grad()
    b = Tensor([1, 2, 3], requires_grad=False)

    output = a * b
    assert np.allclose(output.data, np.array([10, 20, 30]))
    assert np.allclose(a.data, np.array([10, 10, 10]))
    assert np.allclose(b.data, np.array([1, 2, 3]))

    output.backward(grad=np.array([2, 2, 2]))

    assert np.allclose(output.grad, np.array([2, 2, 2]))
    assert np.allclose(a.grad, np.array([2, 4, 6]))
    assert b.grad is None


###########################################
################ DIV ######################
###########################################


def test_div_tensors():
    a = Tensor([8.0, 12.0, 9.0], requires_grad=False)
    b = Tensor([1.0, 2.0, 3.0], requires_grad=False)

    output = a / b
    assert np.allclose(output.data, np.array([8.0, 6.0, 3.0]))
    assert np.allclose(a.data, np.array([8.0, 12.0, 9.0]))
    assert np.allclose(b.data, np.array([1.0, 2.0, 3.0]))

    output.backward()

    assert output.grad is None
    assert a.grad is None
    assert b.grad is None


def test_div_tensors_grad():
    a = Tensor([8.0, 12.0, 9.0], requires_grad=True)
    b = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    output = a / b
    assert np.allclose(output.data, np.array([8.0, 6.0, 3.0]))
    assert np.allclose(a.data, np.array([8.0, 12.0, 9.0]))
    assert np.allclose(b.data, np.array([1.0, 2.0, 3.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad, np.array([1.0, 1 / 2, 1 / 3]))
    assert np.allclose(b.grad, np.array([-8.0, -3.0, -1.0]))


def test_div_tensors_grad_nograd():
    a = Tensor([8.0, 12.0, 9.0], requires_grad=True)
    b = Tensor([1.0, 2.0, 3.0], requires_grad=False)

    output = a / b
    assert np.allclose(output.data, np.array([8.0, 6.0, 3.0]))
    assert np.allclose(a.data, np.array([8.0, 12.0, 9.0]))
    assert np.allclose(b.data, np.array([1.0, 2.0, 3.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad, np.array([1.0, 1 / 2, 1 / 3]))
    assert b.grad is None


###########################################
################ Transpose ################
###########################################


def test_transpose_tensors():
    a = Tensor([[8.0, 5.0], [12.0, 9.0]], requires_grad=False)

    output = a.T
    assert np.allclose(output.data, np.array([[8.0, 12.0], [5.0, 9.0]]))
    assert np.allclose(a.data, np.array([[8.0, 5.0], [12.0, 9.0]]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_transpose_tensors_grad():
    a = Tensor([[8.0, 5.0, 6.0], [12.0, 9.0, 2.0]], requires_grad=True)

    output = a.T
    assert np.allclose(output.data, np.array([[8.0, 12.0], [5.0, 9.0], [6.0, 2.0]]))
    assert np.allclose(a.data, np.array([[8.0, 5.0, 6.0], [12.0, 9.0, 2.0]]))

    output.backward()

    assert np.allclose(output.grad, np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))
    assert np.allclose(a.grad, np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))


###########################################
################ MatMul ###################
###########################################


def test_matmul_tensors():
    a = Tensor([[8.0, 5.0], [12.0, 9.0]], requires_grad=False)
    b = Tensor([[8.0, 2.0], [12.0, 3.0]], requires_grad=False)

    output = a @ b
    assert np.allclose(output.data, np.array([[124.0, 31.0], [204.0, 51.0]]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_matmul_tensors_grad():
    a = Tensor([[8.0, 5.0], [12.0, 9.0]], requires_grad=True)
    b = Tensor([[8.0, 2.0], [12.0, 3.0]], requires_grad=True)

    output = a @ b
    assert np.allclose(output.data, np.array([[124.0, 31.0], [204.0, 51.0]]))

    output.backward()

    out_grad = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(output.grad, out_grad)
    assert np.allclose(a.grad, out_grad @ b.data.T)
    assert np.allclose(b.grad, a.data.T @ out_grad)


###########################################
################ Exp ######################
###########################################


def test_exp_tensors():
    a = Tensor([10.0, 10.0, 10.0], requires_grad=False)

    output = a.exp()
    assert np.allclose(
        output.data, np.array([math.exp(10.0), math.exp(10.0), math.exp(10.0)])
    )

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_exp_tensors_grad():
    a = Tensor([10.0, 10.0, 10.0], requires_grad=True)

    output = a.exp()
    assert np.allclose(
        output.data, np.array([math.exp(10.0), math.exp(10.0), math.exp(10.0)])
    )

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(
        a.grad,
        np.array([1.0, 1.0, 1.0])
        * np.array([math.exp(10.0), math.exp(10.0), math.exp(10.0)]),
    )


###########################################
################ Pow ######################
###########################################


def test_pow_tensors():
    a = Tensor([10.0, 10.0, 10.0], requires_grad=False)

    output = a**2
    assert np.allclose(output.data, np.array([100.0, 100.0, 100.0]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_pow_tensors_grad():
    a = Tensor([10.0, 10.0, 10.0], requires_grad=True)

    output = a**2
    assert np.allclose(output.data, np.array([100.0, 100.0, 100.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(
        a.grad,
        np.array([20.0, 20.0, 20.0]),
    )


###########################################
################ Log ######################
###########################################


def test_log_tensors():
    a = Tensor([math.exp(2), math.exp(2), math.exp(2)], requires_grad=False)

    output = a.log()
    assert np.allclose(output.data, np.array([2.0, 2.0, 2.0]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_log_tensors_grad():
    a = Tensor([math.exp(2), math.exp(2), math.exp(2)], requires_grad=True)

    output = a.log()
    assert np.allclose(output.data, np.array([2.0, 2.0, 2.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(
        a.grad,
        np.array([1.0 / math.exp(2), 1.0 / math.exp(2), 1.0 / math.exp(2)]),
    )


###########################################
################ Sqrt #####################
###########################################


def test_sqrt_tensors():
    a = Tensor([100.0, 100.0, 100.0], requires_grad=False)

    output = a.sqrt()
    assert np.allclose(output.data, np.array([10.0, 10.0, 10.0]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_sqrt_tensors_grad():
    a = Tensor([100.0, 100.0, 100.0], requires_grad=True)

    output = a.sqrt()
    assert np.allclose(output.data, np.array([10.0, 10.0, 10.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad, np.array([0.05, 0.05, 0.05]))


###########################################
################ Max #####################
###########################################


def test_max_tensors_not_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=False)

    output = a.max(1)
    assert np.allclose(output.data, np.array([6.0, 12.0]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_max_tensors_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=False)

    output = a.max(1, keepdim=True)
    assert np.allclose(output.data, np.array([[6.0], [12.0]]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_max_tensors_grad_not_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=True)

    output = a.max(1)
    assert np.allclose(output.data, np.array([6.0, 12.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0]))
    assert np.allclose(a.grad, np.array([[0.0, 0, 1.0], [0.0, 1.0, 0.0]]))


def test_max_tensors_grad_tie():
    a = Tensor([[6.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=True)

    output = a.max(1)
    assert np.allclose(output.data, np.array([6.0, 12.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0]))
    assert np.allclose(a.grad, np.array([[0.5, 0, 0.5], [0.0, 1.0, 0.0]]))


def test_max_tensors_grad_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=True)

    output = a.max(1, keepdim=True)
    assert np.allclose(output.data, np.array([[6.0], [12.0]]))

    output.backward()

    assert np.allclose(output.grad, np.array([[1.0], [1.0]]))
    assert np.allclose(a.grad, np.array([[0.0, 0, 1.0], [0.0, 1.0, 0.0]]))


###########################################
################ Min #####################
###########################################


def test_min_tensors_not_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=False)

    output = a.min(1)
    assert np.allclose(output.data, np.array([1.0, 2.0]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_min_tensors_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=False)

    output = a.min(1, keepdim=True)
    assert np.allclose(output.data, np.array([[1.0], [2.0]]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_min_tensors_grad_not_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=True)

    output = a.min(1)
    assert np.allclose(output.data, np.array([1.0, 2.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0]))
    assert np.allclose(a.grad, np.array([[1.0, 0, 0.0], [1.0, 0.0, 0.0]]))


def test_min_tensors_grad_tie():
    a = Tensor([[6.0, 5.0, 5.0], [2.0, 12.0, 3.0]], requires_grad=True)

    output = a.min(1)
    assert np.allclose(output.data, np.array([5.0, 2.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0]))
    assert np.allclose(a.grad, np.array([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]]))


def test_min_tensors_grad_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=True)

    output = a.min(1, keepdim=True)
    assert np.allclose(output.data, np.array([[1.0], [2.0]]))

    output.backward()

    assert np.allclose(output.grad, np.array([[1.0], [1.0]]))
    assert np.allclose(a.grad, np.array([[1.0, 0, 0.0], [1.0, 0.0, 0.0]]))


###########################################
################ Slice ####################
###########################################


@pytest.mark.parametrize(
    ("idx", "expected_output"), [(0, np.array(1.0)), ([0, 1], np.array([1.0, 2.0]))]
)
def test_slice_tensors(idx, expected_output: np.ndarray):
    a = Tensor([1.0, 2.0, 3.0], requires_grad=False)

    output = a[idx]
    assert np.allclose(output.data, expected_output)

    output.backward()

    assert output.grad is None
    assert a.grad is None


@pytest.mark.parametrize(
    ("idx", "expected_output", "out_grad", "in_grad"),
    [
        (0, np.array(1.0), np.array(1.0), np.array([1.0, 0.0, 0.0])),
        ([0, 1], np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([1.0, 1.0, 0.0])),
    ],
)
def test_slice_tensors_grad(
    idx, expected_output: np.ndarray, out_grad: np.ndarray, in_grad: np.ndarray
):
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    output = a[idx]
    assert np.allclose(output.data, expected_output)

    output.backward()

    assert np.allclose(output.grad, out_grad)
    assert np.allclose(a.grad, in_grad)


###########################################
################ Reshape ##################
###########################################


@pytest.mark.parametrize(
    ("shape", "expected_output"),
    [
        ((2, 2), np.array([[1.0, 2.0], [3.0, 4.0]])),
        ([1, 4], np.array([[1.0, 2.0, 3.0, 4.0]])),
    ],
)
def test_reshape_tensors(shape: tuple[int], expected_output: np.ndarray):
    a = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=False)

    output = a.reshape(*shape)
    assert np.allclose(output.data, expected_output)

    output.backward()

    assert output.grad is None
    assert a.grad is None


@pytest.mark.parametrize(
    ("shape", "expected_output", "out_grad", "in_grad"),
    [
        (
            (2, 2),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([1.0, 1.0, 1.0, 1.0]),
        ),
        (
            [1, 4],
            np.array([[1.0, 2.0, 3.0, 4.0]]),
            np.array([[1.0, 1.0, 1.0, 1.0]]),
            np.array([1.0, 1.0, 1.0, 1.0]),
        ),
    ],
)
def test_reshape_tensors_grad(
    shape: tuple[int],
    expected_output: np.ndarray,
    out_grad: np.ndarray,
    in_grad: np.ndarray,
):
    a = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)

    output = a.reshape(*shape)
    assert np.allclose(output.data, expected_output)

    output.backward()

    assert np.allclose(output.grad, out_grad)
    assert np.allclose(a.grad, in_grad)


###########################################
################ MaskSet ##################
###########################################


@pytest.mark.parametrize(
    ("mask", "expected_output"),
    [
        (np.array([True, False, False]), np.array([0.0, 2.0, 3.0])),
        (np.array([False, False, False]), np.array([1.0, 2.0, 3.0])),
    ],
)
def test_set_tensors(mask: np.ndarray, expected_output: np.ndarray):
    a = Tensor([1.0, 2.0, 3.0], requires_grad=False)

    output = a.set(mask, 0)
    assert np.allclose(output.data, expected_output)

    output.backward()

    assert output.grad is None
    assert a.grad is None


@pytest.mark.parametrize(
    ("mask", "expected_output", "in_grad", "out_grad"),
    [
        (
            np.array([True, False, False]),
            np.array([0.0, 2.0, 3.0]),
            np.array([0.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
        ),
        (
            np.array([True, True, True]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        ),
    ],
)
def test_set_tensors(
    mask: np.ndarray,
    expected_output: np.ndarray,
    in_grad: np.ndaray,
    out_grad: np.ndarray,
):
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    output = a.set(mask, 0)
    assert np.allclose(output.data, expected_output)

    output.backward()

    assert np.allclose(output.grad, out_grad)
    assert np.allclose(a.grad, in_grad)


###########################################
################ Sum ######################
###########################################


def test_sum_tensors_not_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=False)

    output = a.sum(1)
    assert np.allclose(output.data, np.array([12.0, 17.0]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_sum_tensors_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=False)

    output = a.sum(1, keepdim=True)
    assert np.allclose(output.data, np.array([[12.0], [17.0]]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_sum_tensors_grad_not_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=True)

    output = a.sum(1)
    assert np.allclose(output.data, np.array([12.0, 17.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0]))
    assert np.allclose(a.grad, np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))


def test_sum_tensors_grad_keepdim():
    a = Tensor([[1.0, 5.0, 6.0], [2.0, 12.0, 3.0]], requires_grad=True)

    output = a.sum(1, keepdim=True)
    assert np.allclose(output.data, np.array([[12.0], [17.0]]))

    output.backward()

    assert np.allclose(output.grad, np.array([[1.0], [1.0]]))
    assert np.allclose(a.grad, np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))


###########################################
################ Mean #####################
###########################################


def test_mean_tensors_not_keepdim():
    a = Tensor([[1.0, 3.0, 5.0], [6.0, 12.0, 6.0]], requires_grad=False)

    output = a.mean(1)
    assert np.allclose(output.data, np.array([3.0, 8.0]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_mean_tensors_keepdim():
    a = Tensor([[1.0, 3.0, 5.0], [6.0, 12.0, 6.0]], requires_grad=False)

    output = a.mean(1, keepdim=True)
    assert np.allclose(output.data, np.array([[3.0], [8.0]]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_mean_tensors_grad_not_keepdim():
    a = Tensor([[1.0, 3.0, 5.0], [6.0, 12.0, 6.0]], requires_grad=True)

    output = a.mean(1)
    assert np.allclose(output.data, np.array([3.0, 8.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0]))
    assert np.allclose(a.grad, np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]))


def test_mean_tensors_grad_keepdim():
    a = Tensor([[1.0, 3.0, 5.0], [6.0, 12.0, 6.0]], requires_grad=True)

    output = a.mean(1, keepdim=True)
    assert np.allclose(output.data, np.array([[3.0], [8.0]]))

    output.backward()

    assert np.allclose(output.grad, np.array([[1.0], [1.0]]))
    assert np.allclose(a.grad, np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]))


###########################################
################ Var ######################
###########################################


def test_var_tensors_not_keepdim():
    a = Tensor([[1.0, 3.0, 5.0], [6.0, 12.0, 6.0]], requires_grad=False)

    output = a.var(1)
    assert np.allclose(output.data, np.array([8 / 3, 8.0]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_var_tensors_keepdim():
    a = Tensor([[1.0, 3.0, 5.0], [6.0, 12.0, 6.0]], requires_grad=False)

    output = a.var(1, keepdim=True)
    assert np.allclose(output.data, np.array([[8 / 3], [8.0]]))

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_var_tensors_grad_not_keepdim():
    a = Tensor([[1.0, 3.0, 5.0], [6.0, 12.0, 6.0]], requires_grad=True)

    output = a.var(1)
    assert np.allclose(output.data, np.array([8 / 3, 8.0]))

    output.backward()

    assert np.allclose(output.grad, np.array([1.0, 1.0]))
    assert np.allclose(
        a.grad, np.array([[-4 / 3, 0.0, 4 / 3], [-4 / 3, 8 / 3, -4 / 3]])
    )


def test_var_tensors_grad_keepdim():
    a = Tensor([[1.0, 3.0, 5.0], [6.0, 12.0, 6.0]], requires_grad=True)

    output = a.var(1, keepdim=True)
    assert np.allclose(output.data, np.array([[8 / 3], [8.0]]))

    output.backward()

    assert np.allclose(output.grad, np.array([[1.0], [1.0]]))
    assert np.allclose(
        a.grad, np.array([[-4 / 3, 0.0, 4 / 3], [-4 / 3, 8 / 3, -4 / 3]])
    )


###########################################
################ Cat ######################
###########################################


def test_cat_tensors():
    a = Tensor([1.0, 1.0, 1.0], requires_grad=False)
    b = Tensor([2.0, 2.0, 2.0], requires_grad=False)
    c = Tensor([3.0, 3.0, 3.0], requires_grad=False)

    output = cat([a, b, c], 0)
    assert np.allclose(
        output.data, np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    )

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_cat_tensors_grad():
    a = Tensor([1.0, 1.0, 1.0], requires_grad=True)
    b = Tensor([2.0, 2.0, 2.0], requires_grad=True)
    c = Tensor([3.0, 3.0], requires_grad=True)

    output = cat([a, b, c], 0)
    assert np.allclose(output.data, np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]))

    output.backward(grad=np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]))

    assert np.allclose(output.grad, np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0]))
    assert np.allclose(a.grad, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(b.grad, np.array([2.0, 2.0, 2.0]))
    assert np.allclose(c.grad, np.array([3.0, 3.0]))


###########################################
################ Stack ####################
###########################################


def test_stack_tensors():
    a = Tensor([1.0, 1.0, 1.0], requires_grad=False)
    b = Tensor([2.0, 2.0, 2.0], requires_grad=False)
    c = Tensor([3.0, 3.0, 3.0], requires_grad=False)

    output = stack([a, b, c], 0)
    assert np.allclose(
        output.data, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    )

    output.backward()

    assert output.grad is None
    assert a.grad is None


def test_stack_tensors_grad():
    a = Tensor([1.0, 1.0, 1.0], requires_grad=True)
    b = Tensor([2.0, 2.0, 2.0], requires_grad=True)
    c = Tensor([3.0, 3.0, 3.0], requires_grad=True)

    output = stack([a, b, c], 0)
    assert np.allclose(
        output.data, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    )

    output.backward(grad=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))

    assert np.allclose(
        output.grad, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    )
    assert np.allclose(a.grad, np.array([1.0, 1.0, 1.0]))
    assert np.allclose(b.grad, np.array([2.0, 2.0, 2.0]))
    assert np.allclose(c.grad, np.array([3.0, 3.0, 3.0]))
