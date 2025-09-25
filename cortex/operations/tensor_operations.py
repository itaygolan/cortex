from __future__ import annotations

from collections.abc import Iterable as IterableObj
from typing import Any, Iterable, Optional, Union

import numpy as np

from cortex.operations.operation import Operation
from cortex.tensor import Tensor


class Add(Operation):
    """Element wise addition.

    Forward:
    z = a + b

    Backward:
    dz/da = grad
    dz/db = grad
    """

    def forward(self, a: Tensor, b: Tensor):
        requires_grad = a.requires_grad or b.requires_grad
        add_data = a.data + b.data
        output = Tensor(add_data, requires_grad=requires_grad, operation=self)

        # Add `output` as dependency of a and b, as to compute the gradient of a and b
        # the gradient of output must be computed first.
        a.children.append(output)
        b.children.append(output)

        # Save inputs for backward pass
        self.saved_inputs = (a, b)
        self.parents = (a, b)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, b = self.saved_inputs

        # Add gradient is equal to grad
        if a.requires_grad:
            grad_a = _ensure_grad_dim(grad, a.data)
            a.backward(output, grad_a)

        if b.requires_grad:
            grad_b = _ensure_grad_dim(grad, b.data)
            b.backward(output, grad_b)


class Neg(Operation):
    """Negation.

    Forward:
    z = -a

    Backward:
    dz/da = -grad
    """

    def forward(self, a: Tensor):
        requires_grad = a.requires_grad
        neg_data = -a.data
        output = Tensor(neg_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)
        self.saved_inputs = a
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a = self.saved_inputs

        # Neg gradient is -grad
        if a.requires_grad:
            a.backward(output, -grad)


class Mul(Operation):
    """Element wise multiplication.

    Forward:
    z = a * b

    Backward:
    dz/da = grad * b
    dz/db = grad * a


    """

    def forward(self, a: Tensor, b: Tensor):
        requires_grad = a.requires_grad or b.requires_grad
        mul_data = a.data * b.data
        output = Tensor(mul_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)
        b.children.append(output)

        self.saved_inputs = (a, b)
        self.parents = (a, b)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, b = self.saved_inputs

        if a.requires_grad:
            grad_a = _ensure_grad_dim(grad * b.data, a.data)
            a.backward(output, grad_a)

        if b.requires_grad:
            grad_b = _ensure_grad_dim(grad * a.data, b.data)
            b.backward(output, grad_b)


class Div(Operation):
    """Element wise division.

    Forward:
    z = a / b

    Backward:
    dz/da = grad * (b^-1)
    dz/db = -grad * a * (b^-2)

    """

    def forward(self, a: Tensor, b: Tensor):
        requires_grad = a.requires_grad or b.requires_grad
        div_data = a.data / b.data
        output = Tensor(div_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)
        b.children.append(output)

        self.saved_inputs = (a, b)
        self.parents = (a, b)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, b = self.saved_inputs

        if a.requires_grad:
            da = grad * (1 / b.data)
            grad_a = _ensure_grad_dim(da, a.data)
            a.backward(output, grad_a)

        if b.requires_grad:
            db = grad * (-a.data / (b.data**2))
            grad_b = _ensure_grad_dim(db, b.data)
            b.backward(output, grad_b)


class Transpose(Operation):
    """Transpose

    Forward:
    z = A.T

    Backward:
    dz/da = A.T
    """

    def forward(self, a: Tensor, *dims: int):
        requires_grad = a.requires_grad
        transpose_data = np.swapaxes(a.data, *dims)
        output = Tensor(transpose_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, dims)
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, dims = self.saved_inputs

        if a.requires_grad:
            a.backward(output, np.swapaxes(grad, *dims))


class MatMul(Operation):
    """Matrix multiplication

    Forward:
    z = A @ B
        - A: [m, n]
        - B: [n, p]
        - Z: [m, p]

    Backward:
    dz/da = G @ B^T
    dz/db = A^T @ G
    """

    def forward(self, a: Tensor, b: Tensor):
        requires_grad = a.requires_grad or b.requires_grad
        matmul_data = np.dot(a.data, b.data)
        output = Tensor(matmul_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)
        b.children.append(output)

        self.saved_inputs = (a, b)
        self.parents = (a, b)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, b = self.saved_inputs

        if a.requires_grad:
            da = np.dot(grad, b.data.T)
            grad_a = _ensure_grad_dim(da, a.data)
            a.backward(output, grad_a)

        if b.requires_grad:
            db = np.dot(a.data.T, grad)
            grad_b = _ensure_grad_dim(db, b.data)
            b.backward(output, grad_b)


class Exp(Operation):
    """Exponent

    Forward:
    z = e^a

    Backward:
    dz/da = grad * z
    """

    def forward(self, a: Tensor):
        requires_grad = a.requires_grad
        exp_data = np.exp(a.data)
        output = Tensor(exp_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, exp_data)
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, exp_data = self.saved_inputs

        if a.requires_grad:
            a.backward(output, grad * exp_data)


class Pow(Operation):
    """Power

    Forward:
    z = a^b

    Backward:
    dz/da = grad * b * a ^ {b-1}
    """

    def forward(self, a: Tensor, b: Tensor):
        requires_grad = a.requires_grad
        pow_data = np.pow(a.data, b.data)
        output = Tensor(pow_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, b)
        self.parents = (a, b)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, b = self.saved_inputs

        if a.requires_grad:
            da = b.data * np.pow(a.data, b.data - 1)
            a.backward(output, grad * da)


class Log(Operation):
    """Log

    Forward:
    z = log(a)

    Backward:
    dz/da = 1 / a
    """

    def forward(self, a: Tensor):
        requires_grad = a.requires_grad
        log_data = np.log(a.data)
        output = Tensor(log_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = a
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a = self.saved_inputs

        if a.requires_grad:
            a.backward(output, grad * (1 / a.data))


class Sqrt(Operation):
    """Square root

    Forward:
    z = sqrt(a) = a^(1/2)

    Backward:
    dz/da = grad * 0.5 * (a ^ -0.5)
    """

    def forward(self, a: Tensor):
        requires_grad = a.requires_grad
        sqrt_data = np.sqrt(a.data)
        output = Tensor(sqrt_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = a
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a = self.saved_inputs

        if a.requires_grad:
            da = 0.5 * np.pow(a.data, -0.5)
            a.backward(output, grad * da)


class Max(Operation):
    """Max over a dimension

    Forward:
    z = max(a, dim)

    Backward:
    dz/da = grad * (a == max)
    """

    def forward(self, a: Tensor, dim: Union[int, Iterable[int]], keepdim: bool):
        requires_grad = a.requires_grad
        max_data_dim = np.max(a.data, axis=dim, keepdims=True)
        max_data = np.squeeze(max_data_dim, axis=dim) if not keepdim else max_data_dim
        output = Tensor(max_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, max_data_dim, keepdim, dim)
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, max_data, keepdim, dim = self.saved_inputs

        if a.requires_grad:
            # Expand grad to be same shape as input
            grad_expanded = np.expand_dims(grad, axis=dim) if not keepdim else grad
            mask = a.data == max_data

            # Distribute gradient evenly if multiple maxes
            count_max = np.sum(mask, axis=dim, keepdims=True)
            scaled = mask.astype(a.dtype) / count_max
            a.backward(output, grad_expanded * scaled)


class Min(Operation):
    """Min over a dimension

    Forward:
    z = min(a, dim)

    Backward:
    dz/da = grad * (a == min)
    """

    def forward(self, a: Tensor, dim: Union[int, Iterable[int]], keepdim: bool):
        requires_grad = a.requires_grad
        min_data_dim = np.min(a.data, axis=dim, keepdims=True)
        min_data = np.squeeze(min_data_dim, axis=dim) if not keepdim else min_data_dim
        output = Tensor(min_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, min_data_dim, keepdim, dim)
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, min_data, keepdim, dim = self.saved_inputs

        if a.requires_grad:
            # Expand grad to be same shape as input
            grad_expanded = np.expand_dims(grad, axis=dim) if not keepdim else grad
            mask = a.data == min_data

            # Distribute gradient evenly if multiple mins
            count_min = np.sum(mask, axis=dim, keepdims=True)
            scaled = mask.astype(a.dtype) / count_min
            a.backward(output, grad_expanded * scaled)


class Slice(Operation):
    """
    Indexing/Slice operation

    Forward:
    z = a[idx]

    Backward:
    dz/da = grad if in idx else 0
    """

    def forward(self, a: Tensor, idx: Union[int, Iterable]):
        requires_grad = a.requires_grad

        # Convert any sequences to numpy if cortex
        if isinstance(idx, Tensor):
            idx = idx.numpy()
        elif isinstance(idx, tuple):
            new_idx = []
            for item in idx:
                if isinstance(item, Tensor):
                    new_idx.append(item.numpy())
                else:
                    new_idx.append(item)
            idx = tuple(new_idx)

        slice_data = a.data[idx]
        output = Tensor(slice_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, idx)
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, idx = self.saved_inputs

        if a.requires_grad:
            da = np.zeros_like(a.data)
            da[idx] = grad
            a.backward(output, da)


class MaskSet(Operation):
    """
    Indexing/set operation

    Forward:
    a[mask] = val

    Backward:
    dz/da = grad if idx != mask else 0
    """

    def forward(self, a: Tensor, mask: Tensor, value: Tensor):
        requires_grad = a.requires_grad

        new_data = a.data.copy()
        new_data[mask.data] = value.data
        output = Tensor(new_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, mask)
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, mask = self.saved_inputs

        if a.requires_grad:
            da = grad.copy()
            # Overwritten values do not recieve a gradient
            # This is because these values are no longer dependent on the output
            # they are always masked to a given value. Thus, they have a gradient of 0.
            da[mask.data] = 0
            a.backward(output, da)


class Reshape(Operation):
    """
    Indexing/Slice operation

    Forward:
    z = a.reshape(...)

    Backward:
    dz/da = grad.reshape(...)
    """

    def forward(self, a: Tensor, *shape: int):
        requires_grad = a.requires_grad
        reshape_data = a.data.reshape(shape)
        output = Tensor(reshape_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = a
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a = self.saved_inputs

        if a.requires_grad:
            a.backward(output, grad.reshape(a.shape))


class Compare(Operation):
    """Comparison operator.

    Comparisons are not differentiable.
    """

    def forward(self, a: Tensor, other: Tensor, op: callable):
        compare_data = op(a.data, other.data)
        return Tensor(compare_data, requires_grad=False, operation=self)

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        # No gradient can be propagated
        return


class Sum(Operation):
    """
    Sum over a dimension

    Forward:
    z = sum(a, dim)

    Backward:

    dz/da = grad

    Why?
    Let z = a + b + ... +
    dz/da = 1
    """

    def forward(self, a: Tensor, dim: Union[int, Iterable[int]], keepdim: bool):
        requires_grad = a.requires_grad
        sum_data_dim = np.sum(a.data, axis=dim, keepdims=True)
        sum_data = np.squeeze(sum_data_dim, axis=dim) if not keepdim else sum_data_dim
        output = Tensor(sum_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, keepdim, dim)
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, keepdim, dim = self.saved_inputs

        if a.requires_grad:
            # Expand grad to be same shape as input
            # The then `sum` gradient has each input have the same gradient
            grad_expanded = np.expand_dims(grad, axis=dim) if not keepdim else grad
            a.backward(output, grad_expanded)


class Mean(Operation):
    """
    Mean over a dimension

    Forward:
    z = mean(a, dim)

    Backward:

    dz/da = grad / num_elements
    """

    def forward(self, a: Tensor, dim: Union[int, Iterable[int]], keepdim: bool):
        requires_grad = a.requires_grad
        mean_data_dim = np.mean(a.data, axis=dim, keepdims=True)
        mean_data = (
            np.squeeze(mean_data_dim, axis=dim) if not keepdim else mean_data_dim
        )
        output = Tensor(mean_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, keepdim, dim)
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, keepdim, dim = self.saved_inputs

        if a.requires_grad:
            # Expand grad to be same shape as input
            grad = grad.copy()
            grad_expanded = np.expand_dims(grad, axis=dim) if not keepdim else grad
            # We divide by the number of elements we are taking the mean over
            # which is the product of the correponding shape of the dims
            grad_expanded /= np.prod(np.take(a.shape, dim))
            a.backward(output, grad_expanded)


class Var(Operation):
    """
    Variance over a dimension

    Forward:
    z = var(a, dim)

    Recall:
    Var = mean [(x - mean(x))**2]

    Backward:

    dz/da = grad * [2 * (x - mean(x)) / num_elements]
    """

    def forward(
        self,
        a: Tensor,
        dim: Union[int, Iterable[int]],
        keepdim: bool,
        ddof: int,
    ):
        requires_grad = a.requires_grad
        var_data_dim = np.var(a.data, axis=dim, ddof=ddof, keepdims=True)
        var_data = np.squeeze(var_data_dim, axis=dim) if not keepdim else var_data_dim
        output = Tensor(var_data, requires_grad=requires_grad, operation=self)

        a.children.append(output)

        self.saved_inputs = (a, keepdim, dim)
        self.parents = (a,)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        a, keepdim, dim = self.saved_inputs

        if a.requires_grad:
            # Expand grad to be same shape as input
            grad = grad.copy()
            grad_expanded = np.expand_dims(grad, axis=dim) if not keepdim else grad

            # Compute variance gradient
            variance_grad = 2 * (a.data - np.mean(a.data, axis=dim, keepdims=True))
            grad_expanded = variance_grad * grad_expanded
            # Then divide the number of elements as we are taking the mean over
            # the vector, which is the product of the correponding shape of the dims
            grad_expanded /= np.prod(np.take(a.shape, dim))
            a.backward(output, grad_expanded)


class Cat(Operation):
    """
    Concatenate along a dimension

    Forward:
    z = concat([x1, x2, ..], 1)

    Backward:

    dz/dx1 = grad[x1_slice]
    """

    def forward(
        self,
        tensors: list[Tensor],
        dim: Union[int, Iterable[int]],
    ):
        requires_grad = any(tensor.requires_grad for tensor in tensors)
        tensor_datas = [tensor.data for tensor in tensors]
        concat_data = np.concatenate(tensor_datas, axis=dim)
        output = Tensor(concat_data, requires_grad=requires_grad, operation=self)

        for tensor in tensors:
            tensor.children.append(output)

        self.saved_inputs = (tensors, dim)
        self.parents = tuple(tensors)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        tensors, dim = self.saved_inputs

        start = 0
        for tensor in tensors:
            if tensor.requires_grad:
                # The gradient of each input is the slice of the output gradient that
                # corresponds to that input
                dim_size = tensor.data.shape[dim]
                indices = list(range(start, start + dim_size))
                tensor_grad = np.take(grad, indices, axis=dim)
                tensor.backward(output, tensor_grad)

            start += dim_size


class Stack(Operation):
    """
    Stack along a dimension

    Forward:
    z = stack([x1, x2, ..], 0)

    Backward:

    dz/dx1 = grad[x1_slice]
    """

    def forward(
        self,
        tensors: list[Tensor],
        dim: Union[int, Iterable[int]],
    ):
        requires_grad = any(tensor.requires_grad for tensor in tensors)
        tensor_datas = [tensor.data for tensor in tensors]
        stack_data = np.stack(tensor_datas, axis=dim)
        output = Tensor(stack_data, requires_grad=requires_grad, operation=self)

        for tensor in tensors:
            tensor.children.append(output)

        self.saved_inputs = (tensors, dim)
        self.parents = tuple(tensors)

        return output

    def backward(
        self,
        output: Optional[Tensor],
        grad: Optional[np.ndarray],
    ):
        tensors, dim = self.saved_inputs

        # Since we stacked the input tensors, we just split it by the number of tensors
        grads = np.split(grad, len(tensors), axis=dim)
        for i, tensor in enumerate(tensors):
            if tensor.requires_grad:
                tensor.backward(output, grads[i].reshape(tensor.data.shape))


def _ensure_grad_dim(grad: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Reduce/squeeze `grad` to have exactly the same shape as `data`.
    This handles general broadcasting semantics:
      - pad `data.shape` on the left with ones to match grad.ndim
      - sum grad over axes where padded_data_shape == 1 and grad.shape > 1
      - finally reshape to exactly data.shape
    Returns a NEW array shaped like `data` (not a view).
    """
    grad = np.asarray(grad)
    data = np.asarray(data)

    # quick path
    if grad.shape == data.shape:
        return grad

    # If grad has fewer dims, pad it on the left so we can compare shapes
    if grad.ndim < data.ndim:
        grad = grad.reshape((1,) * (data.ndim - grad.ndim) + grad.shape)

    # Pad data.shape on the left with ones to match grad.ndim
    padded_data_shape = (1,) * (grad.ndim - data.ndim) + data.shape

    # axes where data had 1 but grad has >1 -> these were broadcasted
    axes = tuple(
        i
        for i, (ds, gs) in enumerate(zip(padded_data_shape, grad.shape))
        if ds == 1 and gs > 1
    )

    if axes:
        # sum over those axes, keepdims so shape becomes ones where needed
        grad = grad.sum(axis=axes, keepdims=True)

    # finally reshape to exactly data.shape (drops the left padding)
    return grad.reshape(data.shape)
