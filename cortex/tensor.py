from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Type, Union

import numpy as np

if TYPE_CHECKING:
    from cortex.operations import Operation


class Tensor:
    def __init__(
        self,
        data: Any,
        dtype: Optional[Type] = None,
        requires_grad: bool = False,
        operation: Operation = None,
    ):
        self.data = np.array(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
        self.operation = operation

        # Children are all tensors computed using the given tensor (self)
        self.children: list[Tensor] = []

        self.grad: Optional[np.ndarray] = None

    def backward(self, z: Optional[Tensor] = None, grad: Optional[np.ndarray] = None):
        """Recursively compute the gradients of the compute graph rooted at the current Tensor."""

        # No backward pass if requires grad is None
        if not self.requires_grad:
            return

        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float64)

        if grad is None:
            # If grad is None, the gradient with respect to itself is 1
            grad = np.ones_like(self.data, dtype=np.float64)

        # Step 1: Set gradient of tensor to passed gradient
        self.grad += grad

        # Remove z from Tensor dependencies as z was computed previously in the last step.
        if z is not None:
            for i, child in enumerate(self.children):
                if child is z:
                    self.children.pop(i)
                    break

        # Step 2: Compute gradients of compute graph
        # If there exists a backward operation AND all children are computed
        # then we can compute the gradient of the inputs to the current tensor
        if self.operation and not self.children:
            self.operation.backward(self, self.grad)

    def __repr__(self):
        tensor_str = f"tensor({self.data})"
        grad_str = f", grad_fn={self.operation}" if self.operation is not None else ""

        return f"{tensor_str}{grad_str}" if self.requires_grad else tensor_str

    def size(self, index: Optional[int] = None):
        shape = self.data.shape
        return shape[index] if index is not None else shape

    @property
    def shape(self):
        return self.size()

    @property
    def ndim(self):
        return self.data.ndim

    def tolist(self):
        return self.data.tolist()

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def to(self, dtype: Type):
        self.data = self.data.astype(dtype)
        self.dtype = self.data.dtype
        return self

    def zero_grad(self, set_to_none: bool = False):
        self.grad = None if set_to_none else np.zeros_like(self.data, dtype=np.float64)

    def zero_grad_tree(self, set_to_none: bool = False):
        """Zero grad the entire compute graph."""

        self.zero_grad(set_to_none)
        if self.operation is not None:
            for inp in self.operation.parents:
                inp.zero_grad_tree(set_to_none)

    def transpose(self, *dims: int):
        from cortex.operations import Transpose

        op = Transpose()
        return op.forward(self, *dims)

    @property
    def T(self):
        return self.transpose(-2, -1)

    def exp(self):
        from cortex.operations import Exp

        op = Exp()
        return op.forward(self)

    def log(self):
        from cortex.operations import Log

        op = Log()
        return op.forward(self)

    def sqrt(self):
        from cortex.operations import Sqrt

        op = Sqrt()
        return op.forward(self)

    def min(self, dim: Optional[int] = None, keepdim: bool = False):
        from cortex.operations import Min

        op = Min()
        if dim is None:
            dim = tuple(range(self.data.ndim))
        return op.forward(self, dim, keepdim)

    def max(self, dim: Optional[int] = None, keepdim: bool = False):
        from cortex.operations import Max

        op = Max()
        if dim is None:
            dim = tuple(range(self.data.ndim))
        return op.forward(self, dim, keepdim)

    def sum(self, dim: Optional[int] = None, keepdim: bool = False):
        from cortex.operations import Sum

        op = Sum()
        if dim is None:
            dim = tuple(range(self.data.ndim))
        return op.forward(self, dim, keepdim)

    def mean(self, dim: Optional[int] = None, keepdim: bool = False):
        from cortex.operations import Mean

        op = Mean()
        if dim is None:
            dim = tuple(range(self.data.ndim))
        return op.forward(self, dim, keepdim)

    def var(
        self,
        dim: Optional[int] = None,
        ddof: int = 0,
        keepdim: bool = False,
    ):
        from cortex.operations import Var

        op = Var()
        if dim is None:
            dim = tuple(range(self.data.ndim))
        return op.forward(self, dim, keepdim, ddof)

    def reshape(self, *shape: int):
        from cortex.operations import Reshape

        op = Reshape()
        return op.forward(self, *shape)

    def set(self, mask: Tensor, value: Tensor):
        from cortex.operations import MaskSet

        op = MaskSet()
        return op.forward(self, tensor(mask), tensor(value))

    def __gt__(self, other: Tensor):
        from cortex.operations import Compare

        op = Compare()
        return op.forward(self, tensor(other), op=np.greater)

    def __lt__(self, other: Tensor):
        from cortex.operations import Compare

        op = Compare()
        return op.forward(self, tensor(other), op=np.less)

    def __ge__(self, other: Tensor):
        from cortex.operations import Compare

        op = Compare()
        return op.forward(self, tensor(other), op=np.greater_equal)

    def __le__(self, other: Tensor):
        from cortex.operations import Compare

        op = Compare()
        return op.forward(self, tensor(other), op=np.less_equal)

    def __eq__(self, other: Tensor):
        from cortex.operations import Compare

        op = Compare()
        return op.forward(self, tensor(other), op=np.equal)

    def __ne__(self, other: Tensor):
        from cortex.operations import Compare

        op = Compare()
        return op.forward(self, tensor(other), op=np.not_equal)

    def __getitem__(self, idx: Union[int, Iterable]):
        from cortex.operations import Slice

        op = Slice()
        return op.forward(self, idx)

    def __add__(self, other: Tensor):
        from cortex.operations import Add

        op = Add()
        return op.forward(self, tensor(other))

    def __radd__(self, other):
        """New = other + self"""
        from cortex.operations import Add

        op = Add()
        return op.forward(self, tensor(other))

    def __iadd__(self, other):
        from cortex.operations import Add

        op = Add()
        return op.forward(self, tensor(other))

    def __neg__(self):
        from cortex.operations import Neg

        op = Neg()
        return op.forward(self)

    def __sub__(self, other: Tensor):
        return self + -tensor(other)

    def __rsub__(self, other: Tensor):
        return self + -tensor(other)

    def __isub__(self, other: Tensor):
        return self + -tensor(other)

    def __mul__(self, other: Tensor):
        from cortex.operations import Mul

        op = Mul()
        return op.forward(self, tensor(other))

    def __rmul__(self, other: Tensor):
        from cortex.operations import Mul

        op = Mul()
        return op.forward(self, tensor(other))

    def __imul__(self, other: Tensor):
        from cortex.operations import Mul

        op = Mul()
        return op.forward(self, tensor(other))

    def __truediv__(self, other: Tensor):
        from cortex.operations import Div

        op = Div()
        return op.forward(self, tensor(other))

    def __rtruediv__(self, other: Tensor):
        from cortex.operations import Div

        op = Div()
        return op.forward(self, tensor(other))

    def __itruediv__(self, other: Tensor):
        from cortex.operations import Div

        op = Div()
        return op.forward(self, tensor(other))

    def __matmul__(self, other: Tensor):
        from cortex.operations import MatMul

        op = MatMul()
        return op.forward(self, tensor(other))

    def __pow__(self, other: Tensor):
        from cortex.operations import Pow

        op = Pow()
        return op.forward(self, tensor(other))


#######################################
############### HELPERS ###############
#######################################


def tensor(
    data: Any,
    dtype: Optional[Type] = None,
    requires_grad: bool = False,
):
    """Constructs a tensor."""
    if isinstance(data, Tensor):
        return data
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape: int, dtype: Optional[Type] = None, requires_grad: bool = False):
    data = np.zeros((shape), dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def zeros_like(tensor: Tensor):
    data = np.zeros(tensor.shape)
    return Tensor(
        data,
        requires_grad=tensor.requires_grad,
        dtype=tensor.dtype,
        operation=tensor.operation,
    )


def ones(*shape: int, dtype: Optional[Type] = None, requires_grad: bool = False):
    data = np.ones((shape), dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def ones_like(tensor: Tensor):
    data = np.ones(tensor.shape)
    return Tensor(
        data,
        requires_grad=tensor.requires_grad,
        dtype=tensor.dtype,
        operation=tensor.operation,
    )


def empty(*shape: int, dtype: Optional[Type] = None, requires_grad: bool = False):
    data = np.empty((shape), dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)


def empty_like(tensor: Tensor):
    data = np.empty(tensor.shape)
    return Tensor(
        data,
        requires_grad=tensor.requires_grad,
        dtype=tensor.dtype,
        operation=tensor.operation,
    )


def randn(*shape: int, requires_grad: bool = False):
    data = np.random.randn(*shape)
    return Tensor(data, requires_grad=requires_grad)


def randn_like(tensor: Tensor):
    data = np.random.randn(*tensor.shape)
    return Tensor(
        data,
        requires_grad=tensor.requires_grad,
        dtype=tensor.dtype,
        operation=tensor.operation,
    )


def rand(*shape: int, requires_grad: bool = False):
    data = np.random.rand(*shape)
    return Tensor(data, requires_grad=requires_grad)


def rand_like(tensor: Tensor):
    data = np.random.rand(*tensor.shape)
    return Tensor(
        data,
        requires_grad=tensor.requires_grad,
        dtype=tensor.dtype,
        operation=tensor.operation,
    )


def randint(low: int, high: int, shape: tuple[int], requires_grad: bool = False):
    data = np.random.randint(low, high, shape)
    return Tensor(data, requires_grad=requires_grad)


def randint_like(low: int, high: int, tensor: Tensor):
    data = np.random.randint(low, high, tensor.shape)
    return Tensor(
        data,
        requires_grad=tensor.requires_grad,
        dtype=tensor.dtype,
        operation=tensor.operation,
    )


def arange(low: int, high: int, requires_grad: bool = False):
    data = np.arange(low, high)
    return Tensor(data, requires_grad=requires_grad)


def cat(tensors: list[Tensor], dim: int) -> Tensor:
    """Concatenate tensors along a specific dimension."""

    from cortex.operations import Cat

    op = Cat()
    return op.forward(tensors, dim)


def stack(tensors: list[Tensor], dim: int) -> Tensor:
    """Stack tensors along a specific dimension."""

    from cortex.operations import Stack

    op = Stack()
    return op.forward(tensors, dim)
