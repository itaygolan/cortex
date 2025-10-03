from __future__ import annotations

from typing import Any, Generator, Optional, Type

import cortex
from cortex.operations import Operation


class Parameter(cortex.Tensor):
    """Subclass of tensor that always tracks gradients."""

    def __init__(
        self,
        data: Any,
        dtype: Optional[Type] = None,
        operation: Operation = None,
    ):
        if isinstance(data, cortex.Tensor):
            super().__init__(
                data.data,
                dtype=data.dtype,
                requires_grad=True,
                operation=data.operation,
            )
        else:
            super().__init__(
                data,
                dtype=dtype,
                requires_grad=True,
                operation=operation,
            )


class Module:
    """Generic Module."""

    def __init__(self):
        pass

    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward(*args, **kwargs)

    def parameters(self) -> Generator:
        """Returns all parameters as an iterable.
        Parameters here are defined as all parameters of submodules, all `Parameter`
        objects, or all `Tensor` objects with `requires_grad=True`.
        """
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                yield from param.parameters()
            elif isinstance(param, Parameter):
                yield param
            elif isinstance(param, cortex.Tensor):
                if param.requires_grad:
                    yield param

    def modules(self) -> Generator:
        """Returns all submodules as an iterable."""
        for _, param in self.__dict__.items():
            if isinstance(param, Module):
                yield param

    def named_parameters(self, prefix: str = "") -> Generator:
        for name, param in self.__dict__.items():
            if isinstance(param, Module):
                yield from param.named_parameters(prefix=f"{prefix}{name}.")
            elif isinstance(param, Parameter):
                yield prefix + name, param
            elif isinstance(param, cortex.Tensor):
                if param.requires_grad:
                    yield prefix + name, param

    def train(self):
        self.training = True

        for module in self.modules():
            module.train()

    def eval(self):
        self.training = False

        for module in self.modules():
            module.eval()


class Sequential(Module):

    def __init__(self, *modules: Module):
        super().__init__()

        self.modules: list[Module] = list(modules)

    def parameters(self):
        for module in self.modules:
            yield from module.parameters()

    def named_parameters(self):
        for i, module in enumerate(self.modules):
            yield from module.named_parameters(prefix=f"{i}.")

    def forward(self, *args: Any):
        inputs: tuple[Any, ...] = args
        out: Any
        for step in self.modules:
            out = step(*inputs)
            inputs = out if isinstance(out, tuple) else (out,)
        return out
