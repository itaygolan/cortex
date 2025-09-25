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
        if isinstance(self, cortex.Tensor):
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

    def train(self):
        self.training = True

        for param in self.parameters():
            if isinstance(param, Module):
                param.train()

    def eval(self):
        self.training = False

        for param in self.parameters():
            if isinstance(param, Module):
                param.eval()


class Sequential(Module):

    def __init__(self, *modules: Module):
        super().__init__()

        self.modules: list[Module] = list(modules)

    def parameters(self):
        for module in self.modules:
            yield from module.parameters()

    def forward(self, *args: Any):
        inputs: tuple[Any, ...] = args
        out: Any
        for step in self.modules:
            out = step(*inputs)
            inputs = out if isinstance(out, tuple) else (out,)
        return out
