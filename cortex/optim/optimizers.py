from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from cortex.nn.primatives import Parameter
from cortex.optim.primatives import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Any],
        lr: float = 0.1,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        nesterov: bool = False,
    ):
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov
        )
        super().__init__(parameters, defaults)

    def _step_group(self, group: dict[str, Any]):

        params: list[Parameter] = group["params"]
        lr: float = group["lr"]
        weight_decay: float = group["weight_decay"]
        momentum: float = group["momentum"]
        nesterov: bool = group["nesterov"]

        for param in params:
            if param.grad is None:
                continue

            param_grad = param.grad

            # Apply weight decay (L2): weight_decay * param
            if weight_decay != 0.0:
                param_grad += weight_decay * param.data

            if momentum != 0.0:
                # ensure state for this param
                param_state = self.state.setdefault(id(param), {})
                buffer = param_state.get(
                    "momentum_buffer",
                    np.zeros_like(param),
                )

                buffer = momentum * buffer + param_grad
                # Save buffer back
                param_state["momentum_buffer"] = buffer

                if nesterov:
                    step = param_grad + momentum * buffer
                else:
                    step = buffer
            else:
                step = param_grad

            param.data -= lr * step
            param.data -= lr * step
