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
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
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


class RMSProp(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Any],
        lr: float = 0.1,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(parameters, defaults)

    def _step_group(self, group: dict[str, Any]):

        params: list[Parameter] = group["params"]
        lr: float = group["lr"]
        alpha: float = group["alpha"]
        eps: float = group["eps"]
        weight_decay: float = group["weight_decay"]

        for param in params:
            if param.grad is None:
                continue

            grad = param.grad

            # Apply weight decay (L2): weight_decay * param
            if weight_decay != 0.0:
                grad += weight_decay * param.data

            if alpha != 0.0:
                param_state = self.state.setdefault(id(param), {})

                rms = param_state.get("rms", np.zeros_like(param.data))
                rms = rms * alpha + (1 - alpha) * grad**2

                param_state["rms"] = rms

                step = grad / np.sqrt(rms + eps)
            else:
                step = grad

            param.data -= lr * step


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Any],
        lr: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(parameters, defaults)

    def _step_group(self, group: dict[str, Any]):

        params: list[Parameter] = group["params"]
        lr: float = group["lr"]
        betas: tuple[float, float] = group["betas"]
        beta1, beta2 = betas
        eps: float = group["eps"]
        weight_decay: float = group["weight_decay"]

        for param in params:
            if param.grad is None:
                continue

            grad = param.grad
            param_state = self.state.setdefault(id(param), {})
            t = param_state.get("t", 0) + 1

            # Apply weight decay (L2): weight_decay * param
            if weight_decay != 0.0:
                grad += weight_decay * param.data

            # First Adam moment
            step = grad
            if beta1 != 0.0:
                moment1 = param_state.get("moment1", np.zeros_like(param.data))
                moment1 = moment1 * beta1 + (1 - beta1) * grad
                param_state["moment1"] = moment1

                moment1_bias_corrected = moment1 / (1 - beta1**t)

                step = moment1_bias_corrected

            # Second Adam moment
            if beta2 != 0.0:
                moment2 = param_state.get("moment2", np.zeros_like(param.data))
                moment2 = moment2 * beta2 + (1 - beta2) * grad**2
                param_state["moment2"] = moment2

                moment2_bias_corrected = moment2 / (1 - beta2**t)
                step = step / np.sqrt(moment2_bias_corrected + eps)

            # Update step count
            param_state["t"] = t

            param.data -= lr * step


class AdamW(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Any],
        lr: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(parameters, defaults)

    def _step_group(self, group: dict[str, Any]):

        params: list[Parameter] = group["params"]
        lr: float = group["lr"]
        betas: tuple[float, float] = group["betas"]
        beta1, beta2 = betas
        eps: float = group["eps"]
        weight_decay: float = group["weight_decay"]

        for param in params:
            if param.grad is None:
                continue

            grad = param.grad
            param_state = self.state.setdefault(id(param), {})
            t = param_state.get("t", 0) + 1

            # First Adam moment
            step = grad
            if beta1 != 0.0:
                moment1 = param_state.get("moment1", np.zeros_like(param.data))
                moment1 = moment1 * beta1 + (1 - beta1) * grad
                param_state["moment1"] = moment1

                moment1_bias_corrected = moment1 / (1 - beta1**t)

                step = moment1_bias_corrected

            # Second Adam moment
            if beta2 != 0.0:
                moment2 = param_state.get("moment2", np.zeros_like(param.data))
                moment2 = moment2 * beta2 + (1 - beta2) * grad**2
                param_state["moment2"] = moment2

                moment2_bias_corrected = moment2 / (1 - beta2**t)
                step = step / np.sqrt(moment2_bias_corrected + eps)

            # Update step count
            param_state["t"] = t

            # AdamW: Decouple weight decay from moment calculation when updating parameters
            if weight_decay != 0.0:
                param.data -= (lr * weight_decay) * param.data

            param.data -= lr * step
