from __future__ import annotations

from abc import ABC
from typing import Any, Iterable, Optional


class Optimizer(ABC):
    """
    Abstract class for an `Optimizer`.
    """

    def __init__(
        self,
        params: Iterable[Any],
        defaults: Optional[dict[str, Any]] = None,
    ):
        self.defaults = defaults or {}
        self.param_groups: list[dict[str, Any]] = []
        # per-parameter state, keyd by id(param)
        self.state: dict[int, dict[str, Any]] = {}

        self._init_param_groups(params)

    def _init_param_groups(self, params: Iterable[Any]):
        params = list(params)
        if not params:
            return

        # Params by group
        if isinstance(params[0], dict):
            for group in params:
                assert "params" in group
                # Apply defaults to this group if missing
                for k, v in self.defaults.items():
                    group.setdefault(k, v)
                self.param_groups.append(group)
        # Simple list of params
        else:
            group = {"params": params}
            # Apply defaults to the group if missing
            for k, v in self.defaults.items():
                group.setdefault(k, v)
            self.param_groups.append(group)

    def add_param_group(self, group: dict[str, Any]):
        assert "params" in group
        for k, v in self.defaults.items():
            group.setdefault(k, v)
        self.param_groups.append(group)

    def parameters(self):
        for group in self.param_groups:
            for param in group["params"]:
                yield param

    def step(self, closure=None):
        """
        Perform a single optimization step.
        Subclasses should override `_step_param` or override this method fully.
        If closure is provided, it should be a callable that re-evaluates the model/loss and returns the loss.
        """
        if closure is not None:
            loss = closure()
        else:
            loss = None

        for group in self.param_groups:
            self._step_group(group)

        return loss

    def _step_group(self, group: dict[str, Any]):
        raise NotImplementedError("Optimizer subclasses must implement _step_group")

    def zero_grad(self, set_to_none: bool = False):
        """
        Zero the gradients of all parameters.
        If set_to_none, set grad to `None`
        """
        for param in self.parameters():
            param.zero_grad_tree(set_to_none)
