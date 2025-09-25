from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np

from cortex.tensor import Tensor


class Operation(ABC):

    def __init__(self):
        # Parents are the tensors used to compute the new tensor in the given operation
        self.parents: tuple[Tensor]
        # Saved inputs are the inputs needed for the backward computation
        self.saved_inputs: Union[Any, tuple[Any]]

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def backward(self, output: Optional[Tensor], grad: Optional[np.ndarray]):
        pass
