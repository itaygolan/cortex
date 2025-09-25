from __future__ import annotations

import pytest

from cortex import Tensor


@pytest.fixture
def simple_tensor_grad():
    def _make():
        return Tensor([10, 10, 10], requires_grad=True)

    return _make


@pytest.fixture
def simple_tensor_nograd():
    def _make():
        return Tensor([10, 10, 10], requires_grad=False)

    return _make
