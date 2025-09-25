import numpy as np
import pytest

from cortex import Tensor
from cortex.optim.primatives import Optimizer


class MockOptimizer(Optimizer):
    """A concrete Optimizer for testing the abstract base class."""

    def __init__(self, params, defaults=None):
        super().__init__(params, defaults)

    def _step_group(self, group):
        """A mock step that modifies parameters based on a group-specific value."""
        modifier = group.get("modifier", 1)
        for p in group["params"]:
            if p.grad is not None:
                p.data -= modifier


@pytest.fixture
def params():
    """Provides a simple list of parameters for testing."""
    p1 = Tensor([1, 2], requires_grad=True)
    p2 = Tensor([3, 4], requires_grad=True)
    p1.grad = Tensor([0.1, 0.1])
    p2.grad = Tensor([0.2, 0.2])
    return [p1, p2]


@pytest.fixture
def param_groups():
    """Provides parameter groups for testing."""
    p1 = Tensor([10], requires_grad=True)
    p2 = Tensor([20], requires_grad=True)
    p3 = Tensor([30], requires_grad=True)
    p1.grad = Tensor([1.0])
    p2.grad = Tensor([2.0])
    p3.grad = Tensor([3.0])
    return [
        {"params": [p1], "modifier": 1},
        {"params": [p2, p3], "modifier": 10},
    ]


def test_optimizer_init_simple_list(params):
    """
    Tests optimizer initialization with a simple list of parameters.
    """
    defaults = {"lr": 0.1}
    optimizer = MockOptimizer(params, defaults=defaults)

    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["params"] == params
    assert group["lr"] == 0.1


def test_optimizer_init_param_groups(param_groups):
    """
    Tests optimizer initialization with a list of parameter groups.
    """
    defaults = {"momentum": 0.9}
    optimizer = MockOptimizer(param_groups, defaults=defaults)

    assert len(optimizer.param_groups) == 2
    # Check first group
    assert optimizer.param_groups[0]["params"] == param_groups[0]["params"]
    assert optimizer.param_groups[0]["modifier"] == 1
    assert optimizer.param_groups[0]["momentum"] == 0.9  # Default applied

    # Check second group
    assert optimizer.param_groups[1]["params"] == param_groups[1]["params"]
    assert optimizer.param_groups[1]["modifier"] == 10
    assert optimizer.param_groups[1]["momentum"] == 0.9  # Default applied


def test_add_param_group(params):
    """
    Tests adding a parameter group to an existing optimizer.
    """
    optimizer = MockOptimizer(params)
    assert len(optimizer.param_groups) == 1

    new_param = Tensor([5, 6], requires_grad=True)
    new_group = {"params": [new_param], "lr": 0.001}
    optimizer.add_param_group(new_group)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1] == new_group


def test_parameters_generator(param_groups):
    """
    Tests that the parameters() generator yields all parameters.
    """
    optimizer = MockOptimizer(param_groups)
    all_params = list(optimizer.parameters())
    # param_groups has 3 parameters in total
    assert len(all_params) == 3
    assert all_params[0] is param_groups[0]["params"][0]
    assert all_params[1] is param_groups[1]["params"][0]
    assert all_params[2] is param_groups[1]["params"][1]


def test_zero_grad(params):
    """
    Tests that zero_grad() correctly clears the gradients of all parameters.
    """
    optimizer = MockOptimizer(params)
    assert params[0].grad is not None
    assert params[1].grad is not None

    # Test zeroing gradients
    optimizer.zero_grad()
    assert np.array_equal(params[0].grad.data, np.zeros_like(params[0].data))
    assert np.array_equal(params[1].grad.data, np.zeros_like(params[1].data))

    # Test setting gradients to None
    params[0].grad = Tensor([0.1, 0.1])  # Reset grad
    optimizer.zero_grad(set_to_none=True)
    assert params[0].grad is None


def test_step(param_groups):
    """
    Tests that the step() method correctly updates parameters.
    """
    optimizer = MockOptimizer(param_groups)
    p1, p2, p3 = list(optimizer.parameters())

    optimizer.step()

    # p1 was in group with modifier=1
    assert np.allclose(p1.data, [9])  # 10 - 1
    # p2 and p3 were in group with modifier=10
    assert np.allclose(p2.data, [10])  # 20 - 10
    assert np.allclose(p3.data, [20])  # 30 - 10


def test_step_with_closure():
    """
    Tests the step() method with a closure that re-evaluates the loss.
    """
    optimizer = MockOptimizer([])
    closure_called = False

    def loss_closure():
        nonlocal closure_called
        closure_called = True
        return 123.45  # A mock loss value

    returned_loss = optimizer.step(closure=loss_closure)

    assert closure_called
    assert returned_loss
