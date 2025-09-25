import numpy as np
import pytest

from cortex import Tensor
from cortex.nn.activations import ReLU, Sigmoid
from cortex.nn.primatives import Module, Parameter, Sequential


def test_parameter_creation():
    """
    Tests that a Parameter is created with requires_grad=True.
    """
    p = Parameter([1, 2, 3])
    assert isinstance(p, Tensor)
    assert p.requires_grad
    assert np.array_equal(p.data, np.array([1, 2, 3]))


def test_parameter_from_tensor():
    """
    Tests creating a Parameter from an existing Tensor.
    """
    t_no_grad = Tensor([1, 2, 3], requires_grad=False)
    p = Parameter(t_no_grad)
    assert p.requires_grad
    assert np.array_equal(p.data, t_no_grad.data)


class SimpleModule(Module):
    """A simple module for testing purposes."""

    def __init__(self):
        super().__init__()
        self.param1 = Parameter([1.0, 2.0])
        self.tensor_grad = Tensor([3.0, 4.0], requires_grad=True)
        self.tensor_no_grad = Tensor([5.0, 6.0], requires_grad=False)
        self.other_attribute = "not a parameter"

    def forward(self, x):
        return x + self.param1


class NestedModule(Module):
    """A nested module for testing parameter collection."""

    def __init__(self):
        super().__init__()
        self.sub_module = SimpleModule()
        self.param2 = Parameter([7.0, 8.0])


def test_module_call():
    """
    Tests that calling a module executes its forward pass.
    """
    module = SimpleModule()
    input_tensor = Tensor([10.0, 20.0])
    output = module(input_tensor)
    expected_output = np.array([11.0, 22.0])
    assert np.allclose(output.data, expected_output)


def test_module_parameters():
    """
    Tests that the parameters() method correctly identifies tensors that require gradients.
    """
    module = SimpleModule()
    params = list(module.parameters())
    assert len(params) == 2
    # Check that the correct parameters are returned, regardless of order
    param_datas = [p.data.tolist() for p in params]
    assert [1.0, 2.0] in param_datas
    assert [3.0, 4.0] in param_datas


def test_nested_module_parameters():
    """
    Tests that parameters() method recursively finds parameters in sub-modules.
    """
    nested_module = NestedModule()
    params = list(nested_module.parameters())
    assert len(params) == 3
    param_datas = [p.data.tolist() for p in params]
    assert [1.0, 2.0] in param_datas  # from sub_module.param1
    assert [3.0, 4.0] in param_datas  # from sub_module.tensor_grad
    assert [7.0, 8.0] in param_datas  # from param2


def test_module_train_eval_mode():
    """
    Tests the train() and eval() methods to set the module's mode.
    """
    module = NestedModule()

    # Default should not be set
    assert not hasattr(module, "training")
    assert not hasattr(module.sub_module, "training")

    # Set to train mode
    module.train()
    assert module.training is True
    assert module.sub_module.training is True

    # Set to eval mode
    module.eval()
    assert module.training is False
    assert module.sub_module.training is False


def test_sequential_forward():
    """
    Tests the forward pass of a Sequential module.
    """

    # A simple linear layer for testing
    class Linear(Module):
        def __init__(self):
            super().__init__()
            self.p = Parameter([[2.0]])

        def forward(self, x):
            return x @ self.p

    model = Sequential(Linear(), ReLU())
    input_tensor = Tensor([[-1.0], [1.0]])
    output = model(input_tensor)

    # input @ p = [[-2.0], [2.0]]
    # ReLU(output) = [[0.0], [2.0]]
    expected_output = np.array([[0.0], [2.0]])
    assert np.allclose(output.data, expected_output)


def test_sequential_parameters():
    """
    Tests that a Sequential module correctly gathers parameters from its layers.
    """
    model = Sequential(SimpleModule(), Sigmoid(), NestedModule())
    params = list(model.parameters())
    # SimpleModule has 2, Sigmoid has 0, NestedModule has 3
    assert len(params)
