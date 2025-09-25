"""Tests to compare deep learning framework against PyTorch for numerical correctness"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as torch_nn
import torch.optim as torch_optim

import cortex
import cortex.nn as cortex_nn
import cortex.optim as cortex_optim


class TestCortexPyTorchComparison:
    """Test suite comparing Cortex autograd engine with PyTorch."""

    @pytest.fixture
    def training_data(self):
        """Generate reproducible training data."""
        np.random.seed(42)  # For reproducible tests
        torch.manual_seed(42)

        batch_size = 256
        in_dim = 64

        x = np.random.randn(batch_size, in_dim)
        y = (x.sum(1) > 0).reshape(-1, 1).astype(np.float32)

        return x, y

    @pytest.fixture
    def model_config(self):
        """Model configuration parameters."""
        return {
            "in_dim": 64,
            "hidden_dim": 32,
            "out_dim": 1,
            "lr": 0.001,
            "weight_decay": 1e-2,
            "momentum": 0.9,
            "nesterov": False,
        }

    def create_pytorch_model(self, config):
        """Create PyTorch model with specified configuration."""
        torch_l1 = torch_nn.Linear(config["in_dim"], config["hidden_dim"])
        torch_l2 = torch_nn.Linear(config["hidden_dim"], config["out_dim"])

        model = torch_nn.Sequential(
            torch_l1,
            torch_nn.ReLU(),
            torch_l2,
        )

        criterion = torch_nn.MSELoss()
        optimizer = torch_optim.SGD(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
        )

        return model, criterion, optimizer, (torch_l1, torch_l2)

    def create_cortex_model(self, config, torch_layers):
        """Create Cortex model with weights copied from PyTorch model."""
        torch_l1, torch_l2 = torch_layers

        # Create cortex layers
        cortex_l1 = cortex_nn.Linear(config["in_dim"], config["hidden_dim"])
        cortex_l2 = cortex_nn.Linear(config["hidden_dim"], config["out_dim"])

        # Copy weights from PyTorch to ensure identical initialization
        cortex_l1.weight.data = torch_l1.weight.data.numpy().copy()
        cortex_l1.bias.data = torch_l1.bias.data.numpy().copy()
        cortex_l2.weight.data = torch_l2.weight.data.numpy().copy()
        cortex_l2.bias.data = torch_l2.bias.data.numpy().copy()

        model = cortex_nn.Sequential(
            cortex_l1,
            cortex_nn.ReLU(),
            cortex_l2,
        )

        criterion = cortex_nn.MSELoss()
        optimizer = cortex_optim.SGD(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
        )

        return model, criterion, optimizer

    def train_pytorch_model(self, model, criterion, optimizer, x, y, epochs=10):
        """Train PyTorch model and return loss history."""
        torch_in = torch.tensor(x, dtype=torch.float32)
        torch_targets = torch.tensor(y, dtype=torch.float32)

        losses = []
        outputs = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            torch_out = model(torch_in)
            loss = criterion(torch_out, torch_targets)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch == epochs - 1:  # Store final output for comparison
                outputs.append(torch_out.detach().numpy())

        return losses, outputs[-1] if outputs else None

    def train_cortex_model(self, model, criterion, optimizer, x, y, epochs=10):
        """Train Cortex model and return loss history."""
        cortex_in = cortex.tensor(x, dtype=cortex.float64)
        cortex_targets = cortex.tensor(y, dtype=cortex.float64)

        losses = []
        outputs = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            out = model(cortex_in)
            loss = criterion(out, cortex_targets)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch == epochs - 1:  # Store final output for comparison
                outputs.append(out.data)  # Assuming .data gives numpy array

        return losses, outputs[-1] if outputs else None

    def test_single_forward_pass(self, training_data, model_config):
        """Test that single forward pass produces identical results."""
        x, y = training_data

        # Create models
        torch_model, torch_criterion, torch_optimizer, torch_layers = (
            self.create_pytorch_model(model_config)
        )
        cortex_model, cortex_criterion, cortex_optimizer = self.create_cortex_model(
            model_config, torch_layers
        )

        # Single forward pass
        torch_in = torch.tensor(x, dtype=torch.float32)
        cortex_in = cortex.tensor(x, dtype=cortex.float64)

        torch_out = torch_model(torch_in)
        cortex_out = cortex_model(cortex_in)

        # Compare outputs (allowing for float32 vs float64 differences)
        np.testing.assert_allclose(
            torch_out.detach().numpy(),
            cortex_out.data,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Forward pass outputs don't match between PyTorch and Cortex",
        )

    def test_single_loss_computation(self, training_data, model_config):
        """Test that loss computation produces identical results."""
        x, y = training_data

        # Create models
        torch_model, torch_criterion, torch_optimizer, torch_layers = (
            self.create_pytorch_model(model_config)
        )
        cortex_model, cortex_criterion, cortex_optimizer = self.create_cortex_model(
            model_config, torch_layers
        )

        # Forward pass and loss computation
        torch_in = torch.tensor(x, dtype=torch.float32)
        torch_targets = torch.tensor(y, dtype=torch.float32)
        cortex_in = cortex.tensor(x, dtype=cortex.float64)
        cortex_targets = cortex.tensor(y, dtype=cortex.float64)

        torch_out = torch_model(torch_in)
        torch_loss = torch_criterion(torch_out, torch_targets)

        cortex_out = cortex_model(cortex_in)
        cortex_loss = cortex_criterion(cortex_out, cortex_targets)

        # Compare losses
        np.testing.assert_allclose(
            torch_loss.item(),
            cortex_loss.item(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Loss computation doesn't match between PyTorch and Cortex",
        )

    def test_training_convergence_short(self, training_data, model_config):
        """Test that both frameworks converge similarly over a few epochs."""
        x, y = training_data
        epochs = 5

        # Create and train models
        torch_model, torch_criterion, torch_optimizer, torch_layers = (
            self.create_pytorch_model(model_config)
        )
        cortex_model, cortex_criterion, cortex_optimizer = self.create_cortex_model(
            model_config, torch_layers
        )

        torch_losses, torch_final_out = self.train_pytorch_model(
            torch_model, torch_criterion, torch_optimizer, x, y, epochs
        )
        cortex_losses, cortex_final_out = self.train_cortex_model(
            cortex_model, cortex_criterion, cortex_optimizer, x, y, epochs
        )

        # Compare final losses
        np.testing.assert_allclose(
            torch_losses[-1],
            cortex_losses[-1],
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Final losses don't match: PyTorch={torch_losses[-1]:.6f}, Cortex={cortex_losses[-1]:.6f}",
        )

        # Compare final outputs
        np.testing.assert_allclose(
            torch_final_out,
            cortex_final_out,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Final model outputs don't match between PyTorch and Cortex",
        )

    def test_training_convergence_full(self, training_data, model_config):
        """Test that both frameworks converge similarly over full training (marked as slow test)."""
        x, y = training_data
        epochs = 100

        # Create and train models
        torch_model, torch_criterion, torch_optimizer, torch_layers = (
            self.create_pytorch_model(model_config)
        )
        cortex_model, cortex_criterion, cortex_optimizer = self.create_cortex_model(
            model_config, torch_layers
        )

        torch_losses, torch_final_out = self.train_pytorch_model(
            torch_model, torch_criterion, torch_optimizer, x, y, epochs
        )
        cortex_losses, cortex_final_out = self.train_cortex_model(
            cortex_model, cortex_criterion, cortex_optimizer, x, y, epochs
        )

        # Compare loss trajectories at key points
        checkpoints = [9, 19, 49, 99]  # epochs 10, 20, 50, 100
        for i in checkpoints:
            if i < len(torch_losses):
                np.testing.assert_allclose(
                    torch_losses[i],
                    cortex_losses[i],
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"Loss mismatch at epoch {i+1}: PyTorch={torch_losses[i]:.6f}, Cortex={cortex_losses[i]:.6f}",
                )

        # Compare final outputs
        np.testing.assert_allclose(
            torch_final_out,
            cortex_final_out,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Final model outputs don't match after full training",
        )

        # Ensure both models are actually learning (loss should decrease)
        assert (
            torch_losses[-1] < torch_losses[0]
        ), "PyTorch model didn't learn (loss didn't decrease)"
        assert (
            cortex_losses[-1] < cortex_losses[0]
        ), "Cortex model didn't learn (loss didn't decrease)"

    def test_loss_trajectories_similarity(self, training_data, model_config):
        """Test that loss trajectories are similar between frameworks."""
        x, y = training_data
        epochs = 20

        # Create and train models
        torch_model, torch_criterion, torch_optimizer, torch_layers = (
            self.create_pytorch_model(model_config)
        )
        cortex_model, cortex_criterion, cortex_optimizer = self.create_cortex_model(
            model_config, torch_layers
        )

        torch_losses, _ = self.train_pytorch_model(
            torch_model, torch_criterion, torch_optimizer, x, y, epochs
        )
        cortex_losses, _ = self.train_cortex_model(
            cortex_model, cortex_criterion, cortex_optimizer, x, y, epochs
        )

        # Compare entire loss trajectories
        np.testing.assert_allclose(
            torch_losses,
            cortex_losses,
            rtol=1e-4,
            atol=1e-4,
            err_msg="Loss trajectories diverged between PyTorch and Cortex",
        )


# Additional utility functions for running tests
def test_quick_comparison():
    """Quick test that can be run standalone."""
    test_class = TestCortexPyTorchComparison()

    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)
    x = np.random.randn(64, 32).astype(np.float32)
    y = (x.sum(1) > 0).reshape(-1, 1).astype(np.float32)

    config = {
        "in_dim": 32,
        "hidden_dim": 16,
        "out_dim": 1,
        "lr": 0.001,
        "weight_decay": 1e-2,
        "momentum": 0.9,
        "nesterov": False,
    }

    test_class.test_single_forward_pass((x, y), config)
    test_class.test_single_loss_computation((x, y), config)
    test_class.test_training_convergence_short((x, y), config)


def test_full_comparison():
    """Quick test that can be run standalone."""
    test_class = TestCortexPyTorchComparison()

    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)
    x = np.random.randn(64, 32).astype(np.float32)
    y = (x.sum(1) > 0).reshape(-1, 1).astype(np.float32)

    config = {
        "in_dim": 32,
        "hidden_dim": 16,
        "out_dim": 1,
        "lr": 0.001,
        "weight_decay": 1e-2,
        "momentum": 0.9,
        "nesterov": False,
    }

    test_class.test_training_convergence_full((x, y), config)
    test_class.test_loss_trajectories_similarity((x, y), config)
