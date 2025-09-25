# Custom Autograd Framework

Cortex is a custom autograd framework, similar to PyTorch, built in Python for educational purposes.

## How it works

* The fundmental dataclass in `cortex` is the `Tensor`. 
* Each Tensor contains data (as a NumPy array) and optional gradient information (also as a NumPy array)
* The framework builds a computational graph by tracking operations between tensors
* Gradients are computed using reverse-mode automatic differentiation (backpropagation)

## Getting started

```bash
git clone git@github.com:itaygolan/cortex.git && cd cortex
pip install -e ".[all]"
```

## Key Components

### Tensor Class

The core Tensor class tracks:

* `data`: The actual numerical data (NumPy array)
* `grad`: Accumulated gradients (initially None)
* `requires_grad`: Boolean flag indicating if gradients should be computed
* `operation`: Reference to the function that created this tensor (for backprop)
* `children`: References to tensors that were computed using the current tensor. 

### Function Class

Operations are implemented as `Operation` subclasses with:

* `forward()`: Computes the operation result
* `backward()`: Computes gradients with respect to inputs
* Context saving for storing information needed during backward pass

### Computational Graph

* Built dynamically as operations are performed
* Tracks dependencies between tensors by storing/removing children
* Enables automatic, analytical gradient computation via the chain rule

## Basic Usage

```python
import cortex

# Create tensors
x = cortex.Tensor([2.0, 3.0], requires_grad=True)
y = cortex.Tensor([4.0, 5.0], requires_grad=True)

# Perform operations
z = x * y + x ** 2
loss = z.sum()

# Compute gradients
loss.backward()

print(x.grad)  # Gradients with respect to x
print(y.grad)  # Gradients with respect to y
```

## Visualizing the Compute Graph

The library also allows you to visualize the compute graph easily in a Jupyter notebook. For a given leaf node, you can trace back the entire linage of the computation using: `cortex.visualize_graph`:

```python
import cortex

# Create tensors
x = cortex.Tensor([2.0, 3.0], requires_grad=True)
y = cortex.Tensor([4.0, 5.0], requires_grad=True)

# Perform operations
z = x * y + x ** 2
loss = z.sum()

cortex.visualize_graph(loss)
```

## Deep Learning Usage

The library supports common deep learning functionality to design and train neural networks. 

```python
import cortex
import cortex.nn as nn
import cortex.optim as optim

model = nn.Sequential(
    nn.Linear(in_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, out_dim),
)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), weight_decay=1e-3, momentum=0.9)

for epoch in range(100):
    optimizer.zero_grad()

    out = model(...)
    loss = criterion(...)

    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch: {epoch+1} | Loss: {loss.item():.2f}")
```

## Limitations

This is an educational framework with several limitations:

* Performance is not optimized (pure Python implementation)
* Limited operation set compared to production frameworks
* No GPU support
* No optimization algorithms built-in
* Basic error handling