# NanoNet

**A neural network framework built from scratch using NumPy.**

NanoNet is an educational project exploring the mathematics and mechanics behind neural networks. Rather than relying on PyTorch or TensorFlow for its core implementation, NanoNet builds essential components from the ground up, including automatic differentiation, backpropagation, multilayer perceptrons, and the Adam optimizer.

The goal is to understand how neural networks work internally, from tensor operations and gradient computation to parameter updates and model training.

## Features

* **Automatic differentiation:** A custom Tensor engine that builds a computational graph and propagates gradients through operations.
* **Multilayer perceptron:** A fully connected neural network implemented using NumPy.
* **Manual backpropagation:** An alternative MLP implementation with explicitly calculated gradients.
* **Adam optimizer:** An implementation of adaptive gradient-based optimization.
* **Core tensor operations:** Matrix multiplication, element-wise multiplication, broadcasting, summation, sigmoid, and softmax.
* **Classification:** Cross-entropy loss and prediction for multiclass classification.
* **Numerical gradient checking:** Finite-difference checks to validate analytical gradients.
* **Automated testing:** Unit tests and continuous integration through GitHub Actions.

## Architecture

```text
Input Data
    |
    v
Tensor Operations
    |
    v
Forward Pass
    |
    v
Loss Calculation
    |
    v
Automatic Differentiation
    |
    v
Gradient Computation
    |
    v
Adam Optimizer
    |
    v
Parameter Updates
```

## Project Structure

```text
nanonet/
├── nanonet/
│   ├── autograd.py
│   ├── model.py
│   ├── optim.py
│   └── manual.py
├── examples/
│   ├── mnist.py
│   └── bag_of_words.py
├── tests/
├── .github/
│   └── workflows/
├── requirements.txt
└── README.md
```

## Core Components

| Component                  | Description                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------- |
| `nanonet/autograd.py`      | Tensor operations, computational graph construction, and automatic differentiation    |
| `nanonet/model.py`         | MLP, forward propagation, training, prediction, and cross-entropy loss                |
| `nanonet/optim.py`         | Adam optimizer and optimization utilities                                             |
| `nanonet/manual.py`        | MLP implementation using manually calculated backpropagation                          |
| `examples/mnist.py`        | MNIST training and evaluation example                                                 |
| `examples/bag_of_words.py` | Bag-of-words sentiment classification demonstration                                   |
| `tests/`                   | Automated tests for tensor operations, gradients, optimization, and learning behavior |

## Getting Started

### Requirements

* Python 3
* NumPy
* scikit-learn
* Matplotlib
* pandas

Install the dependencies from the repository root:

```bash
pip install -r requirements.txt
```

### Run the MNIST Example

Run the following command from the repository root:

```bash
python examples/mnist.py
```

The example trains a fully connected neural network to classify handwritten digits.

**Network architecture**

* Input: 784 features representing image pixels
* Hidden layer: 128 units
* Output: 10 classes representing digits 0–9

The script normalizes the pixel values, trains the model, evaluates predictions, and plots the training loss.

**Reported accuracy: 95.2%.**

This is the result from the latest full evaluation run. Accuracy can vary with training configuration and evaluation conditions.

### Run the Bag-of-Words Example

```bash
python examples/bag_of_words.py
```

This example demonstrates a simple bag-of-words approach to text classification. It is a small educational demonstration rather than a benchmark of sentiment-analysis performance.

## Testing

Run the test suite from the repository root:

```bash
pytest -v
```

The tests cover core functionality, including:

* Tensor operations and gradient propagation
* Matrix multiplication and broadcasting
* Automatic differentiation
* Numerical gradient checking
* Optimizer updates
* Neural-network learning behavior, including XOR

GitHub Actions runs the automated test suite to help catch regressions.

## Learning Objectives

NanoNet explores the mechanisms that higher-level machine-learning frameworks abstract away:

1. How computational graphs represent mathematical operations.
2. How the chain rule enables reverse-mode automatic differentiation.
3. How gradients propagate through a neural network.
4. How optimizers use gradients to update parameters.
5. How forward propagation, loss calculation, and backpropagation work together during training.
6. How numerical gradient checking helps validate derivative implementations.

## Limitations

NanoNet is an educational implementation, not a replacement for production machine-learning frameworks.

* Computation uses NumPy and runs on the CPU.
* GPU acceleration is not supported.
* The available operations and architectures are limited compared with mature frameworks.
* The bag-of-words example uses a small demonstration dataset.
* Reported performance depends on the training configuration and evaluation procedure.

## Motivation

NanoNet is an exercise in understanding machine learning from first principles: implementing the mathematics, validating gradients, and observing how the components combine to produce a learning system.
