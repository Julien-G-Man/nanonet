import numpy as np
from typing import Callable
from nanonet.autograd import Tensor


def test_mlp_learns_xor():
    from nanonet.model import MLP

    np.random.seed(42)

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([0, 1, 1, 0])

    model = MLP(n_in=2, n_hidden=8, n_out=2)
    model.train(X, y, epochs=1500, batch=4, lr=0.03, verbose=False)

    predictions = model.predict(X)
    np.testing.assert_array_equal(predictions, y)
    

def test_adam_updates_parameters():
    from nanonet.optim import Adam

    parameter = Tensor(np.array([1.0, -2.0]))
    parameter.grad = np.array([0.5, -0.5])
    
    optimizer = Adam([parameter])
    original = parameter.data.copy()
    optimizer.step()

    assert not np.array_equal(parameter.data, original)
    assert np.all(np.isfinite(parameter.data))
    
    
def test_adam_zero_grad():
    from nanonet.optim import Adam

    parameter = Tensor(np.array([1.0, -2.0]))
    optimizer = Adam([parameter])
    parameter.grad = np.array([3.0, 4.0])
    optimizer.zero_grad()

    np.testing.assert_array_equal(parameter.grad, np.zeros(2))
    
    
def test_addition_backward():
    x = Tensor(np.array([1.0, 2.0]))
    y = Tensor(np.array([3.0, 4.0]))
    
    loss:Tensor = (x + y).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, [1.0, 1.0])
    np.testing.assert_allclose(y.grad, [1.0, 1.0])
    

def test_broadcast_bias_gradient():
    x = Tensor(np.zeros((3, 2)))
    bias = Tensor(np.zeros((2)))
    result = x + bias
    
    result.grad = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    
    result._backward()
    np.testing.assert_allclose(bias.grad, [9.0, 12.0])
    

def test_sum_backward():
    x = Tensor(np.array([2.0, 3.0, 4.0]))
    loss = x.sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, [1.0, 1.0, 1.0])
    

def test_multiplication_backward():
    x = Tensor(np.array([2.0, 3.0]))
    y = Tensor(np.array([4.0, 5.0]))
    
    loss:Tensor = (x @ y).sum()
    loss.backward()
    np.testing.assert_allclose(x.grad, [4.0, 5.0])
    np.testing.assert_allclose(y.grad, [2.0, 3.0])
    
    
def test_matmul_backward():
    x = Tensor(np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ]))
    w = Tensor(np.array([
        [5.0, 6.0],
        [7.0, 8.0]
    ]))
    
    loss: Tensor = (x @ w).sum()
    loss.backward()
    
    expected_x_grad = np.array([
        [11.0, 15.0],
        [11.0, 15.0]
    ])
    expected_w_grad = np.array([
        [4.0, 4.0],
        [6.0, 6.0],
    ])
    np.testing.assert_allclose(x.grad, expected_x_grad)
    np.testing.assert_allclose(w.grad, expected_w_grad)
       

def test_softmax_backward():
    logits = Tensor(np.array([[1.0, 2.0, 3.0]]))
    probs = logits.softmax()
    probs.grad = np.array([[1.0, 0.0, 0.0]])    
    probs._backward()
    
    p = probs.data
    expected = p * (
        probs.grad - (probs.grad * p).sum(axis=1, keepdims=True)
    )
    np.testing.assert_allclose(logits.grad, expected)
  
  
def test_multiplication_gradient_check():
    x = Tensor(np.array([2.0, 3.0]))
    y = Tensor(np.array([4.0, 5.0]))
    
    def loss_fn():
        return np.sum(x.data * y.data)
    
    loss: Tensor = (x @ y).sum()
    loss.backward()
    numerical = numerical_gradient(lambda: loss_fn(), x)
    np.testing.assert_allclose(x.grad, numerical, rtol=1e-5, atol=1e-5)
    
        
def test_softmax_cross_entropy_gradient():
    from nanonet.model import _cross_entropy
    
    logits = Tensor(np.array([
        [0.5, 1.0, 1.5],
        [1.2, 0.3, 0.8]
    ]))
    labels = np.array([2, 0])
    probs = logits.softmax()
    loss: Tensor = _cross_entropy(probs, labels)
    loss.backward()
    
    analytical = logits.grad.copy()
    
    def loss_fn():
        x = logits.data
        max_val = x.max(axis=1, keepdims=True)
        exp_x = np.exp(x - max_val)
        probabilities = exp_x / exp_x.sum(axis=1, keepdims=True)
        return -np.log(
            probabilities[np.arange(len(labels)), labels]
        ).mean()
    
    numerical = numerical_gradient(loss_fn, logits)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-5)
    
    
def numerical_gradient(fn: Callable, x: Tensor, eps=1e-5):
    grad = np.zeros_like(x.data)
    for idx in np.ndindex(x.data.shape):
        original = x.data[idx]
        
        x.data[idx] = original + eps
        positive = fn()
        
        x.data[idx] = original - eps
        negative = fn()
        
        x.data[idx] = original
        grad[idx] = (positive - negative) / (2 * eps)
    return grad

