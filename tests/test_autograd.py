import numpy as np
from nanonet.autograd import Tensor

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