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