import numpy as np

class Tensor:
    def __init__(self, data, _prev=(), _op=''):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_prev)
        self._op   = _op
        
        
    def __matmul__(self, other):
        """Matrix Multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.ndim not in (1, 2) or other.data.ndim not in (1, 2):
            raise ValueError(
                f"Matrix multiplication currently supports 1D and 2D arrays only "
                f"{self.data.shape} and {other.data.shape}"
            )
        
        out   = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            grad = out.grad
            if out.data.ndim == 0:
                grad = out.grad.item()
            elif np.asarray(out.grad).size == 1:
                grad = out.grad.item()
            else:
                grad = np.asarray(out.grad).reshape(out.data.shape)

            if self.data.ndim == 1 and other.data.ndim == 1:
                self.grad += grad * other.data
                other.grad += grad * self.data

            elif self.data.ndim == 2 and other.data.ndim == 2:
                self.grad += grad @ other.data.T
                other.grad += self.data.T @ grad

            elif self.data.ndim == 2 and other.data.ndim == 1:
                self.grad += np.outer(grad, other.data)
                other.grad += self.data.T @ grad

            elif self.data.ndim == 1 and other.data.ndim == 2:
                self.grad += grad @ other.data.T
                other.grad += np.outer(self.data, grad)

            else:
                raise ValueError("Unsupported dimensions for matrix multiplication.")
        
        out._backward = _backward
        return out
        
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out   = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad  += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        
        out._backward = _backward
        return out
        
    
    def sum(self, axis=None, keepdims=True):
        out =  Tensor(self.data.sum(axis=axis, keepdims=True), (self,), 'sum')
        
        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.data.shape)
        out._backward = _backward
        return out
        
    def sigmoid(self):
        s   = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += (s * (1-s)) * out.grad
        out._backward = _backward
        return out
       
    
    def softmax(self):
        x = self.data - self.data.max(axis=1, keepdims=True)
        e = np.exp(x)
        p = e / e.sum(axis=1, keepdims=True)  
        out = Tensor(p, (self, ), 'softmax')
        
        def _backward():
            grad = out.grad
            self.grad += p * (
                grad - (grad * p).sum(axis=1, keepdims=True)
            )
        
        out._backward = _backward
        return out   
            
            
    def backward(self):
        # topological sort
        topo = []
        visited = set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def unbroadcast(grad: Tensor, shape):
    while grad.ndim >  len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size ==1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad