import numpy as np
from nanonet.autograd import Tensor
from nanonet.optim import Adam


class MLP:
    """Multi-Layer Perceptron"""
    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = Tensor(rng.standard_normal((n_in, n_hidden)) *  np.sqrt(2 / n_in))
        self.b1 = Tensor(np.zeros((n_hidden)))
        self.W2 = Tensor(rng.standard_normal((n_hidden, n_out)) * np.sqrt(2 / n_hidden))
        self.b2 = Tensor(np.zeros((n_out)))
        self.params:       list[Tensor] = [self.W1, self.b1, self.W2, self.b2]
        self.loss_history: list[float]  = []

        
    def forward(self, X):
        x = Tensor(X)
        z : Tensor = x @ self.W1 + self.b1
        a = z.sigmoid()
        logits: Tensor = a @ self.W2 + self.b2
        probs = logits.softmax()
        return probs
    
        
    def train(self, X, y, epochs: int=5, batch=128, lr=1e-3, verbose=True):
        opt = Adam(self.params, lr=lr)
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            X_shuf, y_shuf = X[idx], y[idx]
            ep_loss, nb = 0, 0
            
            for i in range(0, len(X), batch):
                xb = X_shuf[i:i + batch]
                yb = y_shuf[i:i + batch]
                opt.zero_grad()
                probs = self.forward(xb)
                loss: Tensor = _cross_entropy(probs, yb)
                loss.backward()
                opt.step()
                ep_loss += loss.data
                nb += 1
            avg = ep_loss / nb
            self.loss_history.append(avg)
                
            if verbose:
                d = 1/5*epochs if epochs < 10 else 1/10*epochs
                if epoch % d == 0:
                    print(f"epoch {epoch+1}/{epochs}, loss: {avg:.3f}")
                    
        return self.loss_history
                
        
    def predict(self, X):
        probs = self.forward(X)
        return probs.data.argmax(axis=1)

        

def _cross_entropy(probs: Tensor, y: np.ndarray):
    batch_size = len(y)
    rows = np.arange(batch_size)
    loss_value = -np.log(probs.data[rows, y] + 1e-15).mean()
    out = Tensor(data=loss_value, _prev=(probs,), _op='cross_entropy')
    
    def _backward():
        grad = np.zeros_like(probs.data)
        grad[rows, y] = -1.0 / (
            batch_size * (probs.data[rows, y] + 1e-15)
        )
        probs.grad += grad * out.grad
        
    out._backward = _backward
    return out

