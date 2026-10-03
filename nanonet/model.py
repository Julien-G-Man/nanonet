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
        self.params: list = [self.W1, self.b1, self.W2, self.b2]
        self.loss_history: list[float] = []

        
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
    """
    probs: Tensor with softmax probs (B, C)
    y: (B,) int labels
    return Tensor loss (scalar) that is differentiable
    """
    B = probs.data.shape[0]
    log_p = np.log(probs.data[np.arange(B), y] + 1e-15)
    loss_data = -log_p.mean()
    out = Tensor(data=loss_data, _prev=(probs,), _op='ce')
    
    def _backward():
        # dL/dprobs -> dL/dlogits fused trick: probs - one_hot
        dlogits = probs.data.copy()
        dlogits[np.arange(B), y] -= 1
        dlogits /= B
        # chain with out.grad (which is 1
        probs.grad += dlogits * out.grad
    out._backward = _backward
    return out

