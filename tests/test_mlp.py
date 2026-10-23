import numpy as np


def test_mlp_training_is_producible():
    from nanonet.model import MLP
    
    X = np.array([
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)
    y = np.array([0, 1, 1, 0])
    
    model1 = MLP(2, 8, 2, seed=42)
    model2 = MLP(2, 8, 2, seed=42)
    
    model1.train(X, y, epochs=20, batch=4, verbose=False)
    model2.train(X, y, epochs=20, batch=4, verbose=False)
    
    for p1, p2 in zip(model1.params, model2.params):
        np.testing.assert_allclose(p1.data, p2.data)
    np.testing.assert_allclose(model1.loss_history, model2.loss_history)