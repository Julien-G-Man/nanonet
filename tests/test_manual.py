import numpy as np
from nanonet.manual import MLP


def test_manual_mlp_updates_parameters():
    np.random.seed(42)

    model = MLP(n_in=2, n_hidden=4, n_out=2)

    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])

    original_W1 = model.W1.copy()
    original_W2 = model.W2.copy()

    model.forward(X)
    loss_before = model.backward(y, lr=0.01)

    assert np.isfinite(loss_before)
    assert not np.array_equal(model.W1, original_W1)
    assert not np.array_equal(model.W2, original_W2)
  
    
def test_manual_mlp_learns_xor():
    np.random.seed(42)

    model = MLP(n_in=2, n_hidden=8, n_out=2)

    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    y = np.array([0, 1, 1, 0])

    model.train(X, y, epochs=1500, batch=4, lr=0.03, verbose=False)

    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, y)