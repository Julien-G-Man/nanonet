from nanonet import ManualMLP as MLP
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main():
    X, y = fetch_openml(
        name='mnist_784', version=1, return_X_y=True, as_frame=False
    )

    X = X / 255.0
    y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    model = MLP(n_in=784, n_hidden=128, n_out=10)
    model.train(X_train, y_train, epochs=5)
    
    predictions = model.predict(X_test[:10])
    
    for i in range(10):
        print(f"""Predicted: {predictions[i]} 
              | true: {y_test[i]} 
              | {'ok' if predictions[i] == y_test[i] else 'x'}""")
    
    acc = (model.predict(X_test[:2000]) == y_test[:2000]).mean()
    print(f"\nTest accuracy: {acc*100:.1f}%")
    plot_loss_curve(model.loss_history)


def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("step")
    plt.ylabel("loss - cross entropy")
    plt.title("nanonet training curve (MNIST)")
    plt.savefig("img/mnist_loss.png")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()