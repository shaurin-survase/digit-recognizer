import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # silences a tensorflow warning

import numpy as np
from utils import relu, softmax

def load_params(path="params.npz"):
    """
    Loads trained parameters from disk.

    Returns:
        params: dict containing W1, b1, W2, b2
    """
    data = np.load(path)

    params = {
        "W1": data["W1"],
        "b1": data["b1"],
        "W2": data["W2"],
        "b2": data["b2"],
        "W3": data["W3"],
        "b3": data["b3"]
    }

    return params


def predict(X, params):
    """
    Arguments:
        X: np.ndarray of shape (784, m)

    Returns:
        preds: np.ndarray of shape (m,)
        probs: np.ndarray of shape (10, m)
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]

    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = relu(Z2)
    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)

    preds = np.argmax(A3, axis=0)

    return preds, A3

# Testing the model's accuracy on test data

def accuracy(preds, y_true):
    """
    Arguments:
        preds: np.ndarray of shape (m,)
        y_true: np.ndarray of shape (m,)

    Returns:
        acc: float
    """
    return np.mean(preds == y_true)

if __name__ == "__main__":
    from keras.datasets import mnist

    # Load test data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    m_test = x_test.shape[0]
    X_test = x_test.reshape(m_test, 784).T / 255.0

    # Load trained model
    params = load_params()

    # Predict
    preds, _ = predict(X_test, params)

    # Compute accuracy
    acc = accuracy(preds, y_test)

    print(f"Test accuracy: {acc * 100:.2f}%")

