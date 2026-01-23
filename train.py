import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # silences a tensorflow warning

import numpy as np
from keras.datasets import mnist
from utils import one_hot, relu, softmax, cross_entropy_loss

def load_data():
    """
    Returns:
        X_train: np.ndarray of shape (784, m)
        Y_train: np.ndarray of shape (10, m)
        X_test:  np.ndarray of shape (784, m_test)
        Y_test:  np.ndarray of shape (10, m_test)
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    m = x_train.shape[0]
    m_test = x_test.shape[0]

    x_train = x_train.reshape(m, 784).T / 255.0
    x_test = x_test.reshape(m_test, 784).T / 255.0

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    return x_train, y_train, x_test, y_test

def initialize_parameters():
    """
    Returns:
        params: dict containing W1, b1, W2, b2, W3, b3
    """
    W1 = np.random.randn(256, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((256, 1))

    W2 = np.random.randn(64, 256) * np.sqrt(2 / 256)
    b2 = np.zeros((64, 1))

    W3 = np.random.randn(10, 64) * np.sqrt(2 / 64)
    b3 = np.zeros((10, 1))

    return {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }

def forward(X, params):
    """
    Arguments:
        X: np.ndarray of shape (784, m)
        params: dict containing W1, b1, W2, b2

    Returns:
        cache: dict containing Z1, A1, Z2, A2
    """
    w1: np.ndarray = params["W1"]
    b1: np.ndarray = params["b1"]
    w2: np.ndarray = params["W2"]
    b2: np.ndarray = params["b2"]
    w3: np.ndarray = params["W3"]
    b3: np.ndarray = params["b3"]
    
    z1 = np.matmul(w1, X) + b1
    a1 = relu(z1)
    z2 = np.matmul(w2, a1) + b2
    a2 = relu(z2)
    z3 = np.matmul(w3, a2) + b3
    a3 = softmax(z3)

    return {
        "Z1": z1,
        "A1": a1,
        "Z2": z2,
        "A2": a2,
        "Z3": z3,
        "A3": a3
    }

def backward(X, Y, params, cache):
    """
    Arguments:
        X: np.ndarray of shape (784, m)
        Y: np.ndarray of shape (10, m)
        params: dict containing W1, b1, W2, b2, W3, b3
        cache: dict containing Z1, A1, Z2, A2, Z3, A3

    Returns:
        grads: dict containing dW1, db1, dW2, db2, dW3, db3
    """
    m = X.shape[1]

    # Unpack parameters
    W2 = params["W2"]
    W3 = params["W3"]

    # Unpack cache
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    # ----- Output layer (softmax + cross-entropy) -----
    dZ3 = A3 - Y                              # (10, m)
    dW3 = (1 / m) * (dZ3 @ A2.T)              # (10, 128)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    # ----- Backprop into 2nd hidden layer -----
    dA2 = W3.T @ dZ3                          # (128, m)
    dZ2 = dA2 * (Z2 > 0)                      # ReLU derivative
    dW2 = (1 / m) * (dZ2 @ A1.T)              # (128, 256)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # ----- Backprop into 1st hidden layer -----
    dA1 = W2.T @ dZ2                          # (256, m)
    dZ1 = dA1 * (Z1 > 0)                      # ReLU derivative
    dW1 = (1 / m) * (dZ1 @ X.T)               # (256, 784)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
        "dW3": dW3,
        "db3": db3,
    }

def update_params(params, grads, alpha):
    """
    Arguments:
        params: dict containing W1, b1, W2, b2
        grads: dict containing dW1, db1, dW2, db2
        lr: float (learning rate)

    Returns:
        params: updated parameters dict
    """

    return {
        "W1": params["W1"] - alpha * grads["dW1"],
        "b1": params["b1"] - alpha * grads["db1"],
        "W2": params["W2"] - alpha * grads["dW2"],
        "b2": params["b2"] - alpha * grads["db2"],
        "W3": params["W3"] - alpha * grads["dW3"],
        "b3": params["b3"] - alpha * grads["db3"],
    }

if __name__ == "__main__":
    # Hyperparameters
    epochs = 100
    learning_rate = 0.1
    batch_size = 64

    # Load data
    X_train, Y_train, X_test, Y_test = load_data()

    # Initialize parameters
    params = initialize_parameters()

    # Training loop
    for epoch in range(epochs):

        # ---- shuffle data ----
        perm = np.random.permutation(X_train.shape[1])
        X_shuffled = X_train[:, perm]
        Y_shuffled = Y_train[:, perm]

        epoch_loss = 0.0

        # ---- mini-batch loop ----
        for i in range(0, X_train.shape[1], batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[:, i:i+batch_size]

            # forward
            cache = forward(X_batch, params)

            # loss
            loss = cross_entropy_loss(cache["A3"], Y_batch)
            epoch_loss += loss

            # backward
            grads = backward(X_batch, Y_batch, params, cache)

            # update
            params = update_params(params, grads, learning_rate)

        # ---- logging ----
        epoch_loss /= (X_train.shape[1] // batch_size)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.6f}")

    # Save trained parameters
    np.savez("params.npz", **params)

    print("Training complete. Parameters saved to params.npz")


