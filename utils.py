import numpy as np

def one_hot(y: np.ndarray, num_classes=10):
    """
    Arguments:
        y: np.ndarray of shape (m,) or (m, 1)
        num_classes: int

    Returns:
        Y: np.ndarray of shape (num_classes, m)
    """
    m = y.shape[0]

    if y.ndim == 2:
        y = y.T[0]

    out = np.empty(shape=(m, num_classes))

    eye = np.eye(num_classes)
    
    for i in range(m):
        out[i] = eye[y[i]]

    return out.T

def relu(Z):
    """
    Arguments:
        Z: np.ndarray of any shape

    Returns:
        A: np.ndarray, same shape as Z
    """
    return np.maximum(0, Z)

def relu_derivative(Z):
    """
    Arguments:
        Z: np.ndarray (pre-activation values)

    Returns:
        dZ: np.ndarray, same shape as Z
    """
    return (Z > 0).astype(np.float32)


def softmax(Z):
    """
    Arguments:
        Z: np.ndarray of shape (num_classes, m)

    Returns:
        A: np.ndarray of same shape
    """
    
    A = Z - np.max(Z, axis=0)
    A = np.exp(A)
    A /= np.sum(A, axis=0)
    return A

def cross_entropy_loss(Y_hat: np.ndarray, Y: np.ndarray):
    """
    Arguments:
        Y_hat: np.ndarray of shape (num_classes, m)
        Y: np.ndarray of shape (num_classes, m)

    Returns:
        loss: float
    """

    m = Y.shape[1]
    epsilon = 1e-8 # safeguard against log(0)

    loss = -np.sum(Y * np.log(Y_hat + epsilon)) / m
    return float(loss)

