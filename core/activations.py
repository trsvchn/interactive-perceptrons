import numpy as np


def heaviside(X, W, b):
    """Heaviside step function."""
    z = X @ W.T + b
    a = np.where(z >= 0, 1, 0)
    return a
