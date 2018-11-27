import numpy as np


def heaviside(z):
    """Heaviside step function."""
    a = np.where(z >= 0, 1, 0)
    return a
