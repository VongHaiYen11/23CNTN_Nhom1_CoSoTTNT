import numpy as np

def sphere(x: np.ndarray):
    """Calculate sphere function value (sum of squares).

    Parameters:
    x (np.ndarray): Input vector

    Returns:
    float: Sum of squares of all elements in x
    """
    return np.sum(x ** 2)
