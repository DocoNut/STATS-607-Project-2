import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f

def kernel_gaussian(z: float) -> float:
    """
    Standard Gaussian kernel.

    Args:
        z (float): Input value.

    Returns:
        float: Kernel value at z.
    """
    return np.exp(-z * z / 2) / np.sqrt(2 * np.pi)


def kernel_derivative(z: float) -> float:
    """
    First derivative of the Gaussian kernel.

    Args:
        z (float): Input value.

    Returns:
        float: Derivative value at z.
    """
    return -z * kernel_gaussian(z)


def kernel_second_derivative(z: float) -> float:
    """
    Second derivative of the Gaussian kernel.

    Args:
        z (float): Input value.

    Returns:
        float: Second derivative value at z.
    """
    return (z * z - 1) * kernel_gaussian(z)


def kde(h: float, data: np.ndarray):
    """
    Construct a kernel density estimator (KDE) function with Gaussian kernel.

    Args:
        h (float): Bandwidth parameter.
        data (np.ndarray): 1D sample points.

    Returns:
        callable: Function f(y) that estimates density at y.
    """
    if not isinstance(h, (int, float)):
        raise TypeError("h is not number")
    
    if not isinstance(data, np.ndarray):
        raise TypeError("data is not np.ndarray")
    
    if h <= 0:
        raise ValueError("bandwidth should be greater than 0")
    
    n = len(data)
    if n < 0:
        raise ValueError("data is empty")

    def density(y: float) -> float:
        # Sum contributions from each data point
        total = 0.0
        for xi in data:
            total += kernel_gaussian((y - xi) / h) / (n * h)
        return total

    return density


def multi_kde(h: float, data: np.ndarray, bandwidth_coef: np.ndarray, d: int = 2, lam: float = 0):
    """
    Construct a multi-bandwidth KDE (linear combination of KDEs at different scales).

    Args:
        h (float): Base bandwidth multiplier.
        data (np.ndarray): 1D sample points.
        bandwidths (np.ndarray): Array of scale factors for bandwidths.
        d (int): Polynomial degree for coefficient calculation.
        lam (float): Regularization parameter.

    Returns:
        callable: Function f(y) that estimates density at y.
    """
    if not isinstance(h, (int, float)):
        raise TypeError("h is not number")
    
    if not isinstance(data, np.ndarray):
        raise TypeError("data is not np.ndarray")
    
    if h <= 0:
        raise ValueError("bandwidth should be greater than 0")
    
    n = len(data)
    if n < 0:
        raise ValueError("data is empty")
    
    from models.mulkde_coef import coef  # imported here to avoid circular import

    def density(y: float) -> float:
        # Compute coefficients for combining KDEs
        coeffs = coef(bandwidth_coef, degree=d, lam=lam)

        # Weighted sum of individual KDEs at scaled bandwidths
        total = 0.0
        for i in range(len(bandwidth_coef)):
            total += coeffs[i] * kde(bandwidth_coef[i] * h, data)(y)
        return total

    return density


def multi_kde_n0(h: float, data: np.ndarray, bandwidths: np.ndarray, d: int = 2, lam: float = 0):
    """
    Multi-bandwidth KDE that enforces non-negativity (clipped at 0).

    Args:
        h (float): Base bandwidth multiplier.
        data (np.ndarray): 1D sample points.
        bandwidths (np.ndarray): Array of scale factors for bandwidths.
        d (int): Polynomial degree for coefficient calculation.
        lam (float): Regularization parameter.

    Returns:
        callable: Function f(y) that estimates density at y (≥ 0).
    """
    def density(y: float) -> float:
        # Compute density but clip negative values to 0
        return np.maximum(multi_kde(h, data, bandwidths, d, lam)(y), 0.0)

    return density