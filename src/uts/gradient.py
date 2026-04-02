# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import numpy as np


def lagrange_derivative(
    x: float, x0: float, x1: float, x2: float, y0: float, y1: float, y2: float
) -> float:
    """
    Three point lagrange derivative.

    It computes the derivative of \\(x\\) based on three points:
    \\([(x_0, y_0), (x_1, y_1), (x_2, y_2)]\\)
    Depending on the value of \\(x\\), we can compute the forward,
    backward or central derivative.

    The first derivative is given by:
    \\[
    f'(x) \\approx y_0 \\frac{2x - x_1 - x_2}{(x_0 - x_1)(x_0 - x_2)} + y_1 \\frac{2x - x_0 - x_2}{(x_1 - x_0)(x_1 - x_2)} + y_2 \\frac{2x - x_0 - x_1}{(x_2 - x_0)(x_2 - x_1)}
    \\]

    Args:
        x (float): the value where the derivative will be computed
        x0 (float): first x value
        x1 (float): second x value
        x2 (float): third x value
        y0 (float): first y value
        y1 (float): second y value
        y2 (float): third y value

    Returns:
        float: the first derivative for \\(x\\) value
    """
    p0 = y0 * (2 * x - x1 - x2) / ((x0 - x1) * (x0 - x2))
    p1 = y1 * (2 * x - x0 - x2) / ((x1 - x0) * (x1 - x2))
    p2 = y2 * (2 * x - x0 - x1) / ((x2 - x0) * (x2 - x1))
    return p0 + p1 + p2


def cfd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the central first order derivative for uneven space sequences.

    This method uses the three point lagrange derivative to compute the
    derivative in uneven time series.
    The first and last elements are computed based on the forward and
    backward definition of the first derivative.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates

    Returns:
        np.ndarray: the first order derivative

    Raises:
        ValueError: If the length of x or y is smaller than 3.
    """
    if len(x) < 3:
        raise ValueError("cfd requires at least 3 points.")

    # Vectorized central derivative
    # For i in [1, n-2]:
    x0, x1, x2 = x[:-2], x[1:-1], x[2:]
    y0, y1, y2 = y[:-2], y[1:-1], y[2:]

    # Lagrange derivative at x1
    d1_central = (
        y0 * (x1 - x2) / ((x0 - x1) * (x0 - x2))
        + y1 * (2 * x1 - x0 - x2) / ((x1 - x0) * (x1 - x2))
        + y2 * (x1 - x0) / ((x2 - x0) * (x2 - x1))
    )

    # Forward derivative for the first point (at x[0])
    x0, x1, x2 = x[0], x[1], x[2]
    y0, y1, y2 = y[0], y[1], y[2]
    d1_first = lagrange_derivative(x0, x0, x1, x2, y0, y1, y2)

    # Backward derivative for the last point (at x[-1])
    x0, x1, x2 = x[-3], x[-2], x[-1]
    y0, y1, y2 = y[-3], y[-2], y[-1]
    d1_last = lagrange_derivative(x2, x0, x1, x2, y0, y1, y2)

    return np.concatenate(([d1_first], d1_central, [d1_last]))


def csd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the central second order derivative for uneven space sequences.

    The computation is based on the second order polynomial fitting.
    The second derivative is constant for each triplet of points.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates

    Returns:
        np.ndarray: the second order derivative

    Raises:
        ValueError: If the length of x or y is smaller than 3.
    """
    if len(x) < 3:
        raise ValueError("csd requires at least 3 points.")

    # Vectorized central second derivative
    x1, x2, x3 = x[:-2], x[1:-1], x[2:]
    y1, y2, y3 = y[:-2], y[1:-1], y[2:]

    # Second derivative formula for unevenly spaced points
    # Derived from Lagrange polynomial: 2 * [ y1/((x1-x2)(x1-x3)) + y2/((x2-x1)(x2-x3)) + y3/((x3-x1)(x3-x2)) ]
    d2_central = 2.0 * (
        y1 / ((x1 - x2) * (x1 - x3))
        + y2 / ((x2 - x1) * (x2 - x3))
        + y3 / ((x3 - x1) * (x3 - x2))
    )

    # First point (forward) and last point (backward) use the same parabola as their neighbors
    return np.concatenate(([d2_central[0]], d2_central, [d2_central[-1]]))


def finite_difference_weights(x0: float, x: np.ndarray, m: int = 1) -> np.ndarray:
    """
    Computes the weights for the m-th derivative at x0 using points in x.
    This implements Fornberg's algorithm for arbitrary grids.

    The algorithm recursively computes the weights \\( w_{i,j,k} \\) for the
    \\( k \\)-th derivative using the first \\( i \\) points.

    Args:
        x0 (float): the point where the derivative is to be estimated
        x (np.ndarray): the coordinates of the points in the stencil
        m (int): the order of the derivative (0 for interpolation, 1 for 1st derivative, etc.)

    Returns:
        np.ndarray: weights for each point in x
    """
    n = len(x)
    w = np.zeros((n, m + 1))
    w[0, 0] = 1.0
    c1 = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2 * c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    w[i, k] = (
                        c1 * (k * w[i - 1, k - 1] - (x[i - 1] - x0) * w[i - 1, k]) / c2
                    )
                w[i, 0] = -c1 * (x[i - 1] - x0) * w[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                w[j, k] = ((x[i] - x0) * w[j, k] - k * w[j, k - 1]) / (x[i] - x[j])
            w[j, 0] = (x[i] - x0) * w[j, 0] / (x[i] - x[j])
        c1 = c2
    return w[:, m]


def fd_derivative(
    x: np.ndarray, y: np.ndarray, m: int = 1, stencil_size: int = 3
) -> np.ndarray:
    """
    Computes the m-th order derivative using a moving stencil of specified size.
    Uses Fornberg's algorithm for weights.

    \\[
    f^{(m)}(x_i) \\approx \\sum_{j \\in \\text{stencil}(i)} w_j y_j
    \\]

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): y values
        m (int): derivative order
        stencil_size (int): number of points in the local stencil

    Returns:
        np.ndarray: the m-th derivative
    """
    n = len(x)
    if n < stencil_size:
        raise ValueError(f"Need at least {stencil_size} points for stencil.")

    half = stencil_size // 2
    deriv = np.zeros(n)

    for i in range(n):
        # Select stencil indices
        start = max(0, min(i - half, n - stencil_size))
        end = start + stencil_size

        stencil_x = x[start:end]
        stencil_y = y[start:end]

        weights = finite_difference_weights(x[i], stencil_x, m)
        deriv[i] = np.dot(weights, stencil_y)

    return deriv
