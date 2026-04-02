# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import numpy as np


def last(values: np.ndarray, tau: float) -> np.ndarray:
    """
    Computes the exponential moving average using the 'last' interpolation scheme.

    The recurrence relation is:
    \\[
    EMA(t_i) = EMA(t_{i-1}) e^{-(t_i - t_{i-1})/\tau} + y_{i-1} (1 - e^{-(t_i - t_{i-1})/\tau})
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        tau (float): half-life of EMA kernel

    Returns:
        np.ndarray: the result array (time, ema)
    """
    n = len(values)
    if n == 0:
        return np.array([])

    rv = np.empty([n, 2])
    rv[0] = values[0]

    t = values[:, 0]
    y = values[:, 1]

    # We use a loop for the recursive part to ensure numerical stability
    # for unevenly spaced time series.
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        w = np.exp(-dt / tau)
        # last interpolation: observation at t_{i-1} is used
        ema_prev = rv[i - 1][1]
        y_prev = y[i - 1]
        ema_curr = ema_prev * w + y_prev * (1.0 - w)
        rv[i] = [t[i], ema_curr]

    return rv


def next(values: np.ndarray, tau: float) -> np.ndarray:
    """
    Computes the exponential moving average using the 'next' interpolation scheme.

    The recurrence relation is:
    \\[
    EMA(t_i) = EMA(t_{i-1}) e^{-(t_i - t_{i-1})/\tau} + y_i (1 - e^{-(t_i - t_{i-1})/\tau})
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        tau (float): half-life of EMA kernel

    Returns:
        np.ndarray: the result array (time, ema)
    """
    n = len(values)
    if n == 0:
        return np.array([])

    rv = np.empty([n, 2])
    rv[0] = values[0]

    t = values[:, 0]
    y = values[:, 1]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        w = np.exp(-dt / tau)
        # next interpolation: observation at t_i is used
        ema_prev = rv[i - 1][1]
        y_curr = y[i]
        ema_curr = ema_prev * w + y_curr * (1.0 - w)
        rv[i] = [t[i], ema_curr]

    return rv


def linear(values: np.ndarray, tau: float) -> np.ndarray:
    """
    Computes the exponential moving average using the 'linear' interpolation scheme.

    The recurrence relation is:
    \\[
    EMA(t_i) = EMA(t_{i-1}) w + y_i (1 - w_2) + y_{i-1} (w_2 - w)
    \\]
    where \\( w = e^{-\\Delta t/\tau} \\) and \\( w_2 = (1 - w) / (\\Delta t/\tau) \\).

    Args:
        values (np.ndarray): array of time series values (x, y)
        tau (float): half-life of EMA kernel

    Returns:
        np.ndarray: the result array (time, ema)
    """
    n = len(values)
    if n == 0:
        return np.array([])

    rv = np.empty([n, 2])
    rv[0] = values[0]

    t = values[:, 0]
    y = values[:, 1]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        tmp = dt / tau
        w = np.exp(-tmp)

        # Numerical stability for small dt/tau using Taylor expansion
        if tmp > 1e-6:
            w2 = (1.0 - w) / tmp
        else:
            w2 = 1.0 - (tmp / 2.0) + (tmp**2 / 6.0) - (tmp**3 / 24.0)

        ema_prev = rv[i - 1][1]
        y_curr = y[i]
        y_prev = y[i - 1]

        ema_curr = ema_prev * w + y_curr * (1.0 - w2) + y_prev * (w2 - w)
        rv[i] = [t[i], ema_curr]

    return rv
