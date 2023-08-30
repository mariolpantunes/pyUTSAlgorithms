# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import math


def next(values: np.ndarray, tau: float) -> np.ndarray:
    """
    Computes the exponential moving average.

    This version uses the next value as a aproximation.

    Args:
        values (np.ndarray): array of time series values
        tau (float): half-life of EMA kernel
    
    Returns:
        np.ndarray: the result array after aplying the EMA kernel
    """

    n = len(values)
    # Trivial case
    if n == 0:
        return np.array([])

    # Calculate ema recursively
    rv = np.empty([n, 2])
    rv[0] = values[0]

    for i in range(1, n):
        w = math.exp(-(values[i][0] - values[i-1][0]) / tau)
        y = rv[i-1][1]*w + values[i][1] * (1-w)
        rv[i] = np.array([values[i][0], y])
    return rv


def last(values: np.ndarray, tau: float) -> np.ndarray:
    """
    Computes the exponential moving average.

    This version uses the last value as a aproximation.

    Args:
        values (np.ndarray): array of time series values
        tau (float): half-life of EMA kernel
    
    Returns:
        np.ndarray: the result array after aplying the EMA kernel
    """

    n = len(values)
    # Trivial case
    if n == 0:
        return np.array([])

    # Calculate ema recursively
    rv = np.empty([n, 2])
    rv[0] = values[0]

    for i in range(1, n):
        w = math.exp(-(values[i][0] - values[i-1][0]) / tau)
        y = rv[i-1][1] * w + values[i-1][1] * (1-w)
        rv[i] = np.array([values[i][0], y])
    return rv



def linear(values: np.ndarray, tau: float) -> np.ndarray:
    """
    Computes the exponential moving average.

    This version uses a linear aproximation.

    Args:
        values (np.ndarray): array of time series values
        tau (float): half-life of EMA kernel
    
    Returns:
        np.ndarray: the result array after aplying the EMA kernel
    """

    n = len(values)

    # Trivial case
    if n == 0:
        return np.array([])

    # Calculate ema recursively
    rv = np.empty([n, 2])
    rv[0] = values[0]
    for i in range(1, n):
        tmp = (values[i][0] - values[i-1][0]) / tau
        w = math.exp(-tmp)
        if tmp > 1e-6:
            w2 = (1 - w) / tmp
        else:
            # Use Taylor expansion for numerical stability
            w2 = 1 - (tmp/2.0) + (tmp*tmp)/6.0 - (tmp*tmp*tmp)/24.0
        y = rv[i-1][1] * w + values[i][1] * \
            (1.0 - w2) + values[i-1][1] * (w2 - w)
        rv[i] = np.array([values[i][0], y])
    return rv
