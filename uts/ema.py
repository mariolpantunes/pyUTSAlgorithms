# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import numpy as np
import math


# EMA_next(X, tau)
def next(values: np.ndarray, tau: float) -> np.ndarray:
    # values     ... array of time series values
    # times      ... array of observation times
    # n          ... number of observations, i.e. length of 'values' and 'times'
    # values_new ... array of length *n to store output time series values
    # tau        ... (positive) half-life of EMA kernel

    # double w;

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


# EMA_last(X, tau)
def last(values: np.ndarray, tau: float) -> np.ndarray:

    # values     ... array of time series values
    # times      ... array of observation times
    # n          ... number of observations, i.e. length of 'values' and 'times'
    # values_new ... array of length *n to store output time series values
    # tau        ... (positive) half-life of EMA kernel

    # double w;

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


# values     ... array of time series values
# tau        ... (positive) half-life of EMA kernel
def linear(values: np.ndarray, tau: float) -> np.ndarray:
    # times      ... array of observation times
    # n          ... number of observations, i.e. length of 'values' and 'times'
    # values_new ... array of length *n to store output time series values

    # double w, w2, tmp;
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
