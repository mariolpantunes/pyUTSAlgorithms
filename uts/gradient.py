# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import numpy as np


def cfd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the central first order derivative for uneven space sequences.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
    
    Returns:
        np.ndarray: the first order derivative
    """
    
    d1 = []

    # compute the first point with the forward definition
    d = (y[1] - y[0])/(x[1] - x[0])
    d1.append(d)

    # compute n-2 points with the central definition
    for i in range(1, len(x) - 1):
        d = (y[i+1] - y[i-1])/(x[i+1] - x[i-1])
        d1.append(d)

    # compute the last point with the backwards definition
    d = (y[-1] - y[-2]) / (x[-1] - x[-2])
    d1.append(d)

    return np.array(d1)


def csd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the central second order derivative for uneven space sequences.

    The computation is based on the second order polynomial fitting.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates
    
    Returns:
        np.ndarray: the second order derivative
    """
    d2 = []
    
    # compute the first point with the forward definition
    d = (y[2] - 2*y[1] + y[0]) / ((x[2]-x[1])*(x[1]-x[0]))
    d2.append(d)

    # compute n-2 points with the central definition
    for i in range(1, len(x) - 1):
        y1, y2, y3 = y[i-1:i+2]
        x1, x2, x3 = x[i-1:i+2]
        d = (2*y1/((x2-x1)*(x3-x1))) - \
            (2*y2/((x3-x2)*(x2-x1)))+(2*y3/((x3-x2)*(x3-x1)))
        d2.append(d)
    
    # compute the last point with the backwards definition
    d = (y[-1] - 2*y[-2] + y[-3]) / ((x[-1]-x[-2])*(x[-2]-x[-3]))
    d2.append(d)

    return np.array(d2)
