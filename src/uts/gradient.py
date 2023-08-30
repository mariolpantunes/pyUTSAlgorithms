# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import numpy as np


def lagrange_derivative(x: float, x0: float, x1: float, x2: float, y0: float, y1: float, y2: float) -> float:
    """
    Three point lagrange derivative.

    It computes the derivative of \\(x\\) based on three points:
    \\([(x_0, y_0), (x_1, y_1), (x_2, y_2)]\\)
    Depending od the value of \\(x\\), we can compute the forward,
    backward or central derivative.

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
    p0 = y0 * (2*x-x1-x2) / ((x0-x1)*(x0-x2))
    p1 = y1 * (2*x-x0-x2) / ((x1-x0)*(x1-x2))
    p2 = y2 * (2*x-x0-x1) / ((x2-x0)*(x2-x1))
    return p0+p1+p2


def cfd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the central first order derivative for uneven space sequences.

    This method uses the three point lagrange derivative to compute the
    derivative in uneven time series.
    The first and last elements are computed beased on the forward and 
    backward definition of the first derivative.

    Args:
        x (np.ndarray): the value of the points in the x axis coordinates
        y (np.ndarray): the value of the points in the y axis coordinates

    Returns:
        np.ndarray: the first order derivative
    """
    d1 = []

    # compute the first point with the forward definition
    y0, y1, y2 = y[0:3]
    x0, x1, x2 = x[0:3]
    d = lagrange_derivative(x0, x0, x1, x2, y0, y1, y2)
    d1.append(d)

    # compute n-2 points with the central definition
    for i in range(1, len(x) - 1):
        y0, y1, y2 = y[i-1:i+2]
        x0, x1, x2 = x[i-1:i+2]
        d = lagrange_derivative(x1, x0, x1, x2, y0, y1, y2)
        d1.append(d)

    # compute the last point with the backwards definition
    y0, y1, y2 = y[-3:len(y)+1]
    x0, x1, x2 = x[-3:len(x)+1]
    d = lagrange_derivative(x2, x0, x1, x2, y0, y1, y2)
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
    y0, y1, y2 = y[0:3]
    x0, x1, x2 = x[0:3]
    d = 2.0 * ((x1-x0)*y2 - (x2-x0)*y1 + (x2-x1)*y0) / ((x1-x0)*(x2-x1)*(x2-x0))
    d2.append(d)

    # compute n-2 points with the central definition
    for i in range(1, len(x) - 1):
        y1, y2, y3 = y[i-1:i+2]
        x1, x2, x3 = x[i-1:i+2]
        d = (2*y1/((x2-x1)*(x3-x1))) - (2*y2/((x3-x2)*(x2-x1)))+(2*y3/((x3-x2)*(x3-x1)))
        d2.append(d)

    # compute the last point with the backwards definition
    y2, y1, y0 = y[-3:len(y)+1]
    x2, x1, x0 = x[-3:len(x)+1]
    d = 2.0 * ((x1-x2)*y0 - (x0-x2)*y1 + (x0-x1)*y2) / ((x1-x2)*(x0-x1)*(x0-x2))
    d2.append(d)

    return np.array(d2)
