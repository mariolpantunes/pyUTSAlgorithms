# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import numpy as np
from typing import Tuple


def weighted_avg_and_std(values: np.ndarray, weights: np.ndarray) -> Tuple[float]:
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    #print(f'values={values} weights={weights}')
    variance = np.average((values-average)**2, weights=weights)
    #print(f'STD = {math.sqrt(variance)}/{np.std(values)}')
    return (average, math.sqrt(variance))


def zscore(xi: float, mean: float, std: float) -> float:
    if std != 0:
        return (xi - mean)/std
    else:
        return xi-mean


def linear_delta_mapping_points(points: np.ndarray) -> Tuple[float]:
    x = points[:, 0]
    y = points[:, 1]
    return linear_delta_mapping(x, y)


def linear_delta_mapping(x: np.ndarray ,y: np.ndarray) -> Tuple[float]:
    tdelta =  x[1:] - x[:-1]
    linear_values = (y[1:] + y[:-1]) / 2.0
    return tdelta, linear_values


def zscore_linear(xi: float, points: np.ndarray) -> float:
    if len(points) <= 1:
        raise Exception('The number of points is smaller than 2')

    weights, values = linear_delta_mapping_points(points)
    mean, std = weighted_avg_and_std(values, weights)
    return zscore(xi, mean, std)


def zscore_array_points(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return zscore_array(x, y)

def zscore_array(x: np.ndarray ,y: np.ndarray) -> np.ndarray:
    weights, values = linear_delta_mapping(x, y)
    mean, std = weighted_avg_and_std(values, weights)
    if std != 0.0:
        return (y - mean)/std
    else:
        return y - mean
