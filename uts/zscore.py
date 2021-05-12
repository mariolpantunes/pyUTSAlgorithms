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
    return (xi-mean)/std


def linear_delta_mapping(points: np.ndarray) -> Tuple[float]:
    x = points[:, 0]
    y = points[:, 1]

    tdelta =  x[1:] - x[:-1]
    linear_values = (y[1:] + y[:-1]) / 2.0

    return tdelta, linear_values


def zscore_linear(xi: float, points: np.ndarray) -> float:
    
    if len(points) <= 1:
        raise Exception('The number of points is smaller than 2')

    weights, values = linear_delta_mapping(points)
    mean, std = weighted_avg_and_std(values, weights)
    return zscore(xi, mean, std)


def zscore_array(points: np.ndarray) -> np.ndarray:
    weights, values = linear_delta_mapping(points)
    mean, std = weighted_avg_and_std(values, weights)
    #print(f'Mean = {mean} and STD = {std}')
    y = points[:, 1]
    scores = (y - mean)/std
    return scores