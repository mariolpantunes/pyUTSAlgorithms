# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import numpy as np


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))


def zscore(xi, mean, std):
    return (xi-mean)/std


def linear_delta_mapping(points):
    xpoints = np.transpose(points)[0]
    ypoints = np.transpose(points)[1]

    tdelta =  xpoints[1:] - xpoints[:-1]
    linear_values = (ypoints[1:] + ypoints[:-1]) / 2.0

    return tdelta, linear_values


def zscore_linear(xi, points):
    
    if len(points) <= 1:
        raise Exception('The number of points is smaller than 2')

    weights, values = linear_delta_mapping(points)
    mean, std = weighted_avg_and_std(values, weights)
    return zscore(xi, mean, std)
