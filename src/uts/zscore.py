# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


from typing import Tuple

import numpy as np


def weighted_avg_and_std(
    values: np.ndarray, weights: np.ndarray
) -> Tuple[float, float]:
    """
    Return the weighted average and standard deviation.

    \\[
    \\mu_w = \\frac{\\sum w_i y_i}{\\sum w_i}, \\quad \\sigma_w = \\sqrt{\\frac{\\sum w_i (y_i - \\mu_w)^2}{\\sum w_i}}
    \\]

    Args:
        values (np.ndarray): numpy array with values
        weights (np.ndarray): numpy array with weights

    Returns:
        Tuple[float, float]: returns a tuple with the (weighted average, weighted standard deviation)
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return float(average), float(np.sqrt(variance))


def zscore(xi: float, mean: float, std: float) -> float:
    """
    Return the z-score for a single value.

    \\[ z = \\frac{x - \\mu}{\\sigma} \\]

    Args:
        xi (float): the single value
        mean (float): mean value from the sequence
        std (float): standard deviation from the sequence

    Returns:
        float: the z-score for a single value
    """
    if std != 0:
        return (xi - mean) / std
    else:
        return xi - mean


def linear_delta_mapping_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a linear mapping from the sequence of points.

    One way to estimate the z-score metric from an uneven sequence
    is to map the values linearly and compute the weight of each new value.
    The weight is proportional to the delta in the x axis.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        Tuple[np.ndarray, np.ndarray]: the weights (time deltas) and the interpolated y-values
    """
    x = points[:, 0]
    y = points[:, 1]
    return linear_delta_mapping(x, y)


def linear_delta_mapping(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a linear mapping from the sequence of points.

    The weights are defined as the time intervals between consecutive observations:
    \\[ w_i = t_{i+1} - t_i \\]
    The values are the averages of consecutive observations (linear interpolation):
    \\[ \\bar{y}_i = \\frac{y_i + y_{i+1}}{2} \\]

    Args:
        x (np.ndarray): values from the x axis
        y (np.ndarray): values from the y axis

    Returns:
        Tuple[np.ndarray, np.ndarray]: the weights (time deltas) and the interpolated y-values
    """
    tdelta = x[1:] - x[:-1]
    linear_values = (y[1:] + y[:-1]) / 2.0
    return tdelta, linear_values


def zscore_linear(xi: float, points: np.ndarray) -> float:
    """
    Return the z-score for a single value, using the linear mapping
    to deal with the uneven sequence of values.

    Args:
        xi (float): the single value
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        float: the z-score for a single value

    Raises:
        ValueError: If the length of points is smaller than 2.
    """
    if len(points) <= 1:
        raise ValueError("The number of points must be at least 2")

    weights, values = linear_delta_mapping_points(points)
    mean, std = weighted_avg_and_std(values, weights)
    return zscore(xi, mean, std)


def zscore_array_points(points: np.ndarray) -> np.ndarray:
    """
    Returns the z-score values for all the values in the sequence.

    It uses linear mapping to deal with the uneven sequence.

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the z-score values for all the values in the sequence
    """
    x = points[:, 0]
    y = points[:, 1]
    return zscore_array(x, y)


def zscore_array(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Returns the z-score values for all the values in the sequence.

    It uses linear mapping to deal with the uneven sequence.

    Args:
        x (np.ndarray): values from the x axis
        y (np.ndarray): values from the y axis

    Returns:
        np.ndarray: the z-score values for all the values in the sequence
    """
    if len(x) < 2:
        return np.zeros_like(y)

    weights, values = linear_delta_mapping(x, y)
    mean, std = weighted_avg_and_std(values, weights)
    if std != 0.0:
        return (y - mean) / std
    else:
        return y - mean
