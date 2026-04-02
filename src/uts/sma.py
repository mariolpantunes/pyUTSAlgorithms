# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import numpy as np


def _trapezoid_left(x1: float, x2: float, x3: float, y1: float, y3: float) -> float:
    if (x2 == x3) or (x2 < x1):
        return (x3 - x2) * y1
    w = (x3 - x2) / (x3 - x1)
    y2 = y1 * w + y3 * (1.0 - w)
    return (x3 - x2) * (y2 + y3) / 2.0


def _trapezoid_right(x1: float, x2: float, x3: float, y1: float, y3: float) -> float:
    if (x2 == x1) or (x2 > x3):
        return (x2 - x1) * y1
    w = (x3 - x2) / (x3 - x1)
    y2 = y1 * w + y3 * (1.0 - w)
    return (x2 - x1) * (y1 + y2) / 2.0


def last(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Computes the simple moving average using the 'last' interpolation scheme.

    \\[
    SMA(t_i) = \\frac{1}{w_b + w_a} \\int_{t_i - w_b}^{t_i + w_a} y_{last}(t) dt
    \\]

    Matches the reference C implementation.

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i

    Returns:
        np.ndarray: the result array (time, sma)
    """
    n = len(values)
    if n == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]

    rv = np.empty((n, 2))
    rv[0] = values[0]

    left = 0
    right = 0
    roll_area = y[0] * (width_before + width_after)
    left_area = roll_area
    right_area = 0.0

    for i in range(1, n):
        roll_area -= left_area + right_area

        t_right_new = times[i] + width_after
        while (right < n - 1) and (times[right + 1] <= t_right_new):
            right += 1
            roll_area += y[right - 1] * (times[right] - times[right - 1])

        t_left_new = times[i] - width_before
        while times[left] < t_left_new:
            roll_area -= y[left] * (times[left + 1] - times[left])
            left += 1

        left_area = y[max(0, left - 1)] * (times[left] - t_left_new)
        right_area = y[right] * (t_right_new - times[right])
        roll_area += left_area + right_area

        rv[i] = [times[i], roll_area / (width_before + width_after)]

    return rv


def next(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Computes the simple moving average using the 'next' interpolation scheme.

    \\[
    SMA(t_i) = \\frac{1}{w_b + w_a} \\int_{t_i - w_b}^{t_i + w_a} y_{next}(t) dt
    \\]

    Matches the reference C implementation.

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i

    Returns:
        np.ndarray: the result array (time, sma)
    """
    n = len(values)
    if n == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]

    rv = np.empty((n, 2))
    rv[0] = values[0]

    left = 0
    right = 0
    roll_area = y[0] * (width_before + width_after)
    left_area = roll_area
    right_area = 0.0

    for i in range(1, n):
        roll_area -= left_area + right_area

        t_right_new = times[i] + width_after
        while (right < n - 1) and (times[right + 1] <= t_right_new):
            right += 1
            roll_area += y[right] * (times[right] - times[right - 1])

        t_left_new = times[i] - width_before
        while times[left] < t_left_new:
            roll_area -= y[left + 1] * (times[left + 1] - times[left])
            left += 1

        left_area = y[left] * (times[left] - t_left_new)
        right_area = y[right] * (t_right_new - times[right])
        roll_area += left_area + right_area

        rv[i] = [times[i], roll_area / (width_before + width_after)]

    return rv


def linear(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Computes the simple moving average using the 'linear' interpolation scheme.

    \\[
    SMA(t_i) = \\frac{1}{w_b + w_a} \\int_{t_i - w_b}^{t_i + w_a} y_{linear}(t) dt
    \\]

    Matches the reference C implementation.

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i

    Returns:
        np.ndarray: the result array (time, sma)
    """
    n = len(values)
    if n == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]

    rv = np.empty((n, 2))
    rv[0] = values[0]

    left = 0
    right = 0
    roll_area = y[0] * (width_before + width_after)
    left_area = roll_area
    right_area = 0.0

    for i in range(1, n):
        roll_area -= left_area + right_area

        t_right_new = times[i] + width_after
        while (right < n - 1) and (times[right + 1] <= t_right_new):
            right += 1
            roll_area += (
                (y[right] + y[right - 1]) / 2.0 * (times[right] - times[right - 1])
            )

        t_left_new = times[i] - width_before
        while times[left] < t_left_new:
            roll_area -= (y[left] + y[left + 1]) / 2.0 * (times[left + 1] - times[left])
            left += 1

        left_area = _trapezoid_left(
            times[max(0, left - 1)],
            t_left_new,
            times[left],
            y[max(0, left - 1)],
            y[left],
        )
        right_area = _trapezoid_right(
            times[right],
            t_right_new,
            times[min(right + 1, n - 1)],
            y[right],
            y[min(right + 1, n - 1)],
        )
        roll_area += left_area + right_area

        rv[i] = [times[i], roll_area / (width_before + width_after)]

    return rv
