# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import numpy as np


def trapezoid_left(x1:float, x2:float, x3:float, y1:float, y3:float) -> float:
    """
    Calculate the area of the trapezoid with corner coordinates (x2, 0), (x2, y2), (x3, 0), (x3, y3),
    where y2 is obtained by linear interpolation of (x1, y1) and (x3, y3) evaluated at x2.

    Args:
        x1 (float): x coordinate
        x2 (float): x coordinate
        x3 (float): x coordinate
        y1 (float): y coordinate
        y3 (float): y coordinate
    
    Returns:
        float: the area of the trapezoid
    """

    # Degenerate cases
    if x2 == x3 or x2 < x1:
        return (x3 - x2) * y1

    # Find y2 using linear interpolation and calculate the trapezoid area
    w = (x3 - x2) / (x3 - x1)
    y2 = y1 * w + y3 * (1 - w)
    return (x3 - x2) * (y2 + y3) / 2


def trapezoid_right(x1,  x2,  x3,  y1,  y3) -> float:
    """
    Calculate the area of the trapezoid with corner coordinates (x1, 0), (x1, y1), (x2, 0), (x2, y2),
    where y2 is obtained by linear interpolation of (x1, y1) and (x3, y3) evaluated at x2.

    Args:
        x1 (float): x coordinate
        x2 (float): x coordinate
        x3 (float): x coordinate
        y1 (float): y coordinate
        y3 (float): y coordinate
    
    Returns:
        float: the area of the trapezoid
    """
    
    # Degenerate cases
    if x2 == x1 or x2 > x3:
        return (x2 - x1) * y1

    # Find y2 using linear interpolation and calculate the trapezoid area
    w = (x3 - x2) / (x3 - x1)
    y2 = y1 * w + y3 * (1 - w)
    return (x2 - x1) * (y1 + y2) / 2


def last(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Computes the simple moving average.

    This version uses the last value as a aproximation.

    Args:
        values (np.ndarray): array of time series values
        width_before (float): (non-negative) width of rolling window before t_i
        width_after (float): (non-negative) width of rolling window after t_i
    
    Returns:
        np.ndarray: the result array after aplying the SMA kernel
    """

    left = right = 0
    t_left_new = t_right_new = right_area = 0

    n = len(values)
    # Trivial case
    if n == 0:
        return np.array([])

    # Initialize output
    rv = np.empty([n, 2])
    rv[0] = values[0]
    roll_area = left_area = values[0][1] * (width_before + width_after)

    # Apply rolling window
    for i in range(1, n):
        # Remove truncated area on left and right end
        roll_area -= (left_area + right_area)

        # Expand interval on right end
        t_right_new = values[i][0] + width_after
        while (right < n - 1) and (values[right + 1][0] <= t_right_new):
            right += 1
            roll_area += values[right - 1][1] * (values[right][0] - values[right - 1][0])

        # Shrink interval on left end
        t_left_new = values[i][0] - width_before
        while values[left][0] < t_left_new:
            roll_area -= values[left][1] * (values[left+1][0] - values[left][0])
            left += 1

        # Add truncated area on left and right end
        left_area = values[max(0, left-1)][1] * (values[left][0] - t_left_new)
        right_area = values[right][1] * (t_right_new - values[right][0])
        roll_area += left_area + right_area

        # Save SMA value for current time window
        y = roll_area / (width_before + width_after)
        rv[i] = np.array([values[i][0], y])
    return rv


def next(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Computes the simple moving average.

    This version uses the next value as a aproximation.

    Args:
        values (np.ndarray): array of time series values
        width_before (float): (non-negative) width of rolling window before t_i
        width_after (float): (non-negative) width of rolling window after t_i
    
    Returns:
        np.ndarray: the result array after aplying the SMA kernel
    """

    left = right = 0
    t_left_new = t_right_new = right_area = 0

    n = len(values)
    # Trivial case
    if n == 0:
        return np.array([])
    
    # Initialize output
    rv = np.empty([n, 2])
    rv[0] = values[0]
    roll_area = left_area = values[0][1] * (width_before + width_after)

    # Apply rolling window
    for i in range(1, n):
        # Remove truncated area on left and right end
        roll_area -= (left_area + right_area)

        # Expand interval on right end
        t_right_new = values[i][0] + width_after
        while (right < n - 1) and (values[right + 1][0] <= t_right_new):
            right += 1
            roll_area += values[right][1] * (values[right][0] - values[right - 1][0])

        # Shrink interval on left end
        t_left_new = values[i][0] - width_before
        while values[left][0] < t_left_new:
            roll_area -= values[left+1][1] * (values[left+1][0] - values[left][0])
            left += 1
    
        # Add truncated area on left and right end
        left_area = values[left][1] * (values[left][0] - t_left_new)
        right_area = values[right][1] * (t_right_new - values[right][0])
        roll_area += left_area + right_area

        # Save SMA value for current time window
        y = roll_area / (width_before + width_after)
        rv[i] = np.array([values[i][0], y])
    return rv
    

def linear(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Computes the simple moving average.

    This version uses a linear aproximation.

    Args:
        values (np.ndarray): array of time series values
        width_before (float): (non-negative) width of rolling window before t_i
        width_after (float): (non-negative) width of rolling window after t_i
    
    Returns:
        np.ndarray: the result array after aplying the SMA kernel
    """

    left = right = 0
    t_left_new = t_right_new = right_area = 0

    n = len(values)
    # Trivial case
    if n == 0:
        return np.array([])
    
    # Initialize output
    rv = np.empty([n, 2])
    rv[0] = values[0]
    roll_area = left_area = values[0][1] * (width_before + width_after)

    # Apply rolling window
    for i in range(1, n):
        # Remove truncated area on left and right end
        roll_area -= (left_area + right_area)

        # Expand interval on right end
        t_right_new = values[i][0] + width_after
        while (right < n - 1) and (values[right + 1][0] <= t_right_new):
            right += 1
            roll_area += (values[right][1] + values[right - 1][1])/2.0 * (values[right][0] - values[right - 1][0])

        # Shrink interval on left end
        t_left_new = values[i][0] - width_before
        while values[left][0] < t_left_new:
            roll_area -= (values[left][1]+values[left+1][1])/2.0 * (values[left+1][0] - values[left][0])
            left += 1
    
        # Add truncated area on left and right end
        left_area = trapezoid_left(values[max(0,left-1)][0], t_left_new, values[left][0], values[max(0,left-1)][1], values[left][1])
        right_area = trapezoid_right(values[right][0], t_right_new, values[min(right+1, n-1)][0], values[right][1], values[min(right+1, n-1)][1]) 
        roll_area += left_area + right_area

        # Save SMA value for current time window
        y = roll_area / (width_before + width_after)
        rv[i] = np.array([values[i][0], y])
    return rv
