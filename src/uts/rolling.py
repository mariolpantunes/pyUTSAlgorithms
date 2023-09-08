# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import numpy as np


def _compensated_addition(sum_current: float, addend: float, comp:float) -> (float, float):
    """
    Compensated addition using Kahan (1965) summation algorithm.

    Args:
        sum_current (float): sum calculated so far
        addend (float): value to be added to 'sum'
        comp (float): accumulated numeric error so far
    
    Returns:
        (float, float): new values for sum_current and comp
    """
    
    addend -= comp
    sum_new = sum_current + addend
    comp = (sum_new - sum_current) - addend
    sum_current = sum_new

    return sum_current, comp


def num_obs(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Rolling number of observation values.

    Args:
        values (np.ndarray): array of time series values
        width_before (float): (non-negative) width of rolling window before t_i
        width_after (float): (non-negative) width of rolling window after t_i
    
    Returns:
        np.ndarray: array with time series values
    """

    n = len(values)
    # Trivial case
    if n == 0:
        return np.array([])
    
    # Initialize output
    rv = np.empty([n, 2])
    
    left = 0
    right = -1

    for i in range(0, n):
        # Expand window on the right
        while ((right < (n - 1)) and (values[(right + 1)][0] <= values[i][0] + width_after)):
            right += 1
        
        # Shrink window on the left
        while ((left < n) and (values[left][0] <= values[i][0] - width_before)):
            left+=1
        
        # Number of observations is equal to length of window
        rv[i] = np.array([values[i][0], (right - left + 1)])

    return rv


def sum(values: np.ndarray, width_before: float, width_after: float) -> float:
    """
    Rolling sum of observation values.

    Args:
        values (np.ndarray): array of time series values
        width_before (float): (non-negative) width of rolling window before t_i
        width_after (float): (non-negative) width of rolling window after t_i
    
    Returns:
        np.ndarray: array with time series values
    """

    n = len(values)
    # Trivial case
    if n == 0:
        return np.array([])
    
    # Initialize output
    rv = np.empty([n, 2])

    left = 0
    right = -1

    roll_sum = 0.0

    for i in range(0, n):
        # Expand window on the right
        while ((right < (n - 1)) and (values[(right + 1)][0] <= values[i][0] + width_after)):
            right += 1
            roll_sum += values[right][1]
        
        # Shrink window on the left
        while ((left < n) and (values[left][0] <= values[i][0] - width_before)):
            roll_sum -= values[left][1]
            left+=1
        
        # Update rolling sum
        rv[i] = np.array([values[i][0], roll_sum])

    return rv


def sum_stable(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Same as rolling_sum, but use Kahan (1965) summation algorithm to reduce numerical error.

    Args:
        values (np.ndarray): array of time series values
        width_before (float): (non-negative) width of rolling window before t_i
        width_after (float): (non-negative) width of rolling window after t_i
    
    Returns:
        np.ndarray: array with time series values
    """

    n = len(values)
    # Trivial case
    if n == 0:
        return np.array([])
    
    # Initialize output
    rv = np.empty([n, 2])

    left = 0
    right = -1

    roll_sum = 0.0
    comp = 0.0

    for i in range(0, n):
        # Expand window on the right
        while ((right < (n - 1)) and (values[(right + 1)][0] <= values[i][0] + width_after)):
            right += 1
            roll_sum, comp = _compensated_addition(roll_sum, values[right][1], comp)

        # Shrink window on the left
        while ((left < n) and (values[left][0] <= values[i][0] - width_before)):
            roll_sum, comp = _compensated_addition(roll_sum, -values[left][1], comp)
            left+=1
        
        # Update rolling sum
        rv[i] = np.array([values[i][0], roll_sum])

    return rv


def product(values: np.ndarray, width_before: float, width_after: float,
eps:float = np.finfo(np.float32).eps) -> np.ndarray:
    """
    Rolling product of observation values.

    Args:
        values (np.ndarray): array of time series values
        width_before (float): (non-negative) width of rolling window before t_i
        width_after (float): (non-negative) width of rolling window after t_i
        eps (float): epsilon (default: np.finfo(np.float32).eps)
    
    Returns:
        np.ndarray: array with time series values
    """

    n = len(values)
    # Trivial case
    if n == 0:
        return np.array([])
    
    # Initialize output
    rv = np.empty([n, 2])

    left = 0
    right = -1
    most_recent_zero = -1
    roll_product = 1.0

    for i in range(0, n):
        # Expand window on the right
        while ((right < (n - 1)) and (values[(right + 1)][0] <= values[i][0] + width_after)):
            right += 1
            roll_product *= values[right][1]

            # Save position of most recent zero
            if (math.fabs(0.0 - values[right][1]) <= eps):
                most_recent_zero = right
        
        # Shrink window on the left
        while ((left < n) and (values[left][0] <= values[i][0] - width_before)):
            # Don't need to update rolling product if zero drops out, because calculated from scratch below
            if (math.fabs(0.0-values[left][1]) > eps):
                roll_product /= values[left][1]
            left += 1
        
        # Update rolling product
        # -) need to calculate from scratch in case a zero dropped out of the window
        if ((roll_product == 0.0) and (most_recent_zero < left)):
            roll_product = 1.0
            #for (int pos=left; pos <= right; pos++)
            for pos in range(left, right+1):
                roll_product *= values[pos][1]
        
        rv[i] = np.array([values[i][0], roll_product])
    
    return rv