# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np


def num_obs(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Rolling number of observation values

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