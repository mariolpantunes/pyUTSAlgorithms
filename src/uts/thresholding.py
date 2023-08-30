# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np


def isodata(array: np.ndarray) -> float:
    """
    Returns optimal threshold for dividing the sequence of values.

    Args:
        array (np.ndarray): numpy array with the values
    
    Returns:
        float: optimal threshold for dividing the sequence of values
    """
    mean = array.mean()
    previous_mean = 0.0 
    
    while mean != previous_mean:
        mean_left = array[array <= mean].mean()
        mean_right = array[array > mean].mean()

        previous_mean = mean
        mean = (mean_left+mean_right) / 2.0

    return mean