# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import numpy as np


def isodata(array: np.ndarray, eps: float = 1e-6, max_iter: int = 100) -> float:
    """
    Returns optimal threshold for dividing the sequence of values using the ISODATA algorithm.

    The algorithm works by iteratively updating the threshold:
    \\[
    T_{k+1} = \\frac{1}{2} (\\text{mean}(y \\le T_k) + \\text{mean}(y > T_k))
    \\]

    Args:
        array (np.ndarray): numpy array with the values
        eps (float): tolerance for the threshold convergence
        max_iter (int): maximum number of iterations

    Returns:
        float: optimal threshold for dividing the sequence of values
    """
    if array.size == 0:
        return 0.0

    threshold = np.mean(array)

    for _ in range(max_iter):
        left_mask = array <= threshold
        right_mask = array > threshold

        if not np.any(left_mask) or not np.any(right_mask):
            break

        mean_left = np.mean(array[left_mask])
        mean_right = np.mean(array[right_mask])

        new_threshold = (mean_left + mean_right) / 2.0

        if abs(new_threshold - threshold) < eps:
            threshold = new_threshold
            break

        threshold = new_threshold

    return float(threshold)
