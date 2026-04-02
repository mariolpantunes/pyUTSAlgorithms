# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


import math
from typing import Optional

import numpy as np

from . import thresholding, zscore


def all_peaks(points: np.ndarray) -> np.ndarray:
    """
    Returns the index of all the peaks in a 2D curve.

    A peak (y) is defined by the following expression: \\( y_{i-1} < y_i > y_{i+1} \\)

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the indexes of the peak points
    """
    if len(points) < 3:
        return np.array([], dtype=int)

    y = points[:, 1]
    # Vectorized peak detection
    peaks = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
    return np.where(peaks)[0] + 1


def all_valleys(points: np.ndarray) -> np.ndarray:
    """
    Returns the index of all the valleys in a 2D curve.

    A valley (y) is defined by the following expression: \\( y_{i-1} > y_i < y_{i+1} \\)

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the indexes of the peak points
    """
    if len(points) < 3:
        return np.array([], dtype=int)

    y = points[:, 1]
    # Vectorized valley detection
    valleys = (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
    return np.where(valleys)[0] + 1


def highest_peak(points: np.ndarray, peaks_idx: np.ndarray) -> Optional[int]:
    """
    Returns the index of the highest peak in a curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes

    Returns:
        Optional[int]: the index of the highest peak, or None if no peaks
    """
    if peaks_idx.size == 0:
        return None

    peaks_values = points[peaks_idx, 1]
    return peaks_idx[np.argmax(peaks_values)]


def significant_peaks(
    points: np.ndarray, peaks_idx: np.ndarray, h: float = 1.0
) -> np.ndarray:
    """
    Returns the index of the significant peaks in a 2D curve.

    A significant peak (y) is defined by: \\( y > \\overline{peaks} + h \\times \\sigma(peaks) \\)

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes
        h (float): weight of the standard deviation

    Returns:
        np.ndarray: the indexes of the significant peak points
    """
    if peaks_idx.size == 0:
        return np.array([], dtype=int)

    peaks_values = points[peaks_idx, 1]
    m = np.mean(peaks_values)
    s = np.std(peaks_values)

    mask = peaks_values >= (m + h * s)
    return peaks_idx[mask]


def find_next_tau(points: np.ndarray, i: int, tau: float) -> int:
    """
    Returns the index of the next step.

    Finds the index of the next element given a specified tau value.
    This is necessary since the sequence is uneven.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        i (int): index where the search starts
        tau (float): time duration that needs to be exceeded

    Returns:
        int: the index of the next step
    """
    n = len(points)
    if i >= n - 1:
        return n - 1

    times = points[i:, 0]
    cumulative_durations = times - times[0]

    # Find the first index where cumulative_duration > tau
    # We ignore the first element (which is 0.0)
    mask = cumulative_durations[1:] > tau
    if not np.any(mask):
        return n - 1

    idx = np.argmax(mask)
    return i + int(idx) + 1


def zscore_peaks_values(points: np.ndarray, peaks_idx: np.ndarray) -> np.ndarray:
    """
    Returns the z-score values of each peak.

    The z-score values are computed within a neighborhood, based on the distance from the previous peak.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes

    Returns:
        np.ndarray: z-score values of each peak
    """
    if peaks_idx.size == 0:
        return np.array([])

    scores = []

    # Process each peak
    for j, idx in enumerate(peaks_idx):
        if j == 0:
            # First peak: neighborhood from start to tau after peak
            tau = points[idx, 0] - points[0, 0]
            left_idx = 0
        else:
            # Subsequent peaks: neighborhood from previous peak to tau after current peak
            tau = points[idx, 0] - points[peaks_idx[j - 1], 0]
            left_idx = peaks_idx[j - 1]

        right_idx = find_next_tau(points, idx, tau)
        score = math.fabs(
            zscore.zscore_linear(points[idx, 1], points[left_idx : right_idx + 1])
        )
        scores.append(score)

    return np.array(scores)


def significant_zscore_peaks(
    points: np.ndarray, peaks_idx: np.ndarray, t: float = 1.0
) -> np.ndarray:
    """
    Returns the index of the significant peaks in a 2D curve.

    This method uses the zscore metric to select a significant peak.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes
        t (float): threshold of the z-score metric

    Returns:
        np.ndarray: the indexes of the peak points
    """
    if peaks_idx.size == 0:
        return np.array([], dtype=int)

    scores = zscore_peaks_values(points, peaks_idx)
    return peaks_idx[scores > t]


def significant_zscore_peaks_iso(
    points: np.ndarray, peaks_idx: np.ndarray
) -> np.ndarray:
    """
    Returns the index of the significant peaks in a 2D curve.

    This method uses the zscore metric to select a significant peak.
    The threshold is computed based on the iso_data thresholding algorithm.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes

    Returns:
        np.ndarray: the indexes of the peak points
    """
    if peaks_idx.size == 0:
        return np.array([], dtype=int)

    scores = zscore_peaks_values(points, peaks_idx)
    positive_scores = scores[scores > 0]

    if positive_scores.size == 0:
        return np.array([], dtype=int)

    t = thresholding.isodata(positive_scores)
    return peaks_idx[scores > t]


def kneedle_peak_detection(
    points: np.ndarray, peaks_idx: np.ndarray, s: float = 1.0
) -> np.ndarray:
    """
    Returns the index of the significant peaks in a 2D curve.

    This method uses the [Kneedle](https://ieeexplore.ieee.org/document/5961514) algorithm to select a significant peak.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes
        s (float): sensitivity

    Returns:
        np.ndarray: the indexes of the peak points
    """
    if peaks_idx.size == 0:
        return np.array([], dtype=int)

    x = points[:, 0]
    y = points[:, 1]
    y_peaks = y[peaks_idx]
    valleys_idx = set(all_valleys(points))
    peaks_idx_set = set(peaks_idx)

    # Calculate threshold for each peak
    # The sensitivity s determines how much the value must drop after a peak
    t_lm = y_peaks - s * np.abs(np.diff(x).mean())

    knee_points_index = []
    threshold = 0.0
    knee_idx = -1

    peak_count = 0
    for i in range(len(y)):
        if i in peaks_idx_set:
            threshold = t_lm[peak_count]
            peak_count += 1
            knee_idx = i

        if i in valleys_idx:
            threshold = 0.0

        if knee_idx != -1 and y[i] <= threshold:
            knee_points_index.append(knee_idx)
            knee_idx = -1

    return np.array(knee_points_index, dtype=int)


def peak_prominence(points: np.ndarray, peaks_idx: np.ndarray) -> np.ndarray:
    """
    Calculates the prominence of each peak.
    The prominence of a peak is the vertical distance between the peak and its lowest contour line.

    For each peak \\( y_i \\), its prominence is defined as:
    \\[
    P_i = y_i - \\max(\\text{left\\_min}, \\text{right\\_min})
    \\]
    where left\\_min and right\\_min are the minimum values between the peak and the first points to its
    left and right that are higher than the peak itself.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes

    Returns:
        np.ndarray: prominence values for each peak
    """
    if peaks_idx.size == 0:
        return np.array([])

    y = points[:, 1]
    n = len(y)
    prominences = np.zeros(len(peaks_idx))

    for i, idx in enumerate(peaks_idx):
        # Find left base
        left_min = y[idx]
        for j in range(idx - 1, -1, -1):
            if y[j] > y[idx]:
                break
            if y[j] < left_min:
                left_min = y[j]

        # Find right base
        right_min = y[idx]
        for j in range(idx + 1, n):
            if y[j] > y[idx]:
                break
            if y[j] < right_min:
                right_min = y[j]

        # Prominence is the height above the higher of the two bases
        base = max(left_min, right_min)
        prominences[i] = y[idx] - base

    return prominences


def significant_prominence_peaks(
    points: np.ndarray, peaks_idx: np.ndarray, threshold: float
) -> np.ndarray:
    """
    Returns the index of the significant peaks based on prominence.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes
        threshold (float): prominence threshold

    Returns:
        np.ndarray: the indexes of the significant peak points
    """
    if peaks_idx.size == 0:
        return np.array([], dtype=int)

    prominences = peak_prominence(points, peaks_idx)
    return peaks_idx[prominences >= threshold]
