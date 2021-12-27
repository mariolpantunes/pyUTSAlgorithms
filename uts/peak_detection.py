# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import math
import numpy as np
from uts import zscore, thresholding


def all_peaks(points: np.ndarray) -> np.ndarray:
    """
    Returns the index of all the peaks in a 2D curve.

    A peak (y) is defined by the following expression: \\( y_i-1 < y_i > y_i+1 \\)

    Args:
        points (np.ndarray): numpy array with the points (x, y)

    Returns:
        np.ndarray: the indexes of the peak points
    """

    peaks_idx = []

    for i in range(1, len(points) - 1):
        y0 = points[i-1][1]
        y = points[i][1]
        y1 = points[i+1][1]

        if y0 < y and y > y1:
            peaks_idx.append(i)

    return np.array(peaks_idx)


def highest_peak(points: np.ndarray, peaks_idx: np.ndarray) -> int:
    """
    Returns the index of the highest peak in a curve.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes

    Returns:
        int: the index of the highest peak
    """

    if peaks_idx.size != 0:
        peaks = points[peaks_idx][:, 1]
        idx = np.argmax(peaks)

        return peaks_idx[idx]
    else:
        return None


def significant_peaks(points: np.ndarray, peaks_idx: np.ndarray, h: float = 1.0) -> np.ndarray:
    """
    Returns the index of the significant peaks in a 2D curve.

    A significant peak (y) is defined by the following definition: \\( y > \\overline{peaks} + h \\times \\sigma(peaks) \\)

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes
        h (float): weight of the standart deviation

    Returns:
        np.ndarray: the indexes of the peak points
    """
    if peaks_idx.size != 0:
        peaks = points[peaks_idx][:, 1]
        m = np.mean(peaks)
        s = np.std(peaks)

        significant = []

        for i in range(0, len(peaks)):
            if peaks[i] >= (m+h*s):
                significant.append(peaks_idx[i])

        return np.array(significant)
    else:
        return np.array([])


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

    if i == len(points)-1:
        return i

    durations = points[i+1:, 0] - points[i:-1, 0]
    cumulative_durations = np.cumsum(durations)
    idx = cumulative_durations[cumulative_durations > tau]
    idx = np.argmax(cumulative_durations > tau)

    if idx == 0:
        return len(points)-1

    return i+idx+1


def zscore_peaks_values(points: np.ndarray, peaks_idx: np.ndarray) -> np.ndarray:
    """
    Returns the z-score values of each peak.

    The z-score values are computed within a neighborhood, based on the distance from the previous peak.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes

    Returns:
        np.ndarry: z-score values of each peak
    """
    peaks = points[peaks_idx]
    scores = []

    # first peak
    tau = peaks[0][0] - points[0][0]
    right = find_next_tau(points, peaks_idx[0], tau)
    score = math.fabs(zscore.zscore_linear(peaks[0][1], points[0: right+1]))
    scores.append(score)

    for i in range(1, len(peaks)):
        tau = peaks[i][0] - peaks[i-1][0]
        right = find_next_tau(points, peaks_idx[i], tau)
        score = math.fabs(zscore.zscore_linear(
            peaks[i][1], points[peaks_idx[i-1]: right+1]))
        scores.append(score)

    return np.array(scores)


def significant_zscore_peaks(points: np.ndarray, peaks_idx: np.ndarray, t: float = 1.0) -> np.ndarray:
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
    if peaks_idx.size != 0:
        scores = zscore_peaks_values(points, peaks_idx)
        return peaks_idx[scores > t]
    else:
        return np.array([])


def significant_zscore_peaks_iso(points: np.ndarray, peaks_idx: np.ndarray) -> np.ndarray:
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
    if peaks_idx.size != 0:
        scores = zscore_peaks_values(points, peaks_idx)
        positive_scores = scores[scores > 0]
        t = thresholding.isodata(positive_scores)
        return peaks_idx[scores > t]
    else:
        return np.array([])


def kneedle_peak_detection(points: np.ndarray, peaks_idx: np.ndarray, s: float = 1.0) -> np.ndarray:
    """
    Returns the index of the significant peaks in a 2D curve.

    This method uses the [Kneedle](https://ieeexplore.ieee.org/document/5961514) algorithm to select a significant peak.

    Args:
        points (np.ndarray): numpy array with the points (x, y)
        peaks_idx (np.ndarray): numpy array with the peak indexes
        t (float): sensitivity

    Returns:
        np.ndarray: the indexes of the peak points
    """
    if peaks_idx.size != 0:

        x = points[:, 0]
        y = points[:, 1]
        y_peaks = y[peaks_idx]
        n = len(x)

        t_lm = y_peaks - s * np.sum(np.diff(x)) / (n - 1)
        knee_points_index = []

        for index, i in enumerate(peaks_idx):
            for j in range(0, len(y)):
                if j > i and y[j] <= t_lm[index]:
                    knee_points_index.append(i)
                    break

        return np.array(knee_points_index)
    else:
        return np.array([])
