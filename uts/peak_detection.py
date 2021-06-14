# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import numpy as np
import math


from uts import zscore, thresholding


def all_peaks(points: np.ndarray) -> np.ndarray:
    """
    Returns the index of all the peaks in a 2D curve.

    A peak (y) is defined by the following expression: \( y_i-1 < y_i > y_i+1 \)

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

    peaks = points[peaks_idx][:,1]
    idx = np.argmax(peaks)
    
    return peaks_idx[idx]

def significant_peaks(points: np.ndarray, peaks_idx: np.ndarray, h: float = 1.0) -> np.ndarray:
    """
    Returns the index of the significant peaks in a 2D curve.

    A significant peak (y) is defined by the following definition: \( y > \overline{peaks} + h \times \sigma(peaks) \) 
    
    peak (y) is defined by the following expression: 

    Args:
        points (np.ndarray): numpy array with the points (x, y)
    
    Returns:
        np.ndarray: the indexes of the peak points
    """
    peaks = points[peaks_idx][:,1]
    m = np.mean(peaks)
    s = np.std(peaks)

    significant = []

    for i in range(0, len(peaks)):
        if peaks[i] > (m+h*s):
            significant.append(peaks_idx[i])

    return np.array(significant)


def find_next_tau(points, i, tau):
    if i == len(points)-1:
        return i
    
    durations = points[i+1:,0] - points[i:-1,0]
    cumulative_durations = np.cumsum(durations)
    idx = cumulative_durations[cumulative_durations>tau]
    idx = np.argmax(cumulative_durations>tau)
    
    if idx == 0:
        return len(points)-1
    
    return i+idx+1


def zscore_peaks_values(points, peaks_idx):
    peaks = points[peaks_idx]

    scores = []

    # k current peak
    k = 0
    # left part of the sliding window
    left = 0
    # i current point
    for i in range(0, len(points)):
        if peaks_idx[i]:
            # Zscore from first peak (k=0)
            
            if k == 0:
                tau =  peaks[k][0] - points[0][0]
            else:
                tau = peaks[k][0] - points[k-1][0]
            right = find_next_tau(points, i, tau)

            score = math.fabs(zscore.zscore_linear(peaks[k][1], points[left : right+1]))
            scores.append(score)

            # next left
            left = i
            k += 1
        else:
            scores.append(0)

    return np.array(scores)


def significant_zscore_peaks(points, peaks_idx, t=1.0):
    scores = zscore_peaks_values(points, peaks_idx)
    return scores > t


def significant_zscore_peaks_iso(points, peaks_idx):
    scores = zscore_peaks_values(points, peaks_idx)
    positive_scores = scores[scores > 0]
    t = thresholding.isodata(positive_scores)
    return scores > t
