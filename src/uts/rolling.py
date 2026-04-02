# coding: utf-8

__author__ = "Mário Antunes"
__version__ = "0.2"
__email__ = "mariolpantunes@gmail.com"
__status__ = "Development"


from typing import Callable, Optional

import numpy as np


def _get_window_indices(times: np.ndarray, width_before: float, width_after: float):
    """
    Computes the start and end indices for each rolling window.
    Window is (t_i - width_before, t_i + width_after]
    """
    # side='right' for left boundary: first index j such that times[j] > t_i - width_before
    left = np.searchsorted(times, times - width_before, side="right")
    # side='right' for right boundary: first index j such that times[j] > t_i + width_after
    # So the interval [left, right) contains indices j where t_i - width_before < times[j] <= t_i + width_after
    right = np.searchsorted(times, times + width_after, side="right")
    return left, right


def num_obs(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Rolling number of observations.

    \\[
    N(t_i) = \\sum_{j: t_j \\in (t_i - w_b, t_i + w_a]} 1
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i

    Returns:
        np.ndarray: array with (time, count)
    """
    if values.size == 0:
        return np.array([])

    times = values[:, 0]
    left, right = _get_window_indices(times, width_before, width_after)
    counts = right - left

    return np.column_stack((times, counts.astype(float)))


def sum(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Rolling sum of values.

    \\[
    S(t_i) = \\sum_{j: t_j \\in (t_i - w_b, t_i + w_a]} y_j
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i

    Returns:
        np.ndarray: array with (time, sum)
    """
    if values.size == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]
    left, right = _get_window_indices(times, width_before, width_after)

    cumsum = np.concatenate(([0], np.cumsum(y)))
    roll_sum = cumsum[right] - cumsum[left]

    return np.column_stack((times, roll_sum))


def sum_stable(
    values: np.ndarray, width_before: float, width_after: float
) -> np.ndarray:
    """
    Rolling sum of values using a stable algorithm (alias to sum as cumsum is stable).

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i

    Returns:
        np.ndarray: array with (time, sum)
    """
    return sum(values, width_before, width_after)


def mean(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Rolling mean of values.

    \\[
    \\mu(t_i) = \\frac{1}{N(t_i)} \\sum_{j: t_j \\in (t_i - w_b, t_i + w_a]} y_j
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i

    Returns:
        np.ndarray: array with (time, mean)
    """
    if values.size == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]
    left, right = _get_window_indices(times, width_before, width_after)

    cumsum = np.concatenate(([0], np.cumsum(y)))
    roll_sum = cumsum[right] - cumsum[left]
    counts = right - left

    # Handle empty windows with NaN to match C reference
    with np.errstate(divide="ignore", invalid="ignore"):
        roll_mean = np.where(counts > 0, roll_sum / counts, np.nan)

    return np.column_stack((times, roll_mean))


def var(
    values: np.ndarray, width_before: float, width_after: float, ddof: int = 1
) -> np.ndarray:
    """
    Rolling variance of values.

    \\[
    \\sigma^2(t_i) = \\frac{1}{N(t_i) - ddof} \\sum_{j: t_j \\in (t_i - w_b, t_i + w_a]} (y_j - \\mu(t_i))^2
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i
        ddof (int): delta degrees of freedom (reference uses ddof=1 for central moments)

    Returns:
        np.ndarray: array with (time, variance)
    """
    if values.size == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]
    left, right = _get_window_indices(times, width_before, width_after)

    cumsum_y = np.concatenate(([0], np.cumsum(y)))
    cumsum_y2 = np.concatenate(([0], np.cumsum(y**2)))

    sum_y = cumsum_y[right] - cumsum_y[left]
    sum_y2 = cumsum_y2[right] - cumsum_y2[left]
    n = right - left

    # Var = (sum(y^2) - (sum(y)^2)/n) / (n - ddof)
    with np.errstate(divide="ignore", invalid="ignore"):
        roll_var = np.where(n > ddof, (sum_y2 - (sum_y**2) / n) / (n - ddof), np.nan)

    # Clip small negative values due to precision
    roll_var = np.maximum(roll_var, 0.0)

    return np.column_stack((times, roll_var))


def std(
    values: np.ndarray, width_before: float, width_after: float, ddof: int = 1
) -> np.ndarray:
    """
    Rolling standard deviation of values.

    \\[
    \\sigma(t_i) = \\sqrt{\\sigma^2(t_i)}
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i
        ddof (int): delta degrees of freedom

    Returns:
        np.ndarray: array with (time, std)
    """
    v = var(values, width_before, width_after, ddof)
    v[:, 1] = np.sqrt(v[:, 1])
    return v


def max(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Rolling maximum of values.

    \\[
    M(t_i) = \\max \\{y_j : t_j \\in (t_i - w_b, t_i + w_a]\\}
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i

    Returns:
        np.ndarray: array with (time, max)
    """
    if values.size == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]
    left, right = _get_window_indices(times, width_before, width_after)

    roll_max = np.array(
        [
            np.max(y[low:high]) if low < high else np.nan
            for low, high in zip(left, right)
        ]
    )
    return np.column_stack((times, roll_max))


def min(values: np.ndarray, width_before: float, width_after: float) -> np.ndarray:
    """
    Rolling minimum of values.

    \\[
    m(t_i) = \\min \\{y_j : t_j \\in (t_i - w_b, t_i + w_a]\\}
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i

    Returns:
        np.ndarray: array with (time, min)
    """
    if values.size == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]
    left, right = _get_window_indices(times, width_before, width_after)

    roll_min = np.array(
        [
            np.min(y[low:high]) if low < high else np.nan
            for low, high in zip(left, right)
        ]
    )
    return np.column_stack((times, roll_min))


def apply(
    values: np.ndarray,
    width_before: float,
    width_after: float,
    func: Callable[[np.ndarray], float],
) -> np.ndarray:
    """
    Applies a function to a rolling window.

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i
        func (Callable): function to apply to the y-values in the window

    Returns:
        np.ndarray: array with (time, result)
    """
    if values.size == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]
    left, right = _get_window_indices(times, width_before, width_after)

    results = np.array(
        [func(y[low:high]) if low < high else np.nan for low, high in zip(left, right)]
    )
    return np.column_stack((times, results))


def product(
    values: np.ndarray,
    width_before: float,
    width_after: float,
    eps: Optional[float] = None,
) -> np.ndarray:
    """
    Rolling product of values.

    \\[
    P(t_i) = \\prod_{j: t_j \\in (t_i - w_b, t_i + w_a]} y_j
    \\]

    Args:
        values (np.ndarray): array of time series values (x, y)
        width_before (float): width of rolling window before t_i
        width_after (float): width of rolling window after t_i
        eps (Optional[float]): epsilon for zero detection (kept for compatibility)

    Returns:
        np.ndarray: array with (time, product)
    """
    if values.size == 0:
        return np.array([])

    times = values[:, 0]
    y = values[:, 1]
    left, right = _get_window_indices(times, width_before, width_after)

    # The original implementation used eps for zero detection.
    # We maintain the signature for backward compatibility.
    results = np.array(
        [
            np.prod(y[low:high]) if low < high else np.nan
            for low, high in zip(left, right)
        ]
    )
    return np.column_stack((times, results))
