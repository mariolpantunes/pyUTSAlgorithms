# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'

import numpy as np
import math


def trapezoid_left(x1,  x2,  x3,  y1,  y3):
    # Degenerate cases
    if x2 == x3 or x2 < x1:
        return (x3 - x2) * y1

    # Find y2 using linear interpolation and calculate the trapezoid area
    w = (x3 - x2) / (x3 - x1)
    y2 = y1 * w + y3 * (1 - w)
    return (x3 - x2) * (y2 + y3) / 2


# Calculate the area of the trapezoid with corner coordinates (x1, 0), (x1, y1), (x2, 0), (x2, y2),
# where y2 is obtained by linear interpolation of (x1, y1) and (x3, y3) evaluated at x2.
def trapezoid_right(x1,  x2,  x3,  y1,  y3):
    # Degenerate cases
    if x2 == x1 or x2 > x3:
        return (x2 - x1) * y1

    # Find y2 using linear interpolation and calculate the trapezoid area
    w = (x3 - x2) / (x3 - x1)
    y2 = y1 * w + y3 * (1 - w)
    return (x2 - x1) * (y1 + y2) / 2


# SMA_last(X, width)

def last(values: np.ndarray, width_before: float, width_after: float):
    # values       ... array of time series values
    # width_before ... (non-negative) width of rolling window before t_i
    # width_after  ... (non-negative) width of rolling window after t_i

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
            roll_area += values[right - 1][1] * \
                (values[right][0] - values[right - 1][0])

        # Shrink interval on left end
        t_left_new = values[i][0] - width_before
        while values[left][0] < t_left_new:
            roll_area -= values[left][1] * \
                (values[left+1][0] - values[left][0])
            left += 1

        # Add truncated area on left and right end
        left_area = values[max(0, left-1)][1] * (values[left][0] - t_left_new)
        right_area = values[right][1] * (t_right_new - values[right][0])
        roll_area += left_area + right_area

        # Save SMA value for current time window
        y = roll_area / (width_before + width_after)
        rv[i] = np.array([values[i][0], y])
    return rv


"""// SMA_next(X, width)
void sma_next(const double values[], const double times[], const int *n, double values_new[],
  const double *width_before, const double *width_after)
{
  // values       ... array of time series values
  // times        ... array of observation times
  // n            ... number of observations, i.e. length of 'values' and 'times'
  // values_new   ... array of length *n to store output time series values
  // width_before ... (non-negative) width of rolling window before t_i
  // width_after  ... (non-negative) width of rolling window after t_i
  
  int left = 0, right = 0;
  double t_left_new, t_right_new, roll_area, left_area, right_area = 0;
  
  // Trivial case
  if (*n == 0)
    return;
  
  // Initialize output
  values_new[0] = values[0];  
  roll_area = left_area = values[0] * (*width_before + *width_after);
  
  // Apply rolling window
  for (int i = 1; i < *n; i++) {
    // Remove truncated area on left and right end
    roll_area -= (left_area + right_area);
    
    // Expand interval on right end
    t_right_new = times[i] + *width_after;
    while ((right < *n - 1) && (times[right + 1] <= t_right_new)) {
      right++;
      roll_area += values[right] * (times[right] - times[right - 1]);
    }
    
    // Shrink interval on left end
    t_left_new = times[i] - *width_before;
    while (times[left] < t_left_new) {
      roll_area -= values[left+1] * (times[left+1] - times[left]);
      left++;  
    }
    
    // Add truncated area on left and rigth end
    left_area = values[left] * (times[left] - t_left_new);
    right_area = values[right] * (t_right_new - times[right]);
    roll_area += left_area + right_area;
    
    // Save SMA value for current time window
    values_new[i] = roll_area / (*width_before + *width_after);
  }
}


// SMA_linear(X, width)
void sma_linear(const double values[], const double times[], const int *n, double values_new[],
  const double *width_before, const double *width_after)
{
  // values       ... array of time series values
  // times        ... array of observation times
  // n            ... number of observations, i.e. length of 'values' and 'times'
  // values_new   ... array of length *n to store output time series values
  // width_before ... (non-negative) width of rolling window before t_i
  // width_after  ... (non-negative) width of rolling window after t_i
  
  int left = 0, right = 0;
  double t_left_new, t_right_new, roll_area, left_area, right_area = 0;
  
  // Trivial case
  if (*n == 0)
    return;
  
  // Initialize output
  values_new[0] = values[0];
  roll_area = left_area = values[0] * (*width_before + *width_after);
  
  // Apply rolling window
  for (int i = 1; i < *n; i++) {   
    // Remove truncated area on left and right end
    roll_area -= (left_area + right_area);
    
    // Expand interval on right end
    t_right_new = times[i] + *width_after;
    while ((right < *n - 1) && (times[right + 1] <= t_right_new)) {
      right++;
      roll_area += (values[right] + values[right - 1])/2 * (times[right] - times[right - 1]);
    }
    
    // Shrink interval on left end
    t_left_new = times[i] - *width_before;
    while (times[left] < t_left_new) {
      roll_area -= (values[left] + values[left+1]) / 2 *
        (times[left+1] - times[left]);
      left++;  
    }
    
    // Add truncated area on left and right end
    left_area = trapezoid_left(times[MAX(0, left-1)], t_left_new, times[left],
      values[MAX(0, left-1)], values[left]);
    right_area = trapezoid_right(times[right], t_right_new, times[MIN(right+1, *n-1)],
      values[right], values[MIN(right+1, *n-1)]);
    roll_area += left_area + right_area;
    
    // Save SMA value for current time window
    values_new[i] = roll_area / (*width_before + *width_after);
  }
}"""
