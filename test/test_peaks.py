import unittest
import numpy as np
import numpy.testing as npt

from uts import peak_detection


class Test_Peaks(unittest.TestCase):

    def test_find_next_tau(self):
        points = np.array(
            [[1.0, 1.0], [3.0, 2.0], [6.0, 3.0], [12.0, 4.0], [24.0, 5.0]])
        result = peak_detection.find_next_tau(points, 1, 5.0)
        desired = 3
        self.assertEqual(result, desired)

    def test_find_next_tau_limit(self):
        points = np.array(
            [[1.0, 1.0], [3.0, 2.0], [6.0, 3.0], [12.0, 4.0], [24.0, 5.0]])
        result = peak_detection.find_next_tau(points, 4, 5.0)
        desired = 4
        self.assertEqual(result, desired)

    def test_all_peaks(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
                           [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0],
                           [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0],
                           [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
                           [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        result = peak_detection.all_peaks(points)
        desired = np.array([2, 10, 13, 15])
        npt.assert_equal(result, desired)

    def test_all_valleys(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
                           [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0],
                           [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0],
                           [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
                           [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        result = peak_detection.all_valleys(points)
        desired = np.array([1, 9, 12, 14, 16])
        npt.assert_equal(result, desired)

    def test_highest_peak(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
                           [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0],
                           [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0],
                           [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
                           [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        peaks_idx = peak_detection.all_peaks(points)
        result = peak_detection.highest_peak(points, peaks_idx)
        desired = 15
        npt.assert_equal(result, desired)

    def test_highest_peak_empty(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
                           [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0],
                           [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0],
                           [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
                           [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        peaks_idx = np.array([])
        result = peak_detection.highest_peak(points, peaks_idx)
        desired = None
        npt.assert_equal(result, desired)

    def test_significant_peaks(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
                           [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0],
                           [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0],
                           [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
                           [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        peaks_idx = peak_detection.all_peaks(points)
        result = peak_detection.significant_peaks(points, peaks_idx)
        desired = np.array([15])
        npt.assert_equal(result, desired)

    def test_zscore_peaks_values(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
                           [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0],
                           [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0],
                           [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
                           [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        peaks_idx = peak_detection.all_peaks(points)
        result = peak_detection.zscore_peaks_values(points, peaks_idx)
        #np.array([2.12, 2.07, 1.21, 2.71])
        desired = np.array([2.12, 2.08, 1.12, 2.97])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_significant_zscore_peaks(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
                           [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0],
                           [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0],
                           [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
                           [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        peaks_idx = peak_detection.all_peaks(points)
        result = peak_detection.significant_zscore_peaks(points, peaks_idx)
        desired = np.array([2, 10, 13, 15])
        npt.assert_equal(result, desired)

    def test_significant_zscore_peaks_iso(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
                           [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0],
                           [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0],
                           [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
                           [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        peaks_idx = peak_detection.all_peaks(points)
        result = peak_detection.significant_zscore_peaks_iso(points, peaks_idx)
        desired = np.array([2, 10, 15])
        npt.assert_equal(result, desired)

    def test_kneedle_peak_detection_00(self):
        points = np.array([[.1, -.5], [.2, 0], [.3, 1.66],
                          [.4, 2.5], [.5, 3], [.6, 3.33],
                          [.7, 3.57], [.8, 3.75], [.9, 3.88]])
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        Dn = (points - pmin)/(pmax - pmin)
        x = Dn[:, 0]
        y = Dn[:, 1]
        y_d = y - x
        Dd = np.column_stack((x, y_d))
        peaks_idx = peak_detection.all_peaks(Dd)
        result = peak_detection.kneedle_peak_detection(Dd, peaks_idx)
        desired = np.array([3])
        npt.assert_equal(result, desired)

    def test_kneedle_peak_detection_01(self):
        points = np.array([[1.0, 5.0], [2.0, -2.0], [3.0, 3.0],
                           [4.0, 1.0], [5.0, 1.0], [6.0, 4.0], [7.0, 4.0],
                           [8.0, -3.0], [9.0, -3.0], [10.0, -5.0], [11.0, 6.0],
                           [12.0, 3.0], [13.0, 0.0], [14.0, 4.0], [15.0, -2.0],
                           [16.0, 7.0], [17.0, -7.0], [18.0, -3.0], [19.0, 0.0], [20.0, 5.0]])
        peaks_idx = peak_detection.all_peaks(points)
        result = peak_detection.kneedle_peak_detection(points, peaks_idx)
        desired = np.array([2, 10, 13, 15])
        npt.assert_equal(result, desired)


if __name__ == '__main__':
    unittest.main()
