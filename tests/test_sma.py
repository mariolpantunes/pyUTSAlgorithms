import unittest

import numpy as np
import numpy.testing as npt

import uts.sma as sma


class TestSMA(unittest.TestCase):
    def test_sma_last(self):
        values = np.array([[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])
        result = sma.sma_last(values, 2.5, 1.0)
        desired = np.array([[0.0, 0.0], [1.0, 1.03], [1.2, 1.26], [2.3, 3.31], [2.9, 4.69], [5, 8.34]])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_sma_next(self):
        values = np.array([[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])
        result = sma.sma_next(values, 2.5, 1.0)
        desired = np.array([[0.0, 0.0], [1.0, 1.71], [1.2, 1.94], [2.3, 4.97], [2.9, 6.11], [5, 9.77]])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_sma_linear(self):
        values = np.array([[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])
        result = sma.sma_linear(values, 2.5, 1.0)
        desired = np.array([[0.0, 0.0], [1.0, 1.54], [1.2, 1.86], [2.3, 4.16], [2.9, 5.60], [5, 9.10]])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_empty_input(self):
        empty = np.array([]).reshape(0, 2)
        self.assertEqual(len(sma.sma_last(empty, 1, 1)), 0)
        self.assertEqual(len(sma.sma_next(empty, 1, 1)), 0)
        self.assertEqual(len(sma.sma_linear(empty, 1, 1)), 0)

    def test_single_point(self):
        single = np.array([[1.0, 10.0]])
        self.assertEqual(sma.sma_last(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(sma.sma_next(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(sma.sma_linear(single, 1, 1)[0, 1], 10.0)

    def test_zero_width(self):
        values = np.array([[0.0, 0.0], [1.0, 2.0], [1.2, 4.0]])
        npt.assert_almost_equal(sma.sma_last(values, 0.0, 0.0), values)
        npt.assert_almost_equal(sma.sma_next(values, 0.0, 0.0), values)
        npt.assert_almost_equal(sma.sma_linear(values, 0.0, 0.0), values)


if __name__ == "__main__":
    unittest.main()
