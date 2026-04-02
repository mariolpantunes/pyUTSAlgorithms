import unittest

import numpy as np
import numpy.testing as npt

from src.uts import sma


class TestSMA(unittest.TestCase):
    def test_sma_last(self):
        values = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]]
        )
        result = sma.last(values, 2.5, 1.0)
        desired = np.array(
            [[0.0, 0.0], [1.0, 1.03], [1.2, 1.26], [2.3, 3.31], [2.9, 4.69], [5, 8.34]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_sma_next(self):
        values = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]]
        )
        result = sma.next(values, 2.5, 1.0)
        desired = np.array(
            [[0.0, 0.0], [1.0, 1.71], [1.2, 1.94], [2.3, 4.97], [2.9, 6.11], [5, 9.77]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_sma_linear(self):
        values = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]]
        )
        result = sma.linear(values, 2.5, 1.0)
        desired = np.array(
            [[0.0, 0.0], [1.0, 1.54], [1.2, 1.86], [2.3, 4.16], [2.9, 5.60], [5, 9.10]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_empty_input(self):
        empty = np.array([]).reshape(0, 2)
        self.assertEqual(len(sma.last(empty, 1, 1)), 0)
        self.assertEqual(len(sma.next(empty, 1, 1)), 0)
        self.assertEqual(len(sma.linear(empty, 1, 1)), 0)

    def test_single_point(self):
        single = np.array([[1.0, 10.0]])
        self.assertEqual(sma.last(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(sma.next(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(sma.linear(single, 1, 1)[0, 1], 10.0)


if __name__ == "__main__":
    unittest.main()
