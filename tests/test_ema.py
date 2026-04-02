import unittest

import numpy as np
import numpy.testing as npt

from src.uts import ema


class TestEMA(unittest.TestCase):
    def test_ema_next(self):
        values = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]]
        )
        result = ema.next(values, 1.5)
        desired = np.array(
            [[0.0, 0.0], [1.0, 0.97], [1.2, 1.35], [2.3, 3.77], [2.9, 5.16], [5, 8.81]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_ema_last(self):
        values = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]]
        )
        result = ema.last(values, 1.5)
        desired = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.2, 0.25], [2.3, 2.20], [2.9, 3.45], [5, 6.88]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_ema_linear(self):
        values = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]]
        )
        result = ema.linear(values, 1.5)
        desired = np.array(
            [[0.0, 0.0], [1.0, 0.54], [1.2, 0.85], [2.3, 3.07], [2.9, 4.39], [5, 8.03]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_empty_input(self):
        empty = np.array([]).reshape(0, 2)
        self.assertEqual(len(ema.last(empty, 1)), 0)
        self.assertEqual(len(ema.next(empty, 1)), 0)
        self.assertEqual(len(ema.linear(empty, 1)), 0)

    def test_single_point(self):
        single = np.array([[1.0, 10.0]])
        self.assertEqual(ema.last(single, 1)[0, 1], 10.0)
        self.assertEqual(ema.next(single, 1)[0, 1], 10.0)
        self.assertEqual(ema.linear(single, 1)[0, 1], 10.0)


if __name__ == "__main__":
    unittest.main()
