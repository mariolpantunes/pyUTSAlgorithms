import unittest

import numpy as np
import numpy.testing as npt

import uts.rolling as rolling


class TestRolling(unittest.TestCase):
    def setUp(self):
        self.values = np.array([[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])

    def test_num_obs(self):
        result = rolling.rolling_num_obs(self.values, 2.5, 1.0)
        desired = np.array([[0.0, 2.0], [1.0, 3.0], [1.2, 3.0], [2.3, 5.0], [2.9, 4.0], [5, 2.0]])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_sum(self):
        result = rolling.rolling_sum(self.values, 2.5, 1.0)
        desired = np.array([[0.0, 2.0], [1.0, 6.0], [1.2, 6.0], [2.3, 20.0], [2.9, 20.0], [5, 18.0]])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_sum_stable(self):
        result = rolling.rolling_sum_stable(self.values, 2.5, 1.0)
        desired = np.array([[0.0, 2.0], [1.0, 6.0], [1.2, 6.0], [2.3, 20.0], [2.9, 20.0], [5, 18.0]])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_product(self):
        result = rolling.rolling_product(self.values, 2.5, 1.0, eps=1e-10)
        desired = np.array([[0.0, 0.0], [1.0, 0.0], [1.2, 0.0], [2.3, 0.0], [2.9, 384.0], [5, 80.0]])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_rolling_mean(self):
        result = rolling.rolling_mean(self.values, 2.5, 1.0)
        # Window for t=2.3: (2.3-2.5, 2.3+1.0] = (-0.2, 3.3] ->
        # points at [0, 1, 1.2, 2.3, 2.9] -> values [0, 2, 4, 6, 8] -> mean 4.0
        self.assertEqual(result[3, 1], 4.0)

    def test_rolling_max(self):
        result = rolling.rolling_max(self.values, 2.5, 1.0)
        self.assertEqual(result[3, 1], 8.0)

    def test_rolling_min(self):
        result = rolling.rolling_min(self.values, 2.5, 1.0)
        self.assertEqual(result[3, 1], 0.0)

    def test_rolling_var(self):
        result = rolling.rolling_var(self.values, 2.5, 1.0)
        # values at t=2.3: [0, 2, 4, 6, 8] -> mean 4.0 -> var = (16+4+0+4+16)/4 = 10.0
        self.assertAlmostEqual(result[3, 1], 10.0)

    def test_rolling_apply(self):
        result = rolling.rolling_apply(self.values, 2.5, 1.0, np.median)
        self.assertEqual(result[3, 1], 4.0)

    def test_empty_input(self):
        empty = np.array([]).reshape(0, 2)
        self.assertEqual(len(rolling.rolling_sum(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.rolling_num_obs(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.rolling_mean(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.rolling_var(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.rolling_max(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.rolling_min(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.rolling_apply(empty, 1, 1, np.mean)), 0)
        self.assertEqual(len(rolling.rolling_product(empty, 1, 1)), 0)

    def test_single_point(self):
        single = np.array([[1.0, 10.0]])
        self.assertEqual(rolling.rolling_sum(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(rolling.rolling_num_obs(single, 1, 1)[0, 1], 1.0)
        self.assertEqual(rolling.rolling_mean(single, 1, 1)[0, 1], 10.0)

        self.assertTrue(np.isnan(rolling.rolling_var(single, 1, 1)[0, 1]))
        self.assertEqual(rolling.rolling_max(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(rolling.rolling_min(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(rolling.rolling_apply(single, 1, 1, np.mean)[0, 1], 10.0)
        self.assertEqual(rolling.rolling_product(single, 1, 1)[0, 1], 10.0)


if __name__ == "__main__":
    unittest.main()
