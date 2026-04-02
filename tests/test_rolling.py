import unittest

import numpy as np
import numpy.testing as npt

from src.uts import rolling


class Test_Rolling(unittest.TestCase):
    def setUp(self):
        self.values = np.array(
            [[0.0, 0.0], [1.0, 2.0], [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]]
        )

    def test_num_obs(self):
        result = rolling.num_obs(self.values, 2.5, 1.0)
        desired = np.array(
            [[0.0, 2.0], [1.0, 3.0], [1.2, 3.0], [2.3, 5.0], [2.9, 4.0], [5, 2.0]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_sum(self):
        result = rolling.sum(self.values, 2.5, 1.0)
        desired = np.array(
            [[0.0, 2.0], [1.0, 6.0], [1.2, 6.0], [2.3, 20.0], [2.9, 20.0], [5, 18.0]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_sum_stable(self):
        result = rolling.sum_stable(self.values, 2.5, 1.0)
        desired = np.array(
            [[0.0, 2.0], [1.0, 6.0], [1.2, 6.0], [2.3, 20.0], [2.9, 20.0], [5, 18.0]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_product(self):
        result = rolling.product(self.values, 2.5, 1.0, eps=1e-10)
        desired = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.2, 0.0], [2.3, 0.0], [2.9, 384.0], [5, 80.0]]
        )
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_rolling_mean(self):
        result = rolling.mean(self.values, 2.5, 1.0)
        # Window for t=2.3: (2.3-2.5, 2.3+1.0] = (-0.2, 3.3] -> points at [0, 1, 1.2, 2.3, 2.9] -> values [0, 2, 4, 6, 8] -> mean 4.0
        self.assertEqual(result[3, 1], 4.0)

    def test_rolling_max(self):
        result = rolling.max(self.values, 2.5, 1.0)
        self.assertEqual(result[3, 1], 8.0)

    def test_rolling_min(self):
        result = rolling.min(self.values, 2.5, 1.0)
        self.assertEqual(result[3, 1], 0.0)

    def test_rolling_var(self):
        result = rolling.var(self.values, 2.5, 1.0)
        # values at t=2.3: [0, 2, 4, 6, 8] -> mean 4.0 -> var = (16+4+0+4+16)/4 = 10.0
        self.assertAlmostEqual(result[3, 1], 10.0)

    def test_rolling_apply(self):
        result = rolling.apply(self.values, 2.5, 1.0, np.median)
        self.assertEqual(result[3, 1], 4.0)

    def test_empty_input(self):
        empty = np.array([]).reshape(0, 2)
        self.assertEqual(len(rolling.sum(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.num_obs(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.mean(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.var(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.max(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.min(empty, 1, 1)), 0)
        self.assertEqual(len(rolling.apply(empty, 1, 1, np.mean)), 0)
        self.assertEqual(len(rolling.product(empty, 1, 1)), 0)

    def test_single_point(self):
        single = np.array([[1.0, 10.0]])
        self.assertEqual(rolling.sum(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(rolling.num_obs(single, 1, 1)[0, 1], 1.0)
        self.assertEqual(rolling.mean(single, 1, 1)[0, 1], 10.0)

        self.assertTrue(np.isnan(rolling.var(single, 1, 1)[0, 1]))
        self.assertEqual(rolling.max(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(rolling.min(single, 1, 1)[0, 1], 10.0)
        self.assertEqual(rolling.apply(single, 1, 1, np.mean)[0, 1], 10.0)
        self.assertEqual(rolling.product(single, 1, 1)[0, 1], 10.0)


if __name__ == "__main__":
    unittest.main()
