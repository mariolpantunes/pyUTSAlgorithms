import unittest
import numpy as np
import numpy.testing as npt

from uts import rolling


class Test_Rolling(unittest.TestCase):
    def test_ema_next(self):
        values = np.array([[0.0, 0.0], [1.0, 2.0],
        [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])
        result = rolling.num_obs(values, 2.5, 1.0)
        desired = np.array([[0.0, 2.0], [1.0, 3.0],
        [1.2, 3.0], [2.3, 5.0], [2.9, 4.0], [5, 2.0]])
        npt.assert_almost_equal(result, desired, decimal=2)