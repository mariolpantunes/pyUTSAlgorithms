import unittest
import numpy as np
import numpy.testing as npt

from uts import ema


class TestEMA(unittest.TestCase):
    def test_list_int(self):
        values = np.array([[0.0, 0.0], [1.0, 2.0],
        [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])
        result = ema.linear(values, 1.5)
        desired = np.array([[0.0, 0.0], [1.0, .54],
        [1.2, .85], [2.3, 3.07], [2.9, 4.39], [5, 8.03]])
        npt.assert_almost_equal(result, desired, decimal=2)
        #self.assertEqual(result, 6)

if __name__ == '__main__':
    unittest.main()