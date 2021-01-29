import unittest
import numpy as np
import numpy.testing as npt

from uts import sma


class TestEMA(unittest.TestCase):
    def test_sma_last(self):
        values = np.array([[0.0, 0.0], [1.0, 2.0],
        [1.2, 4.0], [2.3, 6], [2.9, 8], [5, 10]])
        result = sma.last(values, 2.5, 1.0)
        desired = np.array([[0.0, 0.0], [1.0, 1.03],
        [1.2, 1.26], [2.3, 3.31], [2.9, 4.69], [5, 8.34]])
        npt.assert_almost_equal(result, desired, decimal=2)


if __name__ == '__main__':
    unittest.main()