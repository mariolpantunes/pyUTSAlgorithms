import unittest
import numpy as np
import numpy.testing as npt

from uts import gradient


class TestGradient(unittest.TestCase):
    def test_gradient_cfd_even(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])
        result = gradient.cfd(x, y)
        desired = np.array([3., 4.,  6.,  8., 10., 12., 14., 16., 17.])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_gradient_csd_even(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])
        result = gradient.csd(x, y)
        desired = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_gradient_cfd_uneven(self):
        x = np.array([1, 3, 4, 5, 6, 8, 9])
        y = np.array([1, 9, 16, 25, 36, 64, 81])
        result = gradient.cfd(x, y)
        desired = np.array([4.,  5.,  8., 10., 13., 15., 17.])
        npt.assert_almost_equal(result, desired, decimal=2)
    
    def test_gradient_csd_uneven(self):
        x = np.array([1, 3, 4, 5, 6, 8, 9])
        y = np.array([1, 9, 16, 25, 36, 64, 81])
        result = gradient.csd(x, y)
        desired = np.array([-0.5,  2. ,  2. ,  2. ,  2. ,  2. , -5.5])
        npt.assert_almost_equal(result, desired, decimal=2)

if __name__ == '__main__':
    unittest.main()
