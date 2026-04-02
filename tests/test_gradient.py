import unittest

import numpy as np
import numpy.testing as npt

from src.uts import gradient


class TestGradient(unittest.TestCase):
    def test_gradient_cfd_even(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])
        result = gradient.cfd(x, y)
        desired = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_gradient_cfd_even_2(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = x**3
        result = gradient.cfd(x, y)
        desired = np.array([1, 13, 28, 49, 76, 109, 148, 193, 241])
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
        desired = np.array([2, 6, 8, 10, 12, 16, 18])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_gradient_csd_uneven(self):
        x = np.array([1, 3, 4, 5, 6, 8, 9])
        y = np.array([1, 9, 16, 25, 36, 64, 81])
        result = gradient.csd(x, y)
        desired = np.array([2, 2, 2, 2, 2, 2, 2])
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_gradient_small_input(self):
        small = np.array([[1.0, 10.0], [2.0, 20.0]])
        with self.assertRaises(ValueError):
            gradient.cfd(small[:, 0], small[:, 1])

    def test_fd_derivative(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81])
        # Second derivative of y = x^2 is 2.
        result = gradient.fd_derivative(x, y, m=2, stencil_size=3)
        desired = np.full(9, 2.0)
        npt.assert_almost_equal(result, desired, decimal=2)

    def test_finite_difference_weights(self):
        x = np.array([0.0, 1.0, 2.0])
        x0 = 1.0
        m = 1
        # Central difference weights for 1st derivative at x=1 should be [-0.5, 0, 0.5]
        weights = gradient.finite_difference_weights(x0, x, m)
        desired = np.array([-0.5, 0.0, 0.5])
        npt.assert_almost_equal(weights, desired)


if __name__ == "__main__":
    unittest.main()
