import unittest
import numpy as np
import numpy.testing as npt

from uts import zscore


class TestZScore(unittest.TestCase):
    def test_linear_mapping(self):
        values = np.array([[1.0, 1.0], [3.0, 2.0], [6.0, 3.0]])
        result = zscore.linear_delta_mapping_points(values)
        desired = (np.array([2.0, 3.0]), np.array([1.5, 2.5]))
        npt.assert_almost_equal(result, desired, decimal=2)
    
    def test_zscore(self):
        x = m = s = 1.0
        desired = 0.0
        result = zscore.zscore(x,m,s)
        self.assertEqual(result, desired)

    def test_zscore_linear(self):
        x = 1.0
        points = np.array([[1.0, 1.0], [3.0, 2.0], [6.0, 3.0]])
        result = zscore.zscore_linear(x, points)
        desired = -2.2453655975512468
        self.assertAlmostEqual(result, desired, places=6)
    
    def test_zscore_array(self):
        points = np.array([[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]])
        result = zscore.zscore_array_points(points)
        desired = np.array([-1.76776695, -1.06066017, -0.35355339,  0.35355339,  1.06066017,  1.76776695])
        #print(result)
        np.testing.assert_array_almost_equal(result, desired, decimal=6)

if __name__ == '__main__':
    unittest.main()