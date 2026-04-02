import unittest

import numpy as np

from src.uts import thresholding


class TestThresholding(unittest.TestCase):
    def test_isodata(self):
        array = np.array([1, 2, 3, 10, 11, 12])
        result = thresholding.isodata(array)
        # mean is 6.5. left: [1,2,3] mean 2. right: [10,11,12] mean 11. next threshold: (2+11)/2 = 6.5. stable.
        self.assertAlmostEqual(result, 6.5)

    def test_empty_input(self):
        empty = np.array([])
        self.assertEqual(thresholding.isodata(empty), 0.0)


if __name__ == "__main__":
    unittest.main()
