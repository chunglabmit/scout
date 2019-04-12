import unittest
import itertools
import torch
import numpy as np
from scout import detection


class TestTorchImpl(unittest.TestCase):
    def test_gradient(self):
        a = np.random.RandomState(1).uniform(size=(10, 10, 10))\
              .astype(np.float32)
        dz, dy, dx = [_.numpy() for _ in
                      detection.gradient(torch.from_numpy(a))]
        g = detection.gradient_numpy(a)[1:-1, 1:-1, 1:-1]
        np.testing.assert_almost_equal(dz, g[..., 0], 4)
        np.testing.assert_almost_equal(dy, g[..., 1], 4)
        np.testing.assert_almost_equal(dx, g[..., 2], 4)

    def test_structure_tensor(self):
        a = np.random.RandomState(2).uniform(size=(10, 10, 10))\
              .astype(np.float32)
        S_gpu = detection.structure_tensor(
            *detection.gradient(torch.from_numpy(a)))
        S_gpu = [[_.numpy() for _ in __] for __ in S_gpu]
        S_cpu = detection.structure_tensor_numpy(detection.gradient_numpy(a))
        for i, j in itertools.product(range(3), range(3)):
            np.testing.assert_almost_equal(
                S_gpu[i][j], S_cpu[2:-2, 2:-2, 2:-2,i, j])

    def test_inverse(self):
        r = np.random.RandomState(3)
        a = [[r.uniform(size=10) for _ in range(3)]
             for __ in range(3)]
        aa = [[torch.from_numpy(_) for _ in __] for __ in a]
        inv = [[_.numpy() for _ in __] for __ in detection._inverse(aa)]
        for i in range(10):
            matrix = np.array([[_[i] for _ in __] for __ in a])
            gpuinv = np.array([[_[i] for _ in __] for __ in inv])
            npinv = np.linalg.inv(matrix)
            np.testing.assert_almost_equal(gpuinv, npinv, 4)

    def test_weingarten(self):
        a = np.random.RandomState(4).uniform(size=(10, 10, 10))
        w_gpu = detection.weingarten(torch.from_numpy(a))
        w_gpu = [[_.numpy() for _ in __] for __ in w_gpu]
        w_cpu = detection.weingarten_numpy(a)
        for i, j in itertools.product(range(3), range(3)):
            np.testing.assert_almost_equal(w_gpu[i][j],
                                           w_cpu[2:-2, 2:-2, 2:-2, i, j],
                                           4)

    def test_eigen3(self):
        r = np.random.RandomState(5)
        a = [[None for __ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(i, 3):
                a[i][j] = a[j][i] = torch.from_numpy(r.uniform(size=10))
        e1, e2, e3 = [_.numpy() for _ in detection.eigen3(a)]
        for i in range(10):
            matrix = np.array([[_[i].numpy() for _ in __] for __ in a])
            ce1, ce2, ce3 = np.linalg.eigvalsh(matrix)
            self.assertAlmostEqual(e1[i], ce1, 4)
            self.assertAlmostEqual(e2[i], ce2, 4)
            self.assertAlmostEqual(e3[i], ce3, 4)


if __name__ == '__main__':
    unittest.main()
