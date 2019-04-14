import unittest
import itertools
import torch
import numpy as np
from scout import curvature


class TestEigenvaulesOfWeingartenNumpy(unittest.TestCase):

    def test_isotropic(self):
        # Make a sphere
        sphere = np.clip(
            50 - np.sqrt(np.sum(np.square(np.mgrid[-50:51, -50:51, -50:51]), 0)),
            0, 50)
        e = curvature.eigvals_of_weingarten_numpy(sphere)
        for eidx in range(3):
            minidx = np.argmin(e[..., eidx])
            minz = minidx // 101 // 101
            miny = (minidx // 101) % 101
            minx = minidx % 101
            self.assertEqual(minz, 50)
            self.assertEqual(miny, 50)
            self.assertEqual(minx, 50)

    def test_anisotropic(self):
        grid = np.mgrid[-50:51, -50:51, -50:51].astype(np.float32) * \
            np.array([2.0, 1.0, .5]).reshape(3, 1, 1, 1)
        sphere = np.clip(
            25 - np.sqrt(np.sum(np.square(grid), 0)),
            0, 25)
        e = curvature.eigvals_of_weingarten_numpy(sphere, zum=2.0, yum=1.0, xum=.5)

        # The value at z = 4 should be about the same as y = 8 and x = 16
        # because the micron distance is similar.

        for eidx in range(3):
            zval = e[54, 50, 50, eidx]
            yval = e[50, 58, 50, eidx]
            xval = e[50, 50, 66, eidx]
            self.assertAlmostEqual(zval, yval, 2)
            self.assertAlmostEqual(zval, xval, 2)


class TestTorchImpl(unittest.TestCase):
    def test_gradient(self):
        a = np.random.RandomState(1).uniform(size=(10, 10, 10))\
              .astype(np.float32)
        dz, dy, dx = [_.numpy() for _ in
                      curvature.gradient(torch.from_numpy(a))]
        g = curvature.gradient_numpy(a)[1:-1, 1:-1, 1:-1]
        np.testing.assert_almost_equal(dz, g[..., 0], 4)
        np.testing.assert_almost_equal(dy, g[..., 1], 4)
        np.testing.assert_almost_equal(dx, g[..., 2], 4)

    def test_structure_tensor(self):
        a = np.random.RandomState(2).uniform(size=(10, 10, 10))\
              .astype(np.float32)
        S_gpu = curvature.structure_tensor(
            *curvature.gradient(torch.from_numpy(a)))
        S_gpu = [[_.numpy() for _ in __] for __ in S_gpu]
        S_cpu = curvature.structure_tensor_numpy(curvature.gradient_numpy(a))
        for i, j in itertools.product(range(3), range(3)):
            np.testing.assert_almost_equal(
                S_gpu[i][j], S_cpu[2:-2, 2:-2, 2:-2,i, j])

    def test_inverse(self):
        r = np.random.RandomState(3)
        a = [[r.uniform(size=10) for _ in range(3)]
             for __ in range(3)]
        aa = [[torch.from_numpy(_) for _ in __] for __ in a]
        inv = [[_.numpy() for _ in __] for __ in curvature._inverse(aa)]
        for i in range(10):
            matrix = np.array([[_[i] for _ in __] for __ in a])
            gpuinv = np.array([[_[i] for _ in __] for __ in inv])
            npinv = np.linalg.inv(matrix)
            np.testing.assert_almost_equal(gpuinv, npinv, 4)

    def test_weingarten(self):
        a = np.random.RandomState(4).uniform(size=(10, 10, 10))
        w_gpu = curvature.weingarten(torch.from_numpy(a))
        w_gpu = [[_.numpy() for _ in __] for __ in w_gpu]
        w_cpu = curvature.weingarten_numpy(a)
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
        e1, e2, e3 = [_.numpy() for _ in curvature.eigen3(a)]
        for i in range(10):
            matrix = np.array([[_[i].numpy() for _ in __] for __ in a])
            ce1, ce2, ce3 = np.linalg.eigvalsh(matrix)
            self.assertAlmostEqual(e1[i], ce1, 4)
            self.assertAlmostEqual(e2[i], ce2, 4)
            self.assertAlmostEqual(e3[i], ce3, 4)


if __name__ == '__main__':
    unittest.main()
