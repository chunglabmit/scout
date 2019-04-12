import unittest
import numpy as np
from scout import synthetic


np.random.seed(123)  # For consistent random point generation


class TestGeneratePoints(unittest.TestCase):
    def test(self):
        n = 100
        shape = (64, 64, 64)
        points = synthetic.random_points(n=n, shape=shape)
        self.assertEqual(len(points), len(shape))
        for dim, lim in zip(points, shape):
            self.assertGreaterEqual(dim.min(), 0)
            self.assertLess(dim.max(), lim)
            self.assertEqual(len(dim), n)


class TestPointsToBinary(unittest.TestCase):
    def test_random(self):
        n = 100
        shape = (128, 128, 128)
        points = synthetic.random_points(n=n, shape=shape)
        binary = synthetic.points_to_binary(points=points, shape=shape)
        self.assertEqual(binary.shape, shape)
        self.assertTrue(np.all(binary[points] == 255))
        self.assertTrue(binary.min() == 0)

    def test_one(self):
        shape = (128, 128, 128)
        points = tuple(np.array(s//2) for s in shape)
        binary = synthetic.points_to_binary(points=points, shape=shape)
        self.assertEqual(binary.shape, shape)
        self.assertTrue(np.all(binary[points] == 255))
        self.assertTrue(np.all(binary[0, 0, 0] == 0))


class TestBinaryToBlobs(unittest.TestCase):
    def test_random(self):
        n = 20
        shape = (128, 128, 128)
        sigmas = [1, 2, 3, 4]
        thresh = 0.7

        points = synthetic.random_points(n, shape)
        binary = synthetic.points_to_binary(points, shape)

        old_nb_foreground = 0
        for sigma in sigmas:
            blobs = synthetic.binary_to_blobs(binary, sigma)

            self.assertEqual(blobs.shape, shape)
            self.assertAlmostEqual(blobs.max(), 1)
            self.assertAlmostEqual(blobs.min(), 0)

            idx = np.where(blobs.ravel() > thresh)[0]
            new_nb_foreground = len(idx)
            self.assertTrue(old_nb_foreground < new_nb_foreground)
            old_nb_foreground = new_nb_foreground

    def test_one(self):
        shape = (64, 64, 64)
        sigmas = [1, 2, 3, 4]
        offset = 1
        targets = [0.22313017, 0.68728930, 0.84648174, 0.91051036]

        points = tuple(np.array(s//2) for s in shape)
        test_points = tuple(np.array(s//2+offset) for s in shape)
        binary = synthetic.points_to_binary(points=points, shape=shape)
        for sigma, target in zip(sigmas, targets):
            blobs = synthetic.binary_to_blobs(binary, sigma)
            self.assertEqual(blobs[points], 1)
            self.assertAlmostEqual(blobs[test_points], target)


class TestRemoveRandomPoints(unittest.TestCase):
    def test_float_outbounds(self):
        n = 20
        shape = (128, 128, 128)
        points = synthetic.random_points(n, shape)
        with self.assertRaises(ValueError):
            sample = synthetic.remove_random_points(points, amount=1.5)
        with self.assertRaises(ValueError):
            sample = synthetic.remove_random_points(points, amount=-0.5)

    def test_float(self):
        n = 20
        shape = (128, 128, 128)
        amount = 0.5
        points = synthetic.random_points(n, shape)
        sample = synthetic.remove_random_points(points, amount)
        self.assertTrue(len(sample[0]) == n//2)
        self.check_subset(points, sample)

    def test_int_outbounds(self):
        n = 20
        shape = (128, 128, 128)
        points = synthetic.random_points(n, shape)
        with self.assertRaises(ValueError):
            sample = synthetic.remove_random_points(points, amount=-1)

    def test_int(self):
        n = 20
        shape = (128, 128, 128)
        amount = 3
        points = synthetic.random_points(n, shape)
        sample = synthetic.remove_random_points(points, amount)
        self.assertTrue(len(sample[0]) == n - amount)
        self.check_subset(points, sample)

    def check_subset(self, points, sample):
        for p, s in zip(points, sample):
            self.assertTrue(np.all(np.isin(s, p)))


if __name__ == '__main__':
    unittest.main()
