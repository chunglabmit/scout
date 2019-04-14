import unittest
import numpy as np
from scout import score


class TestPrecision(unittest.TestCase):

    def test(self):
        tp = [100, 0]
        fp = [25, 100]
        expected = [100. / 125., 0]
        for tpp, fpp, e in zip(tp, fp, expected):
            self.assertAlmostEqual(e, score.precision(tpp, fpp))

    def test_nan(self):
        self.assertTrue(np.isnan(score.precision(0, 0)))


class TestRecall(unittest.TestCase):

    def test(self):
        tp = [100, 0]
        fn = [50, 100]
        expected = [100. / 150., 0]
        for tpp, fnn, e in zip(tp, fn, expected):
            self.assertAlmostEqual(e, score.recall(tpp, fnn))

    def test_nan(self):
        self.assertTrue(np.isnan(score.recall(0, 0)))


class TestMatchCentroids(unittest.TestCase):

    def do_case(self, c1, c2, max_distance, c1_expected, c2_expected):
        c1_result, c2_result = score.match_centroids(c1, c2, max_distance)
        np.testing.assert_equal(c1_result, c1_expected)
        np.testing.assert_equal(c2_result, c2_expected)
        # For completeness, check in reverse too
        c2_result, c1_result = score.match_centroids(c2, c1, max_distance)
        np.testing.assert_equal(c1_result, c1_expected)
        np.testing.assert_equal(c2_result, c2_expected)

    def test_match_one(self):
        self.do_case(np.array([[1., 2., 3.]]),
                     np.array([[1., 2., 3.5]]),
                     .51,
                     np.zeros(1, int),
                     np.zeros(1, int))

    def test_no_match_one(self):
        self.do_case(np.array([[1., 2., 3.]]),
                     np.array([[1., 2., 3.5]]),
                     0.49,
                     -np.ones(1, int),
                     -np.ones(1, int))

    def test_best(self):
        self.do_case(np.array([[1., 2., 3.]]),
                     np.array([[1., 2., 3.5],
                               [1., 2., 3.4]]),
                     .51,
                     np.ones(1, int),
                     np.array([-1, 0]))


class TestScoreCentroids(unittest.TestCase):
    def do_case(self, detected, gt, max_distance, p, r, f):
        stats = score.score_centroids(detected, gt, max_distance)
        self.assertAlmostEqual(p, stats.precision)
        self.assertAlmostEqual(r, stats.recall)
        self.assertAlmostEqual(f, stats.f_score)

    def test_perfect(self):
        self.do_case(
            np.array([[1., 2., 3.],
                      [4., 5., 6.]]),
            np.array([[1., 2., 3.],
                      [4., 5., 6.]]),
            .5, 1.0, 1.0, 1.0)

    def test_one_missing(self):
        self.do_case(
            np.array([[1., 2., 3.]]),
            np.array([[1., 2., 3.],
                      [4., 5., 6.]]),
            .5, 1.0, .5, 1.0 / 1.5)

    def test_one_extra(self):
        self.do_case(
            np.array([[1., 2., 3.],
                      [4., 5., 6.]]),
            np.array([[1., 2., 3.]]),
            .5, .5, 1.0, 1.0 / 1.5)
