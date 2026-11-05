import unittest
import numpy as np

from circe import Weights, bounded, fuse, score_features
from circe.metrics import PreparedMetrics


class ScoringTests(unittest.TestCase):
    def setUp(self):
        # Synthetic test-only coefficients; these are not experimental settings.
        self.weights = Weights(*np.random.default_rng(917).dirichlet(np.ones(3)))

    def test_fixed_weights_and_offset(self):
        w = self.weights
        expected = w.cic * bounded(.25) + w.gs * bounded(.1) + w.likelihood / (1 + .8)
        self.assertAlmostEqual(fuse(.25, .1, .8, weights=w), expected, places=14)
        for x in (-10., 0., 10.):
            self.assertGreater(bounded(x), 0.)
            self.assertLess(bounded(x), 1.)
        x, y, h = .12, -.7, 1.6
        negative = w.cic * bounded(x) + w.gs * bounded(y) - w.likelihood * h / (1 + h)
        self.assertAlmostEqual(fuse(x, y, h, weights=w) - negative, w.likelihood, places=14)

    def test_invalid_inputs(self):
        for weights in ((-1, 1, 1), (0, 0, 0), (float("nan"), 1, 0), (True, 0, 0)):
            with self.assertRaises(ValueError):
                Weights(*weights)
        for h in (-.1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                fuse(.1, .2, h, weights=self.weights)
        with self.assertRaises(ValueError):
            score_features(np.ones((6, 3)), np.ones((6, 3)), np.ones((6, 3)),
                           np.ones((7, 3)), .1, weights=self.weights)

    def test_feature_arithmetic(self):
        rng = np.random.default_rng(17)
        for layers in (6, 28, 32, 48):
            a, m, m0 = rng.normal(size=(3, layers, 15))
            c = rng.normal(size=(layers + 1, 15))
            result = score_features(a, m, m0, c, .4, weights=self.weights)
            q = m - m0
            au = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
            qu = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            cic = np.mean([(au[:-r] * qu[r:]).sum(1).mean() for r in (1, 2, 3, 4)])
            self.assertAlmostEqual(result.cic, cic, places=14)


class MetricTests(unittest.TestCase):
    def test_perfect_and_reversed(self):
        np.testing.assert_allclose(PreparedMetrics([0, 1, 0, 1], [.1, .9, .2, .8]).point(), [1, 1, 0])
        values = PreparedMetrics([0, 1], [1., 0.]).point()
        np.testing.assert_allclose(values, [0, .5, 1])

    def test_ties_and_empirical_threshold(self):
        np.testing.assert_allclose(PreparedMetrics([0, 1], [0., 0.]).point(), [.5, .5, 1])
        # First observed threshold reaching .95 recall, with no interpolation.
        y = [1] * 19 + [0, 1]
        values = PreparedMetrics(y, np.arange(21, 0, -1)).point()
        self.assertEqual(values[2], 0)

    def test_single_class_and_invalid(self):
        self.assertTrue(np.isnan(PreparedMetrics([1, 1], [0., 1.]).point()).all())
        with self.assertRaises(ValueError):
            PreparedMetrics([0, 2], [.1, .2])
        with self.assertRaises(ValueError):
            PreparedMetrics([0, 1], [.1, float("nan")])

    def test_bootstrap_weights_equal_explicit_replication(self):
        y, scores = np.array([0, 1, 0, 1]), np.array([.1, .2, .2, .8])
        weights = np.array([[1, 1, 1, 1], [2, 1, 0, 3], [2, 0, 2, 0]])
        actual = PreparedMetrics(y, scores).weighted(weights)
        for i, w in enumerate(weights):
            ids = np.repeat(np.arange(4), w)
            expected = PreparedMetrics(y[ids], scores[ids]).point()
            np.testing.assert_allclose(actual[i], expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
