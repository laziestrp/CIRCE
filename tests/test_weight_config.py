"""Configuration contract checks using synthetic parameters only."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from circe import Weights, fuse, load_weights


class WeightConfigurationTests(unittest.TestCase):
    def test_no_implicit_fusion_configuration(self):
        with self.assertRaises(TypeError):
            Weights()
        with self.assertRaises(TypeError):
            fuse(0, 0, 0)

    def test_symbolic_parameterization(self):
        for alpha, lambda_ in np.random.default_rng(73).random((20, 2)):
            weights = Weights.from_parameters(alpha=alpha, lambda_=lambda_)
            self.assertEqual(weights.cic, (1 - lambda_) * alpha)
            self.assertEqual(weights.gs, (1 - lambda_) * (1 - alpha))
            self.assertEqual(weights.likelihood, lambda_)
            self.assertAlmostEqual(weights.cic + weights.gs + weights.likelihood, 1)

    def test_parameters_must_be_numeric_bounded_and_finite(self):
        for value in ("xxx", "0", None, True, np.bool_(False), -1, 2, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Weights.from_parameters(alpha=value, lambda_=0)
                with self.assertRaises(ValueError):
                    Weights.from_parameters(alpha=0, lambda_=value)

    def test_boundary_parameters_are_valid(self):
        for alpha in (0, 1):
            for lambda_ in (0, 1):
                weights = Weights.from_parameters(alpha=alpha, lambda_=lambda_)
                self.assertTrue(np.isfinite(fuse(0, 0, 0, weights=weights)))

    def test_external_json_and_placeholder_rejection(self):
        alpha, lambda_ = np.random.default_rng(97).random(2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.json"
            parameters = {"alpha": float(alpha), "lambda": float(lambda_)}
            path.write_text(json.dumps(parameters), encoding="utf-8")
            self.assertEqual(load_weights(path), Weights.from_parameters(alpha=alpha, lambda_=lambda_))
            for invalid in ({"alpha": "xxx", "lambda": "xxx"}, {"alpha": 0},
                            {"alpha": 0, "lambda": 0, "extra": 0}, [0, 0]):
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_weights(path)


if __name__ == "__main__":
    unittest.main()
