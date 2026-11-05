"""NumPy reference parity and explicit device-boundary checks."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from circe import Weights, score_features

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from circe import score_tensors
HAS_CUDA = HAS_TORCH and torch.cuda.is_available()


def fixture(device="cpu", *, layers=6):
    rng = np.random.default_rng(251 + layers)
    a, m, m0 = rng.normal(size=(3, layers, 24))
    chain = rng.normal(size=(layers + 1, 24))
    nll = float(rng.uniform(0, 2))
    weights = Weights(*rng.dirichlet(np.ones(3)))
    expected = score_features(a, m, m0, chain, nll, weights=weights)
    inputs = [torch.as_tensor(value, dtype=torch.float64, device=device)
              for value in (a, m - m0, chain, nll)]
    return inputs, weights, expected


@unittest.skipUnless(HAS_TORCH, "install torch for tensor scoring checks")
class TensorScoringTests(unittest.TestCase):
    def test_numpy_reference_parity(self):
        for layers in (2, 6, 28, 32, 48):
            with self.subTest(layers=layers):
                inputs, weights, expected = fixture(layers=layers)
                actual = score_tensors(*inputs, weights=weights)
                np.testing.assert_allclose(
                    [actual.score, actual.cic, actual.gs, actual.mean_nll],
                    [expected.score, expected.cic, expected.gs, expected.mean_nll], atol=1e-12, rtol=1e-12)

    def test_no_numpy_or_feature_cpu_transfer(self):
        inputs, weights, _ = fixture()
        original_cpu = torch.Tensor.cpu
        transfers = []

        def scalar_cpu(value, *args, **kwargs):
            self.assertEqual(tuple(value.shape), (9,))
            transfers.append(value.numel())
            return original_cpu(value, *args, **kwargs)

        with patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy is not a scoring backend")), \
             patch.object(torch.Tensor, "cpu", scalar_cpu):
            actual = score_tensors(*inputs, weights=weights)
        self.assertIsInstance(actual.score, float)
        self.assertEqual(transfers, [9])

    def test_invalid_shapes_and_lags(self):
        inputs, weights, _ = fixture()
        for lag in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                score_tensors(*inputs, weights=weights, max_lag=lag)
        for index, value in ((0, inputs[0][0]), (1, inputs[1][:-1]),
                             (2, inputs[2][:-1]), (3, inputs[3].reshape(1))):
            bad = list(inputs)
            bad[index] = value
            with self.assertRaises(ValueError):
                score_tensors(*bad, weights=weights)

    def test_finite_nonnegative_and_degenerate_gates(self):
        inputs, weights, _ = fixture()
        bad = [value.clone() for value in inputs]
        bad[1][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            score_tensors(*bad, weights=weights)
        bad = list(inputs)
        bad[3] = torch.tensor(-1., dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            score_tensors(*bad, weights=weights)
        bad = list(inputs)
        bad[2] = torch.ones_like(bad[2])
        with self.assertRaisesRegex(ValueError, "degenerate"):
            score_tensors(*bad, weights=weights)

    def test_explicit_tensor_contract(self):
        inputs, weights, _ = fixture()
        for value in (1., torch.tensor(1)):
            bad = list(inputs)
            bad[3] = value
            with self.assertRaises(TypeError):
                score_tensors(*bad, weights=weights)


@unittest.skipUnless(HAS_CUDA, "CUDA runtime unavailable; GPU execution not verified")
class CudaTensorScoringTests(unittest.TestCase):
    def test_cuda_reference_parity(self):
        inputs, weights, expected = fixture("cuda")
        actual = score_tensors(*inputs, weights=weights)
        np.testing.assert_allclose([actual.score, actual.cic, actual.gs, actual.mean_nll],
                                   [expected.score, expected.cic, expected.gs, expected.mean_nll],
                                   atol=1e-12, rtol=1e-12)
        self.assertTrue(all(value.is_cuda for value in inputs))

    def test_mixed_devices_rejected(self):
        inputs, weights, _ = fixture("cuda")
        inputs[1] = inputs[1].cpu()
        with self.assertRaisesRegex(ValueError, "same device"):
            score_tensors(*inputs, weights=weights)

    def test_cuda_only_scalar_results_cross_to_cpu(self):
        inputs, weights, _ = fixture("cuda")
        original_cpu = torch.Tensor.cpu
        transfers = []

        def scalar_cpu(value, *args, **kwargs):
            self.assertTrue(value.is_cuda)
            self.assertEqual(tuple(value.shape), (9,))
            transfers.append(value.numel())
            return original_cpu(value, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", scalar_cpu), \
             patch.object(torch.Tensor, "numpy", side_effect=AssertionError("unexpected NumPy scoring")):
            score_tensors(*inputs, weights=weights)
        self.assertEqual(transfers, [9])


if __name__ == "__main__":
    unittest.main()
