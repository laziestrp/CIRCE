"""Actual small-decoder capture checks; no checkpoint or data downloads."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from circe import Weights, score_features

HAS_INFERENCE = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))
if HAS_INFERENCE:
    import torch
    from circe import CirceCapture
HAS_CUDA = HAS_INFERENCE and torch.cuda.is_available()


def captured_response(kind="llama", device="cpu"):
    from transformers import (LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
                              Qwen2Config, Qwen2ForCausalLM)
    configs = {"llama": (LlamaConfig, LlamaForCausalLM), "mistral": (MistralConfig, MistralForCausalLM),
               "qwen2": (Qwen2Config, Qwen2ForCausalLM)}
    config_type, model_type = configs[kind]
    torch.set_num_threads(1)
    torch.manual_seed(127)
    config = config_type(vocab_size=32, hidden_size=24, intermediate_size=48, num_hidden_layers=6,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         bos_token_id=1, eos_token_id=2, pad_token_id=2, attn_implementation="eager")
    model = model_type(config).float().to(device).eval()
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    inputs = torch.tensor([[1, 4, 5, 6, 7]], device=device)
    generation = dict(input_ids=inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=4,
                      do_sample=False, num_beams=1, use_cache=True, pad_token_id=2, eos_token_id=2)
    weights = Weights(*np.random.default_rng(131).dirichlet(np.ones(3)))
    with torch.inference_mode():
        baseline = model.generate(**generation)
        with CirceCapture(model, weights=weights) as capture:
            result = model.generate(**generation)
    assert torch.equal(baseline, result), "capture changed generated IDs"
    return capture, result[0, inputs.shape[1]:].cpu().tolist(), weights


@unittest.skipUnless(HAS_INFERENCE, "install inference dependencies for capture checks")
class DeviceCaptureTests(unittest.TestCase):
    def test_three_decoder_families_and_reference_arithmetic(self):
        for kind in ("llama", "mistral", "qwen2"):
            with self.subTest(kind=kind):
                capture, ids, weights = captured_response(kind)
                tensors = capture.pooled_tensors()
                self.assertTrue(all(value.device == capture.sums.device for value in tensors.values()))
                torch.testing.assert_close(tensors["conditioned_ffn"], tensors["ffn"] - tensors["counterfactual_ffn"],
                                            atol=0, rtol=0)
                actual = capture.finalize(ids)
                exported = capture.pooled_features()
                expected = score_features(**{key: exported[key] for key in (
                    "attention", "ffn", "counterfactual_ffn", "hidden_chain", "mean_nll")}, weights=weights)
                np.testing.assert_allclose([actual.score, actual.cic, actual.gs, actual.mean_nll],
                                           [expected.score, expected.cic, expected.gs, expected.mean_nll],
                                           atol=1e-12, rtol=1e-12)
                self.assertEqual(actual.num_decisions, len(ids))
                self.assertEqual(capture.handles, [])

    def test_scoring_never_uses_numpy_export(self):
        capture, ids, _ = captured_response()
        original_cpu = torch.Tensor.cpu
        transfers = []

        def no_feature_cpu(value, *args, **kwargs):
            if value.is_floating_point():
                self.assertEqual(tuple(value.shape), (9,))
                transfers.append(value.numel())
            return original_cpu(value, *args, **kwargs)

        with patch.object(capture, "pooled_features", side_effect=AssertionError("unexpected host export")), \
             patch.object(torch.Tensor, "numpy", side_effect=AssertionError("unexpected NumPy scoring")), \
             patch.object(torch.Tensor, "cpu", no_feature_cpu):
            actual = capture.finalize(ids)
        self.assertIsInstance(actual.score, float)
        self.assertEqual(transfers, [9])


@unittest.skipUnless(HAS_CUDA, "CUDA runtime unavailable; GPU capture not verified")
class CudaCaptureTests(unittest.TestCase):
    def test_cuda_storage_and_no_feature_cpu_scoring(self):
        capture, ids, _ = captured_response(device="cuda")
        tensors = capture.pooled_tensors()
        self.assertTrue(capture.sums.is_cuda and capture.chain.is_cuda)
        self.assertTrue(all(value.is_cuda for value in tensors.values()))
        original_cpu = torch.Tensor.cpu

        def no_feature_cpu(value, *args, **kwargs):
            if value.is_floating_point():
                self.assertTrue(value.is_cuda)
                self.assertEqual(tuple(value.shape), (9,))
            return original_cpu(value, *args, **kwargs)

        with patch.object(capture, "pooled_features", side_effect=AssertionError("unexpected host export")), \
             patch.object(torch.Tensor, "cpu", no_feature_cpu):
            actual = capture.finalize(ids)
        self.assertTrue(np.isfinite(actual.score))


if __name__ == "__main__":
    unittest.main()
