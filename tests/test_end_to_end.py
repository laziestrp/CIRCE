"""Real, randomly initialized local Transformer checkpoints; no model downloads."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HAS_INFERENCE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers"))


@unittest.skipUnless(HAS_INFERENCE, 'install ".[inference]" for inference checks')
class PipelineTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)
        torch.manual_seed(1729)
        vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "user": 3, "assistant": 4,
                 "Answer": 5, "A": 6, "B": 7, "C": 8, "D": 9, ":": 10}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                                bos_token="[BOS]", eos_token="[EOS]", pad_token="[EOS]")
        self.tokenizer.chat_template = "{{ bos_token }} user : {{ messages[0]['content'] }} assistant :"
        config = LlamaConfig(vocab_size=32, hidden_size=24, intermediate_size=48,
                             num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=256, bos_token_id=1, eos_token_id=2,
                             pad_token_id=2, attn_implementation="eager")
        self.model = LlamaForCausalLM(config).float().eval()
        self.checkpoint = self.root / "checkpoint"
        self.model.save_pretrained(self.checkpoint)
        self.tokenizer.save_pretrained(self.checkpoint)
        self.data = self.root / "input.jsonl"
        examples = [dict(id="q1", question="First question?", options=["One", "Two"], gold="A"),
                    dict(id="q2", question="Second question?", options=["One", "Two"], gold="B")]
        self.data.write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
        from circe import load_weights
        alpha, lambda_ = np.random.default_rng(53).random(2)
        self.weight_file = self.root / "weights.json"
        self.weight_file.write_text(json.dumps({"alpha": float(alpha), "lambda": float(lambda_)}),
                                    encoding="utf-8")
        self.weights = load_weights(self.weight_file)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, name="run"):
        from circe.cli import infer_main
        output = self.root / name
        code = infer_main(["--model-path", str(self.checkpoint), "--model-id", "tiny",
                           "--data", str(self.data), "--dataset", "mmlu", "--device", "cpu",
                           "--max-new-tokens", "4", "--output", str(output), "--save-features",
                           "--weights", str(self.weight_file)])
        self.assertEqual(code, 0)
        return output

    def test_checkpoint_to_inference_features_and_evaluation(self):
        from circe.evaluation import evaluate_runs
        from circe.scoring import score_features

        output = self.run_cli()
        for index in range(2):
            record = json.loads((output / "records" / f"{index:08d}.json").read_text())
            generation = json.loads((output / "generations" / f"{index:08d}.json").read_text())
            self.assertEqual(record["output_ids"], generation["output_ids"])
            self.assertEqual(record["tokens"], record["scores"]["num_decisions"])
            with np.load(output / "features" / f"{index:08d}.npz", allow_pickle=False) as features:
                np.testing.assert_allclose(features["conditioned_ffn"], features["ffn"] - features["counterfactual_ffn"], atol=0, rtol=0)
                score = score_features(**{k: features[k] for k in (
                    "attention", "ffn", "counterfactual_ffn", "hidden_chain", "mean_nll")},
                                       weights=self.weights)
            self.assertAlmostEqual(score.score, record["scores"]["score"], places=12)
        summary = evaluate_runs([output], self.root / "evaluation", bootstrap=100)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["conventions"]["positive_class"], "correct")
        self.assertTrue((self.root / "evaluation/labels.csv").is_file())
        labels = self.root / "labels.jsonl"
        labels.write_text("".join(json.dumps(dict(model="tiny", dataset="mmlu", id=qid, correct=y)) + "\n"
                                   for qid, y in (("q1", 0), ("q2", 1))), encoding="utf-8")
        external = evaluate_runs([output], self.root / "external_eval", labels_path=labels, bootstrap=100)
        self.assertIsNotNone(external["macro"][0]["auroc"])
        self.assertEqual(external["conventions"]["truncated_answers"], "fixed external labels supplied by caller")
        # With only one example per class, ordinary resampling often loses a class.
        self.assertFalse(external["macro_ci_supported"])

    def test_generation_survives_numerical_failure(self):
        from circe.data import load_examples
        from circe.inference import infer_examples
        examples = load_examples(self.data, "mmlu", limit=1)
        output = self.root / "failed"
        with patch("circe.capture.CirceCapture.finalize", side_effect=ValueError("numerical gate")):
            with self.assertRaises(ValueError):
                infer_examples(self.model, self.tokenizer, examples, output,
                               weights=self.weights, model_id="tiny", max_new_tokens=4)
        self.assertTrue((output / "generations/00000000.json").is_file())
        self.assertTrue((output / "failure.json").is_file())
        self.assertFalse((output / "completion.json").exists())

    def test_generation_without_feature_export(self):
        from circe import score_greedy
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Question"}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt")
        with patch("circe.capture.CirceCapture.pooled_features", side_effect=AssertionError("unexpected host export")):
            result = score_greedy(self.model, self.tokenizer, ids, weights=self.weights, max_new_tokens=4)
        self.assertIsNone(result.features)
        self.assertIsInstance(result.scores.score, float)

    def test_modified_generation_or_missing_coverage_rejected(self):
        from circe.evaluation import load_run
        output = self.run_cli()
        path = output / "records/00000000.json"
        r = json.loads(path.read_text())
        r["output_ids"][0] = (r["output_ids"][0] + 1) % 32
        path.write_text(json.dumps(r), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "original generation differs"):
            load_run(output)

    def test_placeholder_configuration_stops_before_model_loading(self):
        from circe.cli import infer_main
        self.weight_file.write_text(json.dumps({"alpha": "xxx", "lambda": "xxx"}), encoding="utf-8")
        output = self.root / "unconfigured"
        with patch("circe.models.load_model") as loader, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                infer_main(["--model-path", str(self.checkpoint), "--data", str(self.data),
                            "--dataset", "mmlu", "--output", str(output),
                            "--weights", str(self.weight_file)])
        self.assertEqual(raised.exception.code, 2)
        loader.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
