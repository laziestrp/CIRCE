"""Synthetic artifact fixtures for paired and clustered statistics only."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from circe.evaluation import evaluate_runs, load_run
from circe.inference import write_json
from circe.scoring import Weights, fuse

# Synthetic test-only coefficients, unrelated to experimental parameter values.
TEST_WEIGHTS = Weights(*np.random.default_rng(29).dirichlet(np.ones(3)))


def synthetic_run(root, name, *, single_class=False, clusters=True, weights=TEST_WEIGHTS):
    run = root / name
    (run / "records").mkdir(parents=True)
    (run / "generations").mkdir()
    ids = [f"q{i}" for i in range(10)]
    write_json(run / "run.json", dict(model=name, dataset="race_high", ids=ids, n=10,
                                      weights=asdict(weights)))
    for i, qid in enumerate(ids):
        answer = "A" if i % 2 == 0 or single_class else "B"
        generation = dict(id=qid, model=name, dataset="race_high", cluster_id=str(i // 2) if clusters else None,
                          input=dict(question=f"Question {i}", options=["One", "Two"],
                                     passage=f"Passage {i // 2}", prompt="Synthetic test prompt", prompt_ids=[1, 3]),
                          reference=dict(gold="A", choice_labels=["A", "B"], answer_type=""),
                          output_ids=[3, 2], output_text="Answer: " + answer,
                          raw_output_text="Answer: " + answer + "<|im_end|>", terminated=True,
                          generation_config=dict(eos_token_id=2))
        cic, gs, nll = .1 * i, .2 * i, .1 + .05 * i
        record = dict(generation, tokens=2, truncated=False, scoring_mask=[True, True],
                      scores=dict(score=fuse(cic, gs, nll, weights=weights),
                                  cic=cic, gs=gs, mean_nll=nll, num_decisions=2))
        write_json(run / "records" / f"{i:08d}.json", record)
        write_json(run / "generations" / f"{i:08d}.json", generation)
    write_json(run / "completion.json", dict(status="COMPLETE", records=10))
    return run


class EvaluationTests(unittest.TestCase):
    def test_5000_shared_passage_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one, two = synthetic_run(root, "one"), synthetic_run(root, "two")
            summary = evaluate_runs([one, two], root / "evaluation", bootstrap=5000, seed=1729)
            self.assertEqual(summary["records"], 20)
            self.assertEqual(summary["complete_draws"], 5000)
            self.assertTrue(summary["macro_ci_supported"])
            self.assertEqual(summary["fusion_weights"], asdict(TEST_WEIGHTS))
            with np.load(root / "evaluation/bootstrap.npz", allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays["group_draws"][0], arrays["group_draws"][1])
                self.assertEqual(arrays["group_draws"].shape, (2, 5000, 4, 3))

    def test_single_class_not_dropped_from_macro(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one = synthetic_run(root, "one")
            two = synthetic_run(root, "two", single_class=True)
            summary = evaluate_runs([one, two], root / "evaluation", bootstrap=100)
            self.assertEqual(summary["groups"], 2)
            self.assertEqual(summary["records"], 20)
            self.assertIsNone(summary["macro"][0]["auroc"])
            self.assertFalse(summary["macro_ci_supported"])

    def test_race_requires_clusters_for_intervals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = synthetic_run(root, "one", clusters=False)
            with self.assertRaisesRegex(ValueError, "passage cluster IDs"):
                evaluate_runs([run], root / "bad", bootstrap=100)
            summary = evaluate_runs([run], root / "point", bootstrap=0)
            self.assertIsNotNone(summary["macro"][0]["auroc"])
            self.assertFalse(summary["macro_ci_supported"])

    def test_cross_model_question_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one, two = synthetic_run(root, "one"), synthetic_run(root, "two")
            for kind in ("records", "generations"):
                path = two / kind / "00000000.json"
                value = json.loads(path.read_text())
                value["input"]["question"] = "Changed question"
                path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cross-model examples"):
                evaluate_runs([one, two], root / "bad", bootstrap=0)

    def test_different_supplied_configurations_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            other = Weights(*np.random.default_rng(41).dirichlet(np.ones(3)))
            one = synthetic_run(root, "one")
            two = synthetic_run(root, "two", weights=other)
            with self.assertRaisesRegex(ValueError, "same supplied fusion weights"):
                evaluate_runs([one, two], root / "bad", bootstrap=0)

    def test_recorded_configuration_and_scores_are_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = synthetic_run(root, "one")
            record_path = run / "records/00000000.json"
            value = json.loads(record_path.read_text())
            value["scores"]["score"] += 1
            record_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recorded fusion configuration"):
                load_run(run)
            path = run / "run.json"
            value = json.loads(path.read_text())
            value["weights"]["cic"] = "xxx"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid supplied fusion configuration"):
                load_run(run)


if __name__ == "__main__":
    unittest.main()
