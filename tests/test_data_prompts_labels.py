from dataclasses import replace
import unittest

from circe.data import adapt_example
from circe.labels import correctness
from circe.prompts import build_prompt


def record(dataset, text, gold, *, terminated=True, choices="ABCD", answer_type=""):
    return dict(dataset=dataset, output_text=text, raw_output_text=text + ("<|im_end|>" if terminated else ""),
                terminated=terminated, reference=dict(gold=gold, choice_labels=list(choices), answer_type=answer_type))


class InputPromptTests(unittest.TestCase):
    def test_language_input_and_gold_blind_prompt(self):
        example = adapt_example({"id": "q", "en": "A question", "answer": "2"}, "gsm8k", 0)
        prompt = build_prompt(example)
        self.assertEqual(prompt, build_prompt(replace(example, gold="999")))
        self.assertTrue(prompt.endswith("Question:\nA question\n"))
        self.assertEqual(build_prompt(replace(example, prompt="Exact prompt\n")), "Exact prompt\n")

    def test_full_options_and_passage(self):
        row = {"question": "Which?", "options": [str(i) for i in range(10)], "answer_index": 9}
        example = adapt_example(row, "mmlu_pro", 0)
        self.assertEqual(example.gold, "J")
        self.assertIn("J. 9", build_prompt(example))
        race = adapt_example(dict(question="Which?", article="Same passage", options=["a", "b"], answer="A"), "race_high", 0)
        self.assertIsNotNone(race.cluster_id)
        self.assertIn("Passage:\nSame passage", build_prompt(race))

    def test_official_common_formats(self):
        gsm = adapt_example(dict(question="Q", answer="Steps\n#### 1,234"), "gsm8k", 0)
        self.assertEqual(gsm.gold, "1,234")
        csqa = adapt_example(dict(question="Q", choices={"label": ["A", "B"], "text": ["x", "y"]}, answerKey="B"), "commonsenseqa", 0)
        self.assertEqual(csqa.choice_labels, ("A", "B"))
        logic = adapt_example(dict(question="Q", text="Context", options=["x", "y"], answer=1), "logiqa2", 0)
        self.assertEqual(logic.gold, "B")
        self.assertIn("Context", build_prompt(logic))

    def test_invalid_schema_and_typed_prompt(self):
        with self.assertRaises(ValueError):
            adapt_example(dict(question="Q", gold="A", options=["one"]), "mmlu", 0)
        with self.assertRaises(ValueError):
            adapt_example(dict(question="Q", answer=True), "theoremqa", 0)
        theorem = adapt_example(dict(question="Q", answer="True", answer_type="bool"), "theoremqa", 0)
        self.assertIn("### Answer_type: bool", build_prompt(theorem))


class LabelTests(unittest.TestCase):
    def test_final_and_strict_numeric(self):
        r = record("gsm8k", "Therefore, the answer is -12.", "-12")
        self.assertTrue(correctness(r)["correct"])
        self.assertFalse(correctness(r, "strict")["correct"])
        self.assertTrue(correctness(record("gsm8k", "Answer: 1,234", "1234"))["correct"])

    def test_ambiguous_or_truncated_kept_incorrect(self):
        self.assertFalse(correctness(record("gsm8k", "Answer: 2 or 3", "2"))["correct"])
        self.assertFalse(correctness(record("mmlu", "Answer: A or B", "A"))["correct"])
        self.assertFalse(correctness(record("mmlu", "Answer: A", "A", terminated=False))["correct"])
        self.assertTrue(correctness(record("mmlu", "Answer: A", "A"))["strict_correct"])

    def test_syntactic_math_theorem_and_legacy(self):
        self.assertTrue(correctness(record("math", r"Result: \boxed{\frac{1}{2}}", r"\frac{1}{2}"))["correct"])
        self.assertTrue(correctness(record("theoremqa", "Therefore, the answer is True", "True", answer_type="bool"))["correct"])
        self.assertTrue(correctness(record("mmlu", "Answer: B", "B"), "legacy")["correct"])


if __name__ == "__main__":
    unittest.main()
