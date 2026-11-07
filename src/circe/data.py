from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

DATASETS = ("gsm8k", "math", "commonsenseqa", "mmlu", "belebele", "theoremqa",
            "logiqa2", "mmlu_pro", "race_high")
ALIASES = {"mmmlu": "mmlu", "mgsm": "gsm8k", "gsm8k_reused": "gsm8k"}
CHOICE_COUNTS = {"commonsenseqa": 5, "mmlu": 4, "belebele": 4,
                 "logiqa2": 4, "mmlu_pro": 10, "race_high": 4}


def dataset_name(value):
    value = str(value).lower().replace("-", "_")
    value = ALIASES.get(value, value)
    if value not in DATASETS:
        raise ValueError(f"unsupported dataset {value!r}; choose from {DATASETS}")
    return value


@dataclass(frozen=True)
class Example:
    id: str
    dataset: str
    question: str
    gold: object = None
    options: tuple[str, ...] = ()
    choice_labels: tuple[str, ...] = ()
    passage: str = ""
    answer_type: str = ""
    prompt: str | None = None
    cluster_id: str | None = None

    def reference(self):
        return {"gold": self.gold, "answer_type": self.answer_type,
                "choice_labels": list(self.choice_labels)}


def adapt_example(row, dataset, index, language="en"):
    dataset = dataset_name(dataset)
    if not isinstance(row, dict):
        raise ValueError("each JSONL row must be an object")
    if "dataset" in row and dataset_name(row["dataset"]) != dataset:
        raise ValueError("row dataset differs from the requested dataset")
    question = row.get("question", row.get(language))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("a nonempty question or selected language field is required")
    options = row.get("options", row.get("choices", []))
    supplied_labels = row.get("choice_labels")
    if isinstance(options, dict):
        supplied_labels, options = options.get("label"), options.get("text")
    if not isinstance(options, list) or any(not isinstance(v, str) for v in options):
        raise ValueError("options must be a list of strings, or choices.label/text lists")
    if options and not 2 <= len(options) <= 10:
        raise ValueError("multiple-choice inputs require 2 to 10 options")
    labels = supplied_labels or list("ABCDEFGHIJ"[:len(options)])
    if not options and supplied_labels is None:
        labels = list("ABCDEFGHIJ"[:CHOICE_COUNTS.get(dataset, 0)])
    if not isinstance(labels, (list, tuple)) or list(labels) != list("ABCDEFGHIJ"[:len(labels)]):
        raise ValueError("choice labels must be consecutive uppercase letters starting at A")
    if options and len(labels) != len(options):
        raise ValueError("option and choice-label counts differ")
    gold = row.get("gold", row.get("answer", row.get("answerKey")))
    if dataset == "logiqa2" and "gold" not in row and isinstance(gold, int) and not isinstance(gold, bool):
        if not 0 <= gold < len(labels):
            raise ValueError("LogiQA2 answer index is out of range")
        gold = labels[gold]
    if "answer_index" in row and "gold" not in row:
        answer_index = row["answer_index"]
        if not isinstance(answer_index, int) or isinstance(answer_index, bool) or not 0 <= answer_index < len(labels):
            raise ValueError("answer_index must be a valid zero-based option index")
        indexed_gold = labels[answer_index]
        if gold is not None and str(gold).upper() != indexed_gold:
            raise ValueError("answer and answer_index disagree")
        gold = indexed_gold
    if dataset == "gsm8k" and isinstance(gold, str) and "####" in gold:
        gold = gold.rsplit("####", 1)[1].strip()
    if dataset == "belebele" and gold is not None and str(gold) in {"1", "2", "3", "4"}:
        gold = "ABCD"[int(gold) - 1]
    if dataset in CHOICE_COUNTS and gold is not None and gold not in labels:
        raise ValueError("multiple-choice gold must be an option letter")
    passage = row.get("passage", row.get("article", row.get("text", "") if dataset == "logiqa2" else ""))
    if not isinstance(passage, str):
        raise ValueError("passage must be a string")
    answer_type = row.get("answer_type", "")
    if dataset == "theoremqa" and answer_type not in {
        "bool", "integer", "float", "list of integer", "list of float", "option",
    }:
        raise ValueError("theoremqa requires an explicit supported answer_type")
    prompt = row.get("prompt")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError("prompt must be a nonempty string when supplied")
    qid = str(row.get("id", index))
    if not qid:
        raise ValueError("example ID must be nonempty")
    cluster = row.get("cluster_id")
    if cluster is None and dataset == "race_high" and passage:
        cluster = hashlib.sha256(passage.encode("utf-8")).hexdigest()
    if cluster is None and dataset != "race_high":
        cluster = qid
    return Example(qid, dataset, question, gold, tuple(options), tuple(labels), passage,
                   answer_type, prompt, str(cluster) if cluster is not None else None)


def load_examples(path, dataset, *, language="en", start=0, limit=None):
    if start < 0 or limit is not None and limit < 1:
        raise ValueError("start must be nonnegative and limit must be positive")
    result, seen = [], set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            example = adapt_example(json.loads(line), dataset, len(result), language)
            if example.id in seen:
                raise ValueError(f"duplicate ID at line {line_number}: {example.id}")
            seen.add(example.id)
            result.append(example)
    selected = result[start:] if limit is None else result[start:start + limit]
    if not selected:
        raise ValueError("the selected input range is empty")
    return selected
