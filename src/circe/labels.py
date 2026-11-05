# SPDX-License-Identifier: Apache-2.0
"""Deterministic answer correctness labels, separate from feature extraction."""
from decimal import Decimal, InvalidOperation
import re

from .data import CHOICE_COUNTS
from .legacy_parsing import AnswerParsing
from .numeric_answers import label as numeric_label

LABEL_VERSION = "main_final_answer_v1"
POLICIES = ("final", "strict", "legacy")


def choice_labels(text, gold, choices, terminated):
    """Preserve the main paired experiment's final-option and strict rules."""
    lines = [s.strip() for s in str(text).splitlines() if s.strip()]
    last = lines[-1] if lines else ""
    valid = "".join(choices)
    m = re.fullmatch(r"Answer:\s*([" + valid + r"])", last, re.I)
    strict = bool(m and m[1].upper() == gold and terminated)
    markers = list(re.finditer(r"(?im)^\s*(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?::|is\b\s*:?)\s*", str(text)))
    segment = str(text)[markers[-1].end():].strip() if markers else last
    segment = re.sub(r"(?i)^\s*(?:therefore|thus|hence|so)[,:]?\s*", "", segment)
    found = re.match(r"^\s*(?:\(([" + valid + r"])\)|\[([" + valid + r"])\]|([" + valid + r"]))(?=$|[\s.:,;-])", segment, re.I)
    value = next((x.upper() for x in found.groups() if x), None) if found else None
    tail = segment[found.end():] if found else segment
    ambiguous = bool(re.search(r"(?i)\b(?:or|alternatively|uncertain|cannot|maybe|perhaps|not|incorrect|wrong)\b|\?", segment) or re.search(r"\b[" + valid + r"]\b", tail))
    lists = re.finditer(r"(?i)(?:^|[,;/]|\b(?:and|rather|instead|option|choice)\b)\s*(?:(?:option|choice)\s*)?[\[(]?([" + valid + r"])[\])]?(?=$|[\s.,;:/])", tail)
    for other in lists:
        suffix = tail[other.end():].lstrip()
        article = other[1].upper() == "A" and re.match(r"(?i)^[a-z]{2,}\b", suffix) and not re.match(r"(?i)^(?:and|or|is|was|are|were|seems|would|could|can|must|should|may|might|will)\b", suffix) and not re.search(r"(?i)\b(?:and|rather|instead|option|choice)\b|[\[(]", other[0])
        if not article:
            ambiguous = True
    if ambiguous:
        value = None
    return dict(correct=bool(terminated and value == gold), strict_correct=strict,
                extracted=value, reason="extracted" if value else "invalid_or_ambiguous",
                has_answer_line=bool(m))


def correctness(record, policy="final"):
    if policy not in POLICIES:
        raise ValueError("unknown label policy")
    reference = record["reference"]
    gold = reference["gold"]
    if gold is None:
        raise ValueError("gold is missing; provide fixed binary labels with --labels")
    dataset = record["dataset"]
    text, terminated = record["output_text"], record["terminated"]
    if not isinstance(terminated, bool):
        raise ValueError("terminated must be a boolean")
    if policy == "legacy" or dataset in {"math", "theoremqa"}:
        legacy_dataset = "mmmlu" if dataset == "mmlu" else dataset
        if legacy_dataset not in {"gsm8k", "math", "commonsenseqa", "mmmlu", "belebele", "theoremqa"}:
            raise ValueError("the legacy policy is defined only for the original six datasets")
        legacy_gold = str(gold)
        if dataset == "belebele":
            legacy_gold = str("ABCD".index(str(gold)) + 1)
        sample = {"answer_type": reference.get("answer_type", "")}
        extracted, correct = AnswerParsing(legacy_dataset).dataset_parse(
            record["raw_output_text"], legacy_gold, sample,
        )
        return dict(correct=bool(terminated and correct), strict_correct=bool(terminated and correct),
                    extracted=extracted, reason="legacy_syntactic_exact_match",
                    has_answer_line="Answer:" in text, policy=policy)
    if dataset == "gsm8k":
        found = numeric_label("gsm8k", text, gold)
        lines = [s.strip() for s in text.splitlines() if s.strip()]
        m = re.fullmatch(r"Answer:\s*([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))", lines[-1] if lines else "", re.I)
        try:
            strict = bool(m and Decimal(m[1].replace(",", "")) == Decimal(str(gold).replace(",", "")))
        except InvalidOperation:
            strict = False
        result = dict(correct=bool(found["correct"] and terminated),
                      strict_correct=bool(strict and terminated), extracted=found["value"],
                      reason=found["reason"], has_answer_line=bool(m))
    elif dataset in CHOICE_COUNTS:
        result = choice_labels(text, str(gold), reference["choice_labels"], terminated)
    else:
        raise ValueError("unsupported labeling dataset")
    result["policy"] = policy
    if policy == "strict":
        result["correct"] = result["strict_correct"]
    return result
