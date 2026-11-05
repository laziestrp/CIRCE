# Data and evaluation conventions

The release loads local JSONL files. One command evaluates one dataset; it preserves every selected ID and uses the same examples for all component scores. Download benchmark data separately under the relevant license. The repository contains no benchmark files.

## Canonical input

One object per line:

```json
{"id":"example-1","question":"Question text","gold":"B","options":["First option","Second option"]}
```

This is a schema illustration, not benchmark data. Fields are:

| Field | Meaning |
| --- | --- |
| `id` | Stable question ID; if absent, the zero-based input row is used |
| `question` | Question text; a language-keyed field such as `en` is also accepted |
| `gold` | Reference answer, used only during evaluation; optional during inference |
| `options` | All options in their original order; do not remove distractors |
| `choice_labels` | Optional consecutive uppercase labels starting at A |
| `passage` | Reading-comprehension passage, prepended before the question |
| `answer_type` | Required for TheoremQA: `bool`, `integer`, `float`, `list of integer`, `list of float`, or `option` |
| `prompt` | Optional full prompt override, used verbatim before the checkpoint chat template |
| `cluster_id` | Shared passage ID for questions from one RACE passage |

`answer` / `answerKey` can supply the reference when `gold` is absent. GSM8K's official `answer` field is reduced to the text after `####`. CommonsenseQA's `choices={"label": [...], "text": [...]}` is accepted. MMLU-Pro's explicit zero-based `answer_index` is checked against its answer letter. LogiQA2's official zero-based integer `answer` and `text` passage are adapted. RACE's `article` is an alias for `passage`. Belebele numeric reference labels 1–4 map to A–D. Other numeric multiple-choice encodings must be converted explicitly to letters before loading.

Already formatted inputs can use `en` containing the complete question/passage/options, plus `answer`. Do not also supply `options` or `passage` if they are already embedded in that string. For MMLU-Pro without an options list, supply the exact `choice_labels` if the item has fewer than ten options. `--language` selects the input field; the included templates are the English main-experiment templates, not a multilingual prompt suite.

## Dataset names and prompts

Supported names: `gsm8k`, `math`, `commonsenseqa`, `mmlu`, `belebele`, `theoremqa`, `logiqa2`, `mmlu_pro`, `race_high`. Input aliases `mmmlu`, `mgsm`, and `gsm8k_reused` map to their corresponding adapter. Dataset names do not download data, choose a split, or select a test subset.

The original-six prompt templates are retained in `circe/prompts.py`. The three extension datasets use the fixed general multiple-choice wrapper with every original option. The MATH template preserves the historical literal `\boxed{{}}` instruction. A supplied `prompt` takes precedence; it is saved with its actual token IDs. Formatting already assembled question text twice changes the prompt, so use either structured fields or a complete preformatted question.

`--start` and `--limit` use input order. No correctness labels or scores choose examples. To reproduce a published subset, prepare a file with its exact IDs/order and explicit prompts, rather than treating an arbitrary range as that subset.

## Fusion configuration

Inference requires `--weights path/to/weights.json`. The shipped template is:

```json
{"alpha":"xxx","lambda":"xxx"}
```

Replace each placeholder locally with a numeric α or λ in `[0, 1]`. CIRCE uses these supplied weights to combine the response components into a score. The same weight set applies to every model/dataset group. No numerical experimental weights or default fusion configuration are bundled, and an unresolved placeholder is invalid.

`run.json` records the CIC, GS, and likelihood coefficients used for scoring. Evaluation recomputes CIRCE with those recorded coefficients, checks consistency across runs, and includes the configuration in the local evaluation summary. The recorded configuration supports score and consistency verification. Runtime artifacts can disclose the supplied values and are not intended to be committed. Reproducing a particular coefficient configuration requires its corresponding values.

## Answer correctness policies

The policy is recorded in evaluation outputs and is never selected by performance.

* `final` (default): the frozen final-number extractor for GSM8K and final-option rules for multiple choice. It rejects uncertain/conflicting answers. Final-number parsing supports the main experiment's decimal syntax, not a general mathematical expression solver.
* `strict`: the same responses with strict final-line `Answer: ...` correctness for GSM8K and multiple choice. It is a format sensitivity check.
* `legacy`: the historical syntactic parser, including its special-token termination checks and matching behavior. This is available for historical comparisons, not selected item by item.

MATH and TheoremQA use the retained syntactic parser under both `final` and `strict`: MATH compares the first balanced boxed expression after removing spaces; TheoremQA uses its declared answer type and the original string matching. There is no symbolic algebra, semantic equivalence, tolerance search, or LLM judge. The legacy parser's limitations are preserved in its explicitly named policy. Raw decoding with special tokens is stored for that parser; primary GSM8K/choice evaluation uses explicit token-based termination.

An output that fails to terminate, cannot be extracted, or conflicts with itself remains in the primary evaluation as incorrect. No such row is dropped from the confidence metrics. Score values themselves are computed without reference answers or labels.

For fixed existing binary labels, use `--labels` with a JSONL file containing:

```json
{"model":"model-tag","dataset":"mmlu","id":"example-1","correct":true}
```

Every requested `(model,dataset,id)` must have exactly one label, with no extra rows. These labels override automatic parsing for the entire evaluation, including its completion/invalid-answer rules; the summary explicitly records that the external labels were used as supplied. Their provenance and protocol must be documented by the caller; the code never creates labels with an API/judge.

## Ranking metrics and intervals

Correct answers are positive. `metrics.csv` reports AUROC, correct-positive average precision (`ap`), empirical FPR95, and accuracy/counts for CIRCE and its CIC/GS/negative-NLL components. All confidence directions are fixed as higher = more likely correct. FPR95 is the FPR at the first observed threshold attaining at least 95% TPR; lower FPR95 is better. Ties are grouped before computing curves.

AP and empirical FPR95 are distinct from trapezoidal PR area and interpolated FPR95. Use consistent definitions when comparing numerical results.

By default, 5,000 bootstrap samples are shared across methods and, within a dataset, across models on the same IDs. RACE samples passage clusters, retaining all questions in each sampled passage. Provide `passage` or `cluster_id` for RACE; otherwise use `--bootstrap 0` for point estimates only. The code refuses a cross-model macro evaluation if the underlying question/reference/cluster alignment differs.

Macro values equally weight every supplied model/dataset group. Single-class groups have undefined ranking metrics (`null` / blank CSV entries); they remain in the macro, making the relevant macro undefined. At least 90% of shared draws must be complete across all supplied groups/methods before macro percentile intervals are published. Unsupported intervals remain null. Undefined metrics are not treated as wins, and all supplied groups remain in the evaluation.
