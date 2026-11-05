# CIRCE

End-to-end implementation of a training-free correctness score for a single language-model response. CIRCE combines attention-conditioned FFN interaction (CIC), residual-stream geometry (GS), and mean token negative log-likelihood (NLL). Larger scores indicate greater estimated correctness; scores are not calibrated probabilities.

This anonymous review package includes the complete main-method workflow: local checkpoint loading, JSONL data reading, dataset prompts, greedy inference, attention/FFN/counterfactual extraction, score computation, answer parsing, and confidence evaluation. Model weights, benchmark data, stored responses, supplementary experiments, and cluster job scripts are supplied separately.

The workflow separates input preparation, model inference, component collection, score computation, and answer evaluation. Each stage has a Python interface and a command-line entry point where appropriate.

CIRCE uses a given set of weights to combine CIC, GS, and NLL into a score. The coefficients are supplied through α and λ in the configuration file. This release contains `xxx` placeholders rather than numerical coefficient values; scoring requires a completed configuration.

## Installation

Python 3.10 or newer is required. For feature-level arithmetic only:

```bash
pip install .
```

For the full inference workflow:

```bash
pip install ".[inference]"
```

Install a PyTorch build appropriate for your CPU/CUDA environment. The inference extra uses Transformers 4.57.1. The collector supports the standard pre-norm `llama`, `mistral`, and `qwen2` decoder blocks, including Qwen2.5 checkpoints with `model_type="qwen2"`. Use FP32, eager attention, evaluation mode, and disabled TF32. Other architectures, quantized models, multiple devices, batching, padded prompts, sampling, and beam search require their own validation.

## Run inference and evaluation

Supply a local Hugging Face checkpoint directory and a local JSONL dataset file. See [data formats, prompts, and label policies](docs/DATA_FORMAT.md) for the full schema and supported adapters. Language-keyed rows with `id`, `en`, and `answer` are also accepted. The checkpoint must include a tokenizer chat template.

Copy `weights.example.json` to a local `weights.json` and replace both `xxx` strings with numeric values for α and λ in `[0, 1]`. The same supplied weight set is used across models and datasets. The local `weights.json` is ignored by Git. Unresolved placeholders are invalid, and no fallback weights are supplied.

```bash
python main.py \
  --model-path ./checkpoints/local-model \
  --model-id model-tag \
  --dataset gsm8k \
  --data ./data/gsm8k.jsonl \
  --output ./runs/model_gsm8k \
  --weights ./weights.json \
  --device cuda \
  --max-new-tokens 2048 \
  --save-features

python evaluate.py \
  --runs ./runs/model_gsm8k \
  --output ./runs/eval_gsm8k \
  --label-policy final \
  --bootstrap 5000
```

The examples use Bash line continuation; on PowerShell, put each command on one line. Installed console commands `circe-infer` and `circe-evaluate` accept the same arguments. Use `--device cpu` explicitly for small CPU checks. `--start` / `--limit` select an input-order range without using labels or scores.

The command receives the CIRCE weights through `--weights` and uses them directly for scoring. For every example it performs the following operations:

1. Read the question, passage, and all options; build the dataset prompt and apply the checkpoint chat template.
2. Generate one greedy response while capturing actual attention and FFN outputs and evaluating the local counterfactual FFN.
3. Preserve the exact prompt/output token IDs, normal/raw decoded text, termination flag, and generation configuration.
4. Pool the captured vectors, compute CIC/GS/NLL, and apply the given weights to calculate and save the CIRCE score. Correctness labels do not enter these operations.
5. Optionally export the pooled vectors so the score can be independently recomputed without loading the model.

The evaluation command reads those saved responses, assigns deterministic answer labels, and computes accuracy, AUROC, correct-positive AP, and empirical FPR95. `final`, `strict`, and `legacy` parsing policies are documented explicitly; select the evaluation protocol before running it. Existing fixed binary labels can be supplied with `--labels`. Evaluation itself requires no model, GPU, or external service.

Multiple completed runs can be evaluated together with `--runs RUN1 RUN2 ...`. Each model/dataset group contributes equally to the macro. Across models within one dataset, IDs/questions/references must match exactly for the shared paired bootstrap. RACE uses passage clusters. A single-class group remains undefined, and insufficient complete draws suppress unsupported confidence intervals.

## Output artifacts

```text
runs/model_gsm8k/
  run.json                    IDs, model tag, prompt/version, supplied configuration
  generations/00000000.json   Original prompt/output IDs and raw generation
  records/00000000.json       Same response plus CIC, GS, NLL, CIRCE, and mask
  features/00000000.npz       Optional pooled x, attention, FFN, m0, q, hidden chain
  completion.json            Written only after every requested example is saved

runs/eval_gsm8k/
  labels.csv                 Every answer's label, extraction reason, format/termination
  metrics.csv                Per-model/dataset scores, accuracy, and class counts
  macro_metrics.csv          Equal-group macro metrics and supported 95% intervals
  bootstrap.npz              Optional shared draws and their complete-draw mask
  summary.json               Exact metric definitions, policy, counts, and CI support
```

Feature files contain response-pooled vectors, not every token's full activation tensor. Generation envelopes are written before score finalization; if a numerical gate fails, the raw response is retained and `failure.json` records the interruption. Failed/incomplete runs cannot enter full evaluation. Each command requires a new output directory and does not overwrite existing results or regenerate saved answers automatically.

## Method

For the residual input `x_l`, attention output `a_l`, and actual FFN output `m_l` of a block, compute a local FFN contrast:

```text
m0_l = FFN_l(LN_l(x_l))
q_l  = m_l - m0_l
```

`m0_l` is an additional evaluation of the same FFN with the current attention increment omitted from its input. Collection hooks return `None` and do not replace activations. The extra FFN evaluations have a real computational cost.

Pool attention and FFN vectors within the response before applying cosine normalization. CIC averages `cos(mean(a_l), mean(q_(l+r)))` across layers and forward lags `r=1,...,4`. The code uses `v/(norm(v)+1e-12)` for each vector. For decoders with fewer than five blocks, the available lags are used.

GS is the mean adjacent residual displacement divided by endpoint displacement, minus the mean adjacent angle divided by endpoint angle. Cosine arguments are clipped to `[-1,1]`. Chains with near-zero endpoint distance, endpoint angle, or state norms raise an error rather than receiving an imputed score.

Given α and λ, the CIRCE score is calculated as:

```text
b(z) = 0.5 * (1 + z / (1 + abs(z)))
S    = (1 - λ) * (α * b(CIC) + (1 - α) * b(GS)) + λ / (1 + NLL)
```

α controls the CIC/GS balance and λ controls the likelihood contribution. The supplied coefficients are used directly in this expression, together with the response's CIC, GS, and NLL components. Reference answers do not enter the score calculation. `load_weights` reads the configuration; `Weights.from_parameters(alpha=..., lambda_=...)` provides the equivalent Python interface. Replacing the last term by `-λ*NLL/(1+NLL)` subtracts the same constant λ from every response; it preserves ordering but changes the scalar value.

`run.json` records the coefficients used for CIRCE scoring. Evaluation checks the stored scores against those coefficients and requires the same weight set across runs. This metadata allows the score calculation to be checked consistently. Runtime artifacts can contain the supplied values; keep them local if coefficient disclosure is not intended. The release template does not include numerical experimental weights.

## Token and hidden-state conventions

The collector preserves the conventions of the main implementation:

* Each counted step is a next-token decision. At prefill, CIC uses the final prompt position; later steps use the preceding generated token's position. The generated token is scored from those decision logits.
* GS preserves the historical hidden-chain convention: its first contribution comes from the **first prompt position**, and subsequent cached steps use their last position. The chain contains the embedding/residual input, intermediate block outputs, and the final normalized state. GS and CIC therefore do not use identical first-step positions.
* Every emitted token decision contributes, including the decision that emits an EOS/end-of-turn token. Prompts are unpadded. There is no reasoning-only or answer-only span mask.
* NLL is the mean natural-log loss of the actual greedy token IDs over those same decisions. Raw-logit argmax IDs must exactly match the generated IDs.
* Vector sums accumulate in FP32; pooling and structural arithmetic use float64. Refactoring arithmetic across NumPy/PyTorch backends may introduce floating-point roundoff.

Changing these conventions creates a different scoring protocol. The score does not establish that every reasoning step is true.

## Python interfaces

### Fuse existing components

```python
from circe import fuse, load_weights

weights = load_weights("weights.json")
score = fuse(cic, gs, mean_nll, weights=weights)
```

Scalar inputs return a float; broadcast-compatible NumPy arrays return an array. For already pooled component matrices, call `score_features(attention, ffn, counterfactual_ffn, hidden_chain, mean_nll, weights=weights)`. Component matrices have shape `[layers, hidden_size]`; `hidden_chain` has shape `[layers+1, hidden_size]`. Inputs must come from one response using the conventions above.

### Collect during greedy generation

The caller supplies an already loaded model and tokenizer. Load the model with `torch_dtype=torch.float32` and `attn_implementation="eager"`, move it to one device, and then:

```python
import torch
from circe import load_weights, score_greedy

weights = load_weights("weights.json")
model.eval()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain why the sky appears blue."}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
result = score_greedy(model, tokenizer, prompt_ids, weights=weights, max_new_tokens=2048)
print(result.output_text)
print(result.scores.score, result.scores.cic, result.scores.gs, result.scores.mean_nll)
```

`GreedyResult` contains the original output IDs, decoded text, EOS termination flag, and `CirceScore`. A length-limited response remains in the returned result with `terminated=False`; the library does not judge answer correctness. The prompt plus requested token budget must fit the model context.

For integration into an existing greedy generation call, use `with CirceCapture(model, weights=weights) as capture: ...`, then call `capture.finalize(generated_ids)`. Exclude prompt IDs from `generated_ids`. Use cached generation with a single unpadded prompt and neutral logits processing. The capture validates exact raw-greedy alignment, so a sampled or processed token sequence can raise an error. Hooks are removed on normal completion and on exceptions. Each capture is single-use. `capture.pooled_features()` exposes the within-response pooled vectors for inspection.

## Files

| File | Purpose |
| --- | --- |
| `main.py` / `evaluate.py` | Inference and evaluation entry points |
| `src/circe/models.py` | Local FP32/eager checkpoint and tokenizer loading |
| `src/circe/data.py` | Canonical and language-keyed JSONL adapters |
| `src/circe/prompts.py` | Main-experiment prompt templates and chat tokenization |
| `src/circe/inference.py` | Full inference and artifact-saving workflow |
| `src/circe/scoring.py` | CIC, GS, bounded transform, and explicitly configured fusion |
| `weights.example.json` | Placeholder α/λ configuration; replace `xxx` locally before use |
| `src/circe/capture.py` | Attention/FFN hooks and local counterfactual evaluation |
| `src/circe/generation.py` | Library interface for one greedy response |
| `src/circe/labels.py` / `numeric_answers.py` | Main final-answer and strict-format correctness rules |
| `src/circe/legacy_parsing.py` | Historical syntactic answer parser |
| `src/circe/metrics.py` / `evaluation.py` | Metrics, coverage checks, paired bootstrap, and tables |
| `src/circe/cli.py` | Portable command-line arguments |
| `src/circe/__init__.py` | Public API; arithmetic imports without inference dependencies |
| `tests/` | Arithmetic, input/prompt/label, and local-checkpoint integration checks |
| `docs/DATA_FORMAT.md` | Data schema, prompt assembly, parsing and metric definitions |

## Implementation checks

This extraction was checked on CPU with Python 3.12, PyTorch 2.6.0, NumPy 2.5.4, and Transformers 4.57.1. Checks covered 50 synthetic feature cases against the original arithmetic, and randomly initialized six-layer Llama, Mistral, and Qwen2 decoders against the original generation collector. Greedy output IDs matched; component differences were below `1e-10`. EOS counting, the historical first-step GS position, invalid inputs, and hook removal after exceptions were also checked. These checks validate the refactoring on small models; they are not a rerun of full pretrained-model benchmarks.

Run the included checks after installation:

```bash
python -m unittest discover -s tests -v
```

The integration tests create a small local Transformer checkpoint and tokenizer in a temporary directory, load them through the same model loader, then run prompts, inference, feature export, and evaluation. They use synthetic test-only parameter configurations, not experimental settings, and perform no model/data downloads. Inference tests are explicitly skipped if the inference dependencies are unavailable.

Source code is provided under Apache-2.0; see `LICENSE` and `NOTICE`. Checkpoint and dependency licenses apply separately.
