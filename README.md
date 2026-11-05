# CIRCE

CIRCE is a training-free method for estimating the correctness of a language-model response. It combines attention-conditioned FFN interaction (CIC), residual-stream geometry (GS), and mean token negative log-likelihood (NLL) into a confidence score. Higher CIRCE scores indicate greater estimated correctness.

## Overview

The model generates a response while CIRCE collects attention, FFN, and residual representations. These representations and the response likelihood provide the components used to calculate CIRCE with a given set of weights. During CUDA inference, component collection and score computation remain on the GPU.

## Installation

Python 3.10 or newer is required.

```bash
pip install ".[inference]"
```

Supported decoder families include Llama, Mistral, Qwen2, and Qwen2.5.

## Usage

Provide a local Hugging Face checkpoint and a JSONL dataset. Copy `weights.example.json` to `weights.json` and replace `xxx` with numeric values for α and λ.

```bash
python main.py \
  --model-path ./checkpoints/local-model \
  --data ./data/gsm8k.jsonl \
  --dataset gsm8k \
  --weights ./weights.json \
  --output ./runs/inference \
  --device cuda

python evaluate.py --runs ./runs/inference --output ./runs/evaluation
```

Evaluation reports accuracy, AUROC, correct-positive AP, and FPR95. See the [input and evaluation formats](docs/DATA_FORMAT.md) for supported data schemas and options.

## CIRCE score

CIC captures attention-conditioned FFN interactions, GS describes residual-stream geometry, and NLL measures response likelihood. CIRCE combines these components as:

```text
b(z) = 0.5 * (1 + z / (1 + abs(z)))
S    = (1 - λ) * (α * b(CIC) + (1 - α) * b(GS)) + λ / (1 + NLL)
```

Here, α balances CIC and GS, and λ controls the likelihood contribution.

## Testing

```bash
python -m unittest discover -s tests -v
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
