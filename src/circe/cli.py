import argparse
import json
from pathlib import Path


def infer_main(argv=None):
    from .data import DATASETS, dataset_name, load_examples
    from .inference import infer_examples
    from .models import load_model
    from .scoring import load_weights

    parser = argparse.ArgumentParser(description="Local checkpoint -> prompts -> greedy generation -> CIRCE")
    parser.add_argument("--model-path", required=True, help="local Hugging Face checkpoint directory")
    parser.add_argument("--model-id", default="model", help="identifier recorded in artifacts")
    parser.add_argument("--data", required=True, help="local JSONL input file")
    parser.add_argument("--dataset", type=dataset_name, choices=DATASETS, required=True)
    parser.add_argument("--language", default="en", help="language-keyed input field in JSONL")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--weights", required=True, help="JSON file supplying alpha and lambda for CIRCE scoring; replace xxx")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-features", action="store_true", help="save pooled x/attn/FFN/m0/q/hidden-chain arrays")
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("--output must be a new directory")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    try:
        weights = load_weights(args.weights)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    examples = load_examples(args.data, args.dataset, language=args.language, start=args.start, limit=args.limit)
    model, tokenizer = load_model(args.model_path, device=args.device)
    infer_examples(model, tokenizer, examples, args.output, model_id=args.model_id,
                   weights=weights, max_new_tokens=args.max_new_tokens,
                   save_features=args.save_features, seed=args.seed)
    return 0


def evaluate_main(argv=None):
    from .evaluation import evaluate_runs
    from .labels import POLICIES

    parser = argparse.ArgumentParser(description="Saved original responses -> correctness labels -> ranking metrics")
    parser.add_argument("--runs", nargs="+", required=True, help="one or more completed inference directories")
    parser.add_argument("--output", required=True, help="new evaluation directory")
    parser.add_argument("--label-policy", choices=POLICIES, default="final")
    parser.add_argument("--labels", help="optional fixed binary labels keyed by model, dataset, id")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    summary = evaluate_runs(args.runs, args.output, policy=args.label_policy, labels_path=args.labels,
                            bootstrap=args.bootstrap, seed=args.seed)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))
    return 0
