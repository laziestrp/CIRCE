# SPDX-License-Identifier: Apache-2.0
"""Model inference -> attention/FFN extraction -> immutable per-example artifacts."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .prompts import PROMPT_VERSION, build_prompt, tokenize_prompt
from .scoring import Weights


def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        handle.write("\n")


def infer_examples(model, tokenizer, examples, output, *, weights: Weights,
                   model_id="model", max_new_tokens=2048,
                   save_features=False, seed=0):
    import torch
    import transformers
    from . import __version__
    from .generation import score_greedy

    examples = list(examples)
    if not examples or len({e.id for e in examples}) != len(examples):
        raise ValueError("supply nonempty examples with unique IDs")
    if len({e.dataset for e in examples}) != 1:
        raise ValueError("one dataset per inference run")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("generations", "records"):
        (output / name).mkdir()
    if save_features:
        (output / "features").mkdir()
    torch.manual_seed(seed)
    run = {
        "schema_version": 1, "code_version": __version__, "model": model_id,
        "dataset": examples[0].dataset, "ids": [e.id for e in examples],
        "n": len(examples), "max_new_tokens": max_new_tokens, "seed": seed,
        "weights": asdict(weights), "max_lag": 4, "prompt_version": PROMPT_VERSION,
        "precision": "float32", "attention": "eager", "tf32": False,
        "scoring_backend": "torch", "scoring_device": str(next(model.parameters()).device),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "save_features": save_features, "scoring_mask": "all decisions including EOS",
        "checkpoint": {"model_type": model.config.model_type,
                       "layers": model.config.num_hidden_layers, "hidden_size": model.config.hidden_size},
    }
    write_json(output / "run.json", run)
    completed = 0
    try:
        for index, example in enumerate(examples):
            prompt = build_prompt(example)
            ids = tokenize_prompt(tokenizer, prompt)
            stem = f"{index:08d}"
            base = {
                "schema_version": 1, "id": example.id, "dataset": example.dataset,
                "model": model_id, "cluster_id": example.cluster_id,
                "input": {"question": example.question, "options": list(example.options),
                          "passage": example.passage, "prompt": prompt,
                          "prompt_ids": ids[0].tolist()},
                "reference": example.reference(),
            }

            def preserve_generation(envelope):
                write_json(output / "generations" / (stem + ".json"), dict(base, **envelope))

            start = time.perf_counter()
            result = score_greedy(model, tokenizer, ids, max_new_tokens=max_new_tokens,
                                  weights=weights, return_features=save_features,
                                  on_generation=preserve_generation)
            record = dict(base, output_ids=list(result.output_ids), output_text=result.output_text,
                          raw_output_text=result.raw_output_text, terminated=result.terminated,
                          truncated=not result.terminated, tokens=len(result.output_ids),
                          scoring_mask=[True] * len(result.output_ids),
                          scores=asdict(result.scores), generation_config=result.generation_config,
                          inference_seconds=time.perf_counter() - start)
            if save_features:
                with (output / "features" / (stem + ".npz")).open("xb") as handle:
                    np.savez_compressed(handle, **result.features)
                record["features_file"] = "features/" + stem + ".npz"
            write_json(output / "records" / (stem + ".json"), record)
            completed += 1
            print(json.dumps({"event": "scored", "id": example.id, "done": completed,
                              "total": len(examples), "tokens": len(result.output_ids)}), flush=True)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAILED", "completed": completed,
                                              "error_type": type(error).__name__, "error": str(error)})
        raise
    write_json(output / "completion.json", {"status": "COMPLETE", "records": completed})
    return output
