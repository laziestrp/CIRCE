from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import GenerationConfig

from .capture import CirceCapture
from .scoring import CirceScore, Weights


@dataclass(frozen=True)
class GreedyResult:
    output_ids: tuple[int, ...]
    output_text: str
    terminated: bool
    scores: CirceScore
    raw_output_text: str = ""
    features: dict | None = None
    generation_config: dict | None = None


@torch.inference_mode()
def score_greedy(
    model,
    tokenizer,
    prompt_ids,
    *,
    max_new_tokens: int = 2048,
    weights: Weights,
    return_features: bool = False,
    on_generation=None,
) -> GreedyResult:

    if not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must be a tensor with shape [1, prompt_length]")
    if prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("supply exactly one nonempty unpadded prompt")
    if prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must have dtype torch.long")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if prompt_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus generation budget exceeds model context")
    original = model.generation_config
    eos = original.eos_token_id if original.eos_token_id is not None else tokenizer.eos_token_id
    if eos is None or eos == []:
        raise ValueError("an EOS token configuration is required")
    eos_ids = eos if isinstance(eos, list) else [eos]
    bos = original.bos_token_id if original.bos_token_id is not None else tokenizer.bos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0]
    config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        repetition_penalty=1.0, top_p=1.0, top_k=0, temperature=1.0,
        use_cache=True, eos_token_id=eos, bos_token_id=bos, pad_token_id=pad,
        forced_eos_token_id=None, forced_bos_token_id=None, min_length=0,
        return_dict_in_generate=True, output_scores=False,
        output_hidden_states=False, output_attentions=False,
    )
    ids = prompt_ids.to(next(model.parameters()).device)
    with CirceCapture(model, weights=weights) as capture:
        output = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config,
        )
    generated = tuple(output.sequences[0, ids.shape[1]:].cpu().tolist())
    text = tokenizer.decode(list(generated), skip_special_tokens=True)
    raw_text = tokenizer.decode(list(generated), skip_special_tokens=False)
    terminated = bool(generated and generated[-1] in eos_ids)
    if on_generation is not None:
        on_generation({"output_ids": list(generated), "output_text": text,
                       "raw_output_text": raw_text, "terminated": terminated,
                       "generation_config": config.to_dict()})
    scores = capture.finalize(generated)
    return GreedyResult(
        generated, text, terminated, scores, raw_text,
        capture.pooled_features() if return_features else None, config.to_dict(),
    )
