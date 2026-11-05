# SPDX-License-Identifier: Apache-2.0
"""Load an externally supplied local checkpoint for FP32/eager inference."""
from pathlib import Path


def load_model(model_path, *, device="cuda"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path)
    if not path.is_dir():
        raise ValueError("model_path must be a local checkpoint directory")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("choose a CPU or one CUDA device")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly choose --device cpu if intended")
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if config.model_type not in {"llama", "mistral", "qwen2"}:
        raise ValueError("only llama, mistral, and qwen2 decoder blocks are supported")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager",
    ).to(target).eval()
    model.requires_grad_(False)
    return model, tokenizer
