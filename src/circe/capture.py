from __future__ import annotations

import torch
import torch.nn.functional as F

from .scoring import CirceScore, Weights
from .torch_scoring import score_tensors


class CirceCapture:
    """Capture one unpadded, cached, greedy response from a supported decoder.

    Use as a context manager around model.generate(), then call finalize()
    with the generated IDs, excluding the prompt. Each capture is single-use.
    """

    def __init__(self, model, *, weights: Weights, max_lag: int = 4):
        if model.config.model_type not in {"llama", "mistral", "qwen2"}:
            raise ValueError("supported decoder types: llama, mistral, qwen2")
        if model.training:
            raise ValueError("call model.eval() before collecting features")
        if getattr(model.config, "_attn_implementation", None) != "eager":
            raise ValueError("the method implementation requires eager attention")
        params = list(model.parameters())
        if not params or any(p.dtype != torch.float32 for p in params):
            raise ValueError("the method implementation requires FP32 parameters")
        device = params[0].device
        if any(p.device != device for p in params):
            raise ValueError("one device per model is supported")
        if device.type == "cuda" and (
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
        ):
            raise ValueError("disable CUDA matmul and cuDNN TF32 before inference")
        if not isinstance(max_lag, int) or isinstance(max_lag, bool) or max_lag < 1:
            raise ValueError("max_lag must be a positive integer")
        self.model = model
        self.layers = list(model.model.layers)
        if len(self.layers) < 2:
            raise ValueError("at least two decoder layers are required")
        for i, layer in enumerate(self.layers):
            for attr in ("self_attn", "mlp", "post_attention_layernorm"):
                if not hasattr(layer, attr):
                    raise ValueError(f"layer {i} is missing {attr}")
            if hasattr(layer, "post_feedforward_layernorm"):
                raise ValueError("extra-normalization blocks require a separate implementation")
        self.weights, self.max_lag = weights, max_lag
        n, h = len(self.layers), model.config.hidden_size
        self.sums = torch.zeros((4, n, h), device=device, dtype=torch.float32)
        self.chain = torch.zeros((n + 1, h), device=device, dtype=torch.float32)
        self.nll_sum = torch.zeros((), device=device, dtype=torch.float64)
        self.pending, self.handles, self.argmax_ids = {}, [], []
        self.layer_counts = [0] * n
        self.count = 0
        self.probe_active = False
        self._used = False

    def __enter__(self):
        if self._used:
            raise RuntimeError("create a new capture for every response")
        self._used = True
        try:
            for i, layer in enumerate(self.layers):
                self.handles.append(layer.register_forward_pre_hook(self._pre(i), with_kwargs=True))
                self.handles.append(layer.self_attn.register_forward_hook(self._attention(i)))
                self.handles.append(layer.mlp.register_forward_hook(self._mlp(i)))
                self.handles.append(layer.register_forward_hook(self._done(i)))
            self.handles.append(self.model.register_forward_hook(self._raw_logits))
            self.handles.append(self.model.model.norm.register_forward_hook(self._final_norm))
        except BaseException:
            self.remove()
            raise
        return self

    def __exit__(self, *_exc):
        self.remove()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _pre(self, i):
        def hook(_module, args, kwargs):
            if not self.probe_active:
                hidden = args[0] if args else kwargs["hidden_states"]
                if hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError("only batch size one is supported")
                if self.count and hidden.shape[1] != 1:
                    raise ValueError("cached generation must process one token after prefill")
                self.pending[i] = {
                    "x": hidden[0, -1, :].float(),
                    "legacy_x": hidden[0, 0 if self.count == 0 else -1, :].float(),
                }
            return None
        return hook

    def _attention(self, i):
        def hook(_module, _args, out):
            if not self.probe_active:
                a = out[0] if isinstance(out, tuple) else out
                self.pending[i]["a"] = a[0, -1, :].float()
            return None
        return hook

    def _mlp(self, i):
        def hook(_module, _args, out):
            if not self.probe_active:
                self.pending[i]["m"] = out[0, -1, :].float()
            return None
        return hook

    def _done(self, i):
        def hook(layer, _args, out):
            if self.probe_active:
                return None
            value = out[0] if isinstance(out, tuple) else out
            x, m = self.pending[i]["x"], self.pending[i]["m"]
            self.probe_active = True
            try:
                with torch.inference_mode():
                    probe = x.view(1, 1, -1).to(next(layer.parameters()).dtype)
                    m0 = layer.mlp(layer.post_attention_layernorm(probe))[0, 0, :].float()
            finally:
                self.probe_active = False
            a = self.pending[i]["a"]
            for k, vec in enumerate((x, a, m, m0)):
                self.sums[k, i].add_(vec)
            if i == 0:
                self.chain[0].add_(self.pending[i]["legacy_x"])
            if i + 1 < len(self.layers):
                self.chain[i + 1].add_(value[0, 0 if self.count == 0 else -1, :].float())
            self.layer_counts[i] += 1
            self.pending.pop(i)
            return None
        return hook

    def _final_norm(self, _module, _args, out):
        if not self.probe_active:
            self.chain[-1].add_(out[0, 0 if self.count == 0 else -1, :].float())
        return None

    def _raw_logits(self, _model, _args, out):
        logits = out.logits[0, -1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        argmax = logits.argmax()
        self.argmax_ids.append(argmax)
        self.nll_sum.add_(-logp[argmax].double())
        self.count += 1
        return None

    def pooled_tensors(self):
        """Return pooled float64 tensors on the model device, including q."""
        if self.count == 0 or any(c != self.count for c in self.layer_counts):
            raise ValueError("incomplete token/layer capture")
        v = (self.sums.double() / self.count).detach()
        chain = (self.chain.double() / self.count).detach()
        if not bool(torch.isfinite(v).all() & torch.isfinite(chain).all()):
            raise ValueError("nonfinite collected features")
        return {
            "residual_input": v[0], "conditioned_ffn": v[2] - v[3],
            "attention": v[1], "ffn": v[2], "counterfactual_ffn": v[3],
            "hidden_chain": chain, "mean_nll": (self.nll_sum / self.count).detach(),
        }

    def pooled_features(self):
        """Explicit host export for inspection or saving; not used for scoring."""
        features = self.pooled_tensors()
        return {key: float(value.cpu()) if key == "mean_nll" else value.cpu().numpy()
                for key, value in features.items()}

    def finalize(self, generated_ids) -> CirceScore:
        ids = [int(token) for token in generated_ids]
        if not ids or len(ids) != self.count:
            raise ValueError("generated token count differs from decision count")
        if torch.stack(self.argmax_ids).cpu().tolist() != ids:
            raise ValueError("generated IDs differ from raw greedy decisions")
        features = self.pooled_tensors()
        result = score_tensors(**{k: features[k] for k in (
            "attention", "conditioned_ffn", "hidden_chain", "mean_nll",
        )}, weights=self.weights, max_lag=self.max_lag)
        return CirceScore(result.score, result.cic, result.gs, result.mean_nll, self.count)
