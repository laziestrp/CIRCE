# SPDX-License-Identifier: Apache-2.0
"""CIRCE arithmetic on the device holding the pooled component tensors."""
from __future__ import annotations

import torch

from .scoring import EPS, CirceScore, Weights


@torch.inference_mode()
def score_tensors(attention, conditioned_ffn, hidden_chain, mean_nll, *, weights: Weights,
                  max_lag: int = 4) -> CirceScore:
    """Score same-device pooled tensors without copying feature arrays to CPU.

    attention and conditioned_ffn have shape [L, H]; hidden_chain is [L+1, H].
    conditioned_ffn is the difference of separately pooled FFN and m0 vectors.
    mean_nll is a scalar tensor. Only final scalar values and validation flags
    are returned to the host, after all component and fusion arithmetic.
    """
    tensors = (attention, conditioned_ffn, hidden_chain, mean_nll)
    if not all(isinstance(value, torch.Tensor) and value.is_floating_point() for value in tensors):
        raise TypeError("supply floating-point tensors for every scoring input")
    if any(value.device != attention.device for value in tensors):
        raise ValueError("all scoring tensors must be on the same device")
    if not isinstance(weights, Weights):
        raise TypeError("weights must be a supplied Weights configuration")
    if attention.ndim != 2 or min(attention.shape) == 0 or attention.shape[0] < 2:
        raise ValueError("components must have shape [L, H], with L >= 2 and H >= 1")
    if conditioned_ffn.shape != attention.shape:
        raise ValueError("all component shapes must match")
    if hidden_chain.shape != (attention.shape[0] + 1, attention.shape[1]):
        raise ValueError("hidden_chain must have shape [L+1, H]")
    if mean_nll.ndim != 0:
        raise ValueError("score_tensors requires a scalar mean_nll")
    if not isinstance(max_lag, int) or isinstance(max_lag, bool) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer")

    a, q, c, nll = (value.detach().to(dtype=torch.float64) for value in tensors)
    finite_inputs = torch.stack([torch.isfinite(value).all() for value in (a, q, c, nll)]).all()
    au = a / (torch.linalg.vector_norm(a, dim=1, keepdim=True) + EPS)
    qu = q / (torch.linalg.vector_norm(q, dim=1, keepdim=True) + EPS)
    lag_count = min(max_lag, a.shape[0] - 1)
    cic = torch.stack([
        (au[:-lag] * qu[lag:]).sum(dim=1).mean()
        for lag in range(1, lag_count + 1)
    ]).mean()

    norms = torch.linalg.vector_norm(c, dim=1)
    endpoint = torch.linalg.vector_norm(c[-1] - c[0])
    end_cos = torch.dot(c[-1], c[0]) / (norms[-1] * norms[0])
    end_angle = torch.acos(end_cos.clamp(-1.0, 1.0))
    adjacent = (c[1:] * c[:-1]).sum(dim=1) / (norms[1:] * norms[:-1])
    magnitude = torch.linalg.vector_norm(torch.diff(c, dim=0), dim=1).mean() / endpoint
    angle = torch.acos(adjacent.clamp(-1.0, 1.0)).mean() / end_angle
    gs = magnitude - angle

    def bounded(value):
        return 0.5 * (1.0 + value / (1.0 + value.abs()))

    score = weights.cic * bounded(cic) + weights.gs * bounded(gs) + weights.likelihood / (1.0 + nll)
    values = torch.stack((score, cic, gs, nll))
    checks = torch.stack((finite_inputs, nll >= 0, (norms > EPS).all() & (endpoint > EPS),
                          end_angle > EPS, torch.isfinite(values).all()))
    # A single small transfer: no q, attention, FFN, or residual array leaves the device.
    result = torch.cat((values, checks.to(dtype=values.dtype))).cpu().tolist()
    messages = ("scoring inputs must contain only finite values", "mean_nll must be nonnegative",
                "degenerate residual chain", "degenerate residual endpoint angle", "score must be finite")
    for valid, message in zip(result[4:], messages):
        if not valid:
            raise ValueError(message)
    return CirceScore(*result[:4])
