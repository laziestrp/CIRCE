# SPDX-License-Identifier: Apache-2.0
"""Response-local CIRCE arithmetic; extracted and reorganized for this release."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Weights:
    """Nonnegative fusion weights, in CIC / GS / likelihood order."""

    cic: float
    gs: float
    likelihood: float

    def __post_init__(self):
        values = (self.cic, self.gs, self.likelihood)
        if not all(isinstance(v, Real) and not isinstance(v, (bool, np.bool_))
                   and math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("weights must be finite and nonnegative")
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("weights must sum to one")

    @classmethod
    def from_parameters(cls, *, alpha: float, lambda_: float) -> Weights:
        """Convert caller-supplied alpha and lambda to component coefficients."""
        for value in (alpha, lambda_):
            if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("alpha and lambda must be numeric values in [0, 1]; replace xxx")
        alpha, lambda_ = float(alpha), float(lambda_)
        return cls((1 - lambda_) * alpha, (1 - lambda_) * (1 - alpha), lambda_)


def load_weights(path) -> Weights:
    """Read an explicit frozen configuration; the release provides no defaults."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) != {"alpha", "lambda"}:
        raise ValueError("weight configuration must contain exactly alpha and lambda")
    return Weights.from_parameters(alpha=values["alpha"], lambda_=values["lambda"])


@dataclass(frozen=True)
class CirceScore:
    """Higher score means greater estimated correctness, not a probability."""

    score: float
    cic: float
    gs: float
    mean_nll: float
    num_decisions: int | None = None


def _finite(value, name):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def bounded(value):
    """b(z) = (1 + z / (1 + |z|)) / 2; no statistics from other responses."""
    z = _finite(value, "value")
    result = 0.5 * (1.0 + z / (1.0 + np.abs(z)))
    return float(result) if result.ndim == 0 else result


def fuse(cic, gs, mean_nll, weights: Weights):
    """Fuse scalar or broadcast-compatible component arrays with fixed weights."""
    nll = _finite(mean_nll, "mean_nll")
    if np.any(nll < 0):
        raise ValueError("mean_nll must be nonnegative")
    result = (
        weights.cic * bounded(cic)
        + weights.gs * bounded(gs)
        + weights.likelihood / (1.0 + nll)
    )
    result = _finite(result, "score")
    return float(result) if result.ndim == 0 else result


def score_features(
    attention,
    ffn,
    counterfactual_ffn,
    hidden_chain,
    mean_nll: float,
    *,
    weights: Weights,
    max_lag: int = 4,
) -> CirceScore:
    """Score pooled features from one response.

    Component matrices have shape [L, H]; hidden_chain has shape [L+1, H].
    Pool over decisions BEFORE normalizing vectors or computing cosines.
    Preserve the collection positions described in the README.
    """
    a = _finite(attention, "attention")
    m = _finite(ffn, "ffn")
    m0 = _finite(counterfactual_ffn, "counterfactual_ffn")
    c = _finite(hidden_chain, "hidden_chain")
    if a.ndim != 2 or min(a.shape) == 0 or a.shape[0] < 2:
        raise ValueError("components must have shape [L, H], with L >= 2 and H >= 1")
    if m.shape != a.shape or m0.shape != a.shape:
        raise ValueError("all component shapes must match")
    if c.shape != (a.shape[0] + 1, a.shape[1]):
        raise ValueError("hidden_chain must have shape [L+1, H]")
    if not isinstance(max_lag, int) or isinstance(max_lag, bool) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer")
    if np.asarray(mean_nll).ndim != 0:
        raise ValueError("score_features requires a scalar mean_nll")

    q = m - m0
    au = a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
    qu = q / (np.linalg.norm(q, axis=1, keepdims=True) + EPS)
    lag_count = min(max_lag, a.shape[0] - 1)
    cic = float(np.mean([
        np.mean(np.sum(au[:-lag] * qu[lag:], axis=1))
        for lag in range(1, lag_count + 1)
    ]))

    norms = np.linalg.norm(c, axis=1)
    endpoint = float(np.linalg.norm(c[-1] - c[0]))
    if np.min(norms) <= EPS or endpoint <= EPS:
        raise ValueError("degenerate residual chain")
    end_cos = np.dot(c[-1], c[0]) / (norms[-1] * norms[0])
    end_angle = float(np.arccos(np.clip(end_cos, -1.0, 1.0)))
    if end_angle <= EPS:
        raise ValueError("degenerate residual endpoint angle")
    adjacent = np.sum(c[1:] * c[:-1], axis=1) / (norms[1:] * norms[:-1])
    magnitude = np.linalg.norm(np.diff(c, axis=0), axis=1).mean() / endpoint
    angle = np.arccos(np.clip(adjacent, -1.0, 1.0)).mean() / end_angle
    gs = float(magnitude - angle)
    return CirceScore(float(fuse(cic, gs, mean_nll, weights)), cic, gs, float(mean_nll))
