"""CIRCE: component interaction, residual geometry, and likelihood confidence."""

from .scoring import CirceScore, Weights, bounded, fuse, load_weights, score_features

__version__ = "0.4.1"
__all__ = [
    "CirceScore", "Weights", "bounded", "fuse", "load_weights",
    "score_features", "CirceCapture", "GreedyResult", "score_greedy",
]


def __getattr__(name):
    # Pure score arithmetic can be used without installing PyTorch/Transformers.
    if name == "CirceCapture":
        from .capture import CirceCapture
        return CirceCapture
    if name in {"GreedyResult", "score_greedy"}:
        from .generation import GreedyResult, score_greedy
        return {"GreedyResult": GreedyResult, "score_greedy": score_greedy}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
