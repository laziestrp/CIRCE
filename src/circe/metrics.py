import numpy as np

METRICS = ("auroc", "ap", "fpr95")


class PreparedMetrics:
    def __init__(self, labels, scores):
        y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
        if y.ndim != 1 or s.shape != y.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
            raise ValueError("labels must be binary and scores finite, with identical 1D shapes")
        self.y = y.astype(np.int64)
        self.order = np.argsort(-s, kind="stable")
        s = s[self.order]
        self.labels = self.y[self.order]
        self.starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1] if len(y) else np.array([], dtype=int)

    def point(self):
        if not len(self.y):
            return np.full(3, np.nan)
        positive = np.add.reduceat(self.labels, self.starts)
        negative = np.add.reduceat(1 - self.labels, self.starts)
        tp, fp = positive.cumsum(), negative.cumsum()
        p, n = int(tp[-1]), int(fp[-1])
        if not p or not n:
            return np.full(3, np.nan)
        numerator = (((tp - positive) + tp) * negative).sum(dtype=np.int64)
        ap = float((positive * (tp / (tp + fp))).sum() / p)
        index = int(np.flatnonzero(20 * tp >= 19 * p)[0])
        return np.array([numerator / (2 * p * n), ap, fp[index] / n])

    def weighted(self, weights):
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] != len(self.y) or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("invalid bootstrap multiplicities")
        if not len(self.y):
            return np.full((len(w), 3), np.nan)
        w = w[:, self.order]
        positive = np.add.reduceat(w * self.labels, self.starts, axis=1)
        negative = np.add.reduceat(w * (1 - self.labels), self.starts, axis=1)
        tp, fp = positive.cumsum(axis=1), negative.cumsum(axis=1)
        p, n = tp[:, -1], fp[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            auc = ((((tp - positive) + tp) * negative).sum(axis=1) / 2) / (p * n)
            precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
            ap = (positive * precision).sum(axis=1) / p
            index = (20 * tp >= 19 * p[:, None]).argmax(axis=1)
            fpr = fp[np.arange(len(w)), index] / n
        return np.where(((p > 0) & (n > 0))[:, None], np.column_stack([auc, ap, fpr]), np.nan)


def metric_dict(values):
    return {name: float(value) if np.isfinite(value) else None for name, value in zip(METRICS, values)}
