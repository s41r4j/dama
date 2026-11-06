"""Metrics with numpy only (no sklearn needed)."""
from __future__ import annotations
import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def auc(y, s):
    """Area under the ROC curve via the rank-sum formula; None if one class is missing."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return None
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def best_f1(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    best = (0.0, 0.5)
    for t in np.unique(np.round(p, 3)):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        if tp == 0:
            continue
        prec, rec = tp / pred.sum(), tp / (y == 1).sum()
        best = max(best, (2 * prec * rec / (prec + rec), float(t)))
    return {"f1": float(best[0]), "threshold": best[1]}


def ece(prob, correct, bins=15):
    prob, correct = np.asarray(prob, float), np.asarray(correct, float)
    if len(prob) == 0:
        return None
    which = np.minimum((prob * bins).astype(int), bins - 1)
    return float(sum(abs(prob[which == b].mean() - correct[which == b].mean()) * (which == b).mean()
                     for b in range(bins) if (which == b).any()))


def binary_ece(y, p):
    """Calibration of P(y=1): compare predicted probability with observed frequency."""
    return ece(p, y)


def macro_f1(y, pred, n):
    y, pred = np.asarray(y), np.asarray(pred)
    scores = []
    for c in range(n):
        tp = ((pred == c) & (y == c)).sum()
        if (y == c).sum() == 0:
            continue
        prec = tp / max(1, (pred == c).sum())
        rec = tp / (y == c).sum()
        scores.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(scores)) if scores else None


def recall_at_k(ranked_gold_flags: list[list[int]], n_gold: list[int], k: int):
    """Mean fraction of a question's gold turns found in the top-k (gold missing from the pool counts as missed)."""
    vals = [sum(flags[:k]) / g for flags, g in zip(ranked_gold_flags, n_gold) if g]
    return float(np.mean(vals)) if vals else None


def mrr(ranked_gold_flags):
    vals = [next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0) for flags in ranked_gold_flags]
    return float(np.mean(vals)) if vals else None


def fit_platt(logits, y, steps=300):
    """Fit p = sigmoid(a*z + b) on held-out data (keeps ranking, fixes probabilities). Returns (a, b)."""
    import torch
    z = torch.tensor(np.asarray(logits, float), dtype=torch.float64)
    t = torch.tensor(np.asarray(y, float), dtype=torch.float64)
    a = torch.ones((), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + b, t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(a), float(b)


def fit_temperature(logits, y, steps=300):
    """Multi-class temperature scaling. Returns T (> 0)."""
    import torch
    z = torch.tensor(np.asarray(logits, float), dtype=torch.float64)
    t = torch.tensor(np.asarray(y), dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=steps)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def conformal_threshold(min_gold_scores: list[float], alpha: float) -> float:
    """Split-conformal lambda: admit every candidate with score >= lambda to include all gold with prob >= 1-alpha.

    min_gold_scores[i] is the lowest calibrated score among question i's gold turns, or -inf if retrieval
    missed one of them (so retrieval misses count against the guarantee)."""
    s = sorted(min_gold_scores)
    k = math.floor((len(s) + 1) * alpha)
    return -math.inf if k < 1 else s[k - 1]
