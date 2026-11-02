"""Calibration and distribution metrics shared by eval.py."""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = x / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ece(conf, correct, n_bins: int = 15) -> float:
    """Expected calibration error of top-1 confidence, equal-width bins, bin = min(int(conf*n), n-1)
    (same binning as the typed-decisions scorer, bench/score_with_jobs.py, which uses n_bins=10)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(err)


def fit_temperature(logit_list: list[np.ndarray], target_list: list[np.ndarray],
                    grid=np.exp(np.linspace(np.log(0.05), np.log(20), 200))) -> float:
    """Single temperature minimising soft-target NLL over rows with ragged option counts (grid + refine)."""
    def nll(t):
        tot = 0.0
        for lg, tg in zip(logit_list, target_list):
            p = softmax(lg, t)
            tot -= float((tg * np.log(p + 1e-12)).sum())
        return tot / max(len(logit_list), 1)
    best = min(grid, key=nll)
    lo, hi = best / 1.2, best * 1.2
    for _ in range(30):  # golden-section refine
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return float((lo + hi) / 2)


def kl(p_gold: np.ndarray, q_pred: np.ndarray) -> float:
    """KL(gold || pred) exactly as the typed-decisions scorer: sum over gold>0 of g*log(g/max(p,1e-9))."""
    m = p_gold > 0
    return float((p_gold[m] * np.log(p_gold[m] / np.maximum(q_pred[m], 1e-9))).sum())


def brier(p_gold: np.ndarray, q_pred: np.ndarray) -> float:
    return float(((q_pred - p_gold) ** 2).sum())
