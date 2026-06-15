"""Confidence calibration utilities.

Raw neural-net softmax scores are notoriously over-confident, which is exactly
what makes a naive confidence threshold dangerous for an abstention system.
We apply **temperature scaling** (Guo et al., 2017): divide the logits by a
single scalar T learned on a validation set. T > 1 softens the distribution so
that the reported confidence tracks the true accuracy more closely.

We also expose **normalized predictive entropy** in [0, 1], a distribution-shape
signal the gate uses to detect "the model is spreading its bets" situations that
a top-1 threshold alone would miss.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def softmax(logits: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def normalized_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy scaled to [0, 1] (1 = uniform / maximally uncertain)."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    h = -(p * np.log(p)).sum()
    return float(h / math.log(len(probs)))


def fit_temperature(
    logits: List[Sequence[float]],
    labels: Sequence[int],
    lr: float = 0.01,
    max_iter: int = 300,
) -> float:
    """Fit a single temperature T minimizing NLL on a validation set.

    Pure-numpy gradient descent so this has no torch dependency and can run in
    the training pipeline or a notebook. Returns the fitted temperature.
    """
    L = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    log_t = 0.0  # optimize in log-space to keep T > 0
    n = len(y)
    for _ in range(max_iter):
        T = math.exp(log_t)
        z = L / T
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        # dNLL/d(logT): chain rule through z = L / exp(logT)
        correct_logit = L[np.arange(n), y]
        exp_logit = (p * L).sum(axis=1)
        grad = ((correct_logit - exp_logit) / T).mean()  # d/dlogT scales by T cancels
        log_t += lr * grad
        log_t = float(np.clip(log_t, -3.0, 3.0))
    return math.exp(log_t)
