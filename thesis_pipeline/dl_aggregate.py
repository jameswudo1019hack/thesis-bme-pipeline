"""Per-second probabilities -> 30-s epoch scores (Aim 2 DL arm, design section 4).

Candidate aggregators (all applied to the 30 per-second probabilities of one epoch):

    mean   mean of the 30 values
    max    maximum
    k10    10th-largest value (the score is high only if >= 10 seconds are high)
    c10    max over the 21 ten-second sub-windows of the sub-window minimum, i.e. the
           largest p such that >= 10 *consecutive* seconds are >= p. This mirrors
           Olsen's ">= 10 consecutive seconds" rule and the Aim 2 ">= 10 s overlap" label.

Pre-declared selection rule (``choose_aggregator``): compute the validation epoch AUC of
every candidate; if ``mean`` is within ``tie`` (0.002) AUC of the best candidate choose
``mean``, otherwise choose the best. The choice is made once on M seed-42 validation
predictions and frozen for every config and seed.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

AGGREGATORS: tuple[str, ...] = ("mean", "max", "k10", "c10")
EPOCH_S = 30
TIE_AUC = 0.002


def sec_to_epoch(p: np.ndarray, rule: str) -> np.ndarray:
    """Aggregate an (n, 30) matrix of per-second probabilities to (n,) epoch scores."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != EPOCH_S:
        raise ValueError(f"expected (n, {EPOCH_S}); got {p.shape}")
    if rule == "mean":
        return p.mean(axis=1)
    if rule == "max":
        return p.max(axis=1)
    if rule == "k10":
        return np.partition(p, EPOCH_S - 10, axis=1)[:, EPOCH_S - 10]
    if rule == "c10":
        win = np.lib.stride_tricks.sliding_window_view(p, 10, axis=1)  # (n, 21, 10)
        return win.min(axis=2).max(axis=1)
    raise ValueError(f"unknown aggregator {rule!r}; choose from {AGGREGATORS}")


def all_aggregates(p: np.ndarray) -> dict[str, np.ndarray]:
    """All candidate aggregates of an (n, 30) matrix."""
    return {r: sec_to_epoch(p, r) for r in AGGREGATORS}


def choose_aggregator(aucs: Mapping[str, float], tie: float = TIE_AUC) -> dict:
    """Apply the pre-declared rule to validation AUCs keyed by aggregator name.

    Returns {"chosen", "best", "best_auc", "mean_auc", "tie", "aucs"}.
    """
    missing = set(AGGREGATORS) - set(aucs)
    if missing:
        raise ValueError(f"missing aggregator AUCs: {sorted(missing)}")
    finite = {k: float(v) for k, v in aucs.items() if np.isfinite(v)}
    best = max(finite, key=lambda k: (finite[k], k == "mean"))
    chosen = "mean" if finite.get("mean", -np.inf) >= finite[best] - tie else best
    return {
        "chosen": chosen,
        "best": best,
        "best_auc": finite[best],
        "mean_auc": finite.get("mean"),
        "tie": tie,
        "aucs": {k: float(v) for k, v in aucs.items()},
        "rule": "mean if AUC(mean) >= max AUC - tie, else argmax AUC",
    }
