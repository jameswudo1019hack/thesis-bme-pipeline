"""Survival-model evaluation metrics for Aim 4 (right-censored time-to-event).

All functions take plain numpy arrays:

    time   : follow-up time (any unit, > 0)
    event  : 1 = event observed, 0 = censored
    score  : risk score, higher = higher risk
    surv_t : predicted survival probability S(t | x) at a given horizon
    groups : optional stratum label per subject (e.g. parent cohort)

Censoring weights use the Kaplan-Meier estimate of the censoring distribution
G(t) = P(C > t), estimated on the evaluation sample itself (recompute it
inside every bootstrap replicate). Pass `groups` to estimate G separately
within each stratum, which relaxes the assumption that censoring is
independent of stratum (in SHHS, administrative follow-up and loss to
follow-up differ by parent cohort). Subjects with an event at T_i are
weighted with the left limit G(T_i-), so censorings tied with the event time
do not down-weight it.

Implemented:
  harrell_c              pooled Harrell's C (ties in score count 1/2)
  within_stratum_c       Harrell's C over pairs from the same stratum only
  uno_c                  Uno et al. 2011 IPCW C-statistic truncated at tau
  cumulative_dynamic_auc Uno 2007 / Hung & Chiang 2010 IPCW AUC(t)
  brier_score            Graf et al. 1999 IPCW Brier score at t
  integrated_brier_score trapezoid integral of brier_score over a time grid
  km_risk                1 - KM(t) with a log(-log) Greenwood 95 % CI

Tied event/censoring times follow the Harrell convention used by lifelines
and scikit-survival: a subject censored at the same time as an event is taken
to have survived longer.

References: Harrell 1982 JAMA; Uno et al. 2011 Stat Med 30:1105-17;
Uno et al. 2007 JASA 102:527-37; Hung & Chiang 2010 Can J Stat 38:8-26;
Graf et al. 1999 Stat Med 18:2529-45.
"""
from __future__ import annotations

import numpy as np


def _as_arrays(time, event, *others):
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    out = [np.asarray(o, dtype=float) for o in others]
    n = len(t)
    if len(e) != n or any(len(o) != n for o in out):
        raise ValueError("time, event and scores must have the same length")
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("times must be finite and > 0")
    for o in out:
        if np.any(~np.isfinite(o)):
            raise ValueError("scores / predictions must be finite")
    return (t, e, *out)


class _KM:
    """Kaplan-Meier estimate of P(C > t) for one sample.

    Ties: events at time s are taken to occur before censorings at s (the same
    convention as the concordance statistics), so the censoring risk set at s
    is #{T > s} + #{censored at s}. With this convention the IPCW estimator
    mean(1[T <= t, event] / G(T-)) equals 1 - KM(t) exactly.
    """

    def __init__(self, t: np.ndarray, e: np.ndarray):
        cens = ~e
        self.times = np.unique(t[cens])
        order = np.sort(t)
        n_ge = len(t) - np.searchsorted(order, self.times, side="left")
        te = np.sort(t[e])
        n_events_at = np.searchsorted(te, self.times, side="right") - np.searchsorted(te, self.times, side="left")
        at_risk = n_ge - n_events_at
        tc = np.sort(t[cens])
        n_cens = (np.searchsorted(tc, self.times, side="right") - np.searchsorted(tc, self.times, side="left"))
        self.surv = np.cumprod(1.0 - n_cens / at_risk) if self.times.size else np.array([])

    def at(self, t: np.ndarray, left_limit: bool) -> np.ndarray:
        idx = np.searchsorted(self.times, t, side="left" if left_limit else "right") - 1
        g = np.ones_like(t, dtype=float)
        m = idx >= 0
        g[m] = self.surv[idx[m]]
        return g


class CensoringKM:
    """Censoring survival G(t) = P(C > t), marginal or stratified by `groups`."""

    def __init__(self, time, event, groups=None):
        t, e = _as_arrays(time, event)
        if groups is None:
            self.stratified = False
            self._km = {None: _KM(t, e)}
        else:
            g = np.asarray(groups)
            if len(g) != len(t):
                raise ValueError("groups must have the same length as time")
            self.stratified = True
            self._km = {k: _KM(t[g == k], e[g == k]) for k in np.unique(g)}

    def at(self, t, groups=None, left_limit: bool = False) -> np.ndarray:
        """G(t) (right-continuous) or G(t-) when left_limit=True, per element of t."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if not self.stratified:
            return self._km[None].at(t, left_limit)
        if groups is None:
            raise ValueError("stratified censoring model needs the subjects' groups")
        groups = np.broadcast_to(np.asarray(groups), t.shape)
        out = np.ones_like(t)
        for k, km in self._km.items():
            m = groups == k
            if m.any():
                out[m] = km.at(t[m], left_limit)
        unknown = ~np.isin(groups, list(self._km))
        if unknown.any():
            raise ValueError(f"groups not seen when fitting the censoring model: {set(groups[unknown])}")
        return out


def _comparable(t: np.ndarray, e: np.ndarray, i: int) -> np.ndarray:
    """Subjects j comparable with event subject i: T_j > T_i, or censored at T_i."""
    return (t > t[i]) | ((t == t[i]) & ~e)


def _g_and_groups(t, e, g, groups):
    if g is None:
        g = CensoringKM(t, e, groups)
    if g.stratified and groups is None:
        raise ValueError("pass the same groups used to fit the stratified censoring model")
    return g, (None if groups is None else np.asarray(groups))


def _sub(groups, mask):
    return None if groups is None else groups[mask]


def harrell_c(time, event, score) -> float:
    """Pooled Harrell's C: P(score_i > score_j | T_i < T_j, event_i). Ties in score count 1/2."""
    t, e, s = _as_arrays(time, event, score)
    num = den = 0.0
    for i in np.flatnonzero(e):
        comparable = _comparable(t, e, i)
        n = comparable.sum()
        if n == 0:
            continue
        sj = s[comparable]
        num += np.sum(s[i] > sj) + 0.5 * np.sum(s[i] == sj)
        den += n
    return float(num / den) if den > 0 else float("nan")


def within_stratum_c(time, event, score, groups) -> float:
    """Harrell's C using only pairs whose two subjects share a stratum."""
    t, e, s = _as_arrays(time, event, score)
    groups = np.asarray(groups)
    num = den = 0.0
    for i in np.flatnonzero(e):
        comparable = _comparable(t, e, i) & (groups == groups[i])
        n = comparable.sum()
        if n == 0:
            continue
        sj = s[comparable]
        num += np.sum(s[i] > sj) + 0.5 * np.sum(s[i] == sj)
        den += n
    return float(num / den) if den > 0 else float("nan")


def uno_c(time, event, score, tau: float, g: CensoringKM | None = None, groups=None) -> float:
    """Uno's IPCW concordance truncated at tau (pairs whose earlier time is < tau).

    Pair (i, j) is weighted 1 / (G_{z_i}(T_i-) * G_{z_j}(T_i-)), each subject's own
    stratum's censoring survival (Gerds et al. 2013, Stat Med 32:2173). With a
    marginal G this is Uno's 1 / G(T_i-)^2.
    """
    t, e, s = _as_arrays(time, event, score)
    g, groups = _g_and_groups(t, e, g, groups)
    cases = np.flatnonzero(e & (t < tau))
    if not cases.size:
        return float("nan")
    if g.stratified:
        levels, code = np.unique(groups, return_inverse=True)
        gmat = np.column_stack([g.at(t[cases], np.repeat(levels[k:k + 1], cases.size), left_limit=True)
                                for k in range(len(levels))])  # G_k(T_i-) for every case i and stratum k
    else:
        code = np.zeros(len(t), dtype=int)
        gmat = g.at(t[cases], left_limit=True)[:, None]
    num = den = 0.0
    for pos, i in enumerate(cases):
        comp = _comparable(t, e, i)
        if not comp.any():
            continue
        gij = gmat[pos, code[i]] * gmat[pos, code[comp]]
        w = np.where(gij > 0, 1.0 / np.where(gij > 0, gij, 1.0), 0.0)
        sj = s[comp]
        num += np.sum(w * ((s[i] > sj) + 0.5 * (s[i] == sj)))
        den += np.sum(w)
    return float(num / den) if den > 0 else float("nan")


def cumulative_dynamic_auc(time, event, score, t_eval: float, g: CensoringKM | None = None,
                           groups=None) -> float:
    """IPCW cumulative/dynamic AUC at t_eval.

    Cases: T_i <= t_eval with an event, weighted 1/G(T_i-).
    Controls: T_j > t_eval, weighted 1/G(t_eval) (constant unless G is stratified).
    """
    t, e, s = _as_arrays(time, event, score)
    g, groups = _g_and_groups(t, e, g, groups)
    cases = e & (t <= t_eval)
    controls = t > t_eval
    if not cases.any() or not controls.any():
        return float("nan")
    gi = g.at(t[cases], _sub(groups, cases), left_limit=True)
    wi = np.where(gi > 0, 1.0 / gi, 0.0)
    gj = g.at(np.full(controls.sum(), t_eval), _sub(groups, controls))
    vj = np.where(gj > 0, 1.0 / gj, 0.0)
    order = np.argsort(s[controls], kind="mergesort")
    sc, vc = s[controls][order], vj[order]
    cum = np.concatenate([[0.0], np.cumsum(vc)])
    lo = np.searchsorted(sc, s[cases], side="left")
    hi = np.searchsorted(sc, s[cases], side="right")
    frac = (cum[lo] + 0.5 * (cum[hi] - cum[lo])) / cum[-1]
    return float(np.sum(wi * frac) / np.sum(wi))


def brier_score(time, event, surv_t, t_eval: float, g: CensoringKM | None = None, groups=None) -> float:
    """Graf IPCW Brier score at t_eval for predicted survival S(t_eval | x)."""
    t, e, p = _as_arrays(time, event, surv_t)
    g, groups = _g_and_groups(t, e, g, groups)
    died = e & (t <= t_eval)
    alive = t > t_eval
    gi = g.at(t[died], _sub(groups, died), left_limit=True)
    ga = g.at(np.full(alive.sum(), t_eval), _sub(groups, alive))
    total = np.sum(np.where(gi > 0, (p[died] ** 2) / gi, 0.0))
    total += np.sum(np.where(ga > 0, ((1.0 - p[alive]) ** 2) / ga, 0.0))
    return float(total / len(t))


def integrated_brier_score(time, event, surv_grid: np.ndarray, grid: np.ndarray,
                           g: CensoringKM | None = None, groups=None) -> float:
    """Trapezoid integral of the Brier score over `grid`, divided by its span.

    surv_grid has shape (n_subjects, len(grid)): predicted S(grid_k | x_i).
    """
    t, e = _as_arrays(time, event)
    g, groups = _g_and_groups(t, e, g, groups)
    grid = np.asarray(grid, dtype=float)
    surv_grid = np.asarray(surv_grid, dtype=float)
    if surv_grid.shape != (len(t), len(grid)):
        raise ValueError("surv_grid must be (n_subjects, len(grid))")
    bs = np.array([brier_score(t, e, surv_grid[:, k], grid[k], g, groups) for k in range(len(grid))])
    return float(np.trapezoid(bs, grid) / (grid[-1] - grid[0]))


def km_risk(time, event, t_eval: float) -> tuple[float, float, float]:
    """Observed cumulative risk 1 - KM(t_eval) with a log(-log) Greenwood 95 % CI."""
    t, e = _as_arrays(time, event)
    times = np.unique(t[e & (t <= t_eval)])
    order = np.sort(t)
    s, var_sum = 1.0, 0.0
    for u in times:
        n = len(t) - np.searchsorted(order, u, side="left")
        d = np.sum(e & (t == u))
        s *= 1 - d / n
        if n > d:
            var_sum += d / (n * (n - d))
    if s <= 0 or s >= 1 or var_sum == 0:
        return float(1 - s), float("nan"), float("nan")
    se = np.sqrt(var_sum) / abs(np.log(s))
    lo_s, hi_s = s ** np.exp(1.96 * se), s ** np.exp(-1.96 * se)
    return float(1 - s), float(1 - hi_s), float(1 - lo_s)


def ipcw_risk(time, event, t_eval: float, g: CensoringKM | None = None, groups=None) -> float:
    """Observed cumulative risk by t_eval: mean of 1[T_i <= t, event_i] / G(T_i-).

    With a marginal censoring KM this equals 1 - KM(t_eval) exactly; with a
    stratified G it stays unbiased when censoring differs between strata
    (pooled KM does not). Competing deaths are treated as censoring, matching
    a cause-specific model's 1 - S(t).
    """
    t, e = _as_arrays(time, event)
    g, groups = _g_and_groups(t, e, g, groups)
    died = e & (t <= t_eval)
    gi = g.at(t[died], _sub(groups, died), left_limit=True)
    return float(np.sum(np.where(gi > 0, 1.0 / gi, 0.0)) / len(t))
