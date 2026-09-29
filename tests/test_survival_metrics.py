"""Tests for thesis_pipeline.survival_metrics against closed-form or reference cases."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis_pipeline.survival_metrics import (  # noqa: E402
    CensoringKM,
    brier_score,
    cumulative_dynamic_auc,
    harrell_c,
    integrated_brier_score,
    km_risk,
    uno_c,
)


def _sim(n=400, cens_rate=0.5, seed=0, ties=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t_event = rng.exponential(scale=np.exp(-0.8 * x))
    t_cens = rng.exponential(scale=1 / cens_rate, size=n) if cens_rate > 0 else np.full(n, np.inf)
    t = np.minimum(t_event, t_cens)
    e = (t_event <= t_cens).astype(int)
    if ties:
        t = np.ceil(t * 10) / 10
    return t, e, x


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("ties", [False, True])
def test_harrell_matches_lifelines(seed, ties):
    t, e, x = _sim(seed=seed, ties=ties)
    score = np.round(x, 1) if ties else x
    assert harrell_c(t, e, score) == pytest.approx(concordance_index(t, -score, e), abs=1e-12)


def test_uno_equals_harrell_without_censoring():
    t, e, x = _sim(cens_rate=0, seed=3)
    assert e.all()
    assert uno_c(t, e, x, tau=np.inf) == pytest.approx(harrell_c(t, e, x), abs=1e-12)


def test_uno_weights_upweight_late_events():
    # With censoring, Uno's C differs from Harrell's but stays in [0, 1] and near it on random data.
    t, e, x = _sim(seed=4)
    u = uno_c(t, e, x, tau=np.quantile(t, 0.9))
    h = harrell_c(t, e, x)
    assert 0.5 < u < 1 and abs(u - h) < 0.05


def test_censoring_km_matches_lifelines():
    t, e, _ = _sim(seed=5, ties=True)
    g = CensoringKM(t, e)
    # reference: lifelines KM of censoring with events moved just before tied censorings
    kmf = KaplanMeierFitter().fit(np.where(e == 1, t - 1e-9, t), event_observed=1 - e)
    grid = np.linspace(0.05, np.quantile(t, 0.95), 25)
    ref = kmf.survival_function_at_times(grid).to_numpy()
    assert np.allclose(g.at(grid), ref, atol=1e-12)


def test_left_limit_excludes_censorings_at_t():
    t = np.array([1.0, 2.0, 2.0, 3.0])
    e = np.array([1, 0, 1, 1])
    g = CensoringKM(t, e)
    assert g.at(2.0, left_limit=True)[0] == 1.0
    # the event at t=2 precedes the censoring at t=2, so the censoring risk set is {censored@2, T=3}
    assert g.at(2.0)[0] == pytest.approx(1 - 1 / 2)


def test_td_auc_equals_plain_auc_without_censoring():
    t, e, x = _sim(cens_rate=0, seed=6)
    t_eval = np.median(t)
    y = (t <= t_eval).astype(int)
    assert cumulative_dynamic_auc(t, e, x, t_eval) == pytest.approx(roc_auc_score(y, x), abs=1e-12)


def test_brier_without_censoring_is_mse():
    t, e, x = _sim(cens_rate=0, seed=7)
    t_eval = np.median(t)
    rng = np.random.default_rng(0)
    s = rng.uniform(size=len(t))
    alive = (t > t_eval).astype(float)
    assert brier_score(t, e, s, t_eval) == pytest.approx(np.mean((alive - s) ** 2), abs=1e-12)


def test_integrated_brier_constant_prediction():
    t, e, _ = _sim(cens_rate=0, seed=8)
    grid = np.linspace(np.quantile(t, 0.1), np.quantile(t, 0.8), 15)
    surv = np.full((len(t), len(grid)), 0.5)
    ibs = integrated_brier_score(t, e, surv, grid)
    assert ibs == pytest.approx(0.25, abs=1e-12)


def test_km_risk_matches_lifelines():
    t, e, _ = _sim(seed=9)
    t_eval = np.quantile(t, 0.6)
    risk, lo, hi = km_risk(t, e, t_eval)
    kmf = KaplanMeierFitter().fit(t, event_observed=e)
    assert risk == pytest.approx(1 - kmf.survival_function_at_times([t_eval]).iloc[0], abs=1e-12)
    ci = kmf.confidence_interval_survival_function_
    row = ci[ci.index <= t_eval].iloc[-1]
    assert lo == pytest.approx(1 - row.iloc[1], abs=1e-6)
    assert hi == pytest.approx(1 - row.iloc[0], abs=1e-6)


def test_input_validation():
    with pytest.raises(ValueError):
        harrell_c([1, 2], [1], [0.1, 0.2])
    with pytest.raises(ValueError):
        harrell_c([0, 2], [1, 1], [0.1, 0.2])


# ---------------------------------------------------------------- stratified censoring (v2)

from thesis_pipeline.survival_metrics import within_stratum_c  # noqa: E402


def _groups(n, k=3, seed=11):
    return np.random.default_rng(seed).integers(0, k, n).astype(str)


def test_single_group_equals_marginal():
    t, e, x = _sim(seed=12)
    g1 = np.full(len(t), "A")
    tq = np.quantile(t, 0.7)
    s = np.clip(1 - (x - x.min()) / (x.max() - x.min()), 0.01, 0.99)
    assert uno_c(t, e, x, tq, groups=g1) == pytest.approx(uno_c(t, e, x, tq), abs=1e-12)
    assert cumulative_dynamic_auc(t, e, x, tq, groups=g1) == pytest.approx(cumulative_dynamic_auc(t, e, x, tq), abs=1e-12)
    assert brier_score(t, e, s, tq, groups=g1) == pytest.approx(brier_score(t, e, s, tq), abs=1e-12)


def test_stratified_km_matches_lifelines_per_group():
    t, e, _ = _sim(seed=13, ties=True)
    grp = _groups(len(t))
    g = CensoringKM(t, e, grp)
    grid = np.quantile(t, [0.1, 0.3, 0.5, 0.7])
    for k in np.unique(grp):
        m = grp == k
        ref = KaplanMeierFitter().fit(np.where(e[m] == 1, t[m] - 1e-9, t[m]),
                                      event_observed=1 - e[m]).survival_function_at_times(grid).to_numpy()
        assert np.allclose(g.at(grid, np.full(len(grid), k)), ref, atol=1e-12)


def _brute_auc(t, e, s, tt, G):
    num = den = 0.0
    for i in range(len(t)):
        if not (e[i] and t[i] <= tt):
            continue
        wi = 1 / G(t[i], i, True)
        for j in range(len(t)):
            if t[j] > tt:
                vj = 1 / G(tt, j, False)
                num += wi * vj * ((s[i] > s[j]) + 0.5 * (s[i] == s[j]))
                den += wi * vj
    return num / den


def test_stratified_auc_and_brier_brute_force():
    t, e, x = _sim(n=150, seed=14)
    x = np.round(x, 1)  # score ties
    grp = _groups(len(t), seed=15)
    g = CensoringKM(t, e, grp)
    G = lambda u, idx, left: g.at(u, np.array([grp[idx]]), left_limit=left)[0]  # noqa: E731
    tt = np.quantile(t, 0.6)
    assert cumulative_dynamic_auc(t, e, x, tt, g, grp) == pytest.approx(_brute_auc(t, e, x, tt, G), abs=1e-12)
    s = np.random.default_rng(1).uniform(0.05, 0.95, len(t))
    bs = sum(((s[i] ** 2) / G(t[i], i, True) if (e[i] and t[i] <= tt) else 0.0)
             + (((1 - s[i]) ** 2) / G(tt, i, False) if t[i] > tt else 0.0) for i in range(len(t))) / len(t)
    assert brier_score(t, e, s, tt, g, grp) == pytest.approx(bs, abs=1e-12)


def test_within_stratum_c_pools_per_group_pairs():
    t, e, x = _sim(seed=16)
    grp = _groups(len(t), seed=17)
    num = den = 0.0
    for k in np.unique(grp):
        m = grp == k
        tk, ek, xk = t[m], e[m], x[m]
        for i in np.flatnonzero(ek):
            comp = (tk > tk[i]) | ((tk == tk[i]) & (ek == 0))
            num += np.sum(xk[i] > xk[comp]) + 0.5 * np.sum(xk[i] == xk[comp])
            den += comp.sum()
    assert within_stratum_c(t, e, x, grp) == pytest.approx(num / den, abs=1e-12)


def test_stratified_model_requires_groups():
    t, e, x = _sim(seed=18)
    g = CensoringKM(t, e, _groups(len(t)))
    with pytest.raises(ValueError):
        uno_c(t, e, x, 1.0, g=g)


def test_ipcw_risk_equals_km_when_marginal():
    from thesis_pipeline.survival_metrics import ipcw_risk
    t, e, _ = _sim(seed=19, ties=True)
    for q in (0.3, 0.6, 0.9):
        tt = np.quantile(t, q)
        assert ipcw_risk(t, e, tt) == pytest.approx(km_risk(t, e, tt)[0], abs=1e-10)


def test_ipcw_risk_stratified_single_group():
    from thesis_pipeline.survival_metrics import ipcw_risk
    t, e, _ = _sim(seed=20)
    tt = np.quantile(t, 0.5)
    assert ipcw_risk(t, e, tt, groups=np.full(len(t), "A")) == pytest.approx(ipcw_risk(t, e, tt), abs=1e-12)


def test_uno_c_unbiased_under_group_dependent_censoring():
    """Truncated C from uncensored times is the target; stratified Uno C must recover it."""
    rng = np.random.default_rng(21)
    est, truth = [], []
    for _ in range(8):
        n = 2000
        grp = rng.integers(0, 2, n)
        x = rng.normal(size=n) + 1.0 * grp  # group 1 is high risk ...
        t_ev = rng.exponential(np.exp(-0.8 * x))
        t_c = rng.exponential(np.where(grp == 1, 0.8, 8.0))  # ... and heavily censored
        tau = np.quantile(t_ev, 0.7)
        t = np.minimum(t_ev, t_c)
        e = (t_ev <= t_c).astype(int)
        est.append(uno_c(t, e, x, tau, groups=grp.astype(str)))
        truth.append(uno_c(t_ev, np.ones(n, int), x, tau))
    assert abs(np.mean(est) - np.mean(truth)) < 0.01
