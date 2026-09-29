"""Aim 4 — tabular survival baseline: where does CVD prognostic signal come from?

Nested Cox ladder on SHHS-1 (design spec 2026-09-29, amended after critic review):

  FRS-fixed   D'Agostino 2008 published 10-y risk, no refit (external benchmark)
  FRS-recal   stratified Cox on the published FRS linear predictor
  M0          clinical risk factors (13 terms)
  M1          AHI only (log1p rdi4p, the SHHS 4 % definition)
  M2          M0 + AHI
  M2-cat      M0 + AHI categories 5 / 15 / 30 (ref < 5)
  M3          M2 + hypoxaemia (log1p % time SpO2 < 90, mean NREM SpO2)
  M3-swap     M0 + hypoxaemia, no AHI
  M4          M3 + central apnoea, arousal index, TST, WASO, % N3, % REM,
              sleeping heart rate (pipeline ECG)
  M4-pooled   unstratified Cox on the M4 terms + cohort indicators (same pooled
              likelihood and inputs as XGB-M4, so XGB-M4 - M4-pooled isolates
              functional form; also the unstratified sensitivity)
  XGB-M4      XGBoost survival:cox on the M4 terms + cohort indicators

Outcomes: incident CVD composite (primary) and hard CVD = MI, stroke, HF or
CVD death (co-primary), in subjects free of prevalent CVD by adjudicated
records or self-report (P3).

Scores
------
Every Cox model is stratified by inferred parent cohort (ARIC, CHS, FHS-A,
FHS-B, TUC; plus NY in the all-cause-death analysis). Pooled discrimination (Harrell C, Uno C, AUC(t)),
Brier and calibration all use the predicted 10-year risk, which includes the
stratum baseline. Within-cohort C on the linear predictor X*beta measures the
covariates' contribution alone. Paired deltas are reported for both.

Censoring weights (Uno C, AUC, Brier) use a Kaplan-Meier censoring model
estimated within each cohort, because administrative follow-up and loss to
follow-up differ by cohort. The scaled-Brier null model is the cohort-
specific training KM.

Protocol
--------
* Split: GroupShuffleSplit(test_size=0.2, random_state=42) over the 5,793
  subjects with feature parquets (the thesis-wide patient-level split,
  asserted equal to the Aim 2 test subjects), applied before exclusions.
* Imputation (IterativeImputer, cohort dummies as auxiliaries) and
  standardisation are fitted on training subjects only (per fold in CV).
* Default run is TRAIN-ONLY. The held-out test set is scored only with
  --evaluate-test, to be used once after the script is frozen.

Outputs: models/aim4_cox_v1/ (metrics.json, hr_tables/, test_predictions__*.parquet,
         bootstrap__*.npz, diagnostics/, sleep_hr_cache.parquet)
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import click
import lifelines
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from lifelines import AalenJohansenFitter, CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline import cvd_covariates as cc  # noqa: E402
from thesis_pipeline import cvd_outcomes as co  # noqa: E402
from thesis_pipeline import survival_metrics as sm  # noqa: E402

CSV_DIR = CODE_ROOT.parent / "Dataset" / "shhs" / "csv"
FEATURES_DIR = CODE_ROOT / "features"
OUT_ROOT = CODE_ROOT / "models" / "aim4_cox_v1"
AIM2_REFERENCE = CODE_ROOT / "models" / "recovery_2026-05-03" / "aim2_v6_past_only" / "test_predictions.parquet"
HR_TAR = CODE_ROOT / "features-phase1batch-v1.tar"
AIM4_FILES = ["scripts/fit_aim4_cox_baseline.py", "thesis_pipeline/cvd_outcomes.py",
              "thesis_pipeline/cvd_covariates.py", "thesis_pipeline/survival_metrics.py",
              "tests/test_survival_metrics.py", "tests/test_cvd_covariates.py"]

YEAR = 365.25
H5, H10 = 5.0, 10.0
IBS_GRID = np.arange(1.0, 10.0 + 1e-9, 0.5)
PENALIZER = 0.01
SEED = 42

M0 = cc.CLINICAL
HYPOX = ["l_pctlt90", "sao2nrem"]
FULL_PSG = ["l_cai", "l_ai_all", "tst_h", "l_waso", "times34p", "timeremp", "hr_sleep"]
RDI_CATS = ["rdi4p_5_15", "rdi4p_15_30", "rdi4p_30p"]
LADDER = {
    "M0": M0,
    "M1": ["l_rdi4p"],
    "M2": M0 + ["l_rdi4p"],
    "M2-cat": M0 + RDI_CATS,
    "M3": M0 + ["l_rdi4p"] + HYPOX,
    "M3-swap": M0 + HYPOX,
    "M4": M0 + ["l_rdi4p"] + HYPOX + FULL_PSG,
    "FRS-recal": ["frs_lp"],
}
SMALL = ("M0", "M2", "M4")
PAIRS = [("M2", "M0"), ("M3", "M2"), ("M4", "M3"), ("M4", "M0"), ("M3-swap", "M0"),
         ("XGB-M4", "M4-pooled"), ("XGB-M4", "M4"), ("M4-pooled", "M4"), ("M0", "FRS-recal"),
         ("M2-cat", "M0"), ("M0", "FRS-fixed")]
POOLED = "M4-pooled"
IMPUTE_COLS = M0 + ["l_rdi4p", "l_ahi_a0h3a"] + HYPOX + FULL_PSG
UNSCALED = set(cc.BINARY) | set(RDI_CATS) | {"lp_frs", "frs_lp", "prev_cvd_sr"}
FRS_OUTCOMES = {"cvd", "hard_cvd", "any_cvd"}  # FRS predicts general CVD; not scored against CHD, HF or death
CV_REPEATS = 3
XGB_PARAMS = {"objective": "survival:cox", "eval_metric": "cox-nloglik", "learning_rate": 0.03, "max_depth": 3,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10, "reg_lambda": 1.0,
              "seed": SEED, "nthread": 4, "verbosity": 0}
BOOT_KEYS = ["harrell_c", "within_cohort_c", "uno_c_10y", "auc_5y", "auc_10y", "brier_10y", "ibs_1_10y",
             "scaled_brier_10y"]


# ============================================================================ design matrix

class Preprocessor:
    """Train-only imputation + standardisation, then derived terms (FRS LP, AHI categories)."""

    def __init__(self, impute_cols: list[str] = IMPUTE_COLS):
        self.cols = list(impute_cols)

    def fit(self, raw: pd.DataFrame, coh: pd.Series) -> "Preprocessor":
        self.coh_levels = sorted(coh.unique())
        self.imputer = IterativeImputer(max_iter=10, random_state=SEED, sample_posterior=False)
        self.imputer.fit(self._imputer_input(raw, coh))
        imp = self._impute(raw, coh)
        cont = [c for c in self.cols if c not in UNSCALED]
        self.mean = imp[cont].mean()
        self.sd = imp[cont].std(ddof=0)
        return self

    def _imputer_input(self, raw, coh):
        dummies = pd.get_dummies(coh, dtype=float).reindex(columns=self.coh_levels, fill_value=0.0)
        return np.column_stack([raw[self.cols].to_numpy(float), dummies.to_numpy()])

    def _impute(self, raw, coh) -> pd.DataFrame:
        arr = self.imputer.transform(self._imputer_input(raw, coh))[:, :len(self.cols)]
        imp = pd.DataFrame(arr, index=raw.index, columns=self.cols)
        for b in [c for c in self.cols if c in UNSCALED]:
            imp[b] = imp[b].clip(0, 1).round()
        return imp

    def transform(self, raw: pd.DataFrame, coh: pd.Series) -> pd.DataFrame:
        imp = self._impute(raw, coh)
        out = imp.copy()
        cont = list(self.mean.index)
        out[cont] = (imp[cont] - self.mean) / self.sd
        out["lp_frs"] = cc.frs_linear_predictor(np.exp(imp["ln_age"]), np.exp(imp["ln_chol"]), np.exp(imp["ln_hdl"]),
                                                np.exp(imp["ln_sbp"]), imp["trt"], imp["smk"], imp["dm"], imp["male"])
        out["frs_risk10"] = cc.frs_risk10(out["lp_frs"], imp["male"])
        # published 10-y log cumulative hazard incl. the sex-specific baseline: ranks the published risk exactly
        out["frs_lp"] = np.log(-np.log(1 - np.clip(out["frs_risk10"], 1e-12, 1 - 1e-12)))
        rdi = raw["rdi4p"].to_numpy(float)
        out["rdi4p_5_15"] = ((rdi >= 5) & (rdi < 15)).astype(float)
        out["rdi4p_15_30"] = ((rdi >= 15) & (rdi < 30)).astype(float)
        out["rdi4p_30p"] = (rdi >= 30).astype(float)
        out["coh"] = coh.values
        return out


# ============================================================================ models

def fit_cox(df: pd.DataFrame, terms: list[str], penalizer: float = PENALIZER, stratified: bool = True) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    if stratified:
        cph.fit(df[terms + ["coh", "T", "E"]], duration_col="T", event_col="E", strata=["coh"])
    else:
        cph.fit(df[terms + ["T", "E"]], duration_col="T", event_col="E")
    return cph


def cohort_dummies(df: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    """Indicators for every cohort but the first (reference) level."""
    d = pd.get_dummies(df["coh"], dtype=float).reindex(columns=levels, fill_value=0.0)
    return d[levels[1:]].add_prefix("coh_")


def stratified_survival(cph: CoxPHFitter, x: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    """S(t | x) = exp(-H0_stratum(t) * exp(lp)), with the Breslow step-function baseline.

    Computed explicitly because lifelines' predict_survival_function for
    stratified models (a) returns columns grouped by stratum rather than in
    input order and (b) linearly interpolates H0 between event times (and a
    repeated time in `times` duplicates rows). Rows here follow `x`; the
    baseline is right-continuous, as in R's survfit. Identical to lifelines
    when the same linear interpolation is applied (checked 2026-09-29).
    """
    lp = np.asarray(cph.predict_log_partial_hazard(x), dtype=float)
    H = cph.baseline_cumulative_hazard_
    out = np.empty((len(x), len(times)))
    if not cph.strata:  # unstratified model: one baseline for everyone
        h = H.iloc[:, 0]
        k = np.searchsorted(h.index.to_numpy(), times, side="right") - 1
        h0 = np.where(k >= 0, h.to_numpy()[np.clip(k, 0, None)], 0.0)
        return np.exp(-h0[None, :] * np.exp(lp)[:, None])
    strata = x["coh"].to_numpy()
    for st in np.unique(strata):
        if st not in H.columns:
            raise KeyError(f"stratum {st!r} not seen in training")
        h = H[st].dropna()
        k = np.searchsorted(h.index.to_numpy(), times, side="right") - 1
        h0 = np.where(k >= 0, h.to_numpy()[np.clip(k, 0, None)], 0.0)
        m = strata == st
        out[m] = np.exp(-h0[None, :] * np.exp(lp[m])[:, None])
    return out


def cox_predict(cph: CoxPHFitter, df: pd.DataFrame, terms: list[str]) -> dict:
    x = df[terms + ["coh"]] if cph.strata else df[terms]
    lp = np.asarray(cph.predict_log_partial_hazard(x), dtype=float)
    surv = stratified_survival(cph, x, np.concatenate([[H5, H10], IBS_GRID]))
    return {"lp": lp, "s5": surv[:, 0], "s10": surv[:, 1], "s_grid": surv[:, 2:]}


def xgb_label(df):
    return np.where(df["E"].to_numpy() == 1, df["T"].to_numpy(), -df["T"].to_numpy())


def xgb_matrix(df: pd.DataFrame, terms: list[str], coh_levels: list[str]) -> np.ndarray:
    """M4 terms plus cohort indicators, so XGBoost sees what the Cox strata absorb."""
    dummies = pd.get_dummies(df["coh"], dtype=float).reindex(columns=coh_levels, fill_value=0.0)
    return np.column_stack([df[terms].to_numpy(float), dummies.to_numpy()])


def fit_xgb(df: pd.DataFrame, terms: list[str]) -> dict:
    """XGBoost survival:cox; rounds by 5-fold CV; stage-2 Cox fitted on OUT-OF-FOLD margins."""
    levels = sorted(df["coh"].unique())
    X = xgb_matrix(df, terms, levels)
    y = xgb_label(df)
    strat = (df["coh"].astype(str) + "_" + df["E"].astype(str)).to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(df)), strat))
    cv = xgb.cv(XGB_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=1000, folds=folds,
                early_stopping_rounds=50, verbose_eval=False)
    n_rounds = len(cv)
    oof = np.empty(len(df))
    for tr_i, va_i in folds:
        b = xgb.train(XGB_PARAMS, xgb.DMatrix(X[tr_i], label=y[tr_i]), num_boost_round=n_rounds)
        oof[va_i] = b.predict(xgb.DMatrix(X[va_i]), output_margin=True)
    booster = xgb.train(XGB_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=n_rounds)
    stage2 = fit_cox(df.assign(xgb_margin=oof), ["xgb_margin"], penalizer=0.0)
    return {"booster": booster, "stage2": stage2, "levels": levels, "n_rounds": n_rounds,
            "stage2_slope_oof": float(stage2.params_.iloc[0])}


def xgb_margin(model: dict, df: pd.DataFrame, terms: list[str]) -> np.ndarray:
    return model["booster"].predict(xgb.DMatrix(xgb_matrix(df, terms, model["levels"])), output_margin=True)


def fit_all(train: pd.DataFrame, ladder: dict, with_xgb: bool) -> dict:
    fitted = {name: ("cox", fit_cox(train, terms), terms, None) for name, terms in ladder.items()}
    if with_xgb:
        levels = sorted(train["coh"].unique())
        cd = cohort_dummies(train, levels)
        pooled_terms = LADDER["M4"] + list(cd.columns)
        fitted[POOLED] = ("cox_pooled", fit_cox(pd.concat([train, cd], axis=1), pooled_terms, stratified=False),
                          pooled_terms, {"levels": levels})
        model = fit_xgb(train, LADDER["M4"])
        fitted["XGB-M4"] = ("xgb", model, LADDER["M4"],
                            {"n_rounds": model["n_rounds"], "stage2_slope_oof": model["stage2_slope_oof"]})
    return fitted


def predict_all(fitted: dict, df: pd.DataFrame, with_frs: bool = True) -> dict:
    """Per model: lp (covariate-only linear predictor), risk10 (ranking score), s5, s10, s_grid."""
    preds = {}
    for name, (kind, model, terms, extra) in fitted.items():
        if kind == "cox":
            p = cox_predict(model, df, terms)
        elif kind == "cox_pooled":
            p = cox_predict(model, pd.concat([df, cohort_dummies(df, extra["levels"])], axis=1), terms)
        else:
            d = df.assign(xgb_margin=xgb_margin(model, df, terms))
            p = cox_predict(model["stage2"], d, ["xgb_margin"])  # lp = stage-2 slope x margin
        p["risk10"] = 1 - p["s10"]
        preds[name] = p
    if with_frs:
        s10 = 1 - df["frs_risk10"].to_numpy()
        preds["FRS-fixed"] = {"lp": df["frs_lp"].to_numpy(), "risk10": 1 - s10, "s10": s10,
                              "s5": None, "s_grid": None}
    return preds


# ============================================================================ evaluation

class StratumKMNull:
    """Null model for the scaled Brier score: training KM within each cohort."""

    def __init__(self, train: pd.DataFrame):
        self.s10 = {c: float(KaplanMeierFitter().fit(g["T"], g["E"]).survival_function_at_times([H10]).iloc[0])
                    for c, g in train.groupby("coh")}
        self.overall = float(KaplanMeierFitter().fit(train["T"], train["E"]).survival_function_at_times([H10]).iloc[0])

    def predict(self, coh: np.ndarray) -> np.ndarray:
        return np.array([self.s10[c] for c in coh])


def metric_row(T, E, coh, p, g, null10) -> dict:
    """All test metrics for one model. `g` is the cohort-stratified censoring KM of this sample."""
    r = p["risk10"]
    row = {"harrell_c": sm.harrell_c(T, E, r),
           "within_cohort_c": sm.within_stratum_c(T, E, p["lp"], coh),
           "uno_c_10y": sm.uno_c(T, E, r, tau=H10, g=g, groups=coh),
           "auc_5y": sm.cumulative_dynamic_auc(T, E, r, H5, g=g, groups=coh),
           "auc_10y": sm.cumulative_dynamic_auc(T, E, r, H10, g=g, groups=coh),
           "brier_10y": sm.brier_score(T, E, p["s10"], H10, g=g, groups=coh)}
    bs_null = sm.brier_score(T, E, null10, H10, g=g, groups=coh)
    row["scaled_brier_10y"] = 1 - row["brier_10y"] / bs_null
    row["ibs_1_10y"] = (sm.integrated_brier_score(T, E, p["s_grid"], IBS_GRID, g=g, groups=coh)
                        if p.get("s_grid") is not None else np.nan)
    return row


def calibration(test: pd.DataFrame, p: dict) -> dict:
    """10-y calibration by predicted-risk quintile and overall, plus per cohort.

    Observed risk (primary) is the IPCW estimate with a within-cohort censoring
    KM, which stays unbiased when censoring differs between cohorts; pooled KM
    (with 95 % CI) and Aalen-Johansen (non-CVD death as a competing event) are
    reported alongside.
    """
    risk = p["risk10"]
    T, E, ch = test["T"].to_numpy(), test["E"].to_numpy(), test["coh"].to_numpy()
    g = sm.CensoringKM(T, E, ch)
    comp = np.where(E == 1, 1, np.where(test["died_other"].to_numpy() == 1, 2, 0))
    q = pd.qcut(risk, 5, labels=False, duplicates="drop")
    # IPCW per quintile: weights from the whole-sample stratified G, averaged within the quintile
    died10 = (E == 1) & (T <= H10)
    w = np.zeros(len(T))
    gi = g.at(T[died10], ch[died10], left_limit=True)
    w[died10] = np.where(gi > 0, 1.0 / gi, 0.0)
    rows = []
    for k in np.unique(q):
        m = q == k
        obs_km, lo, hi = sm.km_risk(T[m], E[m], H10)
        aj = AalenJohansenFitter(calculate_variance=False, seed=SEED).fit(T[m], comp[m], event_of_interest=1)
        cif = aj.cumulative_density_
        aj_obs = float(cif.loc[:H10].iloc[-1, 0]) if (cif.index <= H10).any() else 0.0
        rows.append({"quintile": int(k) + 1, "n": int(m.sum()), "events": int(E[m].sum()),
                     "pred": float(risk[m].mean()), "obs_ipcw": float(w[m].mean()),
                     "obs_km": obs_km, "obs_km_ci": [lo, hi], "obs_aj": aj_obs})
    obs = sm.ipcw_risk(T, E, H10, g=g, groups=ch)
    by_cohort = {}
    for c in sorted(set(ch)):
        m = ch == c
        by_cohort[c] = {"n": int(m.sum()), "pred": float(risk[m].mean()), "obs_km": sm.km_risk(T[m], E[m], H10)[0]}
        by_cohort[c]["oe"] = by_cohort[c]["obs_km"] / by_cohort[c]["pred"] if by_cohort[c]["pred"] > 0 else float("nan")
    slope_df = test[["coh", "T", "E"]].assign(lp=p["lp"])
    slope = float(CoxPHFitter(penalizer=0.0).fit(slope_df, "T", "E", strata=["coh"]).params_.iloc[0])
    return {"oe": obs / float(risk.mean()), "obs_ipcw": obs, "mean_pred": float(risk.mean()),
            "oe_km_pooled": sm.km_risk(T, E, H10)[0] / float(risk.mean()),
            "slope_lp_within_cohort": slope, "quintiles": rows, "by_cohort": by_cohort}


def bootstrap(test: pd.DataFrame, preds: dict, null: StratumKMNull, n_boot: int) -> dict:
    """Shared subject resamples for every model; censoring KM re-estimated per replicate."""
    rng = np.random.default_rng(SEED)
    n = len(test)
    T, E, coh = test["T"].to_numpy(), test["E"].to_numpy(), test["coh"].to_numpy()
    null10 = null.predict(coh)
    arr = {m: {k: np.full(n_boot, np.nan) for k in BOOT_KEYS} for m in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Tb, Eb, cb = T[idx], E[idx], coh[idx]
        if Eb.sum() < 5:
            continue
        g = sm.CensoringKM(Tb, Eb, cb)
        for m, p in preds.items():
            pb = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in p.items()}
            r = metric_row(Tb, Eb, cb, pb, g, null10[idx])
            for k in BOOT_KEYS:
                arr[m][k][b] = r[k]
    return arr


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a.size else [np.nan, np.nan]


def paired(arr, point, a, b):
    out = {}
    for k in ("harrell_c", "within_cohort_c", "uno_c_10y", "auc_5y", "auc_10y", "brier_10y"):
        d = arr[a][k] - arr[b][k]
        d = d[np.isfinite(d)]
        out[f"delta_{k}"] = [float(point[a][k] - point[b][k]), *ci(d)]
        n = d.size
        out[f"p_{k}"] = (float(min(1.0, 2 * min(((d <= 0).sum() + 1) / (n + 1), ((d >= 0).sum() + 1) / (n + 1))))
                         if n else float("nan"))  # +1 correction: never reported as exactly 0
    return out


# ============================================================================ train-only diagnostics

def cv_train(train_raw, train_out, coh, ladder, with_xgb, impute_cols, with_frs) -> dict:
    strat = (coh.astype(str) + "_" + train_out["E"].astype(str)).to_numpy()
    names = list(ladder) + ([POOLED, "XGB-M4"] if with_xgb else []) + (["FRS-fixed"] if with_frs else [])
    res = {m: {"harrell_c": [], "within_cohort_c": []} for m in names}
    splits = [s for r in range(CV_REPEATS)
              for s in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(np.zeros(len(strat)), strat)]
    for tr, va in splits:
        pre = Preprocessor(impute_cols).fit(train_raw.iloc[tr], coh.iloc[tr])
        dtr = pre.transform(train_raw.iloc[tr], coh.iloc[tr]).assign(T=train_out["T"].iloc[tr].values,
                                                                     E=train_out["E"].iloc[tr].values)
        dva = pre.transform(train_raw.iloc[va], coh.iloc[va]).assign(T=train_out["T"].iloc[va].values,
                                                                     E=train_out["E"].iloc[va].values)
        preds = predict_all(fit_all(dtr, ladder, with_xgb), dva, with_frs)
        for m in names:
            res[m]["harrell_c"].append(sm.harrell_c(dva["T"], dva["E"], preds[m]["risk10"]))
            res[m]["within_cohort_c"].append(sm.within_stratum_c(dva["T"], dva["E"], preds[m]["lp"], dva["coh"]))
    return {m: {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "folds": [float(x) for x in v]}
                for k, v in d.items()} for m, d in res.items()}


def hr_table(train: pd.DataFrame, terms: list[str], sd: pd.Series) -> pd.DataFrame:
    cph = fit_cox(train, terms, penalizer=0.0)
    s = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    s.columns = ["coef", "hr_per_sd_or_unit", "hr_lo", "hr_hi", "p"]
    s["scale"] = ["per SD" if t in sd.index else "per unit" for t in s.index]
    s["sd_raw"] = [float(sd[t]) if t in sd.index else np.nan for t in s.index]
    try:
        ph = proportional_hazard_test(cph, train[terms + ["coh", "T", "E"]], time_transform="rank")
        s["ph_p"] = ph.summary["p"].reindex(s.index).values
    except Exception:  # noqa: BLE001 - record and continue; PH test can fail on degenerate strata
        s["ph_p"] = np.nan
    return s


# ============================================================================ one analysis

def analyse(label, raw, out, coh, is_test, ladder, out_dir: Path, evaluate_test: bool, n_boot: int,
            with_xgb: bool, write_tables: bool, impute_cols: list[str], with_frs: bool) -> dict:
    """One outcome x cohort definition. `out` has columns T (years), E, died_other.

    Test-split outcomes are read only inside the evaluate_test branch.
    """
    tr, te = ~is_test, is_test
    pre = Preprocessor(impute_cols).fit(raw[tr], coh[tr])
    train = pre.transform(raw[tr], coh[tr]).assign(T=out.loc[tr, "T"].values, E=out.loc[tr, "E"].values)
    res = {"n_train": int(tr.sum()), "events_train": int(out.loc[tr, "E"].sum()), "n_test": int(te.sum()),
           "train_by_cohort": {c: [int(((coh == c) & tr).sum()), int(out.loc[(coh == c) & tr, "E"].sum())]
                               for c in sorted(coh.unique())},
           "test_n_by_cohort": {c: int(((coh == c) & te).sum()) for c in sorted(coh.unique())},
           "train_scaling": {"mean": pre.mean.round(6).to_dict(), "sd": pre.sd.round(6).to_dict()},
           "terms": dict(ladder), "strata_levels": sorted(coh.unique())}
    t0 = time.time()
    res["cv_train"] = cv_train(raw[tr], out[tr], coh[tr], ladder, with_xgb, impute_cols, with_frs)
    res["cv_seconds"] = time.time() - t0
    fitted = fit_all(train, ladder, with_xgb)
    if "XGB-M4" in fitted:
        res["xgb"] = fitted["XGB-M4"][3]
    if write_tables:
        (out_dir / "hr_tables").mkdir(parents=True, exist_ok=True)
        ph = {}
        for m in ladder:
            t = hr_table(train, ladder[m], pre.sd)
            t.to_csv(out_dir / "hr_tables" / f"{label}__{m}.csv")
            ph[m] = [i for i, p in t["ph_p"].items() if np.isfinite(p) and p < 0.05]
        res["ph_violations_train"] = ph
    for m, v in res["cv_train"].items():
        print(f"    [{label}] CV train {m:9s} C {v['harrell_c']['mean']:.4f} ± {v['harrell_c']['sd']:.4f}"
              f"   within-cohort {v['within_cohort_c']['mean']:.4f} ± {v['within_cohort_c']['sd']:.4f}")
    if not evaluate_test:
        return res

    res.update({"events_test": int(out.loc[te, "E"].sum()),
                "events_test_5y": int(((out["E"] == 1) & (out["T"] <= H5) & te).sum()),
                "events_test_10y": int(((out["E"] == 1) & (out["T"] <= H10) & te).sum()),
                "test_events_by_cohort": {c: int(out.loc[(coh == c) & te, "E"].sum()) for c in sorted(coh.unique())}})
    test = pre.transform(raw[te], coh[te]).assign(T=out.loc[te, "T"].values, E=out.loc[te, "E"].values,
                                                  died_other=out.loc[te, "died_other"].values)
    preds = predict_all(fitted, test, with_frs)
    T, E, ch = test["T"].to_numpy(), test["E"].to_numpy(), test["coh"].to_numpy()
    g = sm.CensoringKM(T, E, ch)
    g_marg = sm.CensoringKM(T, E)
    null = StratumKMNull(train)
    null10 = null.predict(ch)
    male = raw.loc[te, "male"].to_numpy()
    age = np.exp(raw.loc[te, "ln_age"].to_numpy())
    models = {}
    for m, p in preds.items():
        r = metric_row(T, E, ch, p, g, null10)
        r["scaled_brier_10y_overall_null"] = 1 - r["brier_10y"] / sm.brier_score(
            T, E, np.full(len(T), null.overall), H10, g=g, groups=ch)
        r["uno_c_10y_marginal_censoring"] = sm.uno_c(T, E, p["risk10"], tau=H10, g=g_marg)
        r["auc_10y_marginal_censoring"] = sm.cumulative_dynamic_auc(T, E, p["risk10"], H10, g=g_marg)
        r["brier_10y_marginal_censoring"] = sm.brier_score(T, E, p["s10"], H10, g=g_marg)
        r["calib_10y"] = calibration(test, p)
        r["c_by_cohort_lp"] = {c: sm.harrell_c(T[ch == c], E[ch == c], p["lp"][ch == c]) for c in sorted(set(ch))}
        r["c_by_sex"] = {s: sm.harrell_c(T[male == v], E[male == v], p["risk10"][male == v])
                         for s, v in (("male", 1), ("female", 0))}
        r["c_by_age70"] = {k: {"n": int(msk.sum()), "events": int(E[msk].sum()),
                               "harrell_c": sm.harrell_c(T[msk], E[msk], p["risk10"][msk]),
                               "within_cohort_c": sm.within_stratum_c(T[msk], E[msk], p["lp"][msk], ch[msk])}
                           for k, msk in (("le70", age <= 70), ("gt70", age > 70))}
        models[m] = r
    t0 = time.time()
    arr = bootstrap(test, preds, null, n_boot)
    res["bootstrap_seconds"] = time.time() - t0
    for m in models:
        models[m].update({f"{k}_ci": ci(arr[m][k]) for k in BOOT_KEYS})
    res["models"] = models
    res["paired"] = {f"{a}-{b}": paired(arr, models, a, b) for a, b in PAIRS if a in arr and b in arr}
    if write_tables:
        np.savez(out_dir / f"bootstrap__{label}.npz", **{f"{m}__{k}": v for m, d in arr.items() for k, v in d.items()})
        pred_df = pd.DataFrame({"nsrrid": raw.index[te], "coh": ch, "T_years": T, "E": E})
        for m, p in preds.items():
            pred_df[f"{m}__lp"] = p["lp"]
            pred_df[f"{m}__risk10"] = p["risk10"]
        pred_df.to_parquet(out_dir / f"test_predictions__{label}.parquet", index=False)
    for m in ("FRS-fixed", "FRS-recal", "M0", "M1", "M2", "M3", "M3-swap", "M4", POOLED, "XGB-M4"):
        if m in models:
            r = models[m]
            print(f"    [{label}] TEST {m:9s} C {r['harrell_c']:.4f} {[round(x, 4) for x in r['harrell_c_ci']]}"
                  f"  within {r['within_cohort_c']:.4f}  Uno {r['uno_c_10y']:.4f}  AUC10 {r['auc_10y']:.4f}"
                  f"  Brier10 {r['brier_10y']:.4f}  O/E {r['calib_10y']['oe']:.2f}")
    for k, v in res["paired"].items():
        print(f"    [{label}] Δ {k:16s} C {v['delta_harrell_c'][0]:+.4f} "
              f"[{v['delta_harrell_c'][1]:+.4f}, {v['delta_harrell_c'][2]:+.4f}] p={v['p_harrell_c']:.3f}"
              f"   within {v['delta_within_cohort_c'][0]:+.4f} p={v['p_within_cohort_c']:.3f}")
    return res


def outcome_frame(sel: pd.DataFrame, name: str, truncate_10y: bool = False) -> pd.DataFrame:
    T = sel[f"T_{name}"].to_numpy(float) / YEAR
    E = sel[f"E_{name}"].to_numpy(int)
    if truncate_10y:  # administrative censoring just past 10 y, so 10-y survivors still count at t = 10
        E = np.where(T > H10, 0, E)
        T = np.minimum(T, H10 + 1.0 / YEAR)
    died_other = (((sel["E_death"] == 1) & (E == 0)).astype(int).to_numpy()
                  if name != "death" else np.zeros(len(E), int))
    return pd.DataFrame({"T": T, "E": E, "died_other": died_other}, index=sel.index)


def split_ids(features_dir: Path) -> tuple[np.ndarray, set]:
    ids = np.array(sorted({int(f.name[6:-8]) for f in features_dir.iterdir()
                           if re.fullmatch(r"shhs1-\d+\.parquet", f.name)}))
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(len(ids)), groups=ids))
    return ids, {int(i) for i in ids[te]}


def sleep_hr(source: Path, ids: np.ndarray, cache: Path) -> pd.Series:
    if cache.exists():
        s = pd.read_parquet(cache)["hr_sleep"]
        if set(s.index) == {int(i) for i in ids}:
            print(f"  sleeping HR from cache ({cache.name})")
            return s
    t0 = time.time()
    s = cc.sleeping_hr_from_features(source, ids)
    s.to_frame().to_parquet(cache)
    print(f"  sleeping HR for {len(ids)} subjects from {source.name} in {time.time() - t0:.0f}s (cached)")
    return s


def run_environment() -> dict:
    def _git(*args):
        try:
            return subprocess.run(["git", "-C", str(CODE_ROOT), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    aim4_status = _git("status", "--porcelain", "--", *AIM4_FILES)  # includes untracked (??)
    return {"git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "aim4_files_uncommitted": aim4_status is None or bool(aim4_status),
            "aim4_sha256": {f: hashlib.sha256((CODE_ROOT / f).read_bytes()).hexdigest() for f in AIM4_FILES},
            "lifelines": lifelines.__version__, "xgboost": xgb.__version__, "sklearn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}


@click.command()
@click.option("--evaluate-test", is_flag=True, help="Score the held-out test set. Use once, after the script is frozen.")
@click.option("--n-boot", type=int, default=1000, show_default=True)
@click.option("--scope", type=click.Choice(["primary", "all"]), default="all", show_default=True,
              help="primary = the two co-primary outcomes; all = + secondary outcomes and sensitivities")
@click.option("--out-root", type=click.Path(path_type=Path), default=OUT_ROOT, show_default=True)
@click.option("--features-dir", type=click.Path(path_type=Path), default=FEATURES_DIR, show_default=True)
@click.option("--csv-dir", type=click.Path(path_type=Path), default=CSV_DIR, show_default=True)
@click.option("--hr-source", type=click.Path(path_type=Path), default=None,
              help="Where to read per-epoch HR for sleeping HR. Default: Code/features-phase1batch-v1.tar if "
                   "present, else the features dir (which must not be iCloud-evicted). Same values either way.")
def main(evaluate_test, n_boot, scope, out_root, features_dir, csv_dir, hr_source) -> None:
    t_start = time.time()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "diagnostics").mkdir(exist_ok=True)
    print(f"=== Aim 4 Cox baseline ({'TRAIN + TEST' if evaluate_test else 'TRAIN ONLY'}) ===")

    outcomes = co.build(csv_dir)
    audit = co.audit(outcomes)
    outcomes_keep = co.build(csv_dir, keep_post_censdate=True)
    _, shhs1 = co.load_tables(csv_dir)
    raw_all = cc.build_covariates(shhs1)

    ids, test_ids = split_ids(features_dir)
    assert len(ids) == 5793 and len(test_ids) == 1159, (len(ids), len(test_ids))
    if AIM2_REFERENCE.exists():
        ref = set(pd.read_parquet(AIM2_REFERENCE, columns=["subject_id"])["subject_id"].astype(int).unique())
        assert test_ids == ref, "test subjects differ from the Aim 2 reference split"
        print("  split matches the Aim 2 test subjects exactly (1,159)")

    src = hr_source or (HR_TAR if HR_TAR.is_file() else features_dir)
    if src.is_dir():
        evicted = [i for i in ids if os.stat(src / f"shhs1-{i}.parquet").st_flags & 0x40000000]  # macOS dataless
        if evicted:
            raise click.UsageError(f"{len(evicted)} feature parquets are iCloud-evicted; pass --hr-source {HR_TAR.name}")
    hr = sleep_hr(src, ids, out_root / "sleep_hr_cache__median_sleep_min240.parquet")
    hr_all = hr.reindex(raw_all.index)
    raw_all["hr_sleep"] = hr_all.where((raw_all["hrqual"] != 1) & hr_all.between(35, 120))

    env = run_environment()
    if evaluate_test and env["aim4_files_uncommitted"]:
        raise click.UsageError("Commit the Aim 4 files before scoring the test set, so the run is tied to a commit.")
    summary: dict = {"audit": audit, "split": {"seed": SEED, "test_size": 0.2, "n_ids": int(len(ids))},
                     "evaluate_test": evaluate_test, "penalizer": PENALIZER, "strata": "coh: 5 levels in CVD analyses (NY has no surveillance), 6 in death_all",
                     "imputation": "IterativeImputer(max_iter=10, seed=42), train-only, cohort dummies as auxiliaries",
                     "scores": "pooled metrics on predicted 10-y risk; within-cohort C on linear predictor",
                     "censoring_weights": "Kaplan-Meier within cohort (marginal-censoring versions also reported)",
                     "env": env, "analyses": {}}

    def run(label, outcome, ladder, definition="P3", with_xgb=False, truncate=False, cohort_filter=None,
            swap_ahi=False, write_tables=False, extra_terms=None, impute_cols=IMPUTE_COLS, table=None):
        tab = outcomes if table is None else table
        if definition:
            sel = co.select(tab, ids, definition)
        else:
            sel = tab[tab.index.isin(ids) & (tab["censdate"] > 0)]
        if cohort_filter is not None:
            sel = sel[cohort_filter(sel)]
        raw = raw_all.reindex(sel.index)
        if swap_ahi:
            raw = raw.assign(l_rdi4p=raw["l_ahi_a0h3a"])
        for name, series in (extra_terms or {}).items():
            raw[name] = series.reindex(sel.index).astype(float)
        out = outcome_frame(sel, outcome, truncate)
        is_test = sel.index.isin(test_ids)
        print(f"\n  --- {label}: n={len(sel)} (train {int((~is_test).sum())}, test {int(is_test.sum())}), "
              f"train events {int(out.loc[~is_test, 'E'].sum())}")
        res = analyse(label, raw, out, sel["coh"], is_test, ladder, out_root, evaluate_test, n_boot, with_xgb,
                      write_tables, impute_cols, with_frs=outcome in FRS_OUTCOMES)
        res.update({"cohort_definition": definition, "outcome": outcome, "truncate_10y": truncate})
        summary["analyses"][label] = res
        (out_root / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    flow = {"parquet_ids": int(len(ids)), "in_cvd_summary": int(outcomes.index.isin(ids).sum())}
    base = outcomes[outcomes.index.isin(ids)]
    flow["with_surveillance"] = int(base["surveillance"].sum())
    flow["P2_free_of_prevalent"] = int((base["surveillance"] & ~base["prev_p2"]).sum())
    flow["P3_free_of_prevalent"] = int((base["surveillance"] & ~base["prev_p3"]).sum())
    sel = co.select(outcomes, ids, "P3")
    flow["P3_followup_gt_0"] = int(len(sel))
    flow["P3_train"] = int((~sel.index.isin(test_ids)).sum())
    flow["P3_test"] = int(sel.index.isin(test_ids).sum())
    flow["P3_by_cohort"] = sel["coh"].value_counts().to_dict()
    (out_root / "cohort_flow.json").write_text(json.dumps(flow, indent=2))
    print("  cohort flow:", {k: v for k, v in flow.items() if k != "P3_by_cohort"})
    (out_root / "feature_list.json").write_text(json.dumps({"ladder": LADDER, "impute_cols": IMPUTE_COLS,
                                                            "unscaled": sorted(UNSCALED)}, indent=2))
    run("primary_cvd", "cvd", LADDER, with_xgb=True, write_tables=True)
    run("coprimary_hard_cvd", "hard_cvd", LADDER, with_xgb=True, write_tables=True)
    if scope == "all":
        small = {k: LADDER[k] for k in SMALL}
        for outc in ("chd", "hf", "death"):
            run(f"secondary_{outc}", outc, small)
        run("S1_nsrr_any_cvd", "any_cvd", small)
        run("S2_P2_cohort", "cvd", small, definition="P2")
        complete = raw_all[IMPUTE_COLS].notna().all(axis=1)
        run("S3_complete_case", "cvd", small, cohort_filter=lambda s: complete.reindex(s.index).fillna(False).to_numpy())
        run("S4_ahi_a0h3a", "cvd", small, swap_ahi=True)
        run("S6_truncate_10y", "cvd", small, truncate=True)
        run("S11_keep_post_censdate_events", "cvd", small, table=outcomes_keep)
        osa_pm = ((shhs1["sa15"] == 1) | (shhs1["pacem15"] == 1)).reindex(outcomes.index).fillna(False)
        run("S12_excl_prior_osa_or_pacemaker", "cvd", small,
            cohort_filter=lambda s: ~osa_pm.reindex(s.index).fillna(False).to_numpy())
        # all-cause death in everyone with follow-up: incl. the no-surveillance cohort and prevalent CVD
        expl = {k: LADDER[k] for k in ("M0", "M2")}  # exploratory: few events (spec 2.2); report with 🚧
        run("exploratory_stroke", "stroke", expl)
        run("exploratory_cvd_death", "cvd_death", expl)
        mort = {k: v + ["prev_cvd_sr"] for k, v in small.items()}
        run("secondary_death_all", "death", mort, definition=None,
            extra_terms={"prev_cvd_sr": cc.self_reported_cvd(shhs1)}, impute_cols=IMPUTE_COLS + ["prev_cvd_sr"])
    summary["not_run"] = {"S5": "unstratified Cox: covered by M4-pooled in the primary ladders",
                          "S7_spec": "+ DBP (Gottlieb model 3)", "S8_spec": "FRS with systbp",
                          "age_non_ph": "ln_age x log(t) / age-band strata sensitivity",
                          "S9": "full-staging studies only", "S10": "lenient diabetes"}
    summary["compute"] = {"wall_seconds": time.time() - t_start, "cpu_only": True,
                          "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out_root / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    if evaluate_test:
        for lab in ("primary_cvd", "coprimary_hard_cvd"):
            plot_calibration(summary["analyses"][lab], lab, out_root / "diagnostics" / f"calibration_{lab}.png")
    print(f"\nsaved -> {out_root}  ({(time.time() - t_start) / 60:.1f} min)")


def plot_calibration(res: dict, label: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 0.6], [0, 0.6], ls=":", color="grey")
    for m in ("FRS-fixed", "M0", "M2", "M4", "XGB-M4"):
        if m not in res.get("models", {}):
            continue
        q = res["models"][m]["calib_10y"]["quintiles"]
        ax.plot([r["pred"] for r in q], [r["obs_ipcw"] for r in q], marker="o", label=m)
    ax.set_xlabel("predicted 10-year risk")
    ax.set_ylabel("observed 10-year risk (IPCW, within-cohort censoring)")
    ax.set_title(f"Calibration by predicted-risk quintile, test set\n{label}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
