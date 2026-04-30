"""Aim 2 Phase 1 — cumulative feature additions: sample entropy + PSD + CPC + ECG bands.

Hypothesis
----------
Each Phase 1 feature group adds genuine, non-tautological signal on top of the
v6 contextual baseline. We measure each group's marginal contribution by
running 4 cumulative configs on the same fixed-LightGBM hyperparameters.

Feature lineage
---------------
The audit-v6 → phase1batch-v1 schema bump added four new feature groups:

  Exp 1: SpO₂ + RR sample entropy (Richman & Moorman 2000) — complexity
  Exp 2: SpO₂ Welch PSD apnoea-band power (0.01–0.067 Hz) — frequency content
  Exp 3: Cardiopulmonary coupling proxy via R-peak amplitude variability (CV/IQR)
  Exp 4: Multi-scale ECG band power (5 bands, Daubechies-equivalent)

Each feature group's base cols also have ROLLING_BASE contextual derivatives
(_lag1/_lead1/_roll5_*/_roll11_*) — those propagate automatically.

Configs (cumulative)
--------------------
  exp1: physio + sample entropy
  exp2: exp1 + SpO₂ PSD
  exp3: exp2 + CPC-proxy
  exp4: exp3 + ECG band power (= full phase1batch-v1 feature set)

Same FIXED_PARAMS as v8.5-tax / phase1_exp5 / phase1_exp6 for cross-experiment
comparability. Same seed=42 outer split as all v6/v8/v8.5 work — paired DeLong
or subject-paired bootstrap can chain in afterwards.

Outputs (per config)
--------------------
  models/aim2_phase1_batch/expN/
    metrics.json
    metrics_extended.json
    test_predictions.parquet
    bootstrap_aucs_subject.npy
    bootstrap_auprs_subject.npy
    feature_list.json

Plus models/aim2_phase1_batch/summary.json comparing all 4.

Usage (Colab high-RAM):
  %cd /content/Code
  !python scripts/fit_aim2_phase1_batch.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import click
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.epochs import wake_mask  # noqa: E402
from thesis_pipeline.extended_metrics import write_extended_metrics  # noqa: E402
from thesis_pipeline.feature_groups import feature_subset  # noqa: E402

FEATURES_DIR = CODE_ROOT / "features"
OUT_ROOT = CODE_ROOT / "models" / "aim2_phase1_batch"

NON_FEATURE = {
    "subject_id", "cohort", "epoch_idx", "epoch_start_sec",
    "sleep_stage", "apnoea_label", "features_version",
}

# Phase 1 base feature columns added per experiment (cumulative).
# Their _lag1 / _lead1 / _roll5_mean / _roll5_std / _roll11_mean / _roll11_std
# contextual derivatives are added automatically.
PHASE1_NEW_BASE_COLS: dict[str, list[str]] = {
    "exp1": ["spo2_sampen", "hrv_sampen"],
    "exp2": ["spo2_psd_apnea_band", "spo2_psd_total", "spo2_psd_apnea_ratio"],
    "exp3": ["cpc_amp_cv", "cpc_amp_iqr"],
    "exp4": [
        "ecg_band_low_power", "ecg_band_mid_low_power", "ecg_band_mid_power",
        "ecg_band_mid_high_power", "ecg_band_high_power", "ecg_band_total_power",
        "ecg_band_low_rel", "ecg_band_mid_low_rel", "ecg_band_mid_rel",
        "ecg_band_mid_high_rel", "ecg_band_high_rel",
    ],
}
CONTEXT_SUFFIXES = ("", "_lag1", "_lead1", "_roll5_mean", "_roll5_std", "_roll11_mean", "_roll11_std")
EXP_ORDER = ("exp1", "exp2", "exp3", "exp4")

# Same FIXED LightGBM params as v8.5-tax / phase1_exp5 / phase1_exp6.
# Not tuned per config — this ablation measures feature-group contribution,
# not HP sensitivity. Direct comparability against earlier Phase 1 work.
FIXED_PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 48,
    "min_child_samples": 100,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "objective": "binary",
    "metric": "auc",
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
FIXED_N_ESTIMATORS = 800
FIXED_EARLY_STOPPING = 50


def _expand_with_context(base_cols: list[str]) -> set[str]:
    """Return base cols + all contextual derivative names."""
    return {c + sfx for c in base_cols for sfx in CONTEXT_SUFFIXES}


def features_for_exp(all_cols: list[str], exp_id: str) -> list[str]:
    """Cumulative feature subset: keep cols not added in EXPs LATER than exp_id.

    exp1 → drops cols added in exp2, exp3, exp4
    exp4 → drops nothing (uses everything)
    """
    if exp_id not in EXP_ORDER:
        raise ValueError(f"unknown exp_id: {exp_id!r}")
    later = EXP_ORDER[EXP_ORDER.index(exp_id) + 1:]
    drop_set: set[str] = set()
    for e in later:
        drop_set.update(_expand_with_context(PHASE1_NEW_BASE_COLS[e]))
    return [c for c in all_cols if c not in drop_set]


def load_cohort(features_dir: Path, version: str) -> pd.DataFrame:
    """Load all per-subject parquets, filter to features_version + sleep epochs.

    AppleDouble shadows (._*) explicitly excluded.
    """
    files = [
        f for f in sorted(features_dir.glob("*.parquet"))
        if f.name != "subject_metadata.parquet" and not f.name.startswith("._")
    ]
    if not files:
        raise FileNotFoundError(f"No per-subject parquet files in {features_dir}")
    print(f"  loading {len(files)} parquet files...")

    KEEP_META = {"subject_id", "epoch_idx", "apnoea_label", "features_version", "sleep_stage"}
    DROP_META = {"cohort", "epoch_start_sec"}

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df = df.drop(columns=[c for c in DROP_META if c in df.columns], errors="ignore")
        for c in df.columns:
            if c in KEEP_META:
                continue
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        if "subject_id" in df.columns:
            df["subject_id"] = df["subject_id"].astype(np.int32)
        if "apnoea_label" in df.columns:
            df["apnoea_label"] = df["apnoea_label"].astype(np.int8)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True, copy=False)
    n_before = df["subject_id"].nunique()
    df = df[df["features_version"] == version].reset_index(drop=True)
    n_after = df["subject_id"].nunique()
    print(f"  features_version filter: {version!r} → {n_after}/{n_before} subjects")

    n_epochs_before = len(df)
    mask = wake_mask(df.sleep_stage.values)
    df = df[~mask].reset_index(drop=True)
    print(f"  sleep-only filter: kept {len(df):,}/{n_epochs_before:,} epochs ({100*len(df)/n_epochs_before:.1f}%)")
    return df


def subject_bootstrap(
    df_test: pd.DataFrame, probs: np.ndarray, n_resamples: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Subject-level paired bootstrap CI on AUC + AUC-PR."""
    rng = np.random.default_rng(seed)
    sids = df_test["subject_id"].values
    y = df_test["apnoea_label"].values.astype(np.int8)

    subjects = np.unique(sids)
    n_subj = len(subjects)
    by_subject = {int(s): np.where(sids == s)[0] for s in subjects}
    subjects_int = np.asarray(list(by_subject.keys()), dtype=np.int64)

    aucs = np.full(n_resamples, np.nan)
    auprs = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        sampled = rng.choice(subjects_int, size=n_subj, replace=True)
        idx = np.concatenate([by_subject[int(s)] for s in sampled])
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        pi = probs[idx]
        aucs[i] = roc_auc_score(yi, pi)
        auprs[i] = average_precision_score(yi, pi)

    valid_auc = aucs[np.isfinite(aucs)]
    valid_aupr = auprs[np.isfinite(auprs)]
    ci = {
        "auc_ci_low": float(np.percentile(valid_auc, 2.5)),
        "auc_ci_high": float(np.percentile(valid_auc, 97.5)),
        "aupr_ci_low": float(np.percentile(valid_aupr, 2.5)),
        "aupr_ci_high": float(np.percentile(valid_aupr, 97.5)),
        "n_bootstrap_valid": int(len(valid_auc)),
    }
    return aucs, auprs, ci


def fit_one_config(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    tv_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: Path,
) -> dict:
    """Fit LightGBM with FIXED_PARAMS on the given feature subset, evaluate on test."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*78}\n=== Config: {name}  ({len(feature_cols)} features)\n{'='*78}")

    X = df[feature_cols].values
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    tr_inner_rel, va_inner_rel = next(inner.split(X[tv_idx], y[tv_idx], groups[tv_idx]))
    tr = tv_idx[tr_inner_rel]
    va = tv_idx[va_inner_rel]

    pos = float(np.sum(y[tr])); neg = float(len(tr) - pos)
    spw = neg / max(pos, 1.0)

    print(f"  inner train: {len(tr):,} epochs / {len(np.unique(groups[tr]))} subj  "
          f"({y[tr].mean()*100:.1f}% positive)")
    print(f"  inner val:   {len(va):,} epochs / {len(np.unique(groups[va]))} subj")
    print(f"  test:        {len(test_idx):,} epochs / {len(np.unique(groups[test_idx]))} subj")
    print(f"  scale_pos_weight: {spw:.3f}")

    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=FIXED_N_ESTIMATORS,
        scale_pos_weight=spw,
        **FIXED_PARAMS,
    )
    model.fit(
        X[tr], y[tr],
        eval_set=[(X[va], y[va])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(FIXED_EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
    )
    fit_time = time.time() - t0
    best_iter = int(model.best_iteration_ or FIXED_N_ESTIMATORS)
    print(f"  fit in {fit_time:.0f}s  (best_iter={best_iter})")

    probs = model.predict_proba(X[test_idx])[:, 1]
    test_auc = float(roc_auc_score(y[test_idx], probs))
    test_aupr = float(average_precision_score(y[test_idx], probs))

    va_probs = model.predict_proba(X[va])[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    va_f1s = [f1_score(y[va], (va_probs > t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(va_f1s))])
    preds = (probs > best_thresh).astype(int)
    test_f1 = float(f1_score(y[test_idx], preds, zero_division=0))
    test_p = float(precision_score(y[test_idx], preds, zero_division=0))
    test_r = float(recall_score(y[test_idx], preds, zero_division=0))

    print(f"  Test AUC: {test_auc:.4f}   AUC-PR: {test_aupr:.4f}")
    print(f"  F1@{best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    df_test = df.iloc[test_idx][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True)
    print(f"  computing subject-level bootstrap CI (1000 resamples)...")
    t0 = time.time()
    aucs, auprs, ci = subject_bootstrap(df_test, probs, n_resamples=1000, seed=42)
    print(f"  bootstrap in {time.time()-t0:.0f}s")
    print(f"  AUC subject CI:    [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
    print(f"  AUC-PR subject CI: [{ci['aupr_ci_low']:.4f}, {ci['aupr_ci_high']:.4f}]")

    test_pred_df = pd.DataFrame({
        "subject_id": df.iloc[test_idx]["subject_id"].values,
        "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
        "apnoea_label": y[test_idx],
        "pred_prob": probs,
        "pred_label": preds,
    })
    test_pred_df.to_parquet(out_dir / "test_predictions.parquet", index=False)
    np.save(out_dir / "bootstrap_aucs_subject.npy", aucs)
    np.save(out_dir / "bootstrap_auprs_subject.npy", auprs)
    (out_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2))

    try:
        sm_path = FEATURES_DIR / "subject_metadata.parquet"
        sm = pd.read_parquet(sm_path) if sm_path.exists() else None
        write_extended_metrics(out_dir, test_pred_df, subject_metadata=sm)
    except Exception as _e:
        print(f"  ! extended metrics failed: {_e}")

    metrics = {
        "name": name,
        "model": "lightgbm",
        "n_features": len(feature_cols),
        "n_test_subjects": int(len(np.unique(groups[test_idx]))),
        "n_test_epochs": int(len(test_idx)),
        "best_iter": best_iter,
        "fixed_params": FIXED_PARAMS,
        "scale_pos_weight": spw,
        "best_threshold": best_thresh,
        "test_auc_roc": test_auc,
        "test_auc_pr": test_aupr,
        "test_auc_ci_low": ci["auc_ci_low"],
        "test_auc_ci_high": ci["auc_ci_high"],
        "test_aupr_ci_low": ci["aupr_ci_low"],
        "test_aupr_ci_high": ci["aupr_ci_high"],
        "test_f1_tuned": test_f1,
        "test_precision_tuned": test_p,
        "test_recall_tuned": test_r,
        "fit_seconds": fit_time,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  saved → {out_dir}")
    return metrics


@click.command()
@click.option("--features-version", default="2026-05-01-phase1batch-v1", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True,
              help="MUST be 42 to keep test split identical to v6/v8/v8.5/Phase 1 Exp5/6")
@click.option("--exp", default="all", show_default=True,
              type=click.Choice(["all", "exp1", "exp2", "exp3", "exp4"]),
              help="Run a single config or 'all' for the cumulative ablation")
@click.option("--feature-mode", default="full", show_default=True,
              type=click.Choice(["full", "physio_only", "aasm_only"]),
              help="Apply feature_groups taxonomy filter BEFORE cumulative logic. "
                   "'physio_only' drops AASM-rule features (the decisive test of whether Phase 1 "
                   "additions add signal independent of rule-recapitulation). "
                   "'full' is the unfiltered run (existing exp1-4 dirs).")
def main(features_version: str, seed: int, exp: str, feature_mode: str) -> None:
    # When mode != full, nest under a mode subdir to avoid clobbering the
    # existing aim2_phase1_batch/{exp1..exp4} (which are full-mode results).
    out_root = OUT_ROOT if feature_mode == "full" else (OUT_ROOT / feature_mode)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Aim 2 Phase 1 cumulative feature ablation (Exp 1-4) ===")
    print(f"  Features dir: {FEATURES_DIR}")
    print(f"  Output dir:   {out_root}")
    print(f"  Features version: {features_version}")
    print(f"  Seed: {seed}")
    print(f"  Feature mode: {feature_mode}")
    print(f"  Configs: {exp}\n")

    df = load_cohort(FEATURES_DIR, version=features_version)
    feature_cols_all = [c for c in df.columns if c not in NON_FEATURE]
    print(f"  Total features (excluding meta): {len(feature_cols_all)}")
    if feature_mode != "full":
        before = len(feature_cols_all)
        feature_cols_all = feature_subset(feature_cols_all, mode=feature_mode)
        print(f"  Taxonomy filter (mode={feature_mode}): kept {len(feature_cols_all)}/{before} features")
    print()

    # Outer split — IDENTICAL to all prior v6/v8/v8.5/Phase 1 work
    X_dummy = np.empty(len(df))
    y = df["apnoea_label"].values
    groups_arr = df["subject_id"].values
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, test_idx = next(outer.split(X_dummy, y, groups_arr))
    print(f"  Outer split (seed={seed}):")
    print(f"    TV pool: {len(tv_idx):,} epochs / {len(np.unique(groups_arr[tv_idx]))} subjects")
    print(f"    Test:    {len(test_idx):,} epochs / {len(np.unique(groups_arr[test_idx]))} subjects")

    configs_to_run = EXP_ORDER if exp == "all" else (exp,)
    summary: dict = {
        "cohort": {
            "features_version": features_version,
            "n_subjects_total": int(df["subject_id"].nunique()),
            "n_epochs_total": int(len(df)),
            "n_test_subjects": int(len(np.unique(groups_arr[test_idx]))),
            "n_test_epochs": int(len(test_idx)),
        },
        "feature_mode": feature_mode,
        "phase1_new_cols": PHASE1_NEW_BASE_COLS,
        "configs": {},
    }

    for name in configs_to_run:
        feature_cols = features_for_exp(feature_cols_all, exp_id=name)
        out_dir = out_root / name
        result = fit_one_config(name, df, feature_cols, tv_idx, test_idx, out_dir)
        summary["configs"][name] = result

    print(f"\n\n{'='*78}\n=== COMPACT COMPARISON ({feature_mode}) ===\n{'='*78}\n")
    print(f"{'Config':6s}  {'Features':>9s}  {'Test AUC':>9s}  {'Subject 95% CI':>20s}  {'AUC-PR':>8s}  {'F1':>7s}")
    for name, r in summary["configs"].items():
        ci = f"[{r['test_auc_ci_low']:.4f}, {r['test_auc_ci_high']:.4f}]"
        print(f"{name:6s}  {r['n_features']:>9d}  {r['test_auc_roc']:>9.4f}  {ci:>20s}  {r['test_auc_pr']:>8.4f}  {r['test_f1_tuned']:>7.4f}")

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
