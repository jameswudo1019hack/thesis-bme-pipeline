"""Aim 2 — Lit-2019 literature-comparable baseline.

Approximates the median 2018-2020 SHHS apnoea-detection ML paper:
~20 base features (SpO2 stats, ODI3, HRV time+freq, EEG band power, airflow),
no contextual lag/lead/rolling derivatives, no hypoxic_burden, no audit-v3
trailing-baseline tricks, no position channels.

Why
---
v6/v8/v8.5 LightGBM on our full audit-v6 schema reaches AUC 0.85-0.96, well
above the published 0.85-0.90 band. Two possible explanations:
  (a) our protocol is too permissive (over-optimistic CV, leakage, etc.) → BAD
  (b) our extra feature engineering (audit-v6 fixes + Sprint 1 features +
      contextual rolling) genuinely lifts AUC above literature → GOOD

A Lit-2019-style replication arbitrates: if our cohort + protocol gives AUC
~0.85-0.90 on a literature-comparable feature subset, our protocol is fine
and the v6/v8/v8.5 lift is real engineering. If it gives AUC 0.95 even with
literature features, there's a methodological problem we missed.

Protocol
--------
Same as the v8.5 taxonomy ablation:
  - sleep-only filter (audit-v6 wake_mask)
  - GroupShuffleSplit(test_size=0.2, random_state=42) — IDENTICAL test set
    to v6 / v8 / v8.5 / 8.5-tax for paired comparisons
  - Inner train/val for early stopping (5% of TV pool)
  - LightGBM with FIXED sensible HP (no Optuna — same hyperparams as
    8.5-tax for fair comparison)
  - Subject-level bootstrap CI on AUC + AUC-PR
  - Val-fold-tuned threshold for F1

Output: Code/models/aim2_lit_2019/{metrics.json, test_predictions.parquet,
        bootstrap_aucs_subject.npy, bootstrap_auprs_subject.npy,
        feature_list.json}

Usage (Colab high-RAM):
  %cd /content/Code
  !python scripts/fit_aim2_lit_replication.py
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
from thesis_pipeline.literature_baselines import (  # noqa: E402
    LIT_2019_BASE_FEATURES,
    lit_2019_missing,
    lit_2019_subset,
)

FEATURES_DIR = CODE_ROOT / "features"
OUT_DIR = CODE_ROOT / "models" / "aim2_lit_2019"
NON_FEATURE = {
    "subject_id", "cohort", "epoch_idx", "epoch_start_sec",
    "sleep_stage", "apnoea_label", "features_version",
}

# Same fixed HP as 8.5-tax for direct comparability
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


def load_cohort(features_dir: Path, version: str = "2026-04-26-audit-v6") -> pd.DataFrame:
    files = [f for f in sorted(features_dir.glob("*.parquet"))
             if f.name != "subject_metadata.parquet"]
    if not files:
        raise FileNotFoundError(f"No per-subject parquet files in {features_dir}")

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


def subject_bootstrap(df_test, probs, n_resamples=1000, seed=42):
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
    return aucs, auprs, {
        "auc_ci_low": float(np.percentile(valid_auc, 2.5)),
        "auc_ci_high": float(np.percentile(valid_auc, 97.5)),
        "aupr_ci_low": float(np.percentile(valid_aupr, 2.5)),
        "aupr_ci_high": float(np.percentile(valid_aupr, 97.5)),
        "n_bootstrap_valid": int(len(valid_auc)),
    }


@click.command()
@click.option("--features-version", default="2026-04-26-audit-v6", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def main(features_version: str, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Aim 2 — Lit-2019-style baseline (literature replication) ===")
    print(f"  Features dir:     {FEATURES_DIR}")
    print(f"  Output dir:       {OUT_DIR}")
    print(f"  Features version: {features_version}")
    print(f"  Seed:             {seed}\n")

    df = load_cohort(FEATURES_DIR, version=features_version)
    all_features = [c for c in df.columns if c not in NON_FEATURE]
    lit_features = lit_2019_subset(all_features)
    missing = lit_2019_missing(all_features)

    print(f"\n  Total features in cohort:    {len(all_features)}")
    print(f"  Lit-2019 features (defined): {len(LIT_2019_BASE_FEATURES)}")
    print(f"  Lit-2019 features (found):   {len(lit_features)}")
    if missing:
        print(f"\n  ⚠ Missing from cohort schema (will skip):")
        for f in missing:
            print(f"      {f}")

    print(f"\n  Using these {len(lit_features)} features:")
    for f in lit_features:
        print(f"      {f}")

    if len(lit_features) < 10:
        raise SystemExit(
            f"\n  ! Only {len(lit_features)} of the {len(LIT_2019_BASE_FEATURES)} Lit-2019 features "
            f"are present in the schema. Replication would not be meaningful. Aborting."
        )

    X = df[lit_features].values
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    # IDENTICAL outer split to v6/v8/v8.5/8.5-tax
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, test_idx = next(outer.split(X, y, groups))
    print(f"\n  Outer split (seed={seed}):")
    print(f"    TV pool: {len(tv_idx):,} epochs / {len(np.unique(groups[tv_idx]))} subj")
    print(f"    Test:    {len(test_idx):,} epochs / {len(np.unique(groups[test_idx]))} subj")

    # Inner val carve for early stopping
    inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=seed)
    tr_inner_rel, va_inner_rel = next(inner.split(X[tv_idx], y[tv_idx], groups[tv_idx]))
    tr = tv_idx[tr_inner_rel]
    va = tv_idx[va_inner_rel]

    pos = float(np.sum(y[tr])); neg = float(len(tr) - pos)
    spw = neg / max(pos, 1.0)
    print(f"    inner train: {len(tr):,} / val: {len(va):,}")
    print(f"    scale_pos_weight: {spw:.3f}")

    print(f"\n▶ Fitting LightGBM (fixed HP, n_est={FIXED_N_ESTIMATORS}, early_stopping={FIXED_EARLY_STOPPING})...")
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

    # Val-tuned threshold
    va_probs = model.predict_proba(X[va])[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    va_f1s = [f1_score(y[va], (va_probs > t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(va_f1s))])
    preds = (probs > best_thresh).astype(int)
    test_f1 = float(f1_score(y[test_idx], preds, zero_division=0))
    test_p = float(precision_score(y[test_idx], preds, zero_division=0))
    test_r = float(recall_score(y[test_idx], preds, zero_division=0))

    print(f"\n=== Held-out test set ===")
    print(f"  Test AUC: {test_auc:.4f}   AUC-PR: {test_aupr:.4f}")
    print(f"  F1@{best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    print(f"\n▶ Subject-level bootstrap CI (1000 resamples)...")
    df_test = df.iloc[test_idx][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True)
    t0 = time.time()
    aucs, auprs, ci = subject_bootstrap(df_test, probs)
    print(f"  bootstrap in {time.time()-t0:.0f}s")
    print(f"  AUC subject CI:    [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
    print(f"  AUC-PR subject CI: [{ci['aupr_ci_low']:.4f}, {ci['aupr_ci_high']:.4f}]")

    # Persist
    test_pred_df = pd.DataFrame({
        "subject_id": df.iloc[test_idx]["subject_id"].values,
        "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
        "apnoea_label": y[test_idx],
        "pred_prob": probs,
        "pred_label": preds,
    })
    test_pred_df.to_parquet(OUT_DIR / "test_predictions.parquet", index=False)
    np.save(OUT_DIR / "bootstrap_aucs_subject.npy", aucs)
    np.save(OUT_DIR / "bootstrap_auprs_subject.npy", auprs)
    (OUT_DIR / "feature_list.json").write_text(json.dumps({
        "defined": list(LIT_2019_BASE_FEATURES),
        "used": lit_features,
        "missing_from_schema": missing,
    }, indent=2))

    # Extended metrics (Aim 2 standard set)
    try:
        sm_path = FEATURES_DIR / "subject_metadata.parquet"
        sm = pd.read_parquet(sm_path) if sm_path.exists() else None
        write_extended_metrics(OUT_DIR, test_pred_df, subject_metadata=sm)
    except Exception as _e:
        print(f"  ! extended metrics failed: {_e}")

    metrics = {
        "name": "lit_2019",
        "description": "Literature-comparable baseline approximating the median 2018-2020 SHHS apnoea-detection ML paper feature set",
        "n_features": len(lit_features),
        "n_features_defined": len(LIT_2019_BASE_FEATURES),
        "n_features_missing": len(missing),
        "features_version": features_version,
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
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {OUT_DIR}")

    print(f"\n=== INTERPRETATION ===")
    print(f"  Lit-2019 baseline: Test AUC = {test_auc:.4f} [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
    print(f"  Published 2018-2020 SHHS papers: 0.85-0.90 (typical range)")
    print(f"  Our 8.5-tax (full): 0.9591 [0.9553, 0.9629]")
    print(f"  Our 8.5-tax (physio_only): 0.6737 [0.6626, 0.6859]")
    print(f"")
    print(f"  If Lit-2019 lands in [0.85, 0.90]: protocol is sound; v6/v8/v8.5 lift is real engineering.")
    print(f"  If Lit-2019 lands at 0.95+: methodological problem we missed; investigate.")
    print(f"  If Lit-2019 lands < 0.80: cohort harder than literature for some reason; investigate.")


if __name__ == "__main__":
    main()
