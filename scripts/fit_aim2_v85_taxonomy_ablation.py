"""Aim 2 v8.5 — feature-taxonomy ablation: full / physiological-only / AASM-rule-only.

Motivation: the headline `0.97` v8.5 LightGBM result on the audit-v6 schema is
mostly carried by ``hypoxic_burden_epoch*`` features, which are continuous
re-implementations of the AASM hypopnoea scoring rule (≥3% SpO₂ desat). The
result is real but near-tautological — the model is "predicting" labels that
are defined by SpO₂ desats using SpO₂-desat features.

This script runs three configurations on the same outer split (seed=42) and
the same (sleep-only) cohort:

    full         — all features (headline w/ caveat)
    physio_only  — drop AASM-rule features (defensible novel SOTA)
    aasm_only    — only AASM-rule features (sanity: rule predicts itself)

Hyperparameters are FIXED at sensible defaults (close to v6 best params) — we
are not running Optuna here. The point is to characterise the feature-class
contribution, not to tune. ~10 min per config on Colab high-RAM CPU.

For each config we save:
    metrics.json
    test_predictions.parquet  (subject_id, epoch_idx, apnoea_label, pred_prob, pred_label)
    bootstrap_aucs_subject.npy   1000-resample subject-level bootstrap on AUC
    bootstrap_auprs_subject.npy  same for AUC-PR
    feature_list.json   (which features were used)

Plus a top-level summary report:
    Code/models/aim2_v85_taxonomy/summary.json

Usage (Colab):
    %cd /content/Code
    !python scripts/fit_aim2_v85_taxonomy_ablation.py
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

from thesis_pipeline.epochs import sleep_mask  # noqa: E402
from thesis_pipeline.extended_metrics import write_extended_metrics  # noqa: E402
from thesis_pipeline.feature_groups import feature_subset, split_features  # noqa: E402

FEATURES_DIR = CODE_ROOT / "features"
OUT_ROOT = CODE_ROOT / "models" / "aim2_v85_taxonomy"
NON_FEATURE = {
    "subject_id", "cohort", "epoch_idx", "epoch_start_sec",
    "sleep_stage", "apnoea_label", "features_version",
}

# Fixed sensible LightGBM hyperparameters — close to v6 best.
# NOT tuned per config; this ablation is about feature-class contribution.
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
    """Load all per-subject parquets, filter to features_version + sleep epochs."""
    files = [
        f for f in sorted(features_dir.glob("*.parquet"))
        if f.name != "subject_metadata.parquet"
    ]
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

    # Sleep-only filter via sleep_mask (True for N1/N2/N3/REM, drop the rest)
    n_epochs_before = len(df)
    mask = sleep_mask(df.sleep_stage.values)
    df = df[mask].reset_index(drop=True)
    assert df["sleep_stage"].isin(["N1", "N2", "N3", "REM"]).all(), (
        f"sleep filter failed: found stages {sorted(df['sleep_stage'].unique())} after filter"
    )
    print(f"  sleep-only filter: kept {len(df):,}/{n_epochs_before:,} epochs ({100*len(df)/n_epochs_before:.1f}%)")
    return df


def subject_bootstrap(
    df_test: pd.DataFrame, probs: np.ndarray, n_resamples: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Subject-level paired bootstrap CI on AUC + AUC-PR.

    Returns (auc_bootstrap_array, aupr_bootstrap_array, ci_dict).
    """
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
    """Fit a LightGBM with FIXED_PARAMS on the given feature subset, evaluate on test.

    Saves test_predictions, bootstrap arrays, metrics.json, feature_list.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*78}\n=== Config: {name}  ({len(feature_cols)} features)\n{'='*78}")

    if len(feature_cols) == 0:
        print(f"  ! no features in this config; skipping")
        return {"name": name, "n_features": 0, "skipped": True}

    X = df[feature_cols].values
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    # Internal val carve from TV pool — for early stopping (5% of TV by subject)
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

    # Tuned-threshold F1 — use the inner val for tuning (proper holdout, not TV pool)
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

    # Subject-level bootstrap CI
    df_test = df.iloc[test_idx][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True)
    print(f"  computing subject-level bootstrap CI (1000 resamples)...")
    t0 = time.time()
    aucs, auprs, ci = subject_bootstrap(df_test, probs, n_resamples=1000, seed=42)
    print(f"  bootstrap in {time.time()-t0:.0f}s")
    print(f"  AUC subject CI:  [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
    print(f"  AUC-PR subject CI: [{ci['aupr_ci_low']:.4f}, {ci['aupr_ci_high']:.4f}]")

    # Save
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

    # Extended metrics (Aim 2 standard set — calibration, clinical AHI, severity κ)
    try:
        sm_path = FEATURES_DIR / "subject_metadata.parquet"
        sm = pd.read_parquet(sm_path) if sm_path.exists() else None
        write_extended_metrics(out_dir, test_pred_df, subject_metadata=sm)
    except Exception as _e:
        print(f"  ! extended metrics failed: {_e}")

    metrics = {
        "name": name,
        "filter_used": "sleep-only",
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
        "feature_taxonomy_module": "thesis_pipeline.feature_groups",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  saved → {out_dir}")
    return metrics


@click.command()
@click.option("--features-version", default="2026-04-26-audit-v6", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True,
              help="MUST be 42 to keep test split identical to v6 / v8 / v8.5 for paired tests")
def main(features_version: str, seed: int) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Aim 2 v8.5 feature-taxonomy ablation ===")
    print(f"  Features dir: {FEATURES_DIR}")
    print(f"  Output dir:   {OUT_ROOT}")
    print(f"  Features version: {features_version}")
    print(f"  Seed: {seed}\n")

    df = load_cohort(FEATURES_DIR, version=features_version)
    feature_cols_all = [c for c in df.columns if c not in NON_FEATURE]
    print(f"  Total features (excluding meta): {len(feature_cols_all)}\n")

    # Print taxonomy summary
    groups = split_features(feature_cols_all)
    print(f"  Feature taxonomy:")
    print(f"    AASM-rule features:       {len(groups['aasm'])}")
    for c in groups["aasm"][:10]:
        print(f"      {c}")
    if len(groups["aasm"]) > 10:
        print(f"      ... +{len(groups['aasm'])-10} more")
    print(f"    Physiological features:   {len(groups['physio'])}")
    print(f"    Unknown (default→physio): {len(groups['unknown'])}")
    if groups["unknown"]:
        for c in groups["unknown"][:10]:
            print(f"      {c}")
        if len(groups["unknown"]) > 10:
            print(f"      ... +{len(groups['unknown'])-10} more")

    # Outer split — IDENTICAL to v6 / v8 / v8.5 (seed=42 patient-level)
    X_dummy = np.empty(len(df))
    y = df["apnoea_label"].values
    groups_arr = df["subject_id"].values
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, test_idx = next(outer.split(X_dummy, y, groups_arr))
    print(f"\n  Outer split (seed={seed}):")
    print(f"    TV pool: {len(tv_idx):,} epochs / {len(np.unique(groups_arr[tv_idx]))} subjects")
    print(f"    Test:    {len(test_idx):,} epochs / {len(np.unique(groups_arr[test_idx]))} subjects")

    # Run all three configs
    summary: dict = {
        "cohort": {
            "features_version": features_version,
            "n_subjects_total": int(df["subject_id"].nunique()),
            "n_epochs_total": int(len(df)),
            "n_test_subjects": int(len(np.unique(groups_arr[test_idx]))),
            "n_test_epochs": int(len(test_idx)),
            "feature_taxonomy": {
                "n_aasm": len(groups["aasm"]),
                "n_physio": len(groups["physio"]),
                "n_unknown": len(groups["unknown"]),
                "aasm_features": groups["aasm"],
                "unknown_features": groups["unknown"],
            },
        },
        "configs": {},
    }

    for name in ("full", "physio_only", "aasm_only"):
        feature_cols = feature_subset(feature_cols_all, mode=name)
        out_dir = OUT_ROOT / name
        result = fit_one_config(name, df, feature_cols, tv_idx, test_idx, out_dir)
        summary["configs"][name] = result

    # Final compact comparison
    print(f"\n\n{'='*78}\n=== COMPACT COMPARISON ===\n{'='*78}\n")
    print(f"{'Config':14s}  {'Features':>9s}  {'Test AUC':>9s}  {'Subject 95% CI':>20s}  {'AUC-PR':>8s}  {'F1':>7s}")
    for name, r in summary["configs"].items():
        if r.get("skipped"):
            continue
        ci = f"[{r['test_auc_ci_low']:.4f}, {r['test_auc_ci_high']:.4f}]"
        print(f"{name:14s}  {r['n_features']:>9d}  {r['test_auc_roc']:>9.4f}  {ci:>20s}  {r['test_auc_pr']:>8.4f}  {r['test_f1_tuned']:>7.4f}")

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
