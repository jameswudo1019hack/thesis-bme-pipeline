"""Aim 2 v8 model-family ladder under FIXED LightGBM-style HP, sleep-only.

Replaces the original v8 series (fit_aim2_v8_ablation.py, fit_aim2_v8_cuml.py,
fit_aim2_v8_ensemble.py) which used Optuna and had no sleep filter. This
canonical-protocol version uses the same FIXED_PARAMS as v8.5-tax / Phase 1
batch / v6 past-only for direct apples-to-apples comparison.

Models:
  lightgbm   — same FIXED_PARAMS as 8.5-tax
  xgboost    — same regularisation budget, lr=0.1, max_depth=6, n_est=800
  catboost   — same budget
  rf         — sklearn RandomForestClassifier with reasonable defaults
  logreg     — sklearn LogisticRegression, L2 penalty, balanced class weight
  ensemble   — simple-mean of {lightgbm, xgboost, catboost} predictions

Output: Code/models/aim2_v8_fixedhp_sleep/<model>/{metrics.json,
        test_predictions.parquet, bootstrap_aucs_subject.npy, ...}

Usage (Colab high-RAM):
  %cd /content/Code
  !python scripts/fit_aim2_v8_fixedhp_sleep.py --model all
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
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.epochs import sleep_mask  # noqa: E402
from thesis_pipeline.extended_metrics import write_extended_metrics  # noqa: E402

FEATURES_DIR = CODE_ROOT / "features"
OUT_ROOT = CODE_ROOT / "models" / "aim2_v8_fixedhp_sleep"
NON_FEATURE = {
    "subject_id", "cohort", "epoch_idx", "epoch_start_sec",
    "sleep_stage", "apnoea_label", "features_version",
}

# Fixed sensible LightGBM hyperparameters — identical to v8.5-tax / v6 best.
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

MODEL_ORDER = ["lightgbm", "xgboost", "catboost", "rf", "logreg", "ensemble"]


# ---------------------------------------------------------------------------
# Cohort loading — identical to v85_taxonomy_ablation
# ---------------------------------------------------------------------------

def load_cohort(features_dir: Path, version: str = "2026-05-01-phase1batch-v1") -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Subject-level bootstrap — verbatim from v85_taxonomy_ablation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

def _build_lightgbm(spw: float) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=FIXED_N_ESTIMATORS, scale_pos_weight=spw, **FIXED_PARAMS)


def _build_xgboost(spw: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=FIXED_N_ESTIMATORS,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        scale_pos_weight=spw,
    )


def _build_catboost(spw: float) -> CatBoostClassifier:
    # CatBoost uses class_weights dict; weight class 1 by spw, class 0 by 1
    return CatBoostClassifier(
        iterations=FIXED_N_ESTIMATORS,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        thread_count=-1,
        random_seed=42,
        verbose=False,
        class_weights={0: 1.0, 1: float(spw)},
    )


def _build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )


def _build_logreg() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


# ---------------------------------------------------------------------------
# Core fit function
# ---------------------------------------------------------------------------

def fit_one_model(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    tv_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: Path,
) -> dict:
    """Fit one model on the given feature set, evaluate on test, save artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*78}\n=== Model: {name}  ({len(feature_cols)} features)\n{'='*78}")

    X = df[feature_cols].values
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    # -----------------------------------------------------------------------
    # ENSEMBLE: load base model predictions, average, evaluate
    # -----------------------------------------------------------------------
    if name == "ensemble":
        base_models = ["lightgbm", "xgboost", "catboost"]
        base_probs_list = []
        ref_df = None

        for bm in base_models:
            pred_path = OUT_ROOT / bm / "test_predictions.parquet"
            if not pred_path.exists():
                raise FileNotFoundError(
                    f"Base predictions for '{bm}' not found at {pred_path}. "
                    f"Run --model {bm} first (or --model all with ensemble last)."
                )
            bm_df = pd.read_parquet(pred_path)
            if ref_df is None:
                ref_df = bm_df[["subject_id", "epoch_idx", "apnoea_label"]].copy()
            else:
                # Verify alignment
                assert (ref_df["subject_id"].values == bm_df["subject_id"].values).all(), \
                    f"subject_id mismatch between lightgbm and {bm} predictions"
                assert (ref_df["epoch_idx"].values == bm_df["epoch_idx"].values).all(), \
                    f"epoch_idx mismatch between lightgbm and {bm} predictions"
            base_probs_list.append(bm_df["pred_prob"].values)
            print(f"  loaded {bm}: {len(bm_df):,} test epochs")

        probs = np.mean(np.stack(base_probs_list, axis=0), axis=0)
        y_test = ref_df["apnoea_label"].values.astype(np.int8)
        best_thresh = 0.5
        preds = (probs > best_thresh).astype(int)
        fit_time = 0.0
        best_iter = None
        spw = None

        test_auc = float(roc_auc_score(y_test, probs))
        test_aupr = float(average_precision_score(y_test, probs))
        test_f1 = float(f1_score(y_test, preds, zero_division=0))
        test_p = float(precision_score(y_test, preds, zero_division=0))
        test_r = float(recall_score(y_test, preds, zero_division=0))

        print(f"  Test AUC: {test_auc:.4f}   AUC-PR: {test_aupr:.4f}")
        print(f"  F1@{best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

        print(f"  computing subject-level bootstrap CI (1000 resamples)...")
        t0 = time.time()
        aucs, auprs, ci = subject_bootstrap(ref_df, probs, n_resamples=1000, seed=42)
        print(f"  bootstrap in {time.time()-t0:.0f}s")
        print(f"  AUC subject CI:  [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
        print(f"  AUC-PR subject CI: [{ci['aupr_ci_low']:.4f}, {ci['aupr_ci_high']:.4f}]")

        test_pred_df = pd.DataFrame({
            "subject_id": ref_df["subject_id"].values,
            "epoch_idx": ref_df["epoch_idx"].values,
            "apnoea_label": y_test,
            "pred_prob": probs,
            "pred_label": preds,
        })
        n_test_subjects = int(ref_df["subject_id"].nunique())
        n_test_epochs = int(len(ref_df))

    # -----------------------------------------------------------------------
    # SKLEARN models: rf, logreg — no early stopping; fit on full TV pool
    # -----------------------------------------------------------------------
    elif name in ("rf", "logreg"):
        # Carve inner val for threshold tuning (5% of TV by subject)
        inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
        tr_inner_rel, va_inner_rel = next(inner.split(X[tv_idx], y[tv_idx], groups[tv_idx]))
        tr = tv_idx[tr_inner_rel]
        va = tv_idx[va_inner_rel]

        print(f"  train (TV): {len(tv_idx):,} epochs / {len(np.unique(groups[tv_idx]))} subj  "
              f"({y[tv_idx].mean()*100:.1f}% positive)")
        print(f"  inner val:  {len(va):,} epochs / {len(np.unique(groups[va]))} subj (threshold tuning only)")
        print(f"  test:       {len(test_idx):,} epochs / {len(np.unique(groups[test_idx]))} subj")

        model = _build_rf() if name == "rf" else _build_logreg()
        t0 = time.time()
        # Fit on inner train only (va is held out for threshold tuning)
        model.fit(X[tr], y[tr])
        fit_time = time.time() - t0
        best_iter = None
        spw = None
        print(f"  fit in {fit_time:.0f}s")

        probs = model.predict_proba(X[test_idx])[:, 1]
        y_test = y[test_idx]

        # Threshold tuning on inner val
        va_probs = model.predict_proba(X[va])[:, 1]
        thresholds = np.linspace(0.05, 0.95, 91)
        va_f1s = [f1_score(y[va], (va_probs > t).astype(int), zero_division=0) for t in thresholds]
        best_thresh = float(thresholds[int(np.argmax(va_f1s))])
        preds = (probs > best_thresh).astype(int)

        test_auc = float(roc_auc_score(y_test, probs))
        test_aupr = float(average_precision_score(y_test, probs))
        test_f1 = float(f1_score(y_test, preds, zero_division=0))
        test_p = float(precision_score(y_test, preds, zero_division=0))
        test_r = float(recall_score(y_test, preds, zero_division=0))

        print(f"  Test AUC: {test_auc:.4f}   AUC-PR: {test_aupr:.4f}")
        print(f"  F1@{best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

        df_test_meta = df.iloc[test_idx][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True)
        print(f"  computing subject-level bootstrap CI (1000 resamples)...")
        t0 = time.time()
        aucs, auprs, ci = subject_bootstrap(df_test_meta, probs, n_resamples=1000, seed=42)
        print(f"  bootstrap in {time.time()-t0:.0f}s")
        print(f"  AUC subject CI:  [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
        print(f"  AUC-PR subject CI: [{ci['aupr_ci_low']:.4f}, {ci['aupr_ci_high']:.4f}]")

        test_pred_df = pd.DataFrame({
            "subject_id": df.iloc[test_idx]["subject_id"].values,
            "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
            "apnoea_label": y_test,
            "pred_prob": probs,
            "pred_label": preds,
        })
        n_test_subjects = int(len(np.unique(groups[test_idx])))
        n_test_epochs = int(len(test_idx))

    # -----------------------------------------------------------------------
    # BOOSTING models: lightgbm, xgboost, catboost — early stopping on inner val
    # -----------------------------------------------------------------------
    else:
        # Inner val carve from TV pool for early stopping + threshold tuning
        inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
        tr_inner_rel, va_inner_rel = next(inner.split(X[tv_idx], y[tv_idx], groups[tv_idx]))
        tr = tv_idx[tr_inner_rel]
        va = tv_idx[va_inner_rel]

        pos = float(np.sum(y[tr]))
        neg = float(len(tr) - pos)
        spw = neg / max(pos, 1.0)

        print(f"  inner train: {len(tr):,} epochs / {len(np.unique(groups[tr]))} subj  "
              f"({y[tr].mean()*100:.1f}% positive)")
        print(f"  inner val:   {len(va):,} epochs / {len(np.unique(groups[va]))} subj")
        print(f"  test:        {len(test_idx):,} epochs / {len(np.unique(groups[test_idx]))} subj")
        print(f"  scale_pos_weight: {spw:.3f}")

        t0 = time.time()
        if name == "lightgbm":
            model = _build_lightgbm(spw)
            model.fit(
                X[tr], y[tr],
                eval_set=[(X[va], y[va])],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(FIXED_EARLY_STOPPING, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            best_iter = int(model.best_iteration_ or FIXED_N_ESTIMATORS)

        elif name == "xgboost":
            model = _build_xgboost(spw)
            model.fit(
                X[tr], y[tr],
                eval_set=[(X[va], y[va])],
                verbose=False,
                early_stopping_rounds=FIXED_EARLY_STOPPING,
            )
            best_iter = int(model.best_iteration) if model.best_iteration is not None else FIXED_N_ESTIMATORS

        elif name == "catboost":
            model = _build_catboost(spw)
            model.fit(
                X[tr], y[tr],
                eval_set=(X[va], y[va]),
                early_stopping_rounds=FIXED_EARLY_STOPPING,
            )
            best_iter = int(model.get_best_iteration() or FIXED_N_ESTIMATORS)

        fit_time = time.time() - t0
        print(f"  fit in {fit_time:.0f}s  (best_iter={best_iter})")

        probs = model.predict_proba(X[test_idx])[:, 1]
        y_test = y[test_idx]

        # Threshold tuning on inner val — F1-maximising
        va_probs = model.predict_proba(X[va])[:, 1]
        thresholds = np.linspace(0.05, 0.95, 91)
        va_f1s = [f1_score(y[va], (va_probs > t).astype(int), zero_division=0) for t in thresholds]
        best_thresh = float(thresholds[int(np.argmax(va_f1s))])
        preds = (probs > best_thresh).astype(int)

        test_auc = float(roc_auc_score(y_test, probs))
        test_aupr = float(average_precision_score(y_test, probs))
        test_f1 = float(f1_score(y_test, preds, zero_division=0))
        test_p = float(precision_score(y_test, preds, zero_division=0))
        test_r = float(recall_score(y_test, preds, zero_division=0))

        print(f"  Test AUC: {test_auc:.4f}   AUC-PR: {test_aupr:.4f}")
        print(f"  F1@{best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

        df_test_meta = df.iloc[test_idx][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True)
        print(f"  computing subject-level bootstrap CI (1000 resamples)...")
        t0 = time.time()
        aucs, auprs, ci = subject_bootstrap(df_test_meta, probs, n_resamples=1000, seed=42)
        print(f"  bootstrap in {time.time()-t0:.0f}s")
        print(f"  AUC subject CI:  [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
        print(f"  AUC-PR subject CI: [{ci['aupr_ci_low']:.4f}, {ci['aupr_ci_high']:.4f}]")

        test_pred_df = pd.DataFrame({
            "subject_id": df.iloc[test_idx]["subject_id"].values,
            "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
            "apnoea_label": y_test,
            "pred_prob": probs,
            "pred_label": preds,
        })
        n_test_subjects = int(len(np.unique(groups[test_idx])))
        n_test_epochs = int(len(test_idx))

    # -----------------------------------------------------------------------
    # Save artifacts (all paths)
    # -----------------------------------------------------------------------
    test_pred_df.to_parquet(out_dir / "test_predictions.parquet", index=False)
    np.save(out_dir / "bootstrap_aucs_subject.npy", aucs)
    np.save(out_dir / "bootstrap_auprs_subject.npy", auprs)
    (out_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2))

    # Extended metrics (calibration, clinical AHI, severity κ)
    try:
        sm_path = FEATURES_DIR / "subject_metadata.parquet"
        sm = pd.read_parquet(sm_path) if sm_path.exists() else None
        write_extended_metrics(out_dir, test_pred_df, subject_metadata=sm)
    except Exception as _e:
        print(f"  ! extended metrics failed: {_e}")

    metrics: dict = {
        "model": name,
        "filter_used": "sleep-only",
        "n_features": len(feature_cols),
        "n_test_subjects": n_test_subjects,
        "n_test_epochs": n_test_epochs,
        "best_iter": best_iter,
        "scale_pos_weight": float(spw) if spw is not None else None,
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


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--features-version", default="2026-05-01-phase1batch-v1", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True,
              help="MUST be 42 to keep test split identical to v6 / v8 / v8.5 for paired tests")
@click.option("--model", default="all", show_default=True,
              type=click.Choice(["all", "lightgbm", "xgboost", "catboost", "rf", "logreg", "ensemble"]),
              help="Which model to fit; 'all' runs the full ladder")
def main(features_version: str, seed: int, model: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Aim 2 v8 FIXED-HP sleep-only model ladder ===")
    print(f"  Features dir:     {FEATURES_DIR}")
    print(f"  Output dir:       {OUT_ROOT}")
    print(f"  Features version: {features_version}")
    print(f"  Seed:             {seed}")
    print(f"  Model(s):         {model}\n")

    df = load_cohort(FEATURES_DIR, version=features_version)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    print(f"  Total features (excluding meta): {len(feature_cols)}\n")

    # Outer split — IDENTICAL to v6 / v8 / v8.5 (seed=42, patient-level)
    X_dummy = np.empty(len(df))
    y = df["apnoea_label"].values
    groups_arr = df["subject_id"].values
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, test_idx = next(outer.split(X_dummy, y, groups_arr))
    print(f"  Outer split (seed={seed}):")
    print(f"    TV pool: {len(tv_idx):,} epochs / {len(np.unique(groups_arr[tv_idx]))} subjects")
    print(f"    Test:    {len(test_idx):,} epochs / {len(np.unique(groups_arr[test_idx]))} subjects\n")

    models_to_run = MODEL_ORDER if model == "all" else [model]
    summary: dict = {}

    for m in models_to_run:
        out_dir = OUT_ROOT / m
        try:
            result = fit_one_model(m, df, feature_cols, tv_idx, test_idx, out_dir)
            summary[m] = result
        except FileNotFoundError as exc:
            print(f"\n  ! SKIPPED {m}: {exc}")
            summary[m] = {"model": m, "skipped": True, "reason": str(exc)}
        except Exception as exc:
            print(f"\n  ! FAILED {m}: {exc}")
            summary[m] = {"model": m, "failed": True, "reason": str(exc)}

    # Compact comparison table
    print(f"\n\n{'='*78}\n=== COMPACT COMPARISON ===\n{'='*78}\n")
    print(f"{'Model':10s}  {'Test AUC':>9s}  {'Subject 95% CI':>22s}  {'AUC-PR':>8s}  {'F1':>7s}")
    for m, r in summary.items():
        if r.get("skipped") or r.get("failed"):
            print(f"{m:10s}  {'SKIPPED/FAILED':>43s}")
            continue
        ci_str = f"[{r['test_auc_ci_low']:.4f}, {r['test_auc_ci_high']:.4f}]"
        print(f"{m:10s}  {r['test_auc_roc']:>9.4f}  {ci_str:>22s}  {r['test_auc_pr']:>8.4f}  {r['test_f1_tuned']:>7.4f}")

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
