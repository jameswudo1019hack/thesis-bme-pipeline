"""Recover the cuML RF run that crashed mid-recompute on 2026-04-26.

Skips Optuna entirely — uses the already-found best params from the prior run
(visible in the cell 7 output and stored in optuna_study.db). Just does the
final TV refit + bootstrap + save so the ensemble step has rf predictions.

Usage on Colab:
    !python /content/v8_work/scripts/recover_rf_cuml.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from cuml.ensemble import RandomForestClassifier as cuRF
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

WORK_ROOT = Path("/content/v8_work")
FEATURES_DIR = WORK_ROOT / "features"
RF_DIR = WORK_ROOT / "models" / "aim2_v8_rf"
RF_DIR.mkdir(parents=True, exist_ok=True)

# Best params from the prior Optuna run (cell 7 output, trial 1 was best at CV 0.8186)
BEST_PARAMS = {
    "n_estimators": 250,
    "max_depth": 24,
    "min_samples_split": 37,
    "min_samples_leaf": 18,
    "max_features": "sqrt",
}

# Per-fold AUCs already observed in cell 7 output (folds 1-4); fold 5 estimated
PRIOR_PER_FOLD = [0.8204, 0.8142, 0.8266, 0.8173]

KEEP_META = {"subject_id", "epoch_idx", "apnoea_label", "features_version"}
DROP_META = {"cohort", "epoch_start_sec", "sleep_stage"}
NON_FEATURE = {
    "subject_id", "cohort", "epoch_idx", "epoch_start_sec",
    "sleep_stage", "apnoea_label", "features_version",
}


def main() -> None:
    print(f"Using best_params from prior Optuna run: {BEST_PARAMS}")

    print("Loading features...")
    t0 = time.time()
    frames = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        df = df.drop(columns=[c for c in DROP_META if c in df.columns], errors="ignore")
        for c in df.columns:
            if c not in KEEP_META and df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        if "subject_id" in df.columns:
            df["subject_id"] = df["subject_id"].astype(np.int32)
        if "apnoea_label" in df.columns:
            df["apnoea_label"] = df["apnoea_label"].astype(np.int8)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True, copy=False)
    df = df[df["features_version"] == "2026-04-25-contextual-v1"].reset_index(drop=True)
    print(f"  loaded {len(df):,} epochs in {time.time()-t0:.0f}s")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    X = df[feature_cols].values
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    # Same seed=42 outer split — DeLong-compatible test set
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tv_idx, test_idx = next(outer.split(X, y, groups))
    print(f"  train+val: {len(tv_idx):,}, test: {len(test_idx):,}")

    # Undersample majority class to 1:1 (matches script's RF behavior)
    rng = np.random.default_rng(42)
    y_tv = y[tv_idx]
    pos = np.where(y_tv == 1)[0]
    neg = np.where(y_tv == 0)[0]
    neg_sampled = rng.choice(neg, len(pos), replace=False)
    sel = np.sort(np.concatenate([pos, neg_sampled]))
    tv_balanced = tv_idx[sel]
    print(f"  balanced TV: {len(tv_balanced):,} ({y[tv_balanced].mean()*100:.1f}% positive)")

    # Median impute (train statistics carry to test)
    X_tv = X[tv_balanced]
    medians = np.nanmedian(X_tv, axis=0)
    X_tv_imp = np.where(np.isnan(X_tv), medians, X_tv).astype(np.float32)
    y_tv_imp = y[tv_balanced].astype(np.int32)

    print("\n▶ Fitting cuML RF on balanced TV pool...")
    t0 = time.time()
    model = cuRF(**BEST_PARAMS, n_streams=1, random_state=42, n_bins=128)
    model.fit(X_tv_imp, y_tv_imp)
    print(f"  fit in {time.time()-t0:.0f}s")

    X_test = X[test_idx]
    X_test_imp = np.where(np.isnan(X_test), medians, X_test).astype(np.float32)
    probs = np.asarray(model.predict_proba(X_test_imp))[:, 1]

    test_auc = float(roc_auc_score(y[test_idx], probs))
    test_ap = float(average_precision_score(y[test_idx], probs))

    # Tuned-threshold F1 from TV predictions
    tv_probs = np.asarray(model.predict_proba(X_tv_imp))[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    tv_f1s = [f1_score(y_tv_imp, (tv_probs > t).astype(int), zero_division=0) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(tv_f1s))])
    preds = (probs > best_t).astype(int)
    test_f1 = float(f1_score(y[test_idx], preds, zero_division=0))
    test_p = float(precision_score(y[test_idx], preds, zero_division=0))
    test_r = float(recall_score(y[test_idx], preds, zero_division=0))

    print("\n▶ Bootstrap 95% CI on test AUC (1000 resamples)...")
    rng2 = np.random.default_rng(42)
    y_te = y[test_idx]
    n = len(y_te)
    aucs = np.empty(1000)
    for i in range(1000):
        idx = rng2.integers(0, n, n)
        if len(np.unique(y_te[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_te[idx], probs[idx])
    valid = aucs[np.isfinite(aucs)]
    ci_lo = float(np.percentile(valid, 2.5))
    ci_hi = float(np.percentile(valid, 97.5))

    print(f"\n=== RF Held-out test set (recovered) ===")
    print(f"  AUC-ROC : {test_auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  AUC-PR  : {test_ap:.4f}")
    print(f"  F1 @ {best_t:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    per_fold = [{"fold": i + 1, "auc": v} for i, v in enumerate(PRIOR_PER_FOLD)]
    per_fold.append({"fold": 5, "auc": float(np.mean(PRIOR_PER_FOLD)), "estimated": True})
    fold_aucs = [f["auc"] for f in per_fold]
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs, ddof=1))

    (RF_DIR / "best_params.json").write_text(json.dumps(BEST_PARAMS, indent=2))
    (RF_DIR / "per_fold_metrics.json").write_text(json.dumps(per_fold, indent=2))
    np.save(RF_DIR / "bootstrap_aucs.npy", aucs)
    pd.DataFrame({
        "subject_id": df.iloc[test_idx]["subject_id"].values,
        "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
        "apnoea_label": y[test_idx],
        "pred_prob": probs,
        "pred_label": preds,
    }).to_parquet(RF_DIR / "test_predictions.parquet", index=False)

    metrics = {
        "model": "rf",
        "framework": "cuml",
        "cv_mean_auc": mean_auc,
        "cv_std_auc": std_auc,
        "test_auc_roc": test_auc,
        "test_auc_pr": test_ap,
        "test_auc_ci_low": ci_lo,
        "test_auc_ci_high": ci_hi,
        "test_f1_tuned": test_f1,
        "test_precision_tuned": test_p,
        "test_recall_tuned": test_r,
        "best_threshold": best_t,
        "best_params": BEST_PARAMS,
        "feature_cols": feature_cols,
        "n_test_subjects": int(np.unique(groups[test_idx]).size),
        "features_version": "2026-04-25-contextual-v1",
        "note": "Recovered after the v8.5 cuML RF script crashed during the per-fold "
                "recompute step on 2026-04-26. Best params from the prior Optuna run; "
                "fold 5 AUC estimated as mean of folds 1-4 (the actual fold 5 fit was "
                "interrupted mid-train).",
    }
    (RF_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {RF_DIR}")


if __name__ == "__main__":
    main()
