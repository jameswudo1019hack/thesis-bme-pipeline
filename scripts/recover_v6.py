"""Recover v6 outputs after the fit_aim2_cv crash.

The v6 fit completed Optuna search and CV, but crashed at the very last step
(saving test_predictions) due to a load_cohort bug that dropped epoch_idx.
This script refits with the saved best_params on the same train/test split
and saves the missing files: metrics.json, test_predictions.parquet, model.txt.

No Optuna re-search — uses ``models/aim2_cv_v6/best_params.json`` directly.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

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
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from fit_aim2_cv import bootstrap_auc_ci, load_cohort, NON_FEATURE_COLS  # noqa: E402

OUT = CODE_ROOT / "models" / "aim2_cv_v6"
SEED = 42
FEATURES_VERSION_FILTER = "2026-04-25-contextual-v1"


def main() -> None:
    print("Loading features...")
    df = load_cohort(CODE_ROOT / "features")
    if "features_version" in df.columns:
        df = df[df["features_version"] == FEATURES_VERSION_FILTER].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"  {len(df):,} epochs, {df['subject_id'].nunique()} subjects, {len(feature_cols)} features")

    X = df[feature_cols].values
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    # Same 80/20 split as fit_aim2_cv (seed=42, GroupShuffleSplit)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tv_idx, test_idx = next(outer.split(X, y, groups))
    print(f"  Train+val: {len(tv_idx):,} epochs / {len(np.unique(groups[tv_idx]))} subj")
    print(f"  Test:      {len(test_idx):,} epochs / {len(np.unique(groups[test_idx]))} subj")

    # Load best params + per-fold metrics
    best_params = json.loads((OUT / "best_params.json").read_text())
    per_fold = json.loads((OUT / "per_fold_metrics.json").read_text())
    mean_best_iter = int(np.round(np.mean([f["best_iter"] for f in per_fold])))
    cv_mean_auc = float(np.mean([f["auc"] for f in per_fold]))
    cv_std_auc = float(np.std([f["auc"] for f in per_fold], ddof=1))
    print(f"  best params: {best_params}")
    print(f"  CV mean AUC: {cv_mean_auc:.4f} ± {cv_std_auc:.4f}")
    print(f"  mean best_iter: {mean_best_iter}")

    # Refit on full TV pool
    pos = float(y[tv_idx].sum())
    neg = float(len(tv_idx) - pos)
    final_params = dict(best_params)
    final_params["scale_pos_weight"] = neg / max(pos, 1.0)
    final_params["n_estimators"] = max(100, int(1.1 * mean_best_iter))
    final_params["verbose"] = -1
    final_params["n_jobs"] = -1
    final_params["random_state"] = SEED

    print(f"\nRefitting on full train+val (n_estimators={final_params['n_estimators']})...")
    model = lgb.LGBMClassifier(**final_params)
    model.fit(X[tv_idx], y[tv_idx])
    probs = model.predict_proba(X[test_idx])[:, 1]
    test_auc = float(roc_auc_score(y[test_idx], probs))
    test_ap = float(average_precision_score(y[test_idx], probs))

    # Tuned-threshold F1 from val
    tv_probs = model.predict_proba(X[tv_idx])[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    tv_f1s = [f1_score(y[tv_idx], (tv_probs > t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(tv_f1s))])
    preds = (probs > best_thresh).astype(int)
    test_f1 = float(f1_score(y[test_idx], preds, zero_division=0))
    test_p = float(precision_score(y[test_idx], preds, zero_division=0))
    test_r = float(recall_score(y[test_idx], preds, zero_division=0))

    print(f"\n=== Test set ===")
    print(f"  AUC-ROC : {test_auc:.4f}")
    print(f"  AUC-PR  : {test_ap:.4f}")
    print(f"  F1 @ {best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    print("\nBootstrap 95% CI...")
    bootstrap, (ci_lo, ci_hi) = bootstrap_auc_ci(y[test_idx], probs, n=1000, seed=SEED)
    print(f"  AUC-ROC 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    # Persist
    np.save(OUT / "bootstrap_aucs.npy", bootstrap)
    pd.DataFrame({
        "subject_id": df.iloc[test_idx]["subject_id"].values,
        "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
        "apnoea_label": y[test_idx],
        "pred_prob": probs,
        "pred_label": preds,
    }).to_parquet(OUT / "test_predictions.parquet", index=False)
    model.booster_.save_model(str(OUT / "model.txt"))
    metrics = {
        "cv_mean_auc": cv_mean_auc,
        "cv_std_auc": cv_std_auc,
        "test_auc_roc": test_auc,
        "test_auc_pr": test_ap,
        "test_auc_ci_low": ci_lo,
        "test_auc_ci_high": ci_hi,
        "test_f1_tuned": test_f1,
        "test_precision_tuned": test_p,
        "test_recall_tuned": test_r,
        "best_threshold": best_thresh,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "mean_best_iter": mean_best_iter,
        "n_cohort_subjects": int(df["subject_id"].nunique()),
        "n_test_subjects": int(len(np.unique(groups[test_idx]))),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
