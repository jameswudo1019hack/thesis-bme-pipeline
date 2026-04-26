"""Aim 2 baseline — LightGBM apnoea classifier on the pilot cohort.

Produces the first real performance number for the thesis. This is the
baseline that deep-learning models will be benchmarked against.

Design choices:
    - Patient-level train/val/test splits (no epoch leakage across subjects).
    - NaN passed through to LightGBM (native handling).
    - Class imbalance handled via ``scale_pos_weight``.
    - Reports both epoch-level metrics (AUC / AP / F1) and per-night AHI
      proxy (true apnoea epochs per hour vs predicted).

Usage:
    python scripts/fit_aim2_baseline.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Everything in the parquet that isn't a label/metadata column. Determined at
# runtime after loading so new feature columns appear automatically.
NON_FEATURE_COLS = {
    "subject_id",
    "cohort",
    "epoch_idx",
    "epoch_start_sec",
    "sleep_stage",
    "apnoea_label",
}

CODE_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = CODE_ROOT / "features"
OUT_DIR = CODE_ROOT / "models" / "aim2_baseline_v1"


def load_cohort(features_dir: Path) -> pd.DataFrame:
    files = sorted(features_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {features_dir}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def patient_split(df: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70/15/15 patient-level split. Returns (train_idx, val_idx, test_idx)."""
    y = df["apnoea_label"].values
    groups = df["subject_id"].values

    # First split off 30% for val+test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(gss1.split(df, y, groups))

    # Split the 30% evenly into val/test
    rest_groups = groups[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(gss2.split(rest_idx, y[rest_idx], rest_groups))
    val_idx = rest_idx[val_rel]
    test_idx = rest_idx[test_rel]
    return train_idx, val_idx, test_idx


def report_split(df: pd.DataFrame, train_idx, val_idx, test_idx) -> None:
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        sub = df.iloc[idx]
        print(
            f"  {name:5s}: {len(idx):6,} epochs  "
            f"{sub['subject_id'].nunique():3d} subjects  "
            f"{sub['apnoea_label'].mean()*100:5.1f}% positive"
        )


def main() -> None:
    print("Loading features...")
    df = load_cohort(FEATURES_DIR)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"  {len(df):,} epochs across {df['subject_id'].nunique()} subjects")
    print(f"  {len(feature_cols)} feature columns: {feature_cols}\n")

    X = df[feature_cols].values
    y = df["apnoea_label"].values
    train_idx, val_idx, test_idx = patient_split(df, seed=42)

    print("Patient-level split:")
    report_split(df, train_idx, val_idx, test_idx)

    # Class balance for LightGBM
    train_pos = y[train_idx].sum()
    train_neg = len(train_idx) - train_pos
    scale_pos = train_neg / max(train_pos, 1)
    print(f"\nscale_pos_weight = {scale_pos:.2f}")

    print("\nFitting LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X[train_idx],
        y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # ── Epoch-level metrics ──────────────────────────────────────────────
    val_probs = model.predict_proba(X[val_idx])[:, 1]
    probs = model.predict_proba(X[test_idx])[:, 1]

    # Pick threshold on val set (maximises F1). scale_pos_weight leaves raw
    # probabilities in the original prior range, so 0.5 isn't meaningful.
    thresholds = np.linspace(0.05, 0.95, 91)
    val_f1s = [f1_score(y[val_idx], (val_probs > t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(val_f1s))])
    print(f"\nBest threshold on val set: {best_thresh:.3f} (F1_val={max(val_f1s):.3f})")

    preds = (probs > best_thresh).astype(int)
    auc = roc_auc_score(y[test_idx], probs)
    ap = average_precision_score(y[test_idx], probs)
    f1_at_best = f1_score(y[test_idx], preds)
    f1_at_half = f1_score(y[test_idx], (probs > 0.5).astype(int), zero_division=0)

    print(f"\n=== Test set (epoch-level) ===")
    print(f"  AUC-ROC          : {auc:.3f}   (threshold-independent)")
    print(f"  AUC-PR           : {ap:.3f}   (threshold-independent)")
    print(f"  F1 @ {best_thresh:.2f} (tuned): {f1_at_best:.3f}")
    print(f"  F1 @ 0.50        : {f1_at_half:.3f}   (all-negative at default threshold)")
    print("\n", classification_report(y[test_idx], preds, digits=3))

    # ── Per-subject AHI proxy ────────────────────────────────────────────
    test_df = df.iloc[test_idx].copy()
    test_df["pred_prob"] = probs
    test_df["pred_label"] = preds
    per_subj = test_df.groupby("subject_id").agg(
        epochs=("epoch_idx", "count"),
        true_ap=("apnoea_label", "sum"),
        pred_ap=("pred_label", "sum"),
    )
    per_subj["hours"] = per_subj["epochs"] * 30 / 3600
    per_subj["true_ahi_proxy"] = per_subj["true_ap"] / per_subj["hours"]
    per_subj["pred_ahi_proxy"] = per_subj["pred_ap"] / per_subj["hours"]
    per_subj["ahi_error"] = per_subj["pred_ahi_proxy"] - per_subj["true_ahi_proxy"]

    print("\n=== Per-night AHI proxy (test set) ===")
    print(per_subj[["hours", "true_ahi_proxy", "pred_ahi_proxy", "ahi_error"]].round(1).to_string())
    mae = per_subj["ahi_error"].abs().mean()
    print(f"\n  MAE of AHI proxy:  {mae:.2f} events/h")
    print(f"  true AHI range:    {per_subj['true_ahi_proxy'].min():.1f} – {per_subj['true_ahi_proxy'].max():.1f}")
    print(f"  pred AHI range:    {per_subj['pred_ahi_proxy'].min():.1f} – {per_subj['pred_ahi_proxy'].max():.1f}")

    # ── Feature importance ───────────────────────────────────────────────
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== Feature importance (LightGBM split-count) ===")
    print(fi.to_string())

    # Persist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(OUT_DIR / "model.txt"))
    test_df[["subject_id", "epoch_idx", "apnoea_label", "pred_prob", "pred_label"]].to_parquet(
        OUT_DIR / "test_predictions.parquet", index=False
    )
    metrics = {
        "auc_roc": float(auc),
        "auc_pr": float(ap),
        "f1_at_best_thresh": float(f1_at_best),
        "f1_at_0_5": float(f1_at_half),
        "best_threshold": best_thresh,
        "ahi_mae": float(mae),
        "n_train_subjects": int(df.iloc[train_idx]["subject_id"].nunique()),
        "n_val_subjects": int(df.iloc[val_idx]["subject_id"].nunique()),
        "n_test_subjects": int(df.iloc[test_idx]["subject_id"].nunique()),
        "feature_cols": feature_cols,
    }
    import json

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model, test_predictions.parquet, metrics.json → {OUT_DIR}")


if __name__ == "__main__":
    main()
