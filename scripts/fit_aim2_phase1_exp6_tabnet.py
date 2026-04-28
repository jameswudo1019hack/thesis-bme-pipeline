"""Aim 2 Phase 1 — Experiment 6: TabNet on audit-v6 features.

Hypothesis
----------
Attention-based tabular DL (TabNet) finds non-linear feature interactions that
LightGBM missed on the same 273-feature audit-v6 input, lifting AUC modestly
in each taxonomy tier (full / physio_only / aasm_only).

Specifically, TabNet uses sequential attention to choose which features to
reason about at each decision step, which should help in particular when:
  - Features have non-additive interactions (e.g. HRV-LF × stage interaction)
  - The feature set is highly redundant (273 cols with lag/lead/rolling
    derivatives — TabNet's attention can prune to a sparse subset per epoch)

Expected outcome
----------------
TabNet probably matches or slightly exceeds LightGBM on full and physio_only.
On aasm_only (only 56 features, all already strong), TabNet may underperform
LightGBM due to less data per parameter. Predicted ranges (sleep-only):

  full:        0.95-0.97  (vs LightGBM 0.9591)
  physio_only: 0.67-0.72  (vs LightGBM 0.6737)
  aasm_only:   0.93-0.95  (vs LightGBM 0.9488)

Protocol
--------
Same as 8.5-tax / Phase 1 Exp 5 (canonical sleep-only):
  - sleep-only filter via wake_mask (drops ~28.8% wake epochs)
  - GroupShuffleSplit(test_size=0.2, random_state=42) — IDENTICAL test set
  - Inner train/val for early stopping (5% of TV pool)
  - Median imputation + StandardScaler (TabNet doesn't handle NaN natively)
  - TabNet with default-ish HP (n_d=64, n_a=64, n_steps=5, gamma=1.5)
  - Class-balanced sample weights for imbalance
  - GPU strongly preferred (T4 OK, A100 better)
  - Subject-level bootstrap CI (1000 resamples)

Three configs run sequentially:
  full        — 273 features
  physio_only — 217 features (drop AASM-rule)
  aasm_only   — 56 features (only AASM-rule)

Output: Code/models/aim2_phase1_exp6_tabnet/<config>/{metrics, test_predictions, bootstrap}

Usage (Colab GPU runtime, T4 or A100):
  %cd /content/Code
  !python scripts/fit_aim2_phase1_exp6_tabnet.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import click
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.epochs import wake_mask  # noqa: E402
from thesis_pipeline.feature_groups import feature_subset, split_features  # noqa: E402

FEATURES_DIR = CODE_ROOT / "features"
OUT_ROOT = CODE_ROOT / "models" / "aim2_phase1_exp6_tabnet"
NON_FEATURE = {
    "subject_id", "cohort", "epoch_idx", "epoch_start_sec",
    "sleep_stage", "apnoea_label", "features_version",
}


def load_cohort(features_dir: Path, version: str = "2026-04-26-audit-v6") -> pd.DataFrame:
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


def subject_bootstrap(df_test: pd.DataFrame, probs: np.ndarray,
                     n_resamples: int = 1000, seed: int = 42):
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


def fit_one_config(name: str, df: pd.DataFrame, feature_cols: list[str],
                   tv_idx: np.ndarray, test_idx: np.ndarray,
                   out_dir: Path, device: str) -> dict:
    """Fit TabNet on the given feature subset; evaluate on held-out test."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*78}\n=== Config: {name}  ({len(feature_cols)} features)\n{'='*78}")

    if len(feature_cols) == 0:
        return {"name": name, "n_features": 0, "skipped": True}

    # Lazy import — fail fast if pytorch-tabnet not installed
    from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: E402

    X = df[feature_cols].values.astype(np.float32)
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    tr_inner_rel, va_inner_rel = next(inner.split(X[tv_idx], y[tv_idx], groups[tv_idx]))
    tr = tv_idx[tr_inner_rel]
    va = tv_idx[va_inner_rel]

    pos = float(np.sum(y[tr])); neg = float(len(tr) - pos)
    print(f"  inner train: {len(tr):,} ({y[tr].mean()*100:.1f}% pos)  val: {len(va):,}  test: {len(test_idx):,}")

    # Median impute + standardize (TabNet doesn't handle NaN natively).
    print("  preprocessing: median impute + standard scale...")
    t0 = time.time()
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X[tr])
    X_va = imp.transform(X[va])
    X_te = imp.transform(X[test_idx])
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr).astype(np.float32)
    X_va = sc.transform(X_va).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    print(f"  preprocess in {time.time()-t0:.0f}s")

    # Class-balanced sample weights — sklearn-style 'balanced'
    n = len(y[tr])
    w_pos = n / (2.0 * max(pos, 1.0))
    w_neg = n / (2.0 * max(neg, 1.0))
    weights = {0: w_neg, 1: w_pos}
    print(f"  class weights: neg={w_neg:.3f} pos={w_pos:.3f}")

    print(f"  fitting TabNet on {device}...")
    t0 = time.time()
    clf = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        lambda_sparse=1e-3,
        seed=42,
        device_name=device,
        verbose=0,
    )
    clf.fit(
        X_tr, y[tr].astype(np.int64),
        eval_set=[(X_va, y[va].astype(np.int64))],
        eval_metric=["auc"],
        max_epochs=100,
        patience=10,
        batch_size=8192,
        virtual_batch_size=512,
        weights=weights,
        drop_last=False,
    )
    fit_time = time.time() - t0
    print(f"  fit in {fit_time:.0f}s  (best_epoch={clf.best_epoch})")

    probs = clf.predict_proba(X_te)[:, 1]
    test_auc = float(roc_auc_score(y[test_idx], probs))
    test_aupr = float(average_precision_score(y[test_idx], probs))

    # Val-tuned threshold for F1
    va_probs = clf.predict_proba(X_va)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    va_f1s = [f1_score(y[va], (va_probs > t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(va_f1s))])
    preds = (probs > best_thresh).astype(int)
    test_f1 = float(f1_score(y[test_idx], preds, zero_division=0))
    test_p = float(precision_score(y[test_idx], preds, zero_division=0))
    test_r = float(recall_score(y[test_idx], preds, zero_division=0))

    print(f"\n  Test AUC: {test_auc:.4f}   AUC-PR: {test_aupr:.4f}")
    print(f"  F1@{best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    df_test = df.iloc[test_idx][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True)
    print(f"  computing subject-level bootstrap CI (1000 resamples)...")
    t0 = time.time()
    aucs, auprs, ci = subject_bootstrap(df_test, probs, n_resamples=1000, seed=42)
    print(f"  bootstrap in {time.time()-t0:.0f}s")
    print(f"  AUC subject CI:    [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
    print(f"  AUC-PR subject CI: [{ci['aupr_ci_low']:.4f}, {ci['aupr_ci_high']:.4f}]")

    # Persist
    pd.DataFrame({
        "subject_id": df.iloc[test_idx]["subject_id"].values,
        "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
        "apnoea_label": y[test_idx],
        "pred_prob": probs,
        "pred_label": preds,
    }).to_parquet(out_dir / "test_predictions.parquet", index=False)
    np.save(out_dir / "bootstrap_aucs_subject.npy", aucs)
    np.save(out_dir / "bootstrap_auprs_subject.npy", auprs)
    (out_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2))

    metrics = {
        "name": name,
        "experiment": "phase1_exp6_tabnet",
        "model": "TabNet",
        "n_features": len(feature_cols),
        "n_test_subjects": int(len(np.unique(groups[test_idx]))),
        "n_test_epochs": int(len(test_idx)),
        "best_epoch": int(clf.best_epoch),
        "tabnet_params": {
            "n_d": 64, "n_a": 64, "n_steps": 5, "gamma": 1.5,
            "n_independent": 2, "n_shared": 2, "lambda_sparse": 1e-3,
            "max_epochs": 100, "patience": 10,
            "batch_size": 8192, "virtual_batch_size": 512,
            "device": device, "seed": 42,
        },
        "class_weights": weights,
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
@click.option("--features-version", default="2026-04-26-audit-v6", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--device", default="auto", show_default=True,
              help="'auto' picks cuda if available; or pass 'cuda'/'cpu' explicitly")
def main(features_version: str, seed: int, device: str) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Aim 2 Phase 1 — Exp 6: TabNet on audit-v6 ===\n")

    # Resolve device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"  Device: {device}")
    if device == "cpu":
        print(f"  ⚠ CPU TabNet will be ~5-10× slower than GPU. Consider switching runtime.")

    df = load_cohort(FEATURES_DIR, version=features_version)
    feature_cols_all = [c for c in df.columns if c not in NON_FEATURE]
    groups = split_features(feature_cols_all)
    print(f"\n  Feature taxonomy: AASM {len(groups['aasm'])}  Physio {len(groups['physio'])}  Unknown→physio {len(groups['unknown'])}")

    # IDENTICAL outer split to v6/v8/v8.5/8.5-tax/Lit-2019/Exp 5
    X_dummy = np.empty(len(df))
    y = df["apnoea_label"].values
    groups_arr = df["subject_id"].values
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, test_idx = next(outer.split(X_dummy, y, groups_arr))
    print(f"\n  Outer split (seed={seed}, sleep-only):")
    print(f"    TV pool: {len(tv_idx):,} epochs / {len(np.unique(groups_arr[tv_idx]))} subj")
    print(f"    Test:    {len(test_idx):,} epochs / {len(np.unique(groups_arr[test_idx]))} subj")

    summary = {"experiment": "phase1_exp6_tabnet", "device": device, "configs": {}}
    BASELINES = {
        "full":        ("8.5-tax full LightGBM",        0.9591),
        "physio_only": ("8.5-tax physio_only LightGBM", 0.6737),
        "aasm_only":   ("8.5-tax aasm_only LightGBM",   0.9488),
    }
    for name in ("full", "physio_only", "aasm_only"):
        feature_cols = feature_subset(feature_cols_all, mode=name)
        out_dir = OUT_ROOT / name
        result = fit_one_config(name, df, feature_cols, tv_idx, test_idx, out_dir, device)
        summary["configs"][name] = result

    # Final compact comparison
    print(f"\n\n{'='*78}\n=== EXP 6 COMPACT COMPARISON (TabNet vs 8.5-tax LightGBM)\n{'='*78}\n")
    print(f"{'Config':14s}  {'Features':>9s}  {'TabNet AUC':>10s}  {'Subject 95% CI':>22s}  {'LGB ref':>10s}  {'Δ':>9s}")
    for name, r in summary["configs"].items():
        if r.get("skipped"):
            continue
        ref_name, ref_auc = BASELINES.get(name, ("?", 0.0))
        delta = r["test_auc_roc"] - ref_auc
        ci = f"[{r['test_auc_ci_low']:.4f}, {r['test_auc_ci_high']:.4f}]"
        print(f"{name:14s}  {r['n_features']:>9d}  {r['test_auc_roc']:>10.4f}  {ci:>22s}  {ref_auc:>10.4f}  {delta:>+9.4f}")

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {OUT_ROOT / 'summary.json'}")

    print(f"\n=== HYPOTHESIS CHECK ===")
    print(f"Hypothesis: TabNet matches or modestly exceeds LightGBM in each tier.")
    deltas = {n: summary["configs"][n].get("test_auc_roc", 0) - BASELINES[n][1]
              for n in BASELINES if n in summary["configs"]}
    if all(d > 0.005 for d in deltas.values()):
        print(f"  → CONFIRMED. TabNet beats LightGBM in all 3 tiers: {deltas}")
    elif all(abs(d) <= 0.01 for d in deltas.values()):
        print(f"  → INCONCLUSIVE. TabNet ≈ LightGBM in all 3 tiers: {deltas}")
    elif any(d > 0.01 for d in deltas.values()) and any(d < -0.01 for d in deltas.values()):
        print(f"  → MIXED. TabNet beats LGB in some tiers but loses in others: {deltas}")
    else:
        print(f"  → DISCONFIRMED. TabNet underperforms LightGBM: {deltas}")


if __name__ == "__main__":
    main()
