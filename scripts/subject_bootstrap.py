"""Re-bootstrap test AUC + AUC-PR at the SUBJECT level for all v6/v8/v8.5 models.

Background: the original `bootstrap_auc_ci()` in `fit_aim2_cv.py` and friends
samples test EPOCHS (rows) with replacement, treating each epoch as i.i.d.
That violates the within-subject correlation structure — epochs from the same
subject are NOT independent. The CIs reported in v6/v8/v8.5 metrics.json files
are therefore narrower than the true subject-level CIs. `Methodology.md` line 36
explicitly declares "Subject-level bootstrap" — the code didn't follow.

This script fixes that by:
  1. Loading each model's test_predictions.parquet
  2. Computing subject-level bootstrap CIs (resample SUBJECTS w/ replacement,
     pull all their epochs, compute AUC) — 1000 iterations, seed=42
  3. Saving subject-level bootstrap arrays alongside the original epoch-level ones
  4. Printing a side-by-side comparison table

Outputs (per model directory):
  bootstrap_aucs_subject.npy    — 1000-vector of subject-level test AUCs
  bootstrap_auprs_subject.npy   — 1000-vector of subject-level AUC-PR values
  ci_subject.json               — {test_auc, test_auc_pr, ci_low, ci_high} dict

Usage:
  python scripts/subject_bootstrap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CODE_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = CODE_ROOT / "models"

MODEL_DIRS = {
    "lightgbm_v6":  MODELS_ROOT / "aim2_cv_v6",
    "xgboost_v8":   MODELS_ROOT / "aim2_v8_xgboost",
    "catboost_v8":  MODELS_ROOT / "aim2_v8_catboost",
    "rf_v8.5":      MODELS_ROOT / "aim2_v8_rf",
    "logreg_v8.5":  MODELS_ROOT / "aim2_v8_logreg",
}


def subject_bootstrap(
    df: pd.DataFrame, n_resamples: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Subject-level paired bootstrap.

    For each resample: draw len(unique_subjects) subjects with replacement,
    concatenate all epochs from those subjects, compute AUC and AUC-PR.

    Performance: extracts the three needed columns to NumPy ONCE before the
    loop and uses NumPy fancy indexing (not pandas .loc) per iteration.
    Pandas-based indexing was ~10× slower at 1.17M rows × 1000 iterations.
    """
    rng = np.random.default_rng(seed)

    # Extract once to NumPy — avoids pandas indexing overhead in the inner loop
    sids = df["subject_id"].values
    y_all = df["apnoea_label"].values
    p_all = df["pred_prob"].values

    subjects = np.unique(sids)
    n_subj = len(subjects)

    # Build per-subject row indices ONCE (subjects are integers in our data)
    by_subject: dict = {int(s): np.where(sids == s)[0] for s in subjects}
    subjects_int = np.asarray(list(by_subject.keys()), dtype=np.int64)

    aucs = np.full(n_resamples, np.nan)
    auprs = np.full(n_resamples, np.nan)

    for i in range(n_resamples):
        sampled = rng.choice(subjects_int, size=n_subj, replace=True)
        idx = np.concatenate([by_subject[int(s)] for s in sampled])
        y = y_all[idx]
        if len(np.unique(y)) < 2:
            continue
        p = p_all[idx]
        aucs[i] = roc_auc_score(y, p)
        auprs[i] = average_precision_score(y, p)

    return aucs, auprs


def report_one(name: str, model_dir: Path) -> dict | None:
    pred_path = model_dir / "test_predictions.parquet"
    if not pred_path.exists():
        print(f"  ! {name}: predictions not found at {pred_path}")
        return None

    df = pd.read_parquet(pred_path)
    n_epochs = len(df)
    n_subjects = df["subject_id"].nunique()

    # Point estimates (these are unaffected by which bootstrap unit we use)
    test_auc = float(roc_auc_score(df["apnoea_label"], df["pred_prob"]))
    test_aupr = float(average_precision_score(df["apnoea_label"], df["pred_prob"]))

    # Subject-level bootstrap
    aucs, auprs = subject_bootstrap(df, n_resamples=1000, seed=42)

    valid_auc = aucs[np.isfinite(aucs)]
    valid_aupr = auprs[np.isfinite(auprs)]
    auc_lo, auc_hi = float(np.percentile(valid_auc, 2.5)), float(np.percentile(valid_auc, 97.5))
    aupr_lo, aupr_hi = float(np.percentile(valid_aupr, 2.5)), float(np.percentile(valid_aupr, 97.5))

    # Load the existing (epoch-level) CI from metrics.json for comparison
    epoch_ci = None
    mfile = model_dir / "metrics.json"
    if mfile.exists():
        m = json.loads(mfile.read_text())
        if "test_auc_ci_low" in m and "test_auc_ci_high" in m:
            epoch_ci = (m["test_auc_ci_low"], m["test_auc_ci_high"])

    # Persist
    np.save(model_dir / "bootstrap_aucs_subject.npy", aucs)
    np.save(model_dir / "bootstrap_auprs_subject.npy", auprs)
    out = {
        "test_auc": test_auc,
        "test_auc_pr": test_aupr,
        "auc_ci_subject_low": auc_lo,
        "auc_ci_subject_high": auc_hi,
        "auc_pr_ci_subject_low": aupr_lo,
        "auc_pr_ci_subject_high": aupr_hi,
        "n_test_subjects": int(n_subjects),
        "n_test_epochs": int(n_epochs),
        "n_bootstrap": 1000,
        "seed": 42,
        "method": "subject-level paired bootstrap (resample subjects with replacement, concat all their epochs)",
    }
    (model_dir / "ci_subject.json").write_text(json.dumps(out, indent=2))

    # Print comparison
    print(f"\n{name}  (n_epochs={n_epochs:,}, n_subjects={n_subjects})")
    print(f"  Test AUC: {test_auc:.4f}")
    if epoch_ci is not None:
        epoch_w = epoch_ci[1] - epoch_ci[0]
        print(f"    epoch-level CI:    [{epoch_ci[0]:.4f}, {epoch_ci[1]:.4f}]   width {epoch_w:.4f}")
    subj_w = auc_hi - auc_lo
    print(f"    SUBJECT-level CI:  [{auc_lo:.4f}, {auc_hi:.4f}]   width {subj_w:.4f}")
    if epoch_ci is not None:
        print(f"    → subject CI is {subj_w / epoch_w:.1f}× wider")
    print(f"  Test AUC-PR: {test_aupr:.4f}    SUBJECT CI [{aupr_lo:.4f}, {aupr_hi:.4f}]")

    return out


def main() -> None:
    print("=== Subject-level bootstrap CIs (Methodology.md compliance fix) ===\n")
    print("Resampling 1,158 test subjects (with replacement) × 1000 iterations × 5 models...")
    print("Seed=42 for reproducibility. Each iteration draws subjects, then pulls ALL their epochs.\n")

    summary = {}
    for name, mdir in MODEL_DIRS.items():
        result = report_one(name, mdir)
        if result is not None:
            summary[name] = result

    # Final compact table
    print("\n\n=== COMPACT TABLE (subject-level CIs, the defensible numbers) ===\n")
    print(f"{'Model':16s}  {'Test AUC':>10s}  {'95% CI (subject-level)':>26s}  {'Test AUC-PR':>12s}  {'AUC-PR CI':>20s}")
    for name, r in summary.items():
        ci = f"[{r['auc_ci_subject_low']:.4f}, {r['auc_ci_subject_high']:.4f}]"
        ci_pr = f"[{r['auc_pr_ci_subject_low']:.4f}, {r['auc_pr_ci_subject_high']:.4f}]"
        print(f"{name:16s}  {r['test_auc']:>10.4f}  {ci:>26s}  {r['test_auc_pr']:>12.4f}  {ci_pr:>20s}")

    # Save aggregate report
    report_path = MODELS_ROOT / "subject_bootstrap_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull summary saved to: {report_path}")


if __name__ == "__main__":
    main()
