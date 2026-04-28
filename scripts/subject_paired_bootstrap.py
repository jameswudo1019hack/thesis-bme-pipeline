"""Subject-level paired bootstrap test for pairwise model comparisons.

This is the subject-level equivalent of the DeLong test. For each pair of
models (A, B) with predictions on the same held-out test set:
    1. Resample SUBJECTS with replacement (n iterations).
    2. For each resample, pull all epochs from sampled subjects.
    3. Compute Δ AUC = AUC_A - AUC_B on those rows.
    4. Report the 95% CI on Δ AUC across resamples.
    5. p_two_sided = 2 × min(P(Δ ≤ 0), P(Δ ≥ 0))  (empirical, percentile-based).

Why this matters: the existing DeLong tests treat test rows (epochs) as
independent. Methodology.md declares subject-level inference. This script
computes the proper subject-paired test that matches the declaration.

Output: Code/models/delong_tests/subject_paired_bootstrap.json
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CODE_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = CODE_ROOT / "models"

MODEL_DIRS = {
    "lightgbm_v6":  MODELS_ROOT / "aim2_cv_v6",
    "xgboost_v8":   MODELS_ROOT / "aim2_v8_xgboost",
    "catboost_v8":  MODELS_ROOT / "aim2_v8_catboost",
    "rf_v8.5":      MODELS_ROOT / "aim2_v8_rf",
    "logreg_v8.5":  MODELS_ROOT / "aim2_v8_logreg",
}

# Pairwise comparisons that matter for the writeup
COMPARISONS = [
    ("lightgbm_v6", "xgboost_v8"),    # The +0.0031 SOTA-claim test
    ("xgboost_v8", "catboost_v8"),     # Within-family (Δ=+0.0026, headline finding)
    ("lightgbm_v6", "catboost_v8"),    # GBM equivalence test
    ("xgboost_v8", "rf_v8.5"),         # Boosting vs non-boosting
    ("xgboost_v8", "logreg_v8.5"),     # Best vs linear baseline
    ("rf_v8.5", "logreg_v8.5"),        # Trees vs linear
]


def load_aligned_predictions(name_a: str, name_b: str) -> pd.DataFrame:
    """Load both models' predictions, verify aligned on (subject_id, epoch_idx, label)."""
    df_a = pd.read_parquet(MODEL_DIRS[name_a] / "test_predictions.parquet")
    df_b = pd.read_parquet(MODEL_DIRS[name_b] / "test_predictions.parquet")

    df_a = df_a.sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True)
    df_b = df_b.sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True)

    if not df_a[["subject_id", "epoch_idx", "apnoea_label"]].equals(
        df_b[["subject_id", "epoch_idx", "apnoea_label"]]
    ):
        raise ValueError(f"Predictions not aligned for {name_a} vs {name_b}")

    return pd.DataFrame({
        "subject_id": df_a["subject_id"].values,
        "epoch_idx": df_a["epoch_idx"].values,
        "y": df_a["apnoea_label"].values.astype(np.int8),
        "p_a": df_a["pred_prob"].values.astype(np.float64),
        "p_b": df_b["pred_prob"].values.astype(np.float64),
    })


def subject_paired_bootstrap(
    df: pd.DataFrame, n_resamples: int = 1000, seed: int = 42
) -> dict:
    """Compute Δ AUC = AUC(p_a) - AUC(p_b) under subject-level resampling.

    Returns the bootstrap distribution of Δ AUC plus point estimates and a
    percentile-based two-sided p-value for H0: Δ AUC = 0.
    """
    rng = np.random.default_rng(seed)

    sids = df["subject_id"].values
    y_all = df["y"].values
    p_a_all = df["p_a"].values
    p_b_all = df["p_b"].values

    subjects = np.unique(sids)
    n_subj = len(subjects)
    by_subject = {int(s): np.where(sids == s)[0] for s in subjects}
    subjects_int = np.asarray(list(by_subject.keys()), dtype=np.int64)

    # Point estimates on the actual (un-resampled) test set
    auc_a_point = float(roc_auc_score(y_all, p_a_all))
    auc_b_point = float(roc_auc_score(y_all, p_b_all))
    delta_point = auc_a_point - auc_b_point

    deltas = np.full(n_resamples, np.nan)

    for i in range(n_resamples):
        sampled = rng.choice(subjects_int, size=n_subj, replace=True)
        idx = np.concatenate([by_subject[int(s)] for s in sampled])
        y = y_all[idx]
        if len(np.unique(y)) < 2:
            continue
        auc_a = roc_auc_score(y, p_a_all[idx])
        auc_b = roc_auc_score(y, p_b_all[idx])
        deltas[i] = auc_a - auc_b

    valid = deltas[np.isfinite(deltas)]
    delta_lo, delta_hi = float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))

    # Percentile-based two-sided p-value
    p_left = float(np.mean(valid <= 0.0))
    p_right = float(np.mean(valid >= 0.0))
    p_two_sided = 2.0 * min(p_left, p_right)

    # Effective n (in case some resamples were degenerate)
    n_eff = int(np.sum(np.isfinite(deltas)))

    return {
        "auc_a_point": auc_a_point,
        "auc_b_point": auc_b_point,
        "delta_point": delta_point,
        "delta_ci_low": delta_lo,
        "delta_ci_high": delta_hi,
        "p_two_sided": p_two_sided,
        "p_left": p_left,
        "p_right": p_right,
        "n_resamples_valid": n_eff,
        "n_resamples_total": n_resamples,
        "deltas_all": deltas,  # for saving
    }


def main() -> None:
    print("=== Subject-paired bootstrap tests (subject-level equivalent of DeLong) ===\n")
    print("For each pair: resample subjects 1000× (seed=42), compute Δ AUC = A - B per resample.")
    print("Report point Δ, 95% CI on Δ, and percentile-based two-sided p-value.\n")

    out_dir = MODELS_ROOT / "delong_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}

    for name_a, name_b in COMPARISONS:
        print(f"\n--- {name_a} vs {name_b} ---")
        df = load_aligned_predictions(name_a, name_b)
        result = subject_paired_bootstrap(df, n_resamples=1000, seed=42)

        # Save the per-pair Δ array
        pair_key = f"{name_a}_vs_{name_b}"
        np.save(out_dir / f"subject_paired_{pair_key}_deltas.npy", result["deltas_all"])

        verdict = (
            "✓ significant"
            if result["p_two_sided"] < 0.05
            else "✗ NOT significant at α=0.05"
        )
        print(f"  AUC_a = {result['auc_a_point']:.4f}")
        print(f"  AUC_b = {result['auc_b_point']:.4f}")
        print(f"  Δ AUC = {result['delta_point']:+.4f}  95% CI [{result['delta_ci_low']:+.4f}, {result['delta_ci_high']:+.4f}]")
        print(f"  p_two_sided = {result['p_two_sided']:.4f}  →  {verdict}")
        print(f"  (n_resamples_valid: {result['n_resamples_valid']}/{result['n_resamples_total']})")

        # Strip large array before json save
        summary[pair_key] = {k: v for k, v in result.items() if k != "deltas_all"}

    # Save summary
    summary_path = out_dir / "subject_paired_bootstrap.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Final compact verdict table
    print("\n\n=== COMPACT VERDICT TABLE ===\n")
    print(f"{'Comparison':40s}  {'Δ AUC':>8s}  {'95% CI on Δ':>22s}  {'p_two':>8s}  Verdict")
    for pair_key, r in summary.items():
        ci = f"[{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}]"
        p = r["p_two_sided"]
        verdict = "✓" if p < 0.05 else "✗"
        print(f"{pair_key:40s}  {r['delta_point']:+8.4f}  {ci:>22s}  {p:>8.4f}  {verdict}")

    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
