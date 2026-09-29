"""Subject-paired bootstrap tests for the sleep-filter recovery batch (2026-05-03).

Operates on the frozen recovery snapshot at:
    Code/models/recovery_2026-05-03/

For each pre-registered comparison, resamples subjects 1000× (seed=42),
computes Δ AUC = AUC_a - AUC_b per resample, and reports a percentile-based
two-sided p-value for H0: Δ AUC = 0.

Comparisons (3 mandatory + 4 supporting):
  1. AASM-rule tautology      — v85_taxonomy/full vs physio_only
  2. PSD significance         — phase1_batch/physio_only/exp2 vs exp1
  3. Schema dominance         — v6_past_only vs v6_past_only_strict
  4. Context effect           — v6_future_inclusive vs v6_past_only
  5. TabNet vs LGBM (physio)  — v85_taxonomy/physio_only vs phase1_exp6_tabnet/physio_only
  6. GBDT family equivalence  — v8_fixedhp/lightgbm vs xgboost
  7. Ensemble lift            — v8_fixedhp/ensemble vs lightgbm

Output:
  Code/models/recovery_2026-05-03/_paired_tests.json
  Code/models/recovery_2026-05-03/_paired_test_deltas/<key>.npy

Usage:
  cd Code
  python3 scripts/recovery_paired_bootstrap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CODE_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = CODE_ROOT / "models" / "recovery_2026-05-03"

# (key, path_a_relative, path_b_relative, hypothesis_summary)
COMPARISONS: list[tuple[str, str, str, str]] = [
    (
        "aasm_rule_tautology",
        "aim2_v85_taxonomy/full",
        "aim2_v85_taxonomy/physio_only",
        "AASM-rule features dominate (pre-reg Δ ≥ +0.015)",
    ),
    (
        "psd_significance_physio",
        "aim2_phase1_batch/physio_only/exp2",
        "aim2_phase1_batch/physio_only/exp1",
        "SpO2 PSD adds info on top of v6 physio baseline (Δ > 0)",
    ),
    (
        "schema_dominance_pastonly",
        "aim2_v6_past_only",
        "aim2_v6_past_only_strict",
        "Audit-v6 schema (Sprint 1 cols) beats strict v6 schema, past-only context",
    ),
    (
        "context_effect_audit",
        "aim2_v6_future_inclusive",
        "aim2_v6_past_only",
        "Future-inclusive context > past-only on audit-v6 schema (look-ahead leak)",
    ),
    (
        "tabnet_vs_lgbm_physio",
        "aim2_v85_taxonomy/physio_only",
        "aim2_phase1_exp6_tabnet/physio_only",
        "On honest physiology, LGBM ≈ TabNet (small Δ expected)",
    ),
    (
        "gbdt_family_equivalence",
        "aim2_v8_fixedhp_sleep/lightgbm",
        "aim2_v8_fixedhp_sleep/xgboost",
        "Top GBDTs interchangeable (Δ ≈ 0)",
    ),
    (
        "ensemble_lift_over_lgbm",
        "aim2_v8_fixedhp_sleep/ensemble",
        "aim2_v8_fixedhp_sleep/lightgbm",
        "Ensemble adds little over best single GBDT (Δ ≈ 0)",
    ),
]


def load_aligned(rel_a: str, rel_b: str) -> pd.DataFrame:
    """Load both models' test_predictions.parquet, sort, verify alignment."""
    pa = SNAPSHOT / rel_a / "test_predictions.parquet"
    pb = SNAPSHOT / rel_b / "test_predictions.parquet"
    if not pa.exists():
        raise FileNotFoundError(f"Missing predictions: {pa}")
    if not pb.exists():
        raise FileNotFoundError(f"Missing predictions: {pb}")

    df_a = pd.read_parquet(pa).sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True)
    df_b = pd.read_parquet(pb).sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True)

    key_cols = ["subject_id", "epoch_idx", "apnoea_label"]
    if not df_a[key_cols].equals(df_b[key_cols]):
        raise ValueError(
            f"Test rows not aligned between {rel_a} and {rel_b}. "
            f"Both should be on the same seed=42 patient-level test split."
        )

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
    """Subject-level paired bootstrap on Δ AUC = AUC(p_a) - AUC(p_b).

    Returns point estimates, 95% percentile CI, percentile two-sided p-value,
    and the full bootstrap delta array.
    """
    rng = np.random.default_rng(seed)

    sids = df["subject_id"].values
    y_all = df["y"].values
    pa_all = df["p_a"].values
    pb_all = df["p_b"].values

    subjects = np.unique(sids)
    n_subj = len(subjects)
    by_subject = {int(s): np.where(sids == s)[0] for s in subjects}
    subjects_int = np.asarray(list(by_subject.keys()), dtype=np.int64)

    auc_a_point = float(roc_auc_score(y_all, pa_all))
    auc_b_point = float(roc_auc_score(y_all, pb_all))
    delta_point = auc_a_point - auc_b_point

    deltas = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        sampled = rng.choice(subjects_int, size=n_subj, replace=True)
        idx = np.concatenate([by_subject[int(s)] for s in sampled])
        y = y_all[idx]
        if len(np.unique(y)) < 2:
            continue
        deltas[i] = roc_auc_score(y, pa_all[idx]) - roc_auc_score(y, pb_all[idx])

    valid = deltas[np.isfinite(deltas)]
    delta_lo = float(np.percentile(valid, 2.5))
    delta_hi = float(np.percentile(valid, 97.5))

    p_left = float(np.mean(valid <= 0.0))
    p_right = float(np.mean(valid >= 0.0))
    p_two_sided = 2.0 * min(p_left, p_right)

    return {
        "auc_a_point": auc_a_point,
        "auc_b_point": auc_b_point,
        "delta_point": delta_point,
        "delta_ci_low": delta_lo,
        "delta_ci_high": delta_hi,
        "p_two_sided": p_two_sided,
        "p_left": p_left,
        "p_right": p_right,
        "n_resamples_valid": int(len(valid)),
        "n_resamples_total": n_resamples,
        "deltas_all": deltas,
    }


@click.command()
@click.option("--n-resamples", type=int, default=1000, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def main(n_resamples: int, seed: int) -> None:
    if not SNAPSHOT.exists():
        raise SystemExit(f"Snapshot not found: {SNAPSHOT}")

    out_dir = SNAPSHOT
    deltas_dir = SNAPSHOT / "_paired_test_deltas"
    deltas_dir.mkdir(exist_ok=True)

    print(f"=== Recovery sleep-only paired bootstrap tests ===")
    print(f"  Snapshot: {SNAPSHOT}")
    print(f"  N resamples: {n_resamples}   seed: {seed}\n")

    summary: dict = {}

    for key, rel_a, rel_b, hypothesis in COMPARISONS:
        print(f"\n--- {key} ---")
        print(f"  A: {rel_a}")
        print(f"  B: {rel_b}")
        print(f"  H1: {hypothesis}")
        try:
            df = load_aligned(rel_a, rel_b)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ! SKIPPED: {exc}")
            summary[key] = {"skipped": True, "reason": str(exc), "path_a": rel_a, "path_b": rel_b}
            continue

        result = subject_paired_bootstrap(df, n_resamples=n_resamples, seed=seed)

        np.save(deltas_dir / f"{key}.npy", result["deltas_all"])

        verdict = "✓ significant" if result["p_two_sided"] < 0.05 else "✗ NOT significant (α=0.05)"
        print(f"  AUC_a (A) = {result['auc_a_point']:.4f}")
        print(f"  AUC_b (B) = {result['auc_b_point']:.4f}")
        print(f"  Δ AUC     = {result['delta_point']:+.4f}   "
              f"95% CI [{result['delta_ci_low']:+.4f}, {result['delta_ci_high']:+.4f}]")
        print(f"  p_two     = {result['p_two_sided']:.4f}   →   {verdict}")
        print(f"  (n_valid: {result['n_resamples_valid']}/{result['n_resamples_total']})")

        summary[key] = {
            "path_a": rel_a,
            "path_b": rel_b,
            "hypothesis": hypothesis,
            **{k: v for k, v in result.items() if k != "deltas_all"},
        }

    # Compact verdict table
    print("\n\n" + "=" * 100)
    print("COMPACT VERDICT TABLE")
    print("=" * 100)
    print(f"{'Comparison':30s}  {'AUC_a':>7s}  {'AUC_b':>7s}  {'Δ AUC':>9s}  "
          f"{'95% CI on Δ':>22s}  {'p_two':>8s}  v")
    print("-" * 100)
    for key, r in summary.items():
        if r.get("skipped"):
            print(f"{key:30s}  SKIPPED — {r['reason']}")
            continue
        ci = f"[{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}]"
        v = "✓" if r["p_two_sided"] < 0.05 else "✗"
        print(f"{key:30s}  {r['auc_a_point']:>7.4f}  {r['auc_b_point']:>7.4f}  "
              f"{r['delta_point']:>+9.4f}  {ci:>22s}  {r['p_two_sided']:>8.4f}  {v}")

    out_path = out_dir / "_paired_tests.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {out_path}")
    print(f"Per-comparison delta arrays: {deltas_dir}")


if __name__ == "__main__":
    main()
