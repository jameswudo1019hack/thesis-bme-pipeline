"""Subject-paired bootstrap test: v6 past-only vs v6 future-inclusive vs v7 no-context.

Pre-registration: see [[Experiments/2026-04-30 - v6 past-only context
(pre-registration)]]. Two paired comparisons:

  past-only vs v7 (no-context)        — does past context help vs no context?
  past-only vs v6 (future-inclusive)  — does past context match future context?

Subject-level bootstrap (1000 resamples) on the same test rows aligned by
(subject_id, epoch_idx). Inner-join on rows present in both predictions to
guarantee paired sampling.

Output: Code/models/aim2_v6_past_only/_paired_bootstrap.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

CODE_ROOT = Path(__file__).resolve().parents[1]
MODELS = CODE_ROOT / "models"

PAST = MODELS / "aim2_v6_past_only" / "test_predictions.parquet"
V6 = MODELS / "aim2_cv_v6" / "test_predictions.parquet"
V7 = MODELS / "aim2_cv_v7" / "test_predictions.parquet"


def _load(p: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_parquet(p)[["subject_id", "epoch_idx", "apnoea_label", "pred_prob"]].copy()
    df = df.rename(columns={"pred_prob": f"pred_prob_{suffix}"})
    return df


def _paired_boot(merged: pd.DataFrame, col_a: str, col_b: str, label: str = "apnoea_label",
                 n: int = 1000, seed: int = 42) -> dict:
    """Δ AUC = AUC_a - AUC_b over subject-level resamples. Two-sided percentile p."""
    rng = np.random.default_rng(seed)
    sids = merged["subject_id"].values
    y = merged[label].values.astype(np.int8)
    pa = merged[col_a].values
    pb = merged[col_b].values

    subjects = np.unique(sids)
    by_sub = {int(s): np.where(sids == s)[0] for s in subjects}
    sub_arr = np.asarray(list(by_sub.keys()), dtype=np.int64)

    auc_a_full = float(roc_auc_score(y, pa))
    auc_b_full = float(roc_auc_score(y, pb))

    deltas = np.full(n, np.nan)
    for i in range(n):
        sampled = rng.choice(sub_arr, size=len(sub_arr), replace=True)
        idx = np.concatenate([by_sub[int(s)] for s in sampled])
        ya = y[idx]
        if len(np.unique(ya)) < 2:
            continue
        deltas[i] = roc_auc_score(ya, pa[idx]) - roc_auc_score(ya, pb[idx])

    valid = deltas[np.isfinite(deltas)]
    ci_low, ci_high = np.percentile(valid, [2.5, 97.5])
    p_low = float((valid <= 0).mean())
    p_high = float((valid >= 0).mean())
    return {
        "auc_a": auc_a_full,
        "auc_b": auc_b_full,
        "delta": auc_a_full - auc_b_full,
        "delta_mean_boot": float(valid.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_two_sided": 2 * min(p_low, p_high),
        "n_valid": int(len(valid)),
        "n_pairs": int(len(merged)),
        "n_subjects": int(len(subjects)),
    }


def main() -> None:
    if not PAST.exists():
        raise SystemExit(f"past-only predictions not found at {PAST}")

    past = _load(PAST, "past")
    v6 = _load(V6, "v6")
    v7 = _load(V7, "v7")

    # past-only vs v6
    m_v6 = past.merge(v6.drop(columns=["apnoea_label"]), on=["subject_id", "epoch_idx"], how="inner")
    print(f"\npast-only ∩ v6: {len(m_v6):,} pairs ({m_v6['subject_id'].nunique()} subjects)")
    r_past_vs_v6 = _paired_boot(m_v6, "pred_prob_past", "pred_prob_v6")

    # past-only vs v7
    m_v7 = past.merge(v7.drop(columns=["apnoea_label"]), on=["subject_id", "epoch_idx"], how="inner")
    print(f"past-only ∩ v7: {len(m_v7):,} pairs ({m_v7['subject_id'].nunique()} subjects)")
    r_past_vs_v7 = _paired_boot(m_v7, "pred_prob_past", "pred_prob_v7")

    print(f"\n{'Comparison':30s}  {'AUC_a':>7s}  {'AUC_b':>7s}  {'Δ':>+10s}  {'95 % CI':>22s}  {'p (2-sided)':>11s}")
    print("-" * 95)
    for label, r in (("past-only vs v6 (future-incl)", r_past_vs_v6),
                     ("past-only vs v7 (no-context)", r_past_vs_v7)):
        ci = f"[{r['ci_low']:+7.4f}, {r['ci_high']:+7.4f}]"
        sig = "***" if r["p_two_sided"] < 0.001 else ("**" if r["p_two_sided"] < 0.01 else ("*" if r["p_two_sided"] < 0.05 else "ns"))
        print(f"{label:30s}  {r['auc_a']:>7.4f}  {r['auc_b']:>7.4f}  {r['delta']:>+10.4f}  {ci:>22s}  {r['p_two_sided']:>10.4g} {sig}")

    out = MODELS / "aim2_v6_past_only" / "_paired_bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "past_vs_v6_future_inclusive": r_past_vs_v6,
        "past_vs_v7_no_context": r_past_vs_v7,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
