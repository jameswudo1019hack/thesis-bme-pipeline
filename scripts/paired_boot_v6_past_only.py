"""Subject-paired bootstrap: 2×2 factorial of (schema × context) on the v6 question.

Four matched runs, all on the same 5,793-subject cohort + seed=42 + FIXED_PARAMS:

  aim2_v6_past_only         (audit-v6, past-only)         — 234 cols
  aim2_v6_future_inclusive  (audit-v6, future-inclusive)  — 273 cols
  aim2_v6_past_only_strict  (strict-v6, past-only)        — 174 cols
  aim2_v6_future_inclusive_strict (strict-v6, future-incl) — 203 cols

Six paired-bootstrap comparisons (1000 resamples each):
  Context effect on each schema:
    past_audit  vs  future_audit
    past_strict vs  future_strict
  Schema effect at each context level:
    past_audit  vs  past_strict
    future_audit vs  future_strict
  Cross (lineage check, with caveat that aim2_cv_v6/v7 are different cohort):
    past_audit  vs  aim2_cv_v6
    past_strict vs  aim2_cv_v7

Output: Code/models/aim2_v6_past_only/_paired_bootstrap.json (canonical home).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CODE_ROOT = Path(__file__).resolve().parents[1]
MODELS = CODE_ROOT / "models"

RUNS = {
    "past_audit":      MODELS / "aim2_v6_past_only" / "test_predictions.parquet",
    "future_audit":    MODELS / "aim2_v6_future_inclusive" / "test_predictions.parquet",
    "past_strict":     MODELS / "aim2_v6_past_only_strict" / "test_predictions.parquet",
    "future_strict":   MODELS / "aim2_v6_future_inclusive_strict" / "test_predictions.parquet",
    "aim2_cv_v6":      MODELS / "aim2_cv_v6" / "test_predictions.parquet",
    "aim2_cv_v7":      MODELS / "aim2_cv_v7" / "test_predictions.parquet",
}

# (a, b) — Δ = AUC_a - AUC_b. Positive Δ means model a beats model b.
COMPARISONS = [
    ("future_audit",  "past_audit",   "context effect (audit-v6 schema)"),
    ("future_strict", "past_strict",  "context effect (strict-v6 schema)"),
    ("past_audit",    "past_strict",  "schema effect (past-only)"),
    ("future_audit",  "future_strict","schema effect (future-inclusive)"),
    ("past_audit",    "aim2_cv_v6",   "lineage: past-only audit-v6 vs original v6 ⚠ cross-cohort"),
    ("past_strict",   "aim2_cv_v7",   "lineage: past-only strict-v6 vs original v7 ⚠ cross-cohort"),
]


def _load(p: Path, suffix: str) -> pd.DataFrame:
    df = pd.read_parquet(p)[["subject_id", "epoch_idx", "apnoea_label", "pred_prob"]].copy()
    return df.rename(columns={"pred_prob": f"pred_prob_{suffix}"})


def _paired_boot(merged: pd.DataFrame, col_a: str, col_b: str, n: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    sids = merged["subject_id"].values
    y = merged["apnoea_label"].values.astype(np.int8)
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
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        deltas[i] = roc_auc_score(yi, pa[idx]) - roc_auc_score(yi, pb[idx])

    valid = deltas[np.isfinite(deltas)]
    ci_low, ci_high = np.percentile(valid, [2.5, 97.5])
    p_low = float((valid <= 0).mean())
    p_high = float((valid >= 0).mean())
    return {
        "auc_a": auc_a_full, "auc_b": auc_b_full,
        "delta": auc_a_full - auc_b_full,
        "delta_mean_boot": float(valid.mean()),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "p_two_sided": 2 * min(p_low, p_high),
        "n_valid": int(len(valid)),
        "n_pairs": int(len(merged)), "n_subjects": int(len(subjects)),
    }


def main() -> None:
    loaded = {}
    for tag, path in RUNS.items():
        if path.exists():
            loaded[tag] = _load(path, tag)
            print(f"  ✓ {tag:20s} ({len(loaded[tag]):,} rows)")
        else:
            print(f"  ✗ {tag:20s} MISSING at {path}")

    results: dict = {}
    print(f"\n{'Comparison':50s}  {'AUC_a':>7s}  {'AUC_b':>7s}  {'Δ AUC':>10s}  {'95 % CI':>22s}  {'p (2s)':>9s}  {'n_pairs':>10s}")
    print("-" * 130)

    for a_tag, b_tag, label in COMPARISONS:
        if a_tag not in loaded or b_tag not in loaded:
            print(f"  ! skipped {a_tag} vs {b_tag} (missing predictions)")
            continue
        a, b = loaded[a_tag], loaded[b_tag]
        merged = a.merge(b.drop(columns=["apnoea_label"]), on=["subject_id", "epoch_idx"], how="inner")
        r = _paired_boot(merged, f"pred_prob_{a_tag}", f"pred_prob_{b_tag}")
        sig = "***" if r["p_two_sided"] < 0.001 else ("**" if r["p_two_sided"] < 0.01 else ("*" if r["p_two_sided"] < 0.05 else "ns"))
        ci = f"[{r['ci_low']:+7.4f}, {r['ci_high']:+7.4f}]"
        tag = f"{a_tag} vs {b_tag}"
        print(f"{tag:50s}  {r['auc_a']:>7.4f}  {r['auc_b']:>7.4f}  {r['delta']:>+10.4f}  {ci:>22s}  {r['p_two_sided']:>8.4g} {sig}  {r['n_pairs']:>10,}")
        results[f"{a_tag}_vs_{b_tag}"] = {**r, "label": label}

    out = MODELS / "aim2_v6_past_only" / "_paired_bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
