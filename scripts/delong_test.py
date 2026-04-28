"""Paired DeLong AUC tests for the Aim 2 apnoea-detection models.

Runs four comparisons:
  v6 vs v7   — LightGBM contextual vs no-contextual features (inner-join on test set)
  v6 vs v8a  — LightGBM vs XGBoost (same test set)
  v8a vs v8b — XGBoost vs CatBoost (same test set)
  v6 / v8a / v8b three-way matrix

Algorithm: DeLong-Xu fast algorithm (Sun & Xu, 2014).  The structural
component (placements) are computed in O(n log n) using cumulative-sum
rank arithmetic rather than the naive O(n*m) double loop.  The covariance
matrix of the AUC estimator is then assembled from those placements, and
the z-statistic for the AUC difference is derived directly from the
off-diagonal covariance element.

Reference (algorithm guidance only):
  Sun & Xu (2014). Fast Implementation of DeLong's Algorithm for Comparing
  the Areas Under Correlated Receiver Operating Characteristic Curves.
  IEEE Signal Processing Letters 21(11):1389-1393.

Usage:
    python scripts/delong_test.py
    python scripts/delong_test.py --help
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PATHS = {
    "v6":  CODE_ROOT / "models" / "aim2_cv_v6"       / "test_predictions.parquet",
    "v7":  CODE_ROOT / "models" / "aim2_cv_v7"       / "test_predictions.parquet",
    "v8a": CODE_ROOT / "models" / "aim2_v8_xgboost"  / "test_predictions.parquet",
    "v8b": CODE_ROOT / "models" / "aim2_v8_catboost" / "test_predictions.parquet",
}
DEFAULT_OUT_DIR = CODE_ROOT / "models" / "delong_tests"
V8A_METRICS     = CODE_ROOT / "models" / "aim2_v8_xgboost" / "metrics.json"

# ---------------------------------------------------------------------------
# DeLong-Xu fast algorithm
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Return midranks (1-based) for array *x*.  Ties share the average rank.

    Works on a 1-D float array.  Complexity: O(n log n).
    """
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    # rank positions in the sorted order (0-based)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        # find the run of equal values
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        mid = (i + j - 1) / 2.0 + 1.0   # convert to 1-based midrank
        ranks[order[i:j]] = mid
        i = j
    return ranks


def _structural_components(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the structural component vectors V10 and V01 for one scorer.

    Parameters
    ----------
    predictions : shape (n_samples,)
    labels      : binary {0, 1}, shape (n_samples,)

    Returns
    -------
    V10 : shape (n_pos,)  — placement values for positive instances
    V01 : shape (n_neg,)  — placement values for negative instances

    Each V10[i] is the probability that the i-th positive case is ranked
    above a randomly chosen negative case (analogous to the Wilcoxon
    estimator of P(pos > neg)).

    Derivation: V10[i] = (rank of pos_i in the full combined ranking) - i
                         divided by n_neg.
    Similarly for V01.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    # Combined midranks
    all_ranks = _compute_midrank(predictions)
    pos_ranks = all_ranks[pos_mask]   # ranks of positive samples in full set
    neg_ranks = all_ranks[neg_mask]   # ranks of negative samples in full set

    # Midranks within positive sub-array and within negative sub-array
    # (needed to subtract the within-group rank to get placement values)
    pos_self_ranks = _compute_midrank(predictions[pos_mask])
    neg_self_ranks = _compute_midrank(predictions[neg_mask])

    # Placement values (see Sun & Xu 2014, Eq. 6–7)
    V10 = (pos_ranks - pos_self_ranks) / n_neg   # each pos vs all neg
    V01 = (neg_ranks - neg_self_ranks) / n_pos   # each neg vs all pos

    return V10, V01


def delong_auc_and_var(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Paired DeLong test: AUC(a), AUC(b), covariance matrix, and derived stats.

    All three arrays must have the same length (same paired observations with
    the same binary labels).

    Returns a dict with keys:
      auc_a, auc_b, auc_diff, delong_var,
      z_score, p_value,
      delta_auc_95ci_lower, delta_auc_95ci_upper
    """
    labels = np.asarray(labels, dtype=np.int8)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    n = len(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Labels must contain both classes.")
    if n != len(scores_a) or n != len(scores_b):
        raise ValueError("scores_a, scores_b, and labels must have the same length.")

    # Structural components for both scorers
    V10_a, V01_a = _structural_components(scores_a, labels)
    V10_b, V01_b = _structural_components(scores_b, labels)

    # AUC estimates (mean of placement values)
    auc_a = float(V10_a.mean())
    auc_b = float(V10_b.mean())

    # Covariance matrix of (AUC_a, AUC_b) — DeLong Eq. 5 / Sun & Xu Eq. 11
    # Cov matrix: [[S00, S01], [S10, S11]]
    # S_ij = (1/n_pos)*cov(V10_i, V10_j) + (1/n_neg)*cov(V01_i, V01_j)

    def _cov(u: np.ndarray, v: np.ndarray) -> float:
        # Population covariance (DeLong uses 1/n not 1/(n-1))
        # But variance of AUC is scaled by 1/n_pos + 1/n_neg per the theory;
        # use ddof=1 to match unbiased Wilcoxon estimate
        return float(np.cov(u, v, ddof=1)[0, 1])

    def _var(u: np.ndarray) -> float:
        return float(np.var(u, ddof=1))

    # Variance for each scorer and cross-covariance
    s_aa = _var(V10_a) / n_pos + _var(V01_a) / n_neg
    s_bb = _var(V10_b) / n_pos + _var(V01_b) / n_neg
    s_ab = _cov(V10_a, V10_b) / n_pos + _cov(V01_a, V01_b) / n_neg

    # Variance of (AUC_a - AUC_b)
    var_diff = s_aa + s_bb - 2.0 * s_ab

    if var_diff <= 0.0:
        # Numerically degenerate (e.g. identical predictions)
        z = 0.0
        p = 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var_diff)
        p = float(2.0 * (1.0 - norm.cdf(abs(z))))

    # 95 % CI for the difference auc_b - auc_a (i.e. lift of b over a)
    diff = auc_b - auc_a
    se = np.sqrt(var_diff)
    ci_lo = diff - 1.96 * se
    ci_hi = diff + 1.96 * se

    return {
        "auc_a":                auc_a,
        "auc_b":                auc_b,
        "auc_diff":             diff,        # auc_b - auc_a (positive = b wins)
        "delong_var":           float(var_diff),
        "z_score":              float(z),    # sign: positive if auc_a > auc_b
        "p_value":              p,
        "delta_auc_95ci_lower": float(ci_lo),
        "delta_auc_95ci_upper": float(ci_hi),
    }


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["subject_id"] = df["subject_id"].astype(np.int64)
    df["pred_prob"] = df["pred_prob"].astype(np.float64)
    return df


def inner_join_pair(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    keys: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two aligned DataFrames sharing the same (subject_id, epoch_idx) rows.

    Rows not present in both are dropped.  The two returned DataFrames are
    sorted by the same key order so row i in df_a corresponds to row i in df_b.
    """
    if keys is None:
        keys = ["subject_id", "epoch_idx"]

    merged = df_a[keys + ["apnoea_label", "pred_prob"]].merge(
        df_b[keys + ["pred_prob"]],
        on=keys,
        suffixes=("_a", "_b"),
        how="inner",
    )
    return merged


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def run_comparison(
    name: str,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    *,
    inner_join: bool = False,
) -> dict:
    """Run a single pairwise DeLong comparison and return a result record."""
    click.echo(f"\n  [{name}]  {label_a} vs {label_b}")

    if inner_join:
        merged = inner_join_pair(df_a, df_b)
        n_in_a  = len(df_a)
        n_in_b  = len(df_b)
        n_pairs = len(merged)
        dropped_a = n_in_a - n_pairs
        dropped_b = n_in_b - n_pairs
        click.echo(
            f"    inner-join: {n_in_a:,} (a) × {n_in_b:,} (b) → {n_pairs:,} pairs  "
            f"(dropped {dropped_a:,} from a, {dropped_b:,} from b)"
        )
        labels   = merged["apnoea_label"].to_numpy(np.int8)
        scores_a = merged["pred_prob_a"].to_numpy(np.float64)
        scores_b = merged["pred_prob_b"].to_numpy(np.float64)
    else:
        # Same test set — direct pair; verify labels agree
        if not (df_a["apnoea_label"].values == df_b["apnoea_label"].values).all():
            click.echo(
                "    WARNING: apnoea_label columns differ between the two frames — "
                "something is wrong.  Aborting this comparison.",
                err=True,
            )
            return {}
        n_pairs  = len(df_a)
        labels   = df_a["apnoea_label"].to_numpy(np.int8)
        scores_a = df_a["pred_prob"].to_numpy(np.float64)
        scores_b = df_b["pred_prob"].to_numpy(np.float64)
        click.echo(f"    direct pair: {n_pairs:,} rows")

    result = delong_auc_and_var(scores_a, scores_b, labels)
    result["comparison"] = name
    result["model_a"]    = label_a
    result["model_b"]    = label_b
    result["n_pairs"]    = n_pairs

    p = result["p_value"]
    z = result["z_score"]
    click.echo(
        f"    AUC_a={result['auc_a']:.6f}  AUC_b={result['auc_b']:.6f}  "
        f"Δ={result['auc_diff']:+.6f}  z={z:.3f}  p={p:.3e}"
    )
    if p < 0.001:
        click.echo("    ✓ significant at p<0.001")
    elif p < 0.05:
        click.echo(f"    ✓ significant at p<0.05  (p={p:.4f})")
    else:
        click.echo(f"    ✗ not significant  (p={p:.4f})")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--v6-path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_PATHS["v6"],
    show_default=True,
    help="test_predictions.parquet for v6 (LightGBM + contextual features).",
)
@click.option(
    "--v7-path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_PATHS["v7"],
    show_default=True,
    help="test_predictions.parquet for v7 (LightGBM, no contextual features).",
)
@click.option(
    "--v8a-path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_PATHS["v8a"],
    show_default=True,
    help="test_predictions.parquet for v8a (XGBoost).",
)
@click.option(
    "--v8b-path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_PATHS["v8b"],
    show_default=True,
    help="test_predictions.parquet for v8b (CatBoost).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUT_DIR,
    show_default=True,
    help="Output directory for delong_results.parquet and delong_summary.json.",
)
@click.option(
    "--sanity-tol",
    type=float,
    default=1e-4,
    show_default=True,
    help="Tolerance for the v8a self-AUC sanity check against metrics.json.",
)
def main(
    v6_path: Path,
    v7_path: Path,
    v8a_path: Path,
    v8b_path: Path,
    out_dir: Path,
    sanity_tol: float,
) -> None:
    """Paired DeLong AUC tests for the Aim 2 apnoea-detection models."""

    click.echo("=" * 72)
    click.echo("  Paired DeLong AUC Tests — Aim 2 Apnoea Detection")
    click.echo("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load predictions
    # ------------------------------------------------------------------
    click.echo("\n[1/4] Loading prediction parquets ...")
    df = {}
    for key, path in [("v6", v6_path), ("v7", v7_path), ("v8a", v8a_path), ("v8b", v8b_path)]:
        df[key] = load_parquet(path)
        click.echo(
            f"  {key:3s}: {len(df[key]):>10,} rows  "
            f"{df[key]['subject_id'].nunique():>5} subjects  "
            f"path={path}"
        )

    # ------------------------------------------------------------------
    # 2. Sanity check — v8a self-AUC vs saved metrics.json
    # ------------------------------------------------------------------
    click.echo("\n[2/4] Sanity check: v8a DeLong AUC vs saved metrics.json ...")
    saved_auc = json.loads(V8A_METRICS.read_text())["test_auc_roc"]
    labels_v8a  = df["v8a"]["apnoea_label"].to_numpy(np.int8)
    scores_v8a  = df["v8a"]["pred_prob"].to_numpy(np.float64)

    # Quick AUC estimate using the structural component mean
    V10_check, _ = _structural_components(scores_v8a, labels_v8a)
    computed_auc = float(V10_check.mean())

    diff_sanity = abs(computed_auc - saved_auc)
    click.echo(f"  DeLong-computed AUC : {computed_auc:.8f}")
    click.echo(f"  Saved test_auc_roc  : {saved_auc:.8f}")
    click.echo(f"  Absolute difference : {diff_sanity:.2e}  (tolerance={sanity_tol:.0e})")

    if diff_sanity > sanity_tol:
        click.echo(
            f"  ERROR: sanity check FAILED — AUC difference {diff_sanity:.2e} exceeds "
            f"tolerance {sanity_tol:.0e}.  Aborting.",
            err=True,
        )
        sys.exit(1)
    else:
        click.echo("  ✓ sanity check passed")

    # ------------------------------------------------------------------
    # 3. Run comparisons
    # ------------------------------------------------------------------
    click.echo("\n[3/4] Running pairwise DeLong tests ...")

    results = []

    # v6 vs v7 — inner-join because test sets differ
    r = run_comparison(
        name="v6_vs_v7",
        df_a=df["v6"],
        df_b=df["v7"],
        label_a="v6 (LightGBM+contextual)",
        label_b="v7 (LightGBM, no-contextual)",
        inner_join=True,
    )
    if r:
        results.append(r)

    # v6 vs v8a — same test set
    r = run_comparison(
        name="v6_vs_v8a",
        df_a=df["v6"],
        df_b=df["v8a"],
        label_a="v6 (LightGBM+contextual)",
        label_b="v8a (XGBoost)",
        inner_join=False,
    )
    if r:
        results.append(r)

    # v8a vs v8b — same test set
    r = run_comparison(
        name="v8a_vs_v8b",
        df_a=df["v8a"],
        df_b=df["v8b"],
        label_a="v8a (XGBoost)",
        label_b="v8b (CatBoost)",
        inner_join=False,
    )
    if r:
        results.append(r)

    # Three-way: v6 / v8a / v8b (all same test set)
    for pair_name, ka, kb, la, lb in [
        ("v6_vs_v8b",  "v6",  "v8b",
         "v6 (LightGBM+contextual)", "v8b (CatBoost)"),
    ]:
        r = run_comparison(
            name=pair_name,
            df_a=df[ka],
            df_b=df[kb],
            label_a=la,
            label_b=lb,
            inner_join=False,
        )
        if r:
            results.append(r)

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    click.echo("\n[4/4] Saving outputs ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build DataFrame — ordered columns
    cols_ordered = [
        "comparison", "model_a", "model_b", "n_pairs",
        "auc_a", "auc_b", "auc_diff",
        "z_score", "p_value", "delong_var",
        "delta_auc_95ci_lower", "delta_auc_95ci_upper",
    ]
    results_df = pd.DataFrame(results)[cols_ordered]

    parquet_out = out_dir / "delong_results.parquet"
    results_df.to_parquet(parquet_out, index=False)
    click.echo(f"  Saved: {parquet_out}")

    # JSON summary
    metadata = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "parquet_paths": {
            "v6":  str(v6_path),
            "v7":  str(v7_path),
            "v8a": str(v8a_path),
            "v8b": str(v8b_path),
        },
        "sanity_check": {
            "delong_auc_v8a": round(computed_auc, 8),
            "saved_test_auc_roc_v8a": round(saved_auc, 8),
            "abs_diff": round(diff_sanity, 10),
            "passed": bool(diff_sanity <= sanity_tol),
        },
    }
    summary = {
        "_metadata": metadata,
        "comparisons": [
            {k: (float(v) if isinstance(v, (np.floating, np.float64, float)) else
                 int(v) if isinstance(v, (np.integer, int)) else v)
             for k, v in row.items()}
            for row in results
        ],
    }
    json_out = out_dir / "delong_summary.json"
    json_out.write_text(json.dumps(summary, indent=2))
    click.echo(f"  Saved: {json_out}")

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    click.echo("\n" + "=" * 72)
    click.echo("  FINAL SUMMARY")
    click.echo("=" * 72)
    hdr = f"  {'Comparison':<16}  {'n_pairs':>10}  {'AUC_a':>8}  {'AUC_b':>8}  {'Δ AUC':>8}  {'z':>7}  {'p (2-sided)':>12}  Verdict"
    click.echo(hdr)
    click.echo("  " + "-" * (len(hdr) - 2))
    for row in results:
        p = row["p_value"]
        if p < 0.001:
            verdict = "✓ p<0.001"
        elif p < 0.05:
            verdict = f"✓ p={p:.4f}"
        else:
            verdict = f"✗ p={p:.4f}"
        click.echo(
            f"  {row['comparison']:<16}  {row['n_pairs']:>10,}  "
            f"{row['auc_a']:>8.6f}  {row['auc_b']:>8.6f}  "
            f"{row['auc_diff']:>+8.6f}  {row['z_score']:>7.2f}  "
            f"{p:>12.3e}  {verdict}"
        )
    click.echo("=" * 72)
    click.echo("  Done.")
    click.echo("=" * 72)


if __name__ == "__main__":
    main()
