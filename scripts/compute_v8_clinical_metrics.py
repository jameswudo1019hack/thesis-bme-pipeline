"""Clinical-fidelity metrics for the v8a XGBoost apnoea-detection model.

Loads per-epoch test predictions, aggregates to per-subject AHI proxies,
then compares them against:
  (a) the annotated epoch-based AHI proxy (training target fidelity), and
  (b) the clinical AHI from the official SHHS1 dataset (ahi_a0h3a).

Outputs (written to models/aim2_v8_xgboost/clinical_metrics/):
  per_subject_clinical.parquet   — one row per test subject
  bland_altman_proxy.parquet     — B-A table: pred_ahi vs true_ahi_proxy
  bland_altman_clinical.parquet  — B-A table: pred_ahi vs clinical_ahi
  severity_confmat_proxy.csv     — 4x4 confusion matrix (proxy truth)
  severity_confmat_clinical.csv  — 4x4 confusion matrix (clinical truth)
  summary.json                   — top-level scalars

Usage:
    python scripts/compute_v8_clinical_metrics.py
    python scripts/compute_v8_clinical_metrics.py --help
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = CODE_ROOT / "models" / "aim2_v8_xgboost" / "test_predictions.parquet"
META_PATH = CODE_ROOT / "features" / "subject_metadata.parquet"
SHHS_CSV = Path(
    "/Users/jameswu/Desktop/University/Year_5/Thesis/Dataset/shhs/csv/shhs1-dataset-0.15.0.csv"
)
OUT_DIR = CODE_ROOT / "models" / "aim2_v8_xgboost" / "clinical_metrics"

# ---------------------------------------------------------------------------
# Severity-class binning
# ---------------------------------------------------------------------------
SEVERITY_BINS = [0, 5, 15, 30, np.inf]
SEVERITY_LABELS = ["Normal (<5)", "Mild (5-15)", "Moderate (15-30)", "Severe (>=30)"]


def severity_class(ahi_series: pd.Series) -> pd.Categorical:
    """Bin AHI values into the four standard OSA severity classes."""
    return pd.cut(
        ahi_series,
        bins=SEVERITY_BINS,
        labels=SEVERITY_LABELS,
        right=False,
        include_lowest=True,
    )


# ---------------------------------------------------------------------------
# Bland-Altman helpers
# ---------------------------------------------------------------------------

def bland_altman_table(subj_df: pd.DataFrame, col_ref: str, col_meas: str) -> pd.DataFrame:
    """Return per-subject Bland-Altman data: mean of pair + difference."""
    ba = pd.DataFrame(
        {
            "subject_id": subj_df["subject_id"],
            "mean_of_pair": (subj_df[col_ref] + subj_df[col_meas]) / 2.0,
            "difference": subj_df[col_meas] - subj_df[col_ref],
        }
    ).reset_index(drop=True)
    return ba


def bland_altman_stats(ba: pd.DataFrame) -> dict:
    """Aggregate Bland-Altman statistics: mean diff + 1.96 SD limits of agreement."""
    mean_diff = float(ba["difference"].mean())
    std_diff = float(ba["difference"].std(ddof=1))
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
    }


# ---------------------------------------------------------------------------
# Confusion matrix helpers
# ---------------------------------------------------------------------------

def severity_confusion_matrix(truth_ahi: pd.Series, pred_ahi: pd.Series) -> pd.DataFrame:
    """Build a 4x4 severity confusion matrix (rows=truth, cols=prediction)."""
    truth_cls = severity_class(truth_ahi)
    pred_cls = severity_class(pred_ahi)
    cm = pd.crosstab(
        truth_cls,
        pred_cls,
        rownames=["Truth"],
        colnames=["Prediction"],
        dropna=False,
    )
    # Reindex to ensure all four classes appear even if absent
    cm = cm.reindex(index=SEVERITY_LABELS, columns=SEVERITY_LABELS, fill_value=0)
    return cm


def format_confusion_matrix(cm: pd.DataFrame, title: str) -> str:
    """Return a nicely formatted string for stdout display."""
    lines = [f"\n{title}"]
    lines.append(f"{'':30s}" + "  ".join(f"{c:>17s}" for c in cm.columns))
    for idx_label in cm.index:
        row_vals = "  ".join(f"{cm.loc[idx_label, c]:>17d}" for c in cm.columns)
        lines.append(f"  {idx_label:28s}{row_vals}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--pred-path",
    type=click.Path(exists=True, path_type=Path),
    default=PRED_PATH,
    show_default=True,
    help="Path to test_predictions.parquet.",
)
@click.option(
    "--meta-path",
    type=click.Path(exists=True, path_type=Path),
    default=META_PATH,
    show_default=True,
    help="Path to subject_metadata.parquet.",
)
@click.option(
    "--shhs-csv",
    type=click.Path(exists=True, path_type=Path),
    default=SHHS_CSV,
    show_default=True,
    help="Path to shhs1-dataset-*.csv (must contain nsrrid and ahi_a0h3a).",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=OUT_DIR,
    show_default=True,
    help="Output directory for clinical_metrics artefacts.",
)
def main(
    pred_path: Path,
    meta_path: Path,
    shhs_csv: Path,
    out_dir: Path,
) -> None:
    """Compute clinical-fidelity metrics for the v8a XGBoost predictions."""

    click.echo("=" * 70)
    click.echo("  v8a XGBoost — Clinical-Fidelity Metrics")
    click.echo("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load predictions
    # ------------------------------------------------------------------
    click.echo(f"\n[1/5] Loading predictions from {pred_path} ...")
    pred = pd.read_parquet(pred_path)
    click.echo(f"      Loaded {len(pred):,} rows")

    required_pred_cols = {"subject_id", "epoch_idx", "apnoea_label", "pred_prob", "pred_label"}
    missing = required_pred_cols - set(pred.columns)
    if missing:
        click.echo(
            f"ERROR: test_predictions.parquet is missing expected columns: {missing}\n"
            f"       Actual columns: {list(pred.columns)}",
            err=True,
        )
        sys.exit(1)

    # subject_id in predictions is int32 (e.g. 200009); convert to shhs1-XXXXXX
    pred["subject_id_str"] = "shhs1-" + pred["subject_id"].astype(str)

    n_pred_subjects = pred["subject_id_str"].nunique()
    click.echo(f"      Unique test subjects in predictions: {n_pred_subjects:,}")

    # ------------------------------------------------------------------
    # 2. Load subject metadata
    # ------------------------------------------------------------------
    click.echo(f"\n[2/5] Loading subject metadata from {meta_path} ...")
    meta = pd.read_parquet(meta_path)
    click.echo(f"      Loaded {len(meta):,} rows")

    # Keep only metadata for test subjects
    meta_test = meta[meta["subject_id"].isin(pred["subject_id_str"].unique())].copy()
    click.echo(f"      Test subjects found in metadata: {len(meta_test):,}")

    # ------------------------------------------------------------------
    # 3. Load SHHS CSV — pull nsrrid + ahi_a0h3a
    # ------------------------------------------------------------------
    click.echo(f"\n[3/5] Loading SHHS CSV from {shhs_csv} ...")
    # Read full CSV but only the two columns we need
    shhs_df = pd.read_csv(shhs_csv, usecols=["nsrrid", "ahi_a0h3a"])
    click.echo(f"      Loaded {len(shhs_df):,} rows from CSV")

    # Normalise to shhs1-XXXXXX string for joining
    shhs_df["subject_id"] = "shhs1-" + shhs_df["nsrrid"].astype(int).astype(str)
    shhs_test = shhs_df[shhs_df["subject_id"].isin(pred["subject_id_str"].unique())].copy()
    click.echo(f"      Test subjects found in SHHS CSV: {len(shhs_test):,}")

    # ------------------------------------------------------------------
    # 4. Per-subject aggregation of predicted and true epoch counts
    # ------------------------------------------------------------------
    click.echo("\n[4/5] Aggregating per-subject counts ...")

    agg = (
        pred.groupby("subject_id_str", sort=False)
        .agg(
            pred_apnoea_epochs=("pred_label", lambda s: (s == 1).sum()),
            true_apnoea_epochs=("apnoea_label", lambda s: (s == 1).sum()),
        )
        .reset_index()
        .rename(columns={"subject_id_str": "subject_id"})
    )

    # Join metadata (for tst_min)
    agg = agg.merge(
        meta_test[["subject_id", "tst_min", "n_apnoea_epochs"]],
        on="subject_id",
        how="left",
    )

    # Join SHHS clinical AHI
    agg = agg.merge(
        shhs_test[["subject_id", "ahi_a0h3a"]].rename(columns={"ahi_a0h3a": "clinical_ahi"}),
        on="subject_id",
        how="left",
    )

    # ------------------------------------------------------------------
    # 5. Drop subjects with invalid TST; warn if any
    # ------------------------------------------------------------------
    n_before_tst_drop = len(agg)
    agg = agg[agg["tst_min"].notna() & (agg["tst_min"] > 0)].copy()
    n_dropped_tst = n_before_tst_drop - len(agg)
    if n_dropped_tst > 0:
        click.echo(
            f"      WARNING: dropped {n_dropped_tst} subject(s) with NaN or zero tst_min"
        )

    # Compute AHI proxies (events per hour of sleep)
    agg["tst_hr"] = agg["tst_min"] / 60.0
    agg["pred_ahi_proxy"] = agg["pred_apnoea_epochs"] / agg["tst_hr"]
    agg["true_ahi_proxy"] = agg["true_apnoea_epochs"] / agg["tst_hr"]

    # Subjects with valid clinical AHI (for clinical comparisons)
    agg_clinical = agg[agg["clinical_ahi"].notna()].copy()
    n_dropped_clinical = len(agg) - len(agg_clinical)
    if n_dropped_clinical > 0:
        click.echo(
            f"      WARNING: {n_dropped_clinical} subject(s) dropped from clinical "
            f"comparisons (NaN clinical_ahi)"
        )

    n_total = len(agg)
    n_clinical = len(agg_clinical)
    click.echo(f"      Subjects retained for proxy comparisons: {n_total:,}")
    click.echo(f"      Subjects retained for clinical comparisons: {n_clinical:,}")

    # Sanity check: cross-check true_apnoea_epochs vs n_apnoea_epochs in metadata
    mismatch = (agg["true_apnoea_epochs"] != agg["n_apnoea_epochs"]).sum()
    if mismatch > 0:
        click.echo(
            f"      WARNING: {mismatch} subject(s) have true_apnoea_epochs != "
            f"n_apnoea_epochs from metadata (epoch filtering may differ)"
        )
    else:
        click.echo("      Cross-check PASSED: true_apnoea_epochs matches metadata n_apnoea_epochs for all subjects")

    # ------------------------------------------------------------------
    # 6. Compute metrics
    # ------------------------------------------------------------------
    click.echo("\n[5/5] Computing metrics ...")

    # --- Proxy comparison (pred vs annotated) ---
    mae_proxy = float(np.abs(agg["pred_ahi_proxy"] - agg["true_ahi_proxy"]).mean())

    ba_proxy = bland_altman_table(agg, col_ref="true_ahi_proxy", col_meas="pred_ahi_proxy")
    ba_proxy_stats = bland_altman_stats(ba_proxy)

    cm_proxy = severity_confusion_matrix(agg["true_ahi_proxy"], agg["pred_ahi_proxy"])
    accuracy_proxy = float(
        np.diag(cm_proxy.values).sum() / cm_proxy.values.sum()
    )

    # --- Clinical comparison (pred vs clinical AHI) ---
    mae_clinical = float(np.abs(agg_clinical["pred_ahi_proxy"] - agg_clinical["clinical_ahi"]).mean())

    ba_clinical = bland_altman_table(agg_clinical, col_ref="clinical_ahi", col_meas="pred_ahi_proxy")
    ba_clinical_stats = bland_altman_stats(ba_clinical)

    cm_clinical = severity_confusion_matrix(agg_clinical["clinical_ahi"], agg_clinical["pred_ahi_proxy"])
    accuracy_clinical = float(
        np.diag(cm_clinical.values).sum() / cm_clinical.values.sum()
    )

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    # per_subject_clinical.parquet
    out_subject = out_dir / "per_subject_clinical.parquet"
    agg.to_parquet(out_subject, index=False)
    click.echo(f"\n  Saved: {out_subject}")

    # bland_altman_proxy.parquet
    out_ba_proxy = out_dir / "bland_altman_proxy.parquet"
    ba_proxy.to_parquet(out_ba_proxy, index=False)
    click.echo(f"  Saved: {out_ba_proxy}")

    # bland_altman_clinical.parquet
    out_ba_clinical = out_dir / "bland_altman_clinical.parquet"
    ba_clinical.to_parquet(out_ba_clinical, index=False)
    click.echo(f"  Saved: {out_ba_clinical}")

    # severity_confmat_proxy.csv
    out_cm_proxy = out_dir / "severity_confmat_proxy.csv"
    cm_proxy.to_csv(out_cm_proxy)
    click.echo(f"  Saved: {out_cm_proxy}")

    # severity_confmat_clinical.csv
    out_cm_clinical = out_dir / "severity_confmat_clinical.csv"
    cm_clinical.to_csv(out_cm_clinical)
    click.echo(f"  Saved: {out_cm_clinical}")

    # summary.json
    summary = {
        "n_test_subjects": n_total,
        "n_test_subjects_clinical": n_clinical,
        "mae_proxy": round(mae_proxy, 4),
        "mae_clinical": round(mae_clinical, 4),
        "mean_diff_proxy": round(ba_proxy_stats["mean_diff"], 4),
        "loa_proxy_lower": round(ba_proxy_stats["loa_lower"], 4),
        "loa_proxy_upper": round(ba_proxy_stats["loa_upper"], 4),
        "mean_diff_clinical": round(ba_clinical_stats["mean_diff"], 4),
        "loa_clinical_lower": round(ba_clinical_stats["loa_lower"], 4),
        "loa_clinical_upper": round(ba_clinical_stats["loa_upper"], 4),
        "severity_accuracy_proxy": round(accuracy_proxy, 4),
        "severity_accuracy_clinical": round(accuracy_clinical, 4),
    }
    out_summary = out_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    click.echo(f"  Saved: {out_summary}")

    # ------------------------------------------------------------------
    # 8. Stdout summary
    # ------------------------------------------------------------------
    click.echo("\n" + "=" * 70)
    click.echo("  SUMMARY")
    click.echo("=" * 70)
    click.echo(f"  Test subjects (proxy comparisons): {n_total:,}")
    click.echo(f"  Test subjects (clinical comparisons): {n_clinical:,}")
    click.echo()
    click.echo("  AHI PROXY vs ANNOTATED (prediction vs annotation rule):")
    click.echo(f"    MAE                     : {mae_proxy:.4f} events/hr")
    click.echo(f"    Mean difference         : {ba_proxy_stats['mean_diff']:.4f} events/hr")
    click.echo(f"    Limits of agreement     : [{ba_proxy_stats['loa_lower']:.4f}, {ba_proxy_stats['loa_upper']:.4f}]")
    click.echo(f"    Severity accuracy       : {accuracy_proxy:.4f}")
    click.echo()
    click.echo("  AHI PROXY vs CLINICAL (prediction vs ahi_a0h3a):")
    click.echo(f"    MAE                     : {mae_clinical:.4f} events/hr")
    click.echo(f"    Mean difference         : {ba_clinical_stats['mean_diff']:.4f} events/hr")
    click.echo(f"    Limits of agreement     : [{ba_clinical_stats['loa_lower']:.4f}, {ba_clinical_stats['loa_upper']:.4f}]")
    click.echo(f"    Severity accuracy       : {accuracy_clinical:.4f}")

    # Confusion matrices
    click.echo(
        format_confusion_matrix(
            cm_proxy,
            "  SEVERITY CONFUSION MATRIX — Proxy (rows=annotated, cols=predicted):",
        )
    )
    click.echo(
        format_confusion_matrix(
            cm_clinical,
            "  SEVERITY CONFUSION MATRIX — Clinical (rows=ahi_a0h3a, cols=predicted):",
        )
    )

    click.echo("\n" + "=" * 70)
    click.echo("  Done.")
    click.echo("=" * 70)


if __name__ == "__main__":
    main()
