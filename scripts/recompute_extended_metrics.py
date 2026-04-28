"""Retroactively populate extended Aim 2 metrics on already-saved predictions.

Use case: we ran v6 / v8 / v8.5 / 8.5-tax / Lit-2019 / Phase 1 Exp 5 / Exp 6
before the extended-metrics policy was adopted. Each model dir has a
``test_predictions.parquet`` saved. This script:

  1. Discovers model dirs under ``Code/models/`` (or a passed glob)
  2. For each: loads test_predictions.parquet, optionally subject_metadata.parquet
  3. Calls ``compute_extended_metrics()`` from ``thesis_pipeline.extended_metrics``
  4. Writes ``metrics_extended.json`` next to ``metrics.json`` (keeps original)
  5. Saves bootstrap arrays as .npy
  6. Prints a compact summary table

This is non-destructive — it adds files but does not modify existing
``metrics.json``. To merge old + new fields downstream, scripts can read both.

Usage:
  python scripts/recompute_extended_metrics.py \
      --models 'aim2_cv_v6' 'aim2_v8_xgboost' 'aim2_v85_taxonomy/full' ...

  python scripts/recompute_extended_metrics.py --all   # auto-discover all model dirs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.extended_metrics import compute_extended_metrics  # noqa: E402

MODELS_ROOT = CODE_ROOT / "models"
FEATURES_DIR = CODE_ROOT / "features"


def discover_model_dirs(models_root: Path) -> list[Path]:
    """Find every directory under models/ that contains test_predictions.parquet."""
    out: list[Path] = []
    for p in models_root.rglob("test_predictions.parquet"):
        out.append(p.parent)
    return sorted(out, key=lambda d: str(d))


def load_subject_metadata(features_dir: Path) -> pd.DataFrame | None:
    """Load subject_metadata.parquet from features dir if present."""
    f = features_dir / "subject_metadata.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    if "subject_id" in df.columns:
        # Ensure subject_id matches dtype with predictions (int)
        try:
            df["subject_id"] = df["subject_id"].astype(int)
        except (ValueError, TypeError):
            pass
    return df


def process_one(model_dir: Path, subject_metadata: pd.DataFrame | None) -> dict | None:
    pred_path = model_dir / "test_predictions.parquet"
    if not pred_path.exists():
        print(f"  ! {model_dir.name}: no test_predictions.parquet — skipping")
        return None

    test_pred = pd.read_parquet(pred_path)
    required = {"subject_id", "epoch_idx", "apnoea_label", "pred_prob", "pred_label"}
    missing = required - set(test_pred.columns)
    if missing:
        print(f"  ! {model_dir}: missing cols {missing} — skipping")
        return None

    print(f"\n  → {model_dir.relative_to(MODELS_ROOT)}  "
          f"(n={len(test_pred):,} epochs, {test_pred['subject_id'].nunique()} subj)")

    metrics = compute_extended_metrics(
        test_predictions=test_pred,
        subject_metadata=subject_metadata,
        n_bootstrap_subj=1000,
        seed=42,
    )

    # Strip the bootstrap arrays out of the JSON, save them as .npy
    aucs = np.asarray(metrics.pop("_bootstrap_aucs_subject"))
    auprs = np.asarray(metrics.pop("_bootstrap_auprs_subject"))
    np.save(model_dir / "bootstrap_aucs_subject.npy", aucs)
    np.save(model_dir / "bootstrap_auprs_subject.npy", auprs)

    # Write the extended metrics JSON
    out_path = model_dir / "metrics_extended.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    # Compact print
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and not np.isnan(v) else "—"
    print(
        f"    AUC-ROC {fmt(metrics['auc_roc'])} [{fmt(metrics['auc_roc_ci_low'])}, "
        f"{fmt(metrics['auc_roc_ci_high'])}]   "
        f"AUC-PR {fmt(metrics['auc_pr'])}   "
        f"F1 {fmt(metrics['f1'])}   Sens {fmt(metrics['sensitivity'])}   "
        f"Spec {fmt(metrics['specificity'])}   ECE {fmt(metrics['ece'])}"
    )
    if metrics.get("ahi_mae") is not None:
        print(
            f"    AHI MAE {fmt(metrics['ahi_mae'])}   "
            f"AHI corr {fmt(metrics['ahi_corr'])}   "
            f"Severity acc {fmt(metrics['severity_accuracy'])}   "
            f"Severity κ {fmt(metrics['severity_weighted_kappa'])}   "
            f"(n={metrics.get('n_subjects_with_nsrr_ahi', '?')})"
        )

    return metrics


@click.command()
@click.option("--models", multiple=True, help="Specific model subdirs (relative to Code/models/). Repeat for multiple.")
@click.option("--all", "all_models", is_flag=True, help="Auto-discover every model dir with test_predictions.parquet.")
@click.option("--features-dir", default=str(FEATURES_DIR), show_default=True,
              help="Path to features/ for loading subject_metadata.parquet")
def main(models: tuple[str, ...], all_models: bool, features_dir: str) -> None:
    print(f"\n=== Recompute extended metrics ===")
    print(f"Models root:  {MODELS_ROOT}")
    print(f"Features dir: {features_dir}\n")

    sm = load_subject_metadata(Path(features_dir))
    if sm is None:
        print("⚠ No subject_metadata.parquet — clinical metrics will be None for all models.")
    else:
        print(f"✓ subject_metadata.parquet: {len(sm)} subjects, "
              f"{'ahi_a0h3a' in sm.columns} = ahi_a0h3a present")

    if all_models:
        dirs = discover_model_dirs(MODELS_ROOT)
    elif models:
        dirs = [MODELS_ROOT / m for m in models]
    else:
        print("\n! Provide --all or --models <subdir> [<subdir> ...]")
        sys.exit(1)

    print(f"\nProcessing {len(dirs)} model dir(s):")
    for d in dirs:
        process_one(d, sm)

    print(f"\n=== Done — wrote metrics_extended.json + bootstrap_*_subject.npy in each dir ===")


if __name__ == "__main__":
    main()
