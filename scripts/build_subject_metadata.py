"""Build per-subject metadata table from feature parquets.

Scans ``features/*.parquet`` and computes per-subject TST / Sleep Efficiency
/ WASO / AHI proxy / hypoxic burden per night using
``thesis_pipeline.epochs.subject_metadata``. Optionally joins NSRR
harmonised variables (``nsrr_total_sleep_time``, ``nsrr_sleep_efficiency``,
``nsrr_waso``) from the SHHS CSV for cross-validation; flags rows where
our computed value differs from NSRR by > 5%.

Usage:
    cd Code && python scripts/build_subject_metadata.py [--nsrr-csv PATH]
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

# Add repo root to path for thesis_pipeline import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesis_pipeline.epochs import subject_metadata  # noqa: E402


@click.command()
@click.option(
    "--features-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).resolve().parents[1] / "features",
    help="Directory containing per-subject feature parquets.",
)
@click.option(
    "--nsrr-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional SHHS NSRR CSV for cross-validation (joins nsrr_* harmonised vars).",
)
@click.option(
    "--out-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output parquet path. Default: <features-dir>/subject_metadata.parquet",
)
@click.option(
    "--mismatch-threshold",
    type=float,
    default=0.05,
    help="Flag mismatch if |computed - nsrr| / nsrr > threshold. Default 0.05 (5%).",
)
def main(features_dir: Path, nsrr_csv: Path | None, out_path: Path | None,
         mismatch_threshold: float) -> None:
    parquets = sorted(features_dir.glob("shhs*.parquet"))
    parquets = [p for p in parquets if p.name != "subject_metadata.parquet"]
    if not parquets:
        click.echo(f"No subject parquets found in {features_dir}", err=True)
        sys.exit(1)

    out_path = out_path or features_dir / "subject_metadata.parquet"

    rows: list[dict] = []
    failed: list[str] = []
    for p in tqdm(parquets, desc="Subjects"):
        try:
            df = pd.read_parquet(p)
            md = subject_metadata(df)
            md["subject_id"] = p.stem  # e.g. "shhs1-200001"
            rows.append(md)
        except Exception as e:  # noqa: BLE001
            failed.append(f"{p.stem}: {type(e).__name__}: {e}")

    meta = pd.DataFrame(rows)
    click.echo(f"\nProcessed: {len(meta)} subjects (failed: {len(failed)})")
    for f in failed[:10]:
        click.echo(f"  ✗ {f}")

    # Cohort-level summary
    click.echo("\nCohort summary:")
    for col in ("tst_min", "sleep_efficiency", "waso_min", "ahi_proxy_per_hr",
                "hypoxic_burden_per_night"):
        if col in meta.columns:
            s = meta[col].dropna()
            if not s.empty:
                click.echo(f"  {col}: median={s.median():.2f}, "
                           f"mean={s.mean():.2f}, n_nonnull={len(s)}")

    # Optional NSRR cross-validation
    if nsrr_csv is not None:
        click.echo(f"\nLoading NSRR CSV: {nsrr_csv}")
        nsrr = pd.read_csv(nsrr_csv, low_memory=False)
        # NSRR uses subject IDs without cohort prefix (e.g. "200001" not "shhs1-200001")
        # Try common id columns
        id_col = next((c for c in ("nsrrid", "pptid", "subject_id") if c in nsrr.columns), None)
        if id_col is None:
            click.echo(f"  ✗ Could not find a subject-id column in NSRR CSV. "
                       f"Looked for: nsrrid, pptid, subject_id. "
                       f"Available: {list(nsrr.columns)[:20]}", err=True)
        else:
            click.echo(f"  Using NSRR id column: {id_col}")
            # Build a join key on our side
            meta["_nsrr_join"] = meta["subject_id"].str.replace("shhs1-", "", regex=False)
            nsrr["_nsrr_join"] = nsrr[id_col].astype(str)
            joined = meta.merge(nsrr, on="_nsrr_join", how="left", suffixes=("", "_nsrr"))

            mismatches = []
            checks = [
                ("tst_min", "nsrr_total_sleep_time", 1.0),
                ("sleep_efficiency", "nsrr_sleep_efficiency", 0.01),  # NSRR may report as %
                ("waso_min", "nsrr_waso", 1.0),
            ]
            for ours_col, nsrr_col, scale in checks:
                if ours_col not in joined.columns or nsrr_col not in joined.columns:
                    click.echo(f"  Skipping {ours_col} vs {nsrr_col}: not both present.")
                    continue
                ours = joined[ours_col]
                theirs = joined[nsrr_col] * scale  # rescale if needed
                with pd.option_context("mode.use_inf_as_na", True):
                    rel_err = ((ours - theirs).abs() / theirs.replace(0, pd.NA)).abs()
                n_flagged = (rel_err > mismatch_threshold).sum()
                click.echo(f"  {ours_col} vs {nsrr_col}: "
                           f"{n_flagged} mismatches > {mismatch_threshold*100:.0f}% "
                           f"(of {rel_err.notna().sum()} comparable rows)")
                if n_flagged > 0:
                    mismatches.append((ours_col, n_flagged))

            if mismatches:
                click.echo(f"\n⚠ {len(mismatches)} variable(s) had mismatches > "
                           f"{mismatch_threshold*100:.0f}%. Review the merged frame.")

            # Persist the joined + flagged frame
            meta = joined.drop(columns=["_nsrr_join"])

    meta.to_parquet(out_path)
    click.echo(f"\n✓ Wrote {out_path} ({len(meta)} rows × {len(meta.columns)} cols)")


if __name__ == "__main__":
    main()
