"""Sync Colab-run model artifacts from Drive (downloaded locally) into Code/models/.

The Colab fits land in /MyDrive/Thesis/results/aim2_*/ on Drive. To run
recompute_extended_metrics.py and rebuild the metrics registry locally, those
artifacts need to live under Code/models/<run>/ on the local SSD.

Usage:
    cd Code
    # Step 1 (manual): in your browser, download MyDrive/Thesis/results/ as a
    # zip (or copy individual aim2_* folders) into ~/Downloads/colab_results/.
    # Step 2:
    python3 scripts/sync_colab_results.py ~/Downloads/colab_results

The script walks the source directory, finds every ``test_predictions.parquet``
file, and copies its parent dir's contents into ``Code/models/<dirname>/``
preserving any nested structure (e.g. ``aim2_v85_taxonomy/full/``).

Idempotent: re-running overwrites stale files but leaves untouched local-only
files (like best_params.json from local Optuna runs) intact unless the Colab
side wrote one too.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

CODE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = CODE_ROOT / "models"

# Files we care about pulling back from Colab dirs
ARTIFACT_NAMES = (
    "test_predictions.parquet",
    "metrics.json",
    "metrics_extended.json",
    "bootstrap_aucs_subject.npy",
    "bootstrap_auprs_subject.npy",
    "bootstrap_aucs_epoch.npy",
    "bootstrap_auprs_epoch.npy",
    "feature_list.json",
    "summary.json",
    "best_params.json",
    "ci_subject.json",
    "model.json",
    "model.txt",
)


def _find_run_dirs(src_root: Path) -> list[Path]:
    """Find every dir under src_root containing test_predictions.parquet.

    Returns list of run-result dirs (each one is a leaf containing artifacts).
    """
    return sorted({p.parent for p in src_root.rglob("test_predictions.parquet")})


def _relative_under_aim2(p: Path, src_root: Path) -> Path | None:
    """Map a Colab run dir to its local Code/models/ subpath.

    Strategy: take the path components from the first one matching ``aim2_*``
    onwards. Anything before that (e.g. ``Thesis/results/``) is stripped.
    """
    parts = p.relative_to(src_root).parts
    for i, comp in enumerate(parts):
        if comp.startswith("aim2_") or comp == "aim2_v8_ensemble":
            return Path(*parts[i:])
    return None


def _copy_run(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    """Copy known artifacts from src_dir to dst_dir. Returns (copied, skipped)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for name in ARTIFACT_NAMES:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        # Skip if dst is identical (size + mtime) — cheap idempotency
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


@click.command()
@click.argument("src_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Show what would be copied without doing it")
def main(src_root: Path, dry_run: bool) -> None:
    src_root = src_root.resolve()
    print(f"Source: {src_root}")
    print(f"Dest:   {MODELS_DIR}\n")

    run_dirs = _find_run_dirs(src_root)
    if not run_dirs:
        print("No test_predictions.parquet files found under source — nothing to sync.")
        sys.exit(1)

    print(f"Found {len(run_dirs)} run dirs with predictions:\n")
    total_copied = 0
    total_skipped = 0
    skipped_runs: list[str] = []
    for src in run_dirs:
        rel = _relative_under_aim2(src, src_root)
        if rel is None:
            skipped_runs.append(str(src.relative_to(src_root)))
            continue
        dst = MODELS_DIR / rel
        if dry_run:
            print(f"  [dry] {rel}")
            continue
        copied, skipped = _copy_run(src, dst)
        marker = "+" if copied else "="
        print(f"  {marker} {rel}  ({copied} copied, {skipped} unchanged)")
        total_copied += copied
        total_skipped += skipped

    if skipped_runs:
        print(f"\nSkipped {len(skipped_runs)} dirs not matching aim2_* pattern:")
        for s in skipped_runs:
            print(f"  - {s}")

    print(f"\nDone. {total_copied} files copied, {total_skipped} unchanged.")


if __name__ == "__main__":
    main()
