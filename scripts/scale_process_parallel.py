"""scripts/scale_process_parallel.py — multiprocessing-parallel scale_process.

Each worker independently:
  1. Triggers OneDrive download via 1-byte read
  2. Loads EDF + NSRR XML
  3. Computes 29 features
  4. Atomically writes parquet (tmp → rename)

No shared state across workers. One subject = one worker = one parquet write.
Bit-identical to serial; verify with `scripts/verify_parallel_features.py`.

Force-reprocessing flags (mutually exclusive):

  --force
      Re-extract every subject regardless of whether its parquet exists.
      Use when the extraction code or feature schema has changed and you want
      to rebuild the entire cohort from scratch.

  --force-version-mismatch
      Re-extract only the subjects whose existing parquet has a
      ``features_version`` value different from the current FEATURES_VERSION
      constant in ``thesis_pipeline.features``.  Subjects already at the
      current version are skipped; subjects without a parquet are always
      processed.  Use after a targeted feature-code bump when you want to
      upgrade stale files without touching already-current ones.

Typical usage:
    python scripts/scale_process_parallel.py --workers 4
    python scripts/scale_process_parallel.py --workers 4 --limit 200
    python scripts/scale_process_parallel.py --workers 4 --dry-run
    python scripts/scale_process_parallel.py --workers 4 --force
    python scripts/scale_process_parallel.py --workers 4 --force-version-mismatch
"""

from __future__ import annotations

import os

# Limit thread oversubscription BEFORE NumPy / SciPy / mne import.
# Set in parent so child processes inherit; worker_init also re-applies.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# Allow our project's audit warnings (e.g. read_nsrr_xml flagging non-canonical
# respiratory concepts) through the blanket ignore. Filter by warning category
# (not module) so stacklevel doesn't matter — Codex audit-v6 finding.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesis_pipeline.io import RespiratoryConceptAuditWarning  # noqa: E402
warnings.filterwarnings("always", category=RespiratoryConceptAuditWarning)

import click
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = CODE_ROOT / "features"

# Make package + sibling-script imports work in workers
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "scripts"))


def _worker_init() -> None:
    """Run once at the start of each worker process. Cap BLAS threads + warm imports."""
    # Re-apply env in worker (some BLAS implementations check at first call)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except ImportError:
        pass

    sys.path.insert(0, str(CODE_ROOT))
    sys.path.insert(0, str(CODE_ROOT / "scripts"))
    # Touch hot imports so the first task doesn't pay import cost
    import process_batch  # noqa: F401
    from thesis_pipeline import shhs  # noqa: F401
    from thesis_pipeline import features  # noqa: F401


# Hard per-subject timeout (s). A stuck OneDrive fetch or pathological EDF/XML
# parse can otherwise block a worker forever and stall the whole pool — Codex
# audit-v6 finding. SIGALRM is delivered at the next Python opcode; native
# C calls (mne, pyarrow) may not interrupt instantly but will at the next
# Python statement, which is enough to unblock the pool. Unix-only; macOS OK.
PER_SUBJECT_TIMEOUT_SEC = 300


def _process_one(args: tuple[int, bool]) -> tuple[int, str]:
    """Worker function for one subject. Returns (sid, status).

    Args:
        args: (sid, overwrite). When overwrite is True, an existing parquet is
              re-extracted and overwritten; when False, the subject is skipped
              if its parquet already exists.

    Wraps the body in a SIGALRM timeout so a stuck subject can't hang the run.
    """
    sid, overwrite = args
    import signal as _signal

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"shhs1-{sid}: exceeded {PER_SUBJECT_TIMEOUT_SEC}s")

    old_handler = _signal.signal(_signal.SIGALRM, _timeout_handler)
    _signal.alarm(PER_SUBJECT_TIMEOUT_SEC)

    try:
        from thesis_pipeline import shhs
        from process_batch import process_subject

        sp = shhs.paths_for(sid, cohort="shhs1")
        parquet_path = FEATURES_DIR / f"{sp.display_id}.parquet"
        if parquet_path.exists() and not overwrite:
            return (sid, "skipped_existing")

        # Trigger downloads
        for p in (sp.edf, sp.nsrr_xml):
            try:
                if not shhs.is_local(p):
                    with open(p, "rb") as f:
                        f.read(4096)
            except Exception as e:
                return (sid, f"download_error:{type(e).__name__}:{e}")

        if not shhs.is_subject_local(sp):
            return (sid, "still_cloud_only")

        try:
            frame = process_subject(sp)
        except TimeoutError:
            raise
        except Exception as e:
            return (sid, f"process_error:{type(e).__name__}:{e}")

        # Atomic write
        tmp = parquet_path.with_suffix(".parquet.tmp")
        try:
            frame.to_parquet(tmp, index=False)
            tmp.rename(parquet_path)
        except Exception as e:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
            return (sid, f"write_error:{type(e).__name__}:{e}")

        return (sid, "ok")
    except TimeoutError as e:
        return (sid, f"timeout:{e}")
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old_handler)


def _disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def _read_parquet_version(parquet_path: Path) -> str | None:
    """Read the features_version from a single-column parquet slice.

    Returns the version string, or None if the column is absent or unreadable.
    Uses pyarrow.parquet to read only the one column — fast even for wide files.
    """
    import pyarrow.parquet as pq

    try:
        table = pq.read_table(parquet_path, columns=["features_version"])
        return table.column("features_version")[0].as_py()
    except Exception:
        return None


@click.command()
@click.option("--workers", type=int, default=4, show_default=True,
              help="Number of parallel worker processes")
@click.option("--cohort", default="shhs1", show_default=True,
              help="Subject cohort to process")
@click.option("--limit", type=int, default=None,
              help="Max subjects to process this run (after skipping done)")
@click.option("--start-id", type=int, default=None,
              help="Only consider subject IDs >= this")
@click.option("--min-free-gb", type=float, default=20.0, show_default=True)
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True,
              help="Re-extract every subject regardless of whether its parquet exists. "
                   "Mutually exclusive with --force-version-mismatch.")
@click.option("--force-version-mismatch", is_flag=True,
              help="Re-extract only subjects whose existing parquet has a features_version "
                   "different from the current FEATURES_VERSION. Subjects without a parquet "
                   "are always processed. Mutually exclusive with --force.")
def main(workers: int, cohort: str, limit: int | None, start_id: int | None,
         min_free_gb: float, dry_run: bool, force: bool,
         force_version_mismatch: bool) -> None:
    # Mutual exclusion guard
    if force and force_version_mismatch:
        click.secho(
            "Error: --force and --force-version-mismatch are mutually exclusive. "
            "Use --force to reprocess everything, or --force-version-mismatch to "
            "reprocess only subjects at a stale feature version.",
            fg="red",
            err=True,
        )
        sys.exit(1)

    from thesis_pipeline import shhs
    from thesis_pipeline.features import FEATURES_VERSION

    all_ids = shhs.discover_shhs1_subject_ids() if cohort == "shhs1" else []
    if not all_ids:
        click.secho(f"No subjects discovered for cohort {cohort!r}.", fg="red")
        sys.exit(1)
    if start_id is not None:
        all_ids = [sid for sid in all_ids if sid >= start_id]

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build the set of existing parquet files (keyed by subject ID).
    existing: dict[int, Path] = {}
    for p in FEATURES_DIR.glob(f"{cohort}-*.parquet"):
        try:
            existing[int(p.stem.split("-")[1])] = p
        except ValueError:
            pass

    # Build to_process: each branch is self-contained — no post-hoc filter.
    if force:
        # Overwrite every subject unconditionally.
        to_process = [(sid, True) for sid in all_ids]
    elif force_version_mismatch:
        # Re-extract subjects missing a parquet OR whose version is stale.
        # Probe is serial in the controller so workers don't each import
        # pyarrow and re-read parquet metadata in parallel.
        click.echo(
            f"Probing features_version in {len(existing)} existing parquets "
            f"(current: {FEATURES_VERSION}) ..."
        )
        stale_ids: set[int] = set()
        for sid, p in tqdm(existing.items(), desc="Probing versions", unit="file"):
            if _read_parquet_version(p) != FEATURES_VERSION:
                stale_ids.add(sid)
        to_process = [
            (sid, True)
            for sid in all_ids
            if sid in stale_ids or sid not in existing
        ]
    else:
        # Default: skip any subject that already has a parquet.
        to_process = [
            (sid, False)
            for sid in all_ids
            if sid not in existing
        ]

    if limit is not None:
        to_process = to_process[:limit]

    click.secho(
        f"cohort={cohort}  discovered={len(all_ids)}  already_done={len(existing)}  "
        f"to_process={len(to_process)}  workers={workers}",
        fg="cyan",
    )
    click.echo(f"Free disk: {_disk_free_gb(FEATURES_DIR):.1f} GB (threshold {min_free_gb} GB)")

    if dry_run:
        if to_process:
            click.echo(f"First 10: {[sid for sid, _ in to_process[:10]]}")
        return
    if not to_process:
        click.secho("Nothing to do.", fg="green")
        return

    counts = {"ok": 0, "skipped_existing": 0, "still_cloud_only": 0, "errors": 0}
    t_start = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_worker_init) as pool:
        results = pool.imap_unordered(_process_one, to_process, chunksize=1)
        for sid, status in tqdm(results, total=len(to_process), desc="Processing"):
            if status == "ok":
                counts["ok"] += 1
            elif status == "skipped_existing":
                counts["skipped_existing"] += 1
            elif status == "still_cloud_only":
                counts["still_cloud_only"] += 1
                tqdm.write(f"  ! shhs1-{sid}: download failed (still cloud-only)")
            else:
                counts["errors"] += 1
                tqdm.write(f"  ! shhs1-{sid}: {status}")
            # Disk pressure check every ~25 completions
            total_done_now = sum(counts.values())
            if total_done_now and total_done_now % 25 == 0:
                free = _disk_free_gb(FEATURES_DIR)
                if free < min_free_gb:
                    tqdm.write(
                        f"⚠ free disk {free:.1f} GB < {min_free_gb} threshold; "
                        "consider Free Up Space on OneDrive folder."
                    )

    elapsed = time.time() - t_start
    rate = counts["ok"] / max(elapsed / 60, 1e-9)
    click.secho(
        f"\nDone. ok={counts['ok']}  skipped={counts['skipped_existing']}  "
        f"download_failed={counts['still_cloud_only']}  errors={counts['errors']}  "
        f"elapsed={elapsed/60:.1f} min  rate={rate:.1f}/min",
        fg="green",
    )
    click.echo(f"Total features now: {len(list(FEATURES_DIR.glob('*.parquet')))} parquet files.")


if __name__ == "__main__":
    main()
