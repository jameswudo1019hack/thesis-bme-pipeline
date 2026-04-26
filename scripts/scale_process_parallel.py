"""scripts/scale_process_parallel.py — multiprocessing-parallel scale_process.

Each worker independently:
  1. Triggers OneDrive download via 1-byte read
  2. Loads EDF + NSRR XML
  3. Computes 29 features
  4. Atomically writes parquet (tmp → rename)

No shared state across workers. One subject = one worker = one parquet write.
Bit-identical to serial; verify with `scripts/verify_parallel_features.py`.

Typical usage:
    python scripts/scale_process_parallel.py --workers 4
    python scripts/scale_process_parallel.py --workers 4 --limit 200
    python scripts/scale_process_parallel.py --workers 4 --dry-run
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


def _process_one(sid: int) -> tuple[int, str]:
    """Worker function for one subject. Returns (sid, status)."""
    from thesis_pipeline import shhs
    from process_batch import process_subject

    sp = shhs.paths_for(sid, cohort="shhs1")
    parquet_path = FEATURES_DIR / f"{sp.display_id}.parquet"
    if parquet_path.exists():
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


def _disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


@click.command()
@click.option("--workers", type=int, default=4, show_default=True,
              help="Number of parallel worker processes")
@click.option("--cohort", default="shhs1", show_default=True)
@click.option("--limit", type=int, default=None,
              help="Max subjects to process this run (after skipping done)")
@click.option("--start-id", type=int, default=None,
              help="Only consider subject IDs >= this")
@click.option("--min-free-gb", type=float, default=20.0, show_default=True)
@click.option("--dry-run", is_flag=True)
def main(workers: int, cohort: str, limit: int | None, start_id: int | None,
         min_free_gb: float, dry_run: bool) -> None:
    from thesis_pipeline import shhs

    all_ids = shhs.discover_shhs1_subject_ids() if cohort == "shhs1" else []
    if not all_ids:
        click.secho(f"No subjects discovered for cohort {cohort!r}.", fg="red")
        sys.exit(1)
    if start_id is not None:
        all_ids = [sid for sid in all_ids if sid >= start_id]

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    for p in FEATURES_DIR.glob(f"{cohort}-*.parquet"):
        try:
            done.add(int(p.stem.split("-")[1]))
        except ValueError:
            pass
    remaining = [sid for sid in all_ids if sid not in done]
    to_process = remaining[:limit] if limit is not None else remaining

    click.secho(
        f"cohort={cohort}  discovered={len(all_ids)}  already_done={len(done)}  "
        f"to_process={len(to_process)}  workers={workers}",
        fg="cyan",
    )
    click.echo(f"Free disk: {_disk_free_gb(FEATURES_DIR):.1f} GB (threshold {min_free_gb} GB)")

    if dry_run:
        if to_process:
            click.echo(f"First 10: {to_process[:10]}")
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
