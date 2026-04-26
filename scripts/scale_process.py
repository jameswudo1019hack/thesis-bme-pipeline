"""Scale feature extraction to the whole SHHS1 cohort (or any subset).

Resumable: skips any subject whose parquet already exists. Downloads each
subject from OneDrive on-demand (via 1-byte read), processes, writes
parquet, moves on. The OneDrive Files-On-Demand layer handles its own LRU
cache eviction under disk pressure — we don't need to manually unpin.

Typical usage:

    # Process the next 500 un-processed subjects (recommended batch size):
    python scripts/scale_process.py --limit 500

    # Process everything from scratch (safe to interrupt and resume):
    python scripts/scale_process.py

    # See what would be processed:
    python scripts/scale_process.py --dry-run

    # Start at a specific subject ID:
    python scripts/scale_process.py --start-id 201500 --limit 500
"""

from __future__ import annotations

import concurrent.futures as cf
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from process_batch import process_subject, FEATURES_DIR  # noqa: E402
from thesis_pipeline import shhs  # noqa: E402


def trigger_download(path: Path) -> bool:
    """Read a single byte to force OneDrive to materialise a placeholder.
    Returns True if the file is now local, False otherwise."""
    try:
        if shhs.is_local(path):
            return True
        with open(path, "rb") as f:
            f.read(4096)
        return True
    except Exception:
        return False


def _done_subject_ids(out_dir: Path, cohort: str) -> set[int]:
    """Return the set of subject IDs that already have a parquet file."""
    prefix = f"{cohort}-"
    done: set[int] = set()
    for p in out_dir.glob(f"{prefix}*.parquet"):
        try:
            done.add(int(p.stem.replace(prefix, "")))
        except ValueError:
            pass
    return done


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


@click.command()
@click.option("--cohort", default="shhs1", show_default=True)
@click.option("--limit", type=int, default=None, help="Max number of subjects to process in this run.")
@click.option("--start-id", type=int, default=None, help="Only consider subjects with ID >= this.")
@click.option("--sub-batch", type=int, default=25, show_default=True, help="Download N in parallel, then process sequentially.")
@click.option("--workers", type=int, default=6, show_default=True, help="Parallel download workers per sub-batch.")
@click.option("--min-free-gb", type=float, default=20.0, show_default=True, help="Pause and warn if free disk drops below this.")
@click.option("--dry-run", is_flag=True)
def main(cohort, limit, start_id, sub_batch, workers, min_free_gb, dry_run):
    all_ids = shhs.discover_shhs1_subject_ids() if cohort == "shhs1" else []
    if not all_ids:
        click.secho(f"No subjects discovered for cohort {cohort!r}.", fg="red")
        sys.exit(1)

    if start_id is not None:
        all_ids = [sid for sid in all_ids if sid >= start_id]

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    done = _done_subject_ids(FEATURES_DIR, cohort)
    remaining = [sid for sid in all_ids if sid not in done]
    if limit is not None:
        to_process = remaining[:limit]
    else:
        to_process = remaining

    click.secho(
        f"cohort={cohort}  discovered={len(all_ids)}  "
        f"already_done={len(done)}  to_process={len(to_process)}",
        fg="cyan",
    )
    click.echo(f"Free disk: {_disk_free_gb(FEATURES_DIR):.1f} GB (threshold {min_free_gb} GB)")

    if dry_run:
        if to_process:
            click.echo(f"First 10: {to_process[:10]}")
            if len(to_process) > 10:
                click.echo(f"Last 10:  {to_process[-10:]}")
        return

    if not to_process:
        click.secho("Nothing to do — all subjects already processed.", fg="green")
        return

    t_start = time.time()
    total_ok = 0
    total_err = 0

    for i in range(0, len(to_process), sub_batch):
        chunk = to_process[i : i + sub_batch]

        # Disk safety check
        free = _disk_free_gb(FEATURES_DIR)
        if free < min_free_gb:
            click.secho(
                f"\n⚠ Free disk at {free:.1f} GB (< threshold {min_free_gb} GB). Pausing.",
                fg="yellow",
            )
            click.echo(
                "  → In Finder, select the OneDrive folder → right-click → Free Up Space, "
                "then press Enter here to resume."
            )
            input()

        # Phase 1 — parallel download
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for sid in chunk:
                sp = shhs.paths_for(sid, cohort=cohort)
                futures.append(ex.submit(trigger_download, sp.edf))
                futures.append(ex.submit(trigger_download, sp.nsrr_xml))
            # Drain
            for _ in cf.as_completed(futures):
                pass

        # Phase 2 — process sequentially
        batch_idx = i // sub_batch + 1
        for sid in tqdm(
            chunk,
            desc=f"Batch {batch_idx}/{(len(to_process) + sub_batch - 1) // sub_batch}",
        ):
            sp = shhs.paths_for(sid, cohort=cohort)
            if not shhs.is_subject_local(sp):
                total_err += 1
                tqdm.write(f"  ! {sp.display_id}: download failed (still cloud-only)")
                continue
            try:
                frame = process_subject(sp)
                frame.to_parquet(FEATURES_DIR / f"{sp.display_id}.parquet", index=False)
                total_ok += 1
            except Exception as e:  # noqa: BLE001
                total_err += 1
                tqdm.write(f"  ! {sp.display_id}: {type(e).__name__}: {e}")

    elapsed = time.time() - t_start
    click.secho(
        f"\nDone. processed={total_ok}  errors={total_err}  "
        f"elapsed={elapsed/60:.1f} min  "
        f"rate={(total_ok + total_err)/max(elapsed/60, 1e-9):.1f}/min",
        fg="green",
    )
    click.echo(f"Total features so far: {len(list(FEATURES_DIR.glob('*.parquet')))} parquet files.")


if __name__ == "__main__":
    main()
