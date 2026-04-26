"""Materialise the 100 pilot subjects locally without Finder clicks.

macOS FileProvider downloads a placeholder in full the first time any
process reads from it. Reading a single byte of each pilot file is enough
to force OneDrive to pull the whole thing into its local cache.

This gives you a fast, scripted alternative to right-clicking 200+ files
in Finder. The downloaded files stay cached by OneDrive's files-on-demand
policy until you explicitly "Free Up Space" or the cache is reclaimed under
disk pressure.

Typical usage:

    python scripts/pin_pilot.py                 # ~3.6 GB of pilot downloads
    python scripts/pin_pilot.py --workers 8     # faster parallel download
    python scripts/pin_pilot.py --dry-run       # print plan, do nothing
"""

from __future__ import annotations

import concurrent.futures as cf
import sys
from pathlib import Path

import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesis_pipeline import shhs  # noqa: E402


def trigger_download(path: Path) -> tuple[Path, str]:
    """Touch a placeholder so OneDrive materialises the full file.

    Returns (path, status) where status is "already-local", "downloaded", or "error: ...".
    """
    try:
        if shhs.is_local(path):
            return path, "already-local"
        with open(path, "rb") as f:
            f.read(4096)  # any read triggers a full materialisation
        # After a successful read the dataless flag is cleared. We don't
        # re-stat aggressively because macOS can take a moment to update it.
        return path, "downloaded"
    except FileNotFoundError:
        return path, "error: file not visible in OneDrive folder"
    except Exception as e:  # noqa: BLE001 — keep going for the rest
        return path, f"error: {type(e).__name__}: {e}"


@click.command()
@click.option(
    "--workers",
    type=int,
    default=4,
    show_default=True,
    help="Parallel download workers. Higher = faster but saturates OneDrive.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the files that would be downloaded; do nothing.",
)
@click.option(
    "--include-profusion",
    is_flag=True,
    help="Also download the Profusion-format XML (NSRR XML is always downloaded).",
)
def main(workers: int, dry_run: bool, include_profusion: bool) -> None:
    pilot_ids = shhs.load_pilot_subject_ids()
    targets: list[Path] = []
    for sid in pilot_ids:
        sp = shhs.paths_for(sid, cohort="shhs1")
        targets.append(sp.edf)
        targets.append(sp.nsrr_xml)
        if include_profusion:
            targets.append(sp.profusion_xml)

    click.secho(
        f"{len(pilot_ids)} pilot subjects × "
        f"{'3' if include_profusion else '2'} files = {len(targets)} targets",
        fg="cyan",
    )
    if dry_run:
        for p in targets[:10]:
            click.echo(f"  {p.name}")
        if len(targets) > 10:
            click.echo(f"  ... and {len(targets) - 10} more")
        return

    counts = {"downloaded": 0, "already-local": 0, "errors": 0}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(trigger_download, p) for p in targets]
        for fut in tqdm(
            cf.as_completed(futures),
            total=len(futures),
            desc="Materialising",
        ):
            path, status = fut.result()
            if status == "already-local":
                counts["already-local"] += 1
            elif status == "downloaded":
                counts["downloaded"] += 1
            else:
                counts["errors"] += 1
                tqdm.write(f"  ! {path.name}: {status}")

    click.echo("")
    click.secho(
        f"  downloaded:    {counts['downloaded']}\n"
        f"  already local: {counts['already-local']}\n"
        f"  errors:        {counts['errors']}",
        fg="green",
    )
    if counts["errors"] == 0:
        click.secho(
            "Ready. Run: python scripts/process_batch.py --pilot-only",
            fg="green",
            bold=True,
        )


if __name__ == "__main__":
    main()
