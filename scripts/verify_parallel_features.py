"""scripts/verify_parallel_features.py — prove parallel == serial output.

Picks N already-processed subjects at random, backs up their parquets,
deletes them, regenerates with the parallel scale_process implementation,
and verifies each new parquet is identical to the backup. Restores
backups when done so cohort state is unchanged.

Usage:
    python scripts/verify_parallel_features.py --n 10 --workers 4
"""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import click
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = CODE_ROOT / "features"
BACKUP_DIR = Path("/tmp/v_parallel_verify_backup")
PARALLEL_SCRIPT = CODE_ROOT / "scripts" / "scale_process_parallel.py"


def _diff_summary(a: pd.DataFrame, b: pd.DataFrame) -> str:
    """Return a human-readable diff summary, or 'identical' if equal."""
    if a.equals(b):
        return "identical"
    parts = []
    if a.shape != b.shape:
        parts.append(f"shape differs: {a.shape} vs {b.shape}")
    common_cols = [c for c in a.columns if c in b.columns]
    if set(a.columns) != set(b.columns):
        parts.append(f"columns differ: extra_a={set(a.columns)-set(b.columns)}, "
                     f"extra_b={set(b.columns)-set(a.columns)}")
    for c in common_cols:
        if a[c].dtype.kind in "fiu":
            try:
                # numeric: tolerate tiny float diffs
                eq = (a[c].fillna(-1e9).round(8) == b[c].fillna(-1e9).round(8)).all()
            except Exception:
                eq = False
        else:
            eq = a[c].equals(b[c])
        if not eq:
            parts.append(f"col {c} differs")
    return "; ".join(parts) if parts else "identical"


@click.command()
@click.option("--n", type=int, default=10, show_default=True, help="Subjects to verify")
@click.option("--workers", type=int, default=4, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def main(n: int, workers: int, seed: int) -> None:
    random.seed(seed)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    candidates = sorted(FEATURES_DIR.glob("shhs1-*.parquet"))
    if len(candidates) < n:
        click.secho(f"Only {len(candidates)} parquets exist; cannot sample {n}.", fg="red")
        sys.exit(1)

    sample = random.sample(candidates, n)
    sids = [int(p.stem.split("-")[1]) for p in sample]
    click.secho(f"Sampling {n} subjects for verification: {sids}", fg="cyan")

    # 1. Backup originals + delete from features dir
    backups = {}
    for p in sample:
        b = BACKUP_DIR / p.name
        shutil.copy2(p, b)
        backups[p.name] = b
        p.unlink()
    click.echo(f"Backed up + deleted {n} originals")

    # 2. Regenerate via parallel script (limited to just these subjects via --start-id + --limit)
    #    Simplest: invoke the parallel script and let it process the missing N.
    #    But the script processes in sorted order — so to ensure ONLY these N get touched,
    #    we trust that they're the only missing ones (we just deleted them).
    click.echo(f"Running parallel reprocess (workers={workers}) on the {n} missing subjects...")
    cmd = [
        sys.executable, str(PARALLEL_SCRIPT),
        "--workers", str(workers),
        "--limit", str(n),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        click.secho(f"Parallel script failed:\n{res.stderr}", fg="red")
        # Restore originals before exiting
        for name, b in backups.items():
            shutil.copy2(b, FEATURES_DIR / name)
        sys.exit(1)
    click.echo("Parallel reprocess done. Verifying outputs...")

    # 3. Compare each regenerated file to its backup
    pass_ct, fail_ct = 0, 0
    for name, backup_path in backups.items():
        new_path = FEATURES_DIR / name
        if not new_path.exists():
            click.secho(f"  ✗ {name}: not regenerated", fg="red")
            fail_ct += 1
            continue
        a = pd.read_parquet(backup_path)
        b = pd.read_parquet(new_path)
        diff = _diff_summary(a, b)
        if diff == "identical":
            click.secho(f"  ✓ {name}: identical", fg="green")
            pass_ct += 1
        else:
            click.secho(f"  ✗ {name}: {diff}", fg="red")
            fail_ct += 1

    # 4. Restore originals (so cohort state is exactly what it was before)
    for name, b in backups.items():
        shutil.copy2(b, FEATURES_DIR / name)
    click.echo(f"\nRestored {n} originals from backup.")

    click.secho(f"\nVerification: {pass_ct}/{n} identical, {fail_ct} differ",
                fg="green" if fail_ct == 0 else "red")
    sys.exit(0 if fail_ct == 0 else 2)


if __name__ == "__main__":
    main()
