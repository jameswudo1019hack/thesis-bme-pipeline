"""Freeze the Aim 2 subject split as splits/aim2_seed42.json (+ .sha256).

Reproduces the train / val / test subject sets of every Aim 2 sleep-only
LightGBM fit from the 5,793 sorted subject ids (see thesis_pipeline/splits.py)
and checks them against the canonical test_predictions key file and the
subject metadata before writing. Deterministic output: rerunning on the same
inputs leaves the file (and its sha256) unchanged; a different result is
refused unless --force.

Ids come from features/subject_metadata.parquet (local, 238 KB). Feature
parquets are never read. --check-tar additionally compares the ids with the
member names of the uncompressed features tar (headers only). It refuses a
tar that iCloud has evicted to a placeholder: listing its members would pull
the whole 17 GB back onto a disk near its floor.

Usage:
  python scripts/make_aim2_split.py
  python scripts/make_aim2_split.py --check-tar features-phase1batch-v1.tar
"""

from __future__ import annotations

import re
import sys
import tarfile
from pathlib import Path

import click
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline import shhs, splits  # noqa: E402

DEFAULT_METADATA = CODE_ROOT / "features" / "subject_metadata.parquet"
DEFAULT_KEY = (
    CODE_ROOT / "models" / "recovery_2026-05-03" / "aim2_v6_past_only" / "test_predictions.parquet"
)
DEFAULT_OUT = CODE_ROOT / "splits" / "aim2_seed42.json"


def tar_subject_ids(path: Path) -> np.ndarray:
    ids = []
    with tarfile.open(path) as tf:
        for m in tf:
            hit = re.fullmatch(r"(?:.*/)?shhs1-(\d+)\.parquet", m.name)
            if m.isfile() and hit:
                ids.append(int(hit.group(1)))
    return np.unique(np.asarray(ids, dtype=np.int64))


@click.command()
@click.option("--metadata", type=click.Path(path_type=Path, exists=True), default=DEFAULT_METADATA,
              show_default=True, help="subject_metadata.parquet (source of the 5,793 ids).")
@click.option("--key", type=click.Path(path_type=Path, exists=True), default=DEFAULT_KEY,
              show_default=True, help="Canonical Aim 2 test_predictions.parquet (test-id check).")
@click.option("--out", type=click.Path(path_type=Path), default=DEFAULT_OUT, show_default=True)
@click.option("--check-tar", type=click.Path(path_type=Path, exists=True), default=None,
              help="Also check the ids against the member names of this features tar.")
@click.option("--force", is_flag=True, help="Overwrite an existing, DIFFERENT split file.")
def main(metadata: Path, key: Path, out: Path, check_tar: Path | None, force: bool) -> None:
    if check_tar is not None and not shhs.is_local(check_tar):
        raise click.ClickException(f"{check_tar} is a cloud placeholder (evicted); refusing to read it, which "
                                   f"would re-download {check_tar.stat().st_size / 1e9:.1f} GB. Drop --check-tar.")
    meta = splits.read_subject_metadata(metadata)
    ids = meta["sid"].to_numpy()
    click.echo(f"ids: {ids.size} from {metadata.name}")
    if ids.size != splits.AIM2_EXPECTED_N_IDS:
        raise click.ClickException(f"expected {splits.AIM2_EXPECTED_N_IDS} ids, got {ids.size}")
    if check_tar is not None:
        tids = tar_subject_ids(check_tar)
        same = np.array_equal(tids, np.sort(ids))
        click.echo(f"tar members: {tids.size} subject parquets; identical id set: {same}")
        if not same:
            raise click.ClickException("metadata ids differ from the tar member ids")

    split = splits.aim2_split(ids)
    checked = splits.assert_canonical(split, key, meta)
    click.echo(f"subjects:     {checked['sizes']}")
    click.echo(f"sleep epochs: {checked['sleep_epochs']}")
    click.echo(f"train neg/pos {checked['train_neg_pos']!r} (stored scale_pos_weight {splits.AIM2_TRAIN_NEG_POS!r})")
    click.echo(f"test ids == subject set of {key.relative_to(CODE_ROOT) if key.is_relative_to(CODE_ROOT) else key}")

    rel = lambda p: str(p.relative_to(CODE_ROOT)) if p.is_relative_to(CODE_ROOT) else str(p)  # noqa: E731
    record = splits.split_record(
        split,
        ids_source=rel(metadata),
        n_ids=ids.size,
        extra={
            "all_ids_sha256": splits.ids_sha256(ids),
            "sleep_epochs": checked["sleep_epochs"],
            "train_neg_pos": checked["train_neg_pos"],
            "checked_against": {"key": rel(key), "key_sha256": splits.file_sha256(key)},
        },
    )
    sha, changed = splits.write_split(out, record, force=force)
    click.secho(f"{'wrote' if changed else 'unchanged'} {out}  sha256={sha}", fg="green")
    loaded = splits.load_split(out)
    for k in splits.SPLIT_NAMES:
        assert np.array_equal(loaded[k], split[k])


if __name__ == "__main__":
    main()
