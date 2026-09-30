"""Stage B for the Aim 2 DL arm: derive 4 Hz model inputs from stage A and pack them.

Commands
--------
pack    derive RR / EDR / THOR / ABDO / AIRFLOW / SPO2 at 4 Hz for every subject of the
        requested splits and pack them (design section 2.2 stage B). Resumable.
bench   time the derivation per night on stage-A (or scout prototype) npz files.
verify  re-hash a packed split folder against its MANIFEST.json.

Examples
--------
    cd Code
    python scripts/derive_dl_signals.py pack --splits train,val --workers 6
    python scripts/derive_dl_signals.py bench /path/shhs1-200001.npz --split-json splits/aim2_seed42.json
    python scripts/derive_dl_signals.py verify ~/thesis_dl_cache/model_v1/val

The test split is packed only when named explicitly (``--splits test``); it stays in
its own folder on this Mac and is never uploaded (decision 2026-09-30).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.dl_data import load_split  # noqa: E402
from thesis_pipeline.dl_stage_b import (  # noqa: E402
    assert_not_under_desktop,
    derive_subject,
    env_workers,
    load_stage_a,
    pack_split,
    verify_manifest,
)

DEFAULT_SPLIT = CODE_ROOT / "splits" / "aim2_seed42.json"


def _refuse_cloud(path: Path) -> None:
    s = str(path.expanduser().resolve())
    if "GoogleDrive" in s or "My Drive" in s or "CloudStorage" in s:
        raise click.ClickException(f"refusing to pack into a cloud-synced folder: {s}")


@click.group()
def cli() -> None:
    """Stage-B derivation and packing for the Aim 2 DL arm."""


@cli.command()
@click.option("--raw-dir", default="~/thesis_dl_cache/raw_v1", show_default=True, help="stage-A folder")
@click.option("--out", "out_root", default="~/thesis_dl_cache/model_v1", show_default=True)
@click.option("--split-json", type=click.Path(exists=True, dir_okay=False), default=str(DEFAULT_SPLIT),
              show_default=True)
@click.option("--splits", default="train,val", show_default=True, help="comma list of train,val,test")
@click.option("--workers", type=int, default=env_workers(), show_default=True)
@click.option("--edr", type=click.Choice(["psa", "ramp"]), default="psa", show_default=True)
@click.option("--max-chunk-gb", type=float, default=1.5, show_default=True)
@click.option("--allow-missing", is_flag=True, help="pack the subjects present (pilot / fallback F-X)")
@click.option("--only-ids", type=click.Path(exists=True, dir_okay=False), default=None,
              help="text file with one subject id per line to restrict to (pilot)")
def pack(raw_dir, out_root, split_json, splits, workers, edr, max_chunk_gb, allow_missing, only_ids) -> None:
    """Derive and pack stage B for the requested splits."""
    out = Path(out_root).expanduser()
    assert_not_under_desktop(out)
    _refuse_cloud(out)
    sp = load_split(split_json)
    restrict = None
    if only_ids:
        restrict = {int(x) for x in Path(only_ids).read_text().split() if x.strip()}
    for name in [s.strip() for s in splits.split(",") if s.strip()]:
        if name not in ("train", "val", "test"):
            raise click.BadParameter(f"unknown split {name}")
        if name == "test":
            click.echo("NOTE: packing TEST locally. It must stay on this Mac (never Drive/Colab).")
        ids = sp[name].tolist()
        if restrict is not None:
            ids = [i for i in ids if i in restrict]
        man = pack_split(
            name, ids, raw_dir, out, split_json=split_json, workers=workers, edr=edr,
            max_chunk_bytes=int(max_chunk_gb * 1e9), allow_missing=allow_missing, log=click.echo,
        )
        click.echo(json.dumps({k: man[k] for k in ("split", "n_subjects", "n_epochs", "n_sleep_epochs",
                                                     "missing_subjects", "chunks")}, default=str))


@cli.command()
@click.argument("npz", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--edr", type=click.Choice(["psa", "ramp"]), default="psa", show_default=True)
@click.option("--split-json", type=click.Path(exists=True, dir_okay=False), default=None,
              help="refuse test subjects listed in this split file (default: splits/aim2_seed42.json if present)")
@click.option("--json-out", type=click.Path(dir_okay=False), default=None)
def bench(npz, edr, split_json, json_out) -> None:
    """Time stage-B derivation per night (no packing, nothing written except --json-out)."""
    if split_json is None and DEFAULT_SPLIT.exists():
        split_json = str(DEFAULT_SPLIT)
    if split_json is None:
        raise click.ClickException("pass --split-json so test subjects can be refused")
    test_ids = set(load_split(split_json)["test"].tolist())
    rows = []
    for p in npz:
        t0 = time.perf_counter()
        sa = load_stage_a(p)
        if sa.subject_id in test_ids:
            click.echo(f"skip {sa.subject_id}: test subject")
            continue
        t_load = time.perf_counter() - t0
        res = derive_subject(sa, edr=edr)
        qc = res["qc"]
        row = {
            "subject_id": sa.subject_id,
            "hours": sa.n_sec / 3600.0,
            "t_load_s": round(t_load, 3),
            "t_derive_s": qc["t_derive_s"],
            "t_ecg_filter_s": qc.get("ecg_t_filter_s"),
            "t_ecg_peaks_s": qc.get("ecg_t_peaks_s"),
            "t_ecg_fixpeaks_s": qc.get("ecg_t_fixpeaks_s"),
            "t_ecg_rr_edr_s": qc.get("ecg_t_rr_edr_s"),
            "ok": {k[3:]: v for k, v in qc.items() if k.startswith("ok_")},
            "rr_gap_frac": qc.get("ecg_rr_gap_frac"),
            "kubios_removed_or_moved_frac": qc.get("ecg_kubios_removed_or_moved_frac"),
            "kubios_added_or_moved_to_frac": qc.get("ecg_kubios_added_or_moved_to_frac"),
            "rr4_clipped_frac": qc.get("ecg_rr4_clipped_frac"),
            "hr_median_bpm": qc.get("ecg_hr_median_bpm"),
            "flags": qc.get("flags"),
        }
        rows.append(row)
        click.echo(json.dumps(row, default=str))
    if rows:
        t = np.array([r["t_derive_s"] + r["t_load_s"] for r in rows])
        summ = {"n": len(rows), "edr": edr, "sec_per_night_mean": float(t.mean()),
                "sec_per_night_max": float(t.max())}
        click.echo(json.dumps(summ))
        if json_out:
            Path(json_out).write_text(json.dumps({"rows": rows, "summary": summ}, indent=2, default=str))


@cli.command()
@click.argument("split_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--channels", default=None, help="only check these channels' signal files (+ labels)")
def verify(split_dir, channels) -> None:
    """Check sha256 of a packed split folder against MANIFEST.json."""
    d = Path(split_dir)
    man = json.loads((d / "MANIFEST.json").read_text())
    names = list(man["sha256"])
    if channels:
        keep = [c.strip() for c in channels.split(",")]
        names = [n for n in names if not n.startswith("signals_") or any(f"_{c}_" in n for c in keep)]
    bad = verify_manifest(d, names)
    if bad:
        raise click.ClickException(f"{len(bad)} file(s) fail sha256: {bad[:5]}")
    click.echo(f"OK: {len(names)} files match MANIFEST ({man['split']}, {man['n_subjects']} subjects)")


if __name__ == "__main__":
    cli()
