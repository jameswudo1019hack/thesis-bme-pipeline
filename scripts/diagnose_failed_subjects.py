"""Single-subject reprocess with verbose error logging.

Reprocesses the 4 SHHS1 subjects that failed the full-cohort run, capturing
EDF read, XML parse, and (optionally) feature extraction errors with full
traceback. Surfaces the root cause so the pipeline can be patched.

Usage:
    cd Code && python scripts/diagnose_failed_subjects.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import click

# Allow running as a script from Code/scripts/ without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesis_pipeline.shhs import paths_for, is_local  # noqa: E402
from thesis_pipeline.io import read_edf, read_nsrr_xml  # noqa: E402
from thesis_pipeline.epochs import build_epoch_frame  # noqa: E402

FAILED_SUBJECTS = [
    "shhs1-204465",
    "shhs1-204564",
    "shhs1-204572",
    "shhs1-204598",
]


def _try(label: str, fn):
    """Run ``fn`` under exception capture; print outcome."""
    try:
        result = fn()
        click.echo(f"  [ok] {label}")
        return result
    except Exception as e:  # noqa: BLE001
        click.echo(f"  [FAIL] {label}: {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stdout)
        return None


@click.command()
@click.option("--with-features", is_flag=True, help="Also try basic feature extraction")
def main(with_features: bool) -> None:
    """Diagnose the 4 known-failing SHHS1 subjects."""
    for sid in FAILED_SUBJECTS:
        click.echo(f"\n=== {sid} ===")

        # 1. Path resolution + local-availability check
        try:
            sp = paths_for(int(sid.split("-")[1]), cohort="shhs1")
            edf, xml = sp.edf, sp.nsrr_xml
        except Exception as e:  # noqa: BLE001
            click.echo(f"  [FAIL] paths_for({sid}): {type(e).__name__}: {e}")
            continue
        click.echo(f"  EDF: {edf}")
        click.echo(f"    exists={edf.exists()}, local={is_local(edf) if edf.exists() else 'n/a'}")
        click.echo(f"  XML: {xml}")
        click.echo(f"    exists={xml.exists()}, local={is_local(xml) if xml.exists() else 'n/a'}")

        if not (edf.exists() and is_local(edf)):
            click.echo(f"  -> EDF not local; pin it via Finder before re-diagnosing.")
            continue
        if not (xml.exists() and is_local(xml)):
            click.echo(f"  -> XML not local; pin it via Finder before re-diagnosing.")
            continue

        # 2. EDF read
        raw = _try("read_edf", lambda: read_edf(edf, preload=False))
        if raw is None:
            continue
        click.echo(f"    channels: {raw.ch_names}")
        click.echo(f"    duration: {raw.times[-1]:.1f} s, n_times: {raw.n_times}")
        click.echo(f"    sfreq (first ch): {raw.info['sfreq']} Hz")

        # 3. NSRR XML parse
        parsed = _try("read_nsrr_xml", lambda: read_nsrr_xml(xml))
        if parsed is None:
            continue
        hypno, events = parsed
        click.echo(f"    stages: {len(hypno.stages)} epochs")
        click.echo(f"    respiratory events: {len(events)}")

        # 4. Build epoch frame
        n_epochs = int(raw.times[-1] // 30)
        ef = _try(
            "build_epoch_frame",
            lambda: build_epoch_frame(int(sid.split("-")[1]), "SHHS1", n_epochs, hypno, events),
        )
        if ef is not None:
            click.echo(f"    epoch frame: {len(ef)} rows x {len(ef.columns)} cols")

        # 5. Optional: try feature extraction on a key channel
        if with_features:
            try:
                from thesis_pipeline.features import hrv_features
                ecg_idx = next((i for i, ch in enumerate(raw.ch_names)
                                if ch.upper() in ("ECG", "EKG")), None)
                if ecg_idx is None:
                    click.echo(f"  [FAIL] feature smoke: no ECG channel found in {raw.ch_names}")
                else:
                    raw.load_data()
                    ecg = raw.get_data(picks=[ecg_idx])[0]
                    hrv = hrv_features(ecg, raw.info["sfreq"])
                    n_nan = sum(int((v != v).sum()) for v in hrv.values()) // len(hrv)
                    click.echo(f"  [ok] feature smoke: HRV computed, ~{n_nan} NaN epochs per metric")
            except Exception as e:  # noqa: BLE001
                click.echo(f"  [FAIL] feature smoke: {type(e).__name__}: {e}")
                traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    main()
