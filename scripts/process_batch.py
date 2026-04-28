"""Process every SHHS1 subject that is currently materialised locally.

Typical usage (see Code/README.md for the full batch workflow):

    # After pinning the 100 pilot subjects in Finder:
    python scripts/process_batch.py --pilot-only

    # Process every locally-materialised subject:
    python scripts/process_batch.py

    # Dry run — list what would be processed, do no work:
    python scripts/process_batch.py --dry-run

Only local (non-placeholder) files are touched; placeholders are skipped
so that running this script does NOT trigger on-demand downloads of the
full 212 GB cohort.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm

# Allow running as a script from Code/scripts/ without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from thesis_pipeline import shhs  # noqa: E402
from thesis_pipeline.epochs import build_epoch_frame  # noqa: E402
from thesis_pipeline.features import (  # noqa: E402
    FEATURES_VERSION,
    contextual_features,
    ecg_band_power,
    eeg_band_power,
    hrv_features,
    hrv_freq_features,
    hypoxic_burden_features,
    position_features,
    respiratory_features,
    spo2_features,
)
from thesis_pipeline.io import read_edf, read_nsrr_xml  # noqa: E402


FEATURES_DIR = Path(__file__).resolve().parents[1] / "features"

# SHHS channel aliases — see io.CANONICAL_CHANNELS for documentation.
SPO2_ALIASES = ["SaO2", "SpO2"]
ECG_ALIASES = ["ECG", "EKG"]
EEG_ALIASES = ["EEG", "EEG1", "C4"]  # primary EEG (C4-A1 in SHHS)
THOR_ALIASES = ["THOR RES", "THOR", "Thor"]
ABDO_ALIASES = ["ABDO RES", "ABDO", "Abdo"]
AIRFLOW_ALIASES = ["AIRFLOW", "NEW AIR", "nasal"]
POSITION_ALIASES = ["POSITION", "Position", "position"]


def _pick_channel(raw, aliases: list[str]):
    for name in aliases:
        if name in raw.ch_names:
            return raw.get_data(picks=[name])[0]
    return None


def process_subject(sp: shhs.SubjectPaths) -> pd.DataFrame:
    """Turn one subject's EDF + NSRR XML into a labelled, feature-enriched epoch frame."""
    raw = read_edf(sp.edf, preload=True)
    hypno, events = read_nsrr_xml(sp.nsrr_xml)
    sfreq = float(raw.info["sfreq"])
    # raw.times[-1] is the last *sample* time = (N-1)/sfreq, which underestimates
    # duration by one sample interval; use n_times directly to avoid off-by-one.
    n_epochs = raw.n_times // int(round(30 * sfreq))

    frame = build_epoch_frame(
        subject_id=sp.subject_id,
        cohort=sp.cohort,
        n_epochs=n_epochs,
        hypno=hypno,
        events=events,
    )

    features: dict[str, np.ndarray] = {}

    # Build epoch-level sleep mask (True for N1/N2/N3/REM) for sleep-only ODI.
    sleep_mask = frame["sleep_stage"].isin(["N1", "N2", "N3", "REM"]).to_numpy()

    spo2 = _pick_channel(raw, SPO2_ALIASES)
    if spo2 is not None:
        features.update(spo2_features(spo2, sfreq, sleep_mask=sleep_mask))
        features.update(hypoxic_burden_features(spo2, sfreq, events, n_epochs))  # NEW T4

    ecg = _pick_channel(raw, ECG_ALIASES)
    if ecg is not None:
        features.update(hrv_features(ecg, sfreq))
        features.update(hrv_freq_features(ecg, sfreq))  # NEW T3 — frequency-domain HRV
        features.update(ecg_band_power(ecg, sfreq))     # Phase 1 Exp 4 — multi-scale ECG band power

    eeg = _pick_channel(raw, EEG_ALIASES)
    if eeg is not None:
        features.update(eeg_band_power(eeg, sfreq))

    thor = _pick_channel(raw, THOR_ALIASES)
    abdo = _pick_channel(raw, ABDO_ALIASES)
    airflow = _pick_channel(raw, AIRFLOW_ALIASES)
    features.update(respiratory_features(thor, abdo, airflow, sfreq))

    position = _pick_channel(raw, POSITION_ALIASES)  # NEW T5
    if position is not None:
        features.update(position_features(position, sfreq))

    for col, vals in features.items():
        v = np.asarray(vals)
        if v.size < n_epochs:
            v = np.concatenate([v, np.full(n_epochs - v.size, np.nan)])
        elif v.size > n_epochs:
            v = v[:n_epochs]
        frame[col] = v

    # Contextual features (lag/lead/rolling) over the base feature columns.
    # Per-subject only — never crosses subject boundaries.
    frame = contextual_features(frame)

    frame["features_version"] = FEATURES_VERSION
    return frame


@click.command()
@click.option(
    "--pilot-only",
    is_flag=True,
    help="Restrict to the 100-subject pilot list (pilot_subjects.json).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="List what would be processed and exit without reading EDFs.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=FEATURES_DIR,
    show_default=True,
    help="Directory to write per-subject parquet files into.",
)
def main(pilot_only: bool, dry_run: bool, out_dir: Path) -> None:
    restrict = shhs.load_pilot_subject_ids() if pilot_only else None
    subjects = shhs.local_subjects(cohort="shhs1", restrict_to=restrict)

    if not subjects:
        click.secho(
            "No locally-materialised SHHS1 subjects found." +
            (" (pilot-only)" if pilot_only else ""),
            fg="yellow",
        )
        click.echo(
            "Pin the files in Finder (Always Keep on This Device) and retry."
        )
        sys.exit(1)

    click.secho(
        f"{len(subjects)} subject(s) locally available" +
        (f" out of {len(restrict)} in pilot" if pilot_only else ""),
        fg="green",
    )
    if dry_run:
        for sp in subjects[:20]:
            click.echo(f"  {sp.display_id}")
        if len(subjects) > 20:
            click.echo(f"  ... and {len(subjects) - 20} more")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in tqdm(subjects, desc="Processing"):
        try:
            frame = process_subject(sp)
            frame.to_parquet(out_dir / f"{sp.display_id}.parquet", index=False)
        except Exception as e:  # noqa: BLE001 — surface any per-subject failure, keep going
            click.secho(f"  ! {sp.display_id}: {type(e).__name__}: {e}", fg="red")


if __name__ == "__main__":
    main()
