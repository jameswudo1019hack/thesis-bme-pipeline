"""Readers for SHHS EDF signals and NSRR XML annotations.

EDF reading uses ``mne`` for channel naming + resampling conveniences. NSRR
annotation XMLs use a CVS/XML format documented at
https://github.com/nsrr/edf-editor-translator and in the SHHS scoring Mop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lxml import etree

import mne


# Channel harmonisation: SHHS EDF files use lab-specific channel names. This
# map collapses common aliases to a canonical label used throughout the
# pipeline. Extend as new aliases are discovered.
CANONICAL_CHANNELS: dict[str, list[str]] = {
    "ECG": ["ECG", "EKG"],
    "SpO2": ["SaO2", "SpO2"],
    "HR": ["H.R.", "HR", "Pulse"],
    "EEG-C3": ["EEG(sec)", "EEG2", "C3"],
    "EEG-C4": ["EEG", "EEG1", "C4"],
    "EOG-L": ["EOG(L)", "EOG-L", "LOC"],
    "EOG-R": ["EOG(R)", "EOG-R", "ROC"],
    "EMG": ["EMG"],
    "THOR": ["THOR RES", "THOR", "Thor"],
    "ABDO": ["ABDO RES", "ABDO", "Abdo"],
    "AIRFLOW": ["AIRFLOW", "NEW AIR", "nasal"],
    "POSITION": ["POSITION", "Position"],
    "LIGHT": ["LIGHT", "Light"],
}


@dataclass
class Recording:
    """One night's PSG — raw data + metadata. Kept lazy where sensible."""

    raw: mne.io.BaseRaw
    subject_id: int
    cohort: str
    sfreq_by_channel: dict[str, float]

    @property
    def duration_sec(self) -> float:
        return self.raw.times[-1]


def read_edf(path: Path, preload: bool = False) -> mne.io.BaseRaw:
    """Read an SHHS EDF file via mne. ``preload=False`` keeps memory low."""
    # SHHS EDFs sometimes carry annotations that clash with mne's parser;
    # silencing non-fatal warnings keeps the batch log readable.
    raw = mne.io.read_raw_edf(path, preload=preload, verbose="ERROR")
    return raw


def canonical_channel_name(raw_channel_name: str) -> str | None:
    """Map an EDF channel name to its canonical label, or None if unknown."""
    lowered = raw_channel_name.strip()
    for canon, aliases in CANONICAL_CHANNELS.items():
        if lowered in aliases or lowered.lower() in (a.lower() for a in aliases):
            return canon
    return None


@dataclass
class Hypnogram:
    """Per-epoch sleep stage labels derived from the NSRR XML."""

    stages: np.ndarray  # shape (n_epochs,), dtype object, values in {"W","N1","N2","N3","REM","?"}
    epoch_seconds: int = 30


@dataclass
class RespiratoryEvent:
    """A single annotated respiratory event (apnoea or hypopnoea)."""

    start_sec: float
    duration_sec: float
    kind: str  # "Obstructive Apnea", "Central Apnea", "Hypopnea", "Mixed Apnea"


def read_nsrr_xml(path: Path) -> tuple[Hypnogram, list[RespiratoryEvent]]:
    """Parse an NSRR (Compumedics) XML annotation file.

    In SHHS NSRR XMLs, sleep stages AND respiratory events are both encoded
    as ``<ScoredEvent>`` records:

        <ScoredEvent>
          <EventType>Stages|Stages</EventType>
          <EventConcept>Wake|0</EventConcept>  (or "Stage 2 sleep|2" etc.)
          <Start>0.0</Start>
          <Duration>6480.0</Duration>
        </ScoredEvent>

        <ScoredEvent>
          <EventType>Respiratory|Respiratory</EventType>
          <EventConcept>Obstructive apnea|Obstructive Apnea</EventConcept>
          <Start>35</Start>
          <Duration>16</Duration>
          <SignalLocation>AIRFLOW</SignalLocation>
        </ScoredEvent>

    Stage event durations are always multiples of 30 seconds; each stage
    event expands to ``duration // 30`` consecutive epoch bins.

    Returns (hypnogram, respiratory_events).
    """
    tree = etree.parse(str(path))
    root = tree.getroot()

    # AASM stage codes used in SHHS NSRR XMLs (after the "|" in EventConcept).
    # Stage 4 appears in older R&K-scored recordings; AASM folds it into N3.
    stage_map = {
        "0": "W",
        "1": "N1",
        "2": "N2",
        "3": "N3",
        "4": "N3",
        "5": "REM",
        "9": "?",
    }
    respiratory_kinds = {
        "Obstructive apnea",
        "Central apnea",
        "Mixed apnea",
        "Hypopnea",
    }

    stage_bins: list[tuple[int, int, str]] = []  # (start_epoch, n_epochs, label)
    resp_events: list[RespiratoryEvent] = []
    max_epoch = 0

    for sev in root.iter():
        if etree.QName(sev).localname != "ScoredEvent":
            continue

        event_type = ""
        concept = ""
        start: float | None = None
        duration: float | None = None
        for child in sev:
            name = etree.QName(child).localname
            if name == "EventType":
                event_type = (child.text or "").strip()
            elif name in ("EventConcept", "Name"):
                concept = (child.text or "").strip()
            elif name == "Start":
                try:
                    start = float(child.text)
                except (TypeError, ValueError):
                    start = None
            elif name == "Duration":
                try:
                    duration = float(child.text)
                except (TypeError, ValueError):
                    duration = None

        if start is None or duration is None:
            continue

        if event_type.startswith("Stages"):
            parts = concept.split("|")
            code = parts[1].strip() if len(parts) == 2 else "9"
            label = stage_map.get(code, "?")
            start_ep = int(start // 30)
            n_ep = max(1, int(round(duration / 30)))
            stage_bins.append((start_ep, n_ep, label))
            max_epoch = max(max_epoch, start_ep + n_ep)
            continue

        primary = concept.split("|")[0].strip()
        if primary in respiratory_kinds:
            resp_events.append(
                RespiratoryEvent(start_sec=start, duration_sec=duration, kind=primary)
            )

    stages = np.full(max_epoch, "?", dtype=object)
    for start_ep, n_ep, label in stage_bins:
        stages[start_ep : start_ep + n_ep] = label
    return Hypnogram(stages=stages), resp_events
