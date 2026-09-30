"""Synthetic EDF + NSRR XML writers so the DL-cache tests need no real SHHS files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sig:
    label: str
    fs: int
    digital: np.ndarray  # int16-compatible, length n_records * fs * record_duration
    pmin: float = -1.0
    pmax: float = 1.0
    dmin: int = -128
    dmax: int = 127
    unit: str = ""
    transducer: str = ""


def _f(value, width: int) -> bytes:
    s = value if isinstance(value, str) else (f"{value:g}" if isinstance(value, float) else str(value))
    b = s.encode("latin-1")[:width]
    return b + b" " * (width - len(b))


def write_edf(path: Path, signals: list[Sig], n_records: int, record_duration: float = 1.0,
              start_time: str = "22.00.00", header_n_records: int | None = None,
              truncate_records: int | None = None) -> Path:
    """Write a plain EDF. ``header_n_records`` overrides the declared count;
    ``truncate_records`` writes only that many data records (a truncated file)."""
    ns = len(signals)
    nsamp = [int(round(s.fs * record_duration)) for s in signals]
    for s, n in zip(signals, nsamp):
        assert len(s.digital) == n * n_records, (s.label, len(s.digital), n * n_records)
    hdr = b"".join([
        _f("0", 8), _f("X X X X", 80), _f("Startdate 01-JAN-1985 X X X", 80),
        _f("01.01.85", 8), _f(start_time, 8), _f(256 * (ns + 1), 8), _f("", 44),
        _f(n_records if header_n_records is None else header_n_records, 8),
        _f(float(record_duration) if record_duration != int(record_duration) else int(record_duration), 8),
        _f(ns, 4),
    ])
    for fieldname, width in (("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8), ("pmax", 8),
                             ("dmin", 8), ("dmax", 8)):
        hdr += b"".join(_f(getattr(s, fieldname), width) for s in signals)
    hdr += b"".join(_f("", 80) for _ in signals)  # prefilter
    hdr += b"".join(_f(n, 8) for n in nsamp)
    hdr += b"".join(_f("", 32) for _ in signals)
    assert len(hdr) == 256 * (ns + 1)
    n_write = n_records if truncate_records is None else truncate_records
    blocks = [np.asarray(s.digital, dtype="<i2").reshape(n_records, n) for s, n in zip(signals, nsamp)]
    data = np.concatenate(blocks, axis=1)[:n_write]
    with open(path, "wb") as fh:
        fh.write(hdr)
        fh.write(np.ascontiguousarray(data, dtype="<i2").tobytes())
    return path


STAGE_CONCEPT = {"W": "Wake|0", "N1": "Stage 1 sleep|1", "N2": "Stage 2 sleep|2", "N3": "Stage 3 sleep|3",
                 "N4": "Stage 4 sleep|4", "REM": "REM sleep|5", "?": "Unscored|9"}
EVENT_CONCEPT = {"Obstructive apnea": "Obstructive apnea|Obstructive Apnea",
                 "Central apnea": "Central apnea|Central Apnea",
                 "Mixed apnea": "Mixed apnea|Mixed Apnea",
                 "Hypopnea": "Hypopnea|Hypopnea"}


def write_nsrr_xml(path: Path, stages: list[str], events: list[tuple[float, float, str]],
                   duration_s: float) -> Path:
    """NSRR-style XML: one 30-s stage event per epoch (runs merged) + respiratory events."""
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>', "<PSGAnnotation>",
             "<SoftwareVersion>Compumedics</SoftwareVersion>", "<EpochLength>30</EpochLength>",
             "<ScoredEvents>",
             "<ScoredEvent><EventType/><EventConcept>Recording Start Time</EventConcept><Start>0</Start>"
             f"<Duration>{duration_s}</Duration><ClockTime>00.00.00 22.00.00</ClockTime></ScoredEvent>"]
    i = 0
    while i < len(stages):
        j = i
        while j < len(stages) and stages[j] == stages[i]:
            j += 1
        parts.append(f"<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>{STAGE_CONCEPT[stages[i]]}"
                     f"</EventConcept><Start>{i * 30.0}</Start><Duration>{(j - i) * 30.0}</Duration></ScoredEvent>")
        i = j
    for start, dur, kind in events:
        parts.append(f"<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>"
                     f"{EVENT_CONCEPT[kind]}</EventConcept><Start>{start}</Start><Duration>{dur}</Duration>"
                     "<SignalLocation>AIRFLOW</SignalLocation></ScoredEvent>")
    parts += ["</ScoredEvents>", "</PSGAnnotation>"]
    path.write_text("\n".join(parts) + "\n")
    return path


def spo2_digital(pct_int: np.ndarray) -> np.ndarray:
    """Integer SpO2 % -> SHHS-like 16-bit digital: j = ceil(p * 2.56), d = 256 j - 32769 (dropout -> -32767).

    Approximates the lattice seen on local SHHS-1 files (p=79..97 match exactly, e.g. p=84 -> 22527;
    real files sit one LSB lower near 98-99 %). Only "one digital value per integer %" matters here.
    """
    j = np.ceil(np.asarray(pct_int, dtype=np.float64) * 2.56 - 1e-9).astype(np.int64)
    d = j * 256 - 32768 - 1
    d[j == 0] = -32767
    return np.clip(d, -32768, 32767).astype(np.int16)


def hr_digital(bpm_int: np.ndarray) -> np.ndarray:
    """SHHS-like H.R. digital: a lattice of ~204.8 LSB steps (0.78 bpm under the header scaling, which
    matches 60/RR on real files). ``bpm_int`` here is a lattice index, not bpm."""
    return np.clip(np.rint(np.asarray(bpm_int) * 204.8) - 32768, -32768, 32767).astype(np.int16)


def make_subject(root: Path, sid: int, n_epochs: int = 12, tail_s: int = 17, seed: int = 0,
                 airflow_labels: tuple[str, ...] = ("AIRFLOW", "NEW AIR"), live_airflow: str = "NEW AIR",
                 with_oxstat: bool = True, ecg_fs: int = 125,
                 stages: list[str] | None = None,
                 events: list[tuple[float, float, str]] | None = None,
                 fs_override: dict[str, int] | None = None,
                 extra_signals: tuple[tuple[str, int], ...] = ()) -> dict:
    """Write shhs1-<sid>.edf + shhs1-<sid>-nsrr.xml under ``root``; return the ground truth.

    ``fs_override`` ({label: fs}) re-rates a signal (its samples are regenerated at
    that rate; read the truth from the returned ``signals``); ``extra_signals``
    appends (label, fs) signals of small random samples, e.g. a stray AUX.
    """
    rng = np.random.default_rng(seed)
    n_rec = n_epochs * 30 + tail_s
    ecg = rng.integers(-128, 128, n_rec * ecg_fs).astype(np.int16)
    thor = rng.integers(-60, 60, n_rec * 10).astype(np.int16)
    abdo = rng.integers(-60, 60, n_rec * 10).astype(np.int16)
    pct = rng.integers(85, 101, n_rec)
    pct[5:8] = 0  # dropout
    spo2 = spo2_digital(pct)
    hr = hr_digital(rng.integers(45, 90, n_rec))
    pos = rng.integers(0, 4, n_rec).astype(np.int16)
    sigs = [
        Sig("SaO2", 1, spo2, 0.0, 100.0, -32768, 32767),
        Sig("H.R.", 1, hr, 0.0, 250.0, -32768, 32767),
        Sig("EEG(sec)", 125, rng.integers(-128, 128, n_rec * 125).astype(np.int16), -125.0, 125.0, unit="uV"),
        Sig("ECG", ecg_fs, ecg, -1.25, 1.25, unit="mV"),
        Sig("THOR RES", 10, thor, 1.0, -1.0),
        Sig("ABDO RES", 10, abdo, 1.0, -1.0),
    ]
    air = {}
    for lab in airflow_labels:
        if lab.upper() == live_airflow.upper():
            x = rng.integers(-90, 90, n_rec * 10).astype(np.int16)
        else:
            x = rng.integers(-2, 3, n_rec * 10).astype(np.int16)  # dead: std ~1.4 LSB
        unit, lo, hi = ("", 1.0, -1.0) if lab.upper() == "AIRFLOW" else ("uV", -125.0, 125.0)
        sigs.append(Sig(lab, 10, x, lo, hi, unit=unit))
        air.setdefault(lab, []).append(x)
    sigs.append(Sig("POSITION", 1, pos, 0.0, 3.0, 0, 3))
    if with_oxstat:
        sigs.append(Sig("OX stat", 1, rng.integers(0, 3, n_rec).astype(np.int16), 0.0, 3.0, 0, 3))
    for lab, fs in (fs_override or {}).items():
        for k, sg in enumerate(sigs):
            if sg.label == lab:
                d = rng.integers(max(sg.dmin, -100), min(sg.dmax, 100) + 1, n_rec * fs).astype(np.int16)
                sigs[k] = Sig(sg.label, fs, d, sg.pmin, sg.pmax, sg.dmin, sg.dmax, sg.unit, sg.transducer)
    for lab, fs in extra_signals:
        sigs.append(Sig(lab, fs, rng.integers(-3, 4, n_rec * fs).astype(np.int16), -1.0, 1.0))
    edf = write_edf(root / f"shhs1-{sid}.edf", sigs, n_rec)
    if stages is None:
        stages = (["W"] * 2 + ["N1", "N2", "N2", "N3", "REM", "W", "N2", "N2", "REM", "?"] * 10)[:n_epochs]
    if events is None:
        events = [(35.0, 16.0, "Obstructive apnea"), (95.5, 10.0, "Hypopnea"), (100.0, 12.0, "Central apnea"),
                  (205.0, 25.0, "Mixed apnea")]
    xml = write_nsrr_xml(root / f"shhs1-{sid}-nsrr.xml", stages, events, float(n_rec))
    return {"edf": edf, "xml": xml, "n_rec": n_rec, "n_epochs": n_rec // 30, "ecg": ecg, "thor": thor,
            "abdo": abdo, "spo2": spo2, "hr": hr, "pos": pos, "air": air, "signals": sigs,
            "stages": stages, "events": events}
