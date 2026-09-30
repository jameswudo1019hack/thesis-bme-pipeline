"""Native-rate EDF reader (numpy only) for the Aim 2 raw-signal DL cache.

Why not ``io.read_edf`` (mne)?
    mne resamples every channel to the highest rate in the file (125 Hz in
    SHHS-1), so SaO2 / THOR / airflow come back interpolated and a full-night
    preload is ~450 MB of float64 and 2-6 s per subject. This reader returns
    each signal's *digital* samples at its native rate in ~0.02 s, which makes
    a lossless int8 / uint8 cache possible (the source is 8-bit for ECG and
    the belts).

Conventions
    * Digital samples are little-endian int16 (EDF spec).
    * Physical = (digital - dmin) * gain + pmin, gain = (pmax - pmin) / (dmax - dmin).
      The gain keeps its sign: SHHS THOR RES / ABDO RES / AIRFLOW have an
      inverted physical range [1, -1], so their gain is negative.
    * Signals are addressed by header index, not label, because labels can be
      duplicated (24 of 1,755 local SHHS-1 files have two ``AIRFLOW`` signals).

Airflow picker (fixed before any modelling; Aim 2 DL design section 2.1)
    Candidates are signals whose upper-cased, stripped label is in
    ``AIRFLOW_CANDIDATE_LABELS`` (duplicated labels included). The candidate
    with the highest whole-night standard deviation in LSB wins (ties: lowest
    header index); ``ok = std > 5 LSB``. This replaces the first-exact-match
    picker of ``scripts/process_batch.py:58-66``, which took a dead channel in
    831 of 1,755 local subjects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

UF_DATALESS = 0x40000000  # macOS FileProvider placeholder flag (see shhs.is_local)

AIRFLOW_CANDIDATE_LABELS = frozenset({"AIRFLOW", "NEW AIR", "NEWAIR", "NEW A/F", "AUX"})
AIRFLOW_LIVE_STD_LSB = 5.0

# Canonical channel -> accepted EDF labels (matched case-insensitively, in order).
CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "ECG": ("ECG", "EKG"),
    "THOR": ("THOR RES", "THOR"),
    "ABDO": ("ABDO RES", "ABDO"),
    "SPO2": ("SaO2", "SpO2"),
    "HR": ("H.R.", "HR", "PULSE"),
    "POSITION": ("POSITION",),
    "OXSTAT": ("OX stat", "OXSTAT"),
}
CANON = frozenset(CHANNEL_ALIASES) | {"AIRFLOW"}

# Expected native rates (Hz). Every SHHS-1 header surveyed so far matches
# these; the extractor still checks every file and refuses to guess.
EXPECTED_FS: dict[str, float] = {
    "ECG": 125.0,
    "THOR": 10.0,
    "ABDO": 10.0,
    "AIRFLOW": 10.0,
    "SPO2": 1.0,
    "HR": 1.0,
    "POSITION": 1.0,
    "OXSTAT": 1.0,
}
EXPECTED_RECORD_DURATION = 1.0


class EdfError(RuntimeError):
    """Malformed or unexpected EDF file."""


class DatalessFileError(EdfError):
    """The file is a cloud placeholder; reading it would trigger a download."""


class RateDeviation(EdfError):
    """A channel's native rate (or the record duration) differs from EXPECTED_FS."""


@dataclass(frozen=True)
class EdfSignal:
    index: int
    label: str
    transducer: str
    unit: str
    pmin: float
    pmax: float
    dmin: int
    dmax: int
    prefilter: str
    samples_per_record: int
    record_duration: float

    @property
    def fs(self) -> float:
        return self.samples_per_record / self.record_duration

    @property
    def gain(self) -> float:
        """Physical units per LSB, signed (negative for inverted ranges)."""
        return (self.pmax - self.pmin) / (self.dmax - self.dmin)

    @property
    def offset(self) -> float:
        """physical = digital * gain + offset."""
        return self.pmin - self.dmin * self.gain

    @property
    def sign(self) -> int:
        return 1 if self.gain >= 0 else -1

    def scaling(self) -> dict:
        return {
            "unit": self.unit,
            "pmin": self.pmin,
            "pmax": self.pmax,
            "dmin": self.dmin,
            "dmax": self.dmax,
            "gain": self.gain,
            "offset": self.offset,
            "sign": self.sign,
            "fs": self.fs,
        }


@dataclass(frozen=True)
class EdfHeader:
    path: str
    header_bytes: int
    n_records_header: int
    n_records: int  # complete records actually present in the file
    record_duration: float
    start_time: str
    file_size: int
    signals: tuple[EdfSignal, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_signals(self) -> int:
        return len(self.signals)

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.signals]

    @property
    def record_samples(self) -> int:
        return sum(s.samples_per_record for s in self.signals)

    @property
    def duration_s(self) -> float:
        return self.n_records * self.record_duration

    def indices(self, label: str) -> list[int]:
        """All header indices whose label equals ``label`` (case-insensitive, stripped)."""
        key = label.strip().upper()
        return [s.index for s in self.signals if s.label.strip().upper() == key]

    def find(self, canonical: str) -> list[int]:
        """Header indices for a canonical channel, using the first alias that matches."""
        if canonical == "AIRFLOW":
            return airflow_candidates(self)
        for alias in CHANNEL_ALIASES[canonical]:
            hits = self.indices(alias)
            if hits:
                return hits
        return []


def _is_dataless(path: Path) -> bool:
    return bool(os.stat(path).st_flags & UF_DATALESS)


def _field(raw: bytes, start: int, width: int) -> str:
    return raw[start : start + width].decode("latin-1").strip()


def read_header(path: Path | str, allow_dataless: bool = False) -> EdfHeader:
    """Parse the fixed and per-signal EDF header. Reads at most 256*(ns+1) bytes.

    Refuses cloud placeholders unless ``allow_dataless`` (reading even the
    header of a placeholder triggers a download of the file).
    """
    path = Path(path)
    if not allow_dataless and _is_dataless(path):
        raise DatalessFileError(f"placeholder, refusing to read: {path}")
    file_size = os.stat(path).st_size
    with open(path, "rb") as fh:
        fixed = fh.read(256)
        if len(fixed) < 256:
            raise EdfError(f"{path}: file shorter than the 256-byte fixed header")
        try:
            header_bytes = int(_field(fixed, 184, 8))
            n_records_header = int(_field(fixed, 236, 8))
            record_duration = float(_field(fixed, 244, 8))
            ns = int(_field(fixed, 252, 4))
        except ValueError as exc:
            raise EdfError(f"{path}: unparseable fixed header ({exc})") from exc
        if header_bytes != 256 * (ns + 1):
            raise EdfError(f"{path}: header_bytes={header_bytes} != 256*(ns+1)={256 * (ns + 1)}")
        sig = fh.read(256 * ns)
    if len(sig) < 256 * ns:
        raise EdfError(f"{path}: truncated signal header")

    offset = 0

    def take(width: int) -> list[str]:
        nonlocal offset
        out = [_field(sig, offset + i * width, width) for i in range(ns)]
        offset += width * ns
        return out

    labels = take(16)
    transducer = take(80)
    unit = take(8)
    pmin = take(8)
    pmax = take(8)
    dmin = take(8)
    dmax = take(8)
    prefilter = take(80)
    nsamp = take(8)

    signals = []
    for i in range(ns):
        try:
            s = EdfSignal(
                index=i,
                label=labels[i],
                transducer=transducer[i],
                unit=unit[i],
                pmin=float(pmin[i]),
                pmax=float(pmax[i]),
                dmin=int(dmin[i]),
                dmax=int(dmax[i]),
                prefilter=prefilter[i],
                samples_per_record=int(nsamp[i]),
                record_duration=record_duration,
            )
        except ValueError as exc:
            raise EdfError(f"{path}: signal {i} ({labels[i]!r}) has an unparseable field ({exc})") from exc
        if s.dmax == s.dmin:
            raise EdfError(f"{path}: signal {i} ({s.label!r}) has dmax == dmin")
        signals.append(s)

    record_bytes = 2 * sum(s.samples_per_record for s in signals)
    if record_bytes <= 0:
        raise EdfError(f"{path}: zero-length data record")
    n_complete = (file_size - header_bytes) // record_bytes
    notes: list[str] = []
    if n_records_header < 0:
        notes.append(f"n_records=-1 in header; derived {n_complete} from file size")
        n_records = int(n_complete)
    elif n_complete < n_records_header:
        notes.append(f"truncated: header says {n_records_header} records, file holds {n_complete}")
        n_records = int(n_complete)
    else:
        n_records = n_records_header
        extra = file_size - header_bytes - n_records * record_bytes
        if extra:
            notes.append(f"{extra} trailing bytes after the last declared record")
    return EdfHeader(
        path=str(path),
        header_bytes=header_bytes,
        n_records_header=n_records_header,
        n_records=n_records,
        record_duration=record_duration,
        start_time=_field(fixed, 176, 8),
        file_size=file_size,
        signals=tuple(signals),
        notes=tuple(notes),
    )


def read_digital(
    path: Path | str,
    indices: Iterable[int] | None = None,
    hdr: EdfHeader | None = None,
    allow_dataless: bool = False,
) -> dict[int, np.ndarray]:
    """Digital int16 samples for the requested header indices (all if None), at native rate.

    Returns ``{index: 1-D int16 array of length n_records * samples_per_record}``.
    """
    path = Path(path)
    if not allow_dataless and _is_dataless(path):
        raise DatalessFileError(f"placeholder, refusing to read: {path}")
    if hdr is None:
        hdr = read_header(path, allow_dataless=allow_dataless)
    want = list(range(hdr.n_signals)) if indices is None else sorted(set(int(i) for i in indices))
    for i in want:
        if not 0 <= i < hdr.n_signals:
            raise IndexError(f"signal index {i} out of range (ns={hdr.n_signals})")
    rs = hdr.record_samples
    raw = np.fromfile(path, dtype="<i2", count=hdr.n_records * rs, offset=hdr.header_bytes)
    if raw.size != hdr.n_records * rs:
        raise EdfError(f"{path}: expected {hdr.n_records * rs} samples, read {raw.size}")
    raw = raw.reshape(hdr.n_records, rs)
    starts = np.concatenate([[0], np.cumsum([s.samples_per_record for s in hdr.signals])])
    return {
        i: np.ascontiguousarray(raw[:, starts[i] : starts[i + 1]]).reshape(-1).astype(np.int16, copy=False)
        for i in want
    }


def to_physical(dig: np.ndarray, sig: EdfSignal) -> np.ndarray:
    """Digital -> physical (header units, sign kept) as float32."""
    return ((dig.astype(np.float64) - sig.dmin) * sig.gain + sig.pmin).astype(np.float32)


# --------------------------------------------------------------------------- rates


def check_rates(hdr: EdfHeader, picks: dict[str, list[int]]) -> list[str]:
    """Deviations from EXPECTED_FS / EXPECTED_RECORD_DURATION for the picked signals.

    ``picks`` maps a canonical name to header indices (e.g. all airflow
    candidates). Returns human-readable deviation strings; empty if all match.
    """
    out: list[str] = []
    if hdr.record_duration != EXPECTED_RECORD_DURATION:
        out.append(f"record_duration={hdr.record_duration:g}s (expected {EXPECTED_RECORD_DURATION:g}s)")
    for canon, idxs in picks.items():
        want = EXPECTED_FS[canon]
        for i in idxs:
            fs = hdr.signals[i].fs
            if fs != want:
                out.append(f"{canon} [{i}:{hdr.signals[i].label}] fs={fs:g} Hz (expected {want:g})")
    return out


def assert_rates(hdr: EdfHeader, picks: dict[str, list[int]]) -> None:
    dev = check_rates(hdr, picks)
    if dev:
        raise RateDeviation("; ".join(dev))


# --------------------------------------------------------------------------- airflow


@dataclass(frozen=True)
class AirflowCandidate:
    index: int
    label: str
    std_lsb: float
    fs: float
    unit: str
    pmin: float
    pmax: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "std_lsb": self.std_lsb,
            "fs": self.fs,
            "unit": self.unit,
            "pmin": self.pmin,
            "pmax": self.pmax,
        }


@dataclass(frozen=True)
class AirflowPick:
    index: int | None
    label: str | None
    std_lsb: float
    ok: bool
    candidates: tuple[AirflowCandidate, ...]

    @property
    def rejected(self) -> tuple[AirflowCandidate, ...]:
        return tuple(c for c in self.candidates if c.index != self.index)


def airflow_candidates(hdr: EdfHeader) -> list[int]:
    """Header indices of every airflow-like signal (duplicated labels included)."""
    return [s.index for s in hdr.signals if s.label.strip().upper() in AIRFLOW_CANDIDATE_LABELS]


def pick_airflow(
    hdr: EdfHeader,
    data: dict[int, np.ndarray],
    n_samples: int | None = None,
    live_std_lsb: float = AIRFLOW_LIVE_STD_LSB,
    candidates: Iterable[int] | None = None,
) -> AirflowPick:
    """Highest whole-night std (LSB) among airflow candidates; ``ok = std > live_std_lsb``.

    ``data`` must hold the digital samples of every candidate. ``n_samples``
    optionally restricts the std to the first n samples (the stored segment).
    ``candidates`` restricts the choice to these header indices (default: all
    of ``airflow_candidates(hdr)``; the extractor passes only the on-rate ones).
    Ties go to the lowest header index. No candidate -> ``index=None, ok=False``.
    """
    cands: list[AirflowCandidate] = []
    for i in (airflow_candidates(hdr) if candidates is None else sorted(int(c) for c in candidates)):
        if i not in data:
            raise KeyError(f"digital data for airflow candidate {i} ({hdr.signals[i].label}) not supplied")
        x = data[i] if n_samples is None else data[i][:n_samples]
        s = hdr.signals[i]
        cands.append(
            AirflowCandidate(
                index=i,
                label=s.label,
                std_lsb=float(np.std(x.astype(np.float64))) if x.size else 0.0,
                fs=s.fs,
                unit=s.unit,
                pmin=s.pmin,
                pmax=s.pmax,
            )
        )
    if not cands:
        return AirflowPick(index=None, label=None, std_lsb=0.0, ok=False, candidates=())
    best = max(cands, key=lambda c: (c.std_lsb, -c.index))
    return AirflowPick(
        index=best.index,
        label=best.label,
        std_lsb=best.std_lsb,
        ok=bool(best.std_lsb > live_std_lsb),
        candidates=tuple(cands),
    )
