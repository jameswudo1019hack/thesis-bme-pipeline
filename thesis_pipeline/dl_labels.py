"""Labels for the Aim 2 raw-signal DL arm.

Two resolutions:

* **Epoch labels** (30 s) come ONLY from ``epochs.build_epoch_frame``, the
  exact code that produced the Aim 2 LightGBM labels. Never re-derive them
  from the per-second mask: a ">= 10 s of the mask" rule disagrees with the
  per-event overlap rule (``epochs.apnoea_labels``) on 0.4-0.8 % of sleep
  epochs.
* **Per-second targets** (Olsen 2020 style): second ``s`` covers
  ``[s, s+1)`` and is positive for event kind k if its midpoint ``s + 0.5``
  lies in ``[start, start + duration)`` of an event of that kind. The four
  kinds are the ones ``io.read_nsrr_xml`` returns, stored as a bitmask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .epochs import EPOCH_SECONDS, build_epoch_frame
from .io import Hypnogram, RespiratoryEvent

# Bitmask for the per-second event array (uint8). Keys are the
# ``RespiratoryEvent.kind`` strings emitted by io.read_nsrr_xml.
EVENT_BITS: dict[str, int] = {
    "Obstructive apnea": 1,
    "Central apnea": 2,
    "Mixed apnea": 4,
    "Hypopnea": 8,
}
ANY_EVENT = 1 | 2 | 4 | 8

# int8 stage codes (the NSRR "EventConcept" digits; ? = unscored/unknown).
STAGE_CODES: dict[str, int] = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 5, "?": 9}
SLEEP_CODES = (1, 2, 3, 5)


def encode_stages(stages: np.ndarray) -> np.ndarray:
    """Object array of stage strings -> int8 codes (raises on an unknown string)."""
    s = np.asarray(stages, dtype=object)
    out = np.empty(s.shape, dtype=np.int8)
    for name, code in STAGE_CODES.items():
        out[s == name] = code
    known = np.isin(s, list(STAGE_CODES))
    if not known.all():
        bad = sorted({str(x) for x in s[~known]})
        raise ValueError(f"unknown stage label(s): {bad}")
    return out


def sleep_epochs(epoch_stage: np.ndarray) -> np.ndarray:
    """Boolean per-epoch sleep mask from int8 codes (N1, N2, N3, REM)."""
    return np.isin(np.asarray(epoch_stage), SLEEP_CODES)


def sec_sleep(epoch_stage: np.ndarray, n_sec: int | None = None) -> np.ndarray:
    """Per-second uint8 sleep mask (1 in N1/N2/N3/REM epochs) from int8 epoch codes.

    ``n_sec`` defaults to ``30 * n_epochs``; seconds beyond the hypnogram are 0.
    """
    per_epoch = sleep_epochs(epoch_stage).astype(np.uint8)
    out = np.repeat(per_epoch, EPOCH_SECONDS)
    if n_sec is None:
        return out
    if n_sec <= out.size:
        return out[:n_sec]
    return np.concatenate([out, np.zeros(n_sec - out.size, dtype=np.uint8)])


def sec_event_mask(events: list[RespiratoryEvent], n_sec: int) -> np.ndarray:
    """Per-second uint8 bitmask of event kinds; bit set iff ``s + 0.5`` is in ``[start, end)``.

    Seconds ``s`` with ``start <= s + 0.5 < end`` are exactly
    ``ceil(start - 0.5) <= s < ceil(end - 0.5)``. Events outside
    ``[0, n_sec)`` are clipped.
    """
    out = np.zeros(int(n_sec), dtype=np.uint8)
    for ev in events:
        if ev.kind not in EVENT_BITS:
            raise ValueError(f"unknown respiratory event kind {ev.kind!r}")
        if ev.duration_sec <= 0:
            continue
        a = int(np.ceil(ev.start_sec - 0.5))
        b = int(np.ceil(ev.start_sec + ev.duration_sec - 0.5))
        a = max(a, 0)
        b = min(b, out.size)
        if b > a:
            out[a:b] |= EVENT_BITS[ev.kind]
    return out


def events_table(events: list[RespiratoryEvent]) -> list[list[float | int]]:
    """Compact JSON-able event list: ``[[start_sec, duration_sec, bit], ...]`` in XML order."""
    return [[float(ev.start_sec), float(ev.duration_sec), EVENT_BITS[ev.kind]] for ev in events]


def epoch_labels(
    subject_id: int,
    n_epochs: int,
    hypno: Hypnogram,
    events: list[RespiratoryEvent],
    cohort: str = "shhs1",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(epoch_stage int8, apnoea_label int8, frame) via epochs.build_epoch_frame (Aim 2 code path)."""
    frame = build_epoch_frame(
        subject_id=subject_id, cohort=cohort, n_epochs=n_epochs, hypno=hypno, events=events
    )
    stage = encode_stages(frame["sleep_stage"].to_numpy())
    apnoea = frame["apnoea_label"].to_numpy().astype(np.int8)
    return stage, apnoea, frame
