"""30-second epoch extraction + labelling.

Epoch labelling rule (confirmed with supervisor, 2026-03-21):
    An epoch is labelled APNOEA if ≥10 seconds of its 30-second window
    overlap with any annotated apnoea or hypopnoea event.

Sleep-stage labels come directly from the NSRR hypnogram (one stage per
30-s epoch).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import Hypnogram, RespiratoryEvent

EPOCH_SECONDS = 30
APNOEA_OVERLAP_THRESHOLD = 10.0  # seconds


def apnoea_labels(
    n_epochs: int, events: list[RespiratoryEvent]
) -> np.ndarray:
    """Return a (n_epochs,) int array, 1 if epoch ≥10 s overlaps apnoea/hypopnoea.

    Epoch i spans [i*30, (i+1)*30) seconds. For each event, compute the
    overlap with every epoch it touches; mark an epoch when total overlap
    with any single event ≥ threshold. (AASM-style counting: multiple short
    events in the same epoch do NOT accumulate; threshold applies per event.)
    """
    labels = np.zeros(n_epochs, dtype=np.int8)
    for ev in events:
        ev_end = ev.start_sec + ev.duration_sec
        first = max(0, int(ev.start_sec // EPOCH_SECONDS))
        last = min(n_epochs - 1, int(ev_end // EPOCH_SECONDS))
        for i in range(first, last + 1):
            e_start = i * EPOCH_SECONDS
            e_end = e_start + EPOCH_SECONDS
            overlap = max(0.0, min(ev_end, e_end) - max(ev.start_sec, e_start))
            if overlap >= APNOEA_OVERLAP_THRESHOLD:
                labels[i] = 1
    return labels


def align_hypnogram(hypno: Hypnogram, n_epochs: int) -> np.ndarray:
    """Right-pad / truncate the hypnogram to match the epoch count."""
    stages = hypno.stages
    if len(stages) == n_epochs:
        return stages
    if len(stages) < n_epochs:
        out = np.full(n_epochs, "?", dtype=object)
        out[: len(stages)] = stages
        return out
    return stages[:n_epochs]


def sleep_mask(stages: np.ndarray) -> np.ndarray:
    """Boolean mask for **sleep epochs** — use as ``df[sleep_mask(stages)]`` to KEEP sleep.

    Returns True for AASM sleep stages (N1, N2, N3, REM); False for wake
    ("W") and unknown ("?"). Use this to filter training/evaluation sets
    to sleep-only — apnoea events are clinically scored only during sleep,
    so AHI = events per hour of sleep, not recording.

    Convention follows Olsen 2020, Phan 2022, Perslev 2021.

    Parameters
    ----------
    stages : np.ndarray
        Per-epoch sleep stage labels (object dtype, values in
        {"W", "N1", "N2", "N3", "REM", "?"}).

    Returns
    -------
    np.ndarray
        Boolean array, length == len(stages). True where stage ∈ {N1, N2, N3, REM}.
    """
    return np.isin(np.asarray(stages, dtype=object), ["N1", "N2", "N3", "REM"])


def wake_mask(stages: np.ndarray) -> np.ndarray:
    """DEPRECATED: misleadingly named — returns True for SLEEP epochs (not wake).

    Kept for backwards compatibility with code that does
    ``df.loc[wake_mask(stages)]`` (correct: keeps sleep) — that pattern relies
    on the historical True-for-sleep behaviour.

    New code should prefer :func:`sleep_mask` (semantically clearer name; same
    return value).

    Parameters
    ----------
    stages : np.ndarray
        Per-epoch sleep stage labels.

    Returns
    -------
    np.ndarray
        Boolean array, length == len(stages). True where stage ∈ {N1, N2, N3, REM}.
        IDENTICAL to ``sleep_mask(stages)``.
    """
    return sleep_mask(stages)


def subject_metadata(epoch_frame: "pd.DataFrame") -> dict[str, float]:
    """Per-subject summary metrics from an epoch frame.

    Computed from the hypnogram (sleep_stage column) + apnoea_label +
    optional hypoxic_burden_epoch column. NSRR harmonised values
    (nsrr_total_sleep_time / nsrr_sleep_efficiency / nsrr_waso from the
    SHHS CSV) are authoritative; this function provides a cross-validation
    sanity-check and per-subject row population for our own metadata table.

    Parameters
    ----------
    epoch_frame : pd.DataFrame
        Per-epoch dataframe (one row per 30-s epoch). Required columns:
        ``sleep_stage`` (object), ``apnoea_label`` (int 0/1). Optional
        columns: ``hypoxic_burden_epoch`` (float, %·s — produces
        ``hypoxic_burden_per_night`` if present).

    Returns
    -------
    dict[str, float]
        - ``tst_min``: Total Sleep Time (minutes), N1+N2+N3+REM × 0.5
        - ``tib_min``: Time in Bed (minutes), all epochs × 0.5
        - ``sleep_efficiency``: TST / TIB (0–1)
        - ``waso_min``: Wake After Sleep Onset (minutes), wake epochs after
          first sleep epoch
        - ``n_apnoea_epochs``: sum of apnoea_label
        - ``ahi_proxy_per_hr``: apnoea_epochs × 60 / TST_min  (per hour of sleep)
        - ``hypoxic_burden_per_night``: %·min/h, if input column present
        - ``n_epochs``: total epoch count
    """
    import numpy as np

    n_epochs = len(epoch_frame)
    tib_min = n_epochs * 30 / 60.0

    mask_sleep = sleep_mask(epoch_frame["sleep_stage"].values)
    # Defensive: catch future renames that silently invert this mask
    assert mask_sleep.dtype == bool and len(mask_sleep) == n_epochs, (
        "sleep_mask returned unexpected shape/dtype"
    )
    tst_min = float(mask_sleep.sum() * 30 / 60.0)

    if mask_sleep.any():
        first_sleep = int(np.argmax(mask_sleep))
        post_onset = epoch_frame.iloc[first_sleep:]
        waso_min = float((post_onset["sleep_stage"] == "W").sum() * 30 / 60.0)
    else:
        waso_min = 0.0

    # Count apnoea epochs only during sleep — AHI is events per hour of sleep,
    # so numerator and denominator should both be sleep-only.
    if "apnoea_label" in epoch_frame.columns:
        sleep_idx = epoch_frame["sleep_stage"].isin(["N1", "N2", "N3", "REM"])
        n_apnoea = float(((epoch_frame["apnoea_label"] == 1) & sleep_idx).sum())
    else:
        n_apnoea = 0.0

    out = {
        "n_epochs": float(n_epochs),
        "tib_min": float(tib_min),
        "tst_min": tst_min,
        "sleep_efficiency": float(tst_min / tib_min) if tib_min > 0 else float("nan"),
        "waso_min": waso_min,
        "n_apnoea_epochs": n_apnoea,
        "ahi_proxy_per_hr": (
            n_apnoea * 60.0 / tst_min if tst_min > 0 else float("nan")
        ),
    }

    if "hypoxic_burden_epoch" in epoch_frame.columns and tst_min > 0:
        # Per-night hypoxic burden: convert per-epoch %·s sum → %·min/h sleep
        total_pct_sec = float(epoch_frame["hypoxic_burden_epoch"].sum())
        out["hypoxic_burden_per_night"] = total_pct_sec / 60.0 / (tst_min / 60.0)
    return out


def build_epoch_frame(
    subject_id: int,
    cohort: str,
    n_epochs: int,
    hypno: Hypnogram,
    events: list[RespiratoryEvent],
) -> pd.DataFrame:
    """Construct a one-row-per-epoch DataFrame with labels (no features yet).

    Features from ECG / SpO2 / EEG / etc. are added on top of this frame by
    ``features.py`` — one column per feature, same row ordering.
    """
    stages = align_hypnogram(hypno, n_epochs)
    ap = apnoea_labels(n_epochs, events)
    return pd.DataFrame(
        {
            "subject_id": subject_id,
            "cohort": cohort,
            "epoch_idx": np.arange(n_epochs, dtype=np.int32),
            "epoch_start_sec": np.arange(n_epochs, dtype=np.int32) * EPOCH_SECONDS,
            "sleep_stage": stages,
            "apnoea_label": ap,
        }
    )
