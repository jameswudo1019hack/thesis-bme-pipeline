"""Matched-input LightGBM feature cells for the Aim 2 raw-signal DL comparison.

The DL configs see a restricted set of raw channels (M = RR + EDR from the ECG;
P4 = M + thoracic and abdominal belts). The LightGBM comparator for each config
must see the same modalities, so each cell is a strict subset of the canonical
217-column ``physio_only`` feature list
(``models/recovery_2026-05-03/aim2_v85_taxonomy/physio_only/feature_list.json``):

  physio_only_repro  all 217 columns, unchanged (reproduction of the canonical fit)
  ecg_only           base feature starts with ``hr_`` or ``hrv_``          (8 bases x 7 = 56)
  ecg_belt           ecg_only + resp_thor_rms, resp_abdo_rms,
                     resp_thor_abdo_corr                                   (11 bases x 7 = 77)

A column's *base feature* is its name with the contextual suffix removed
(``_lag1``, ``_lead1``, ``_roll5_mean`` ...); every base carries 7 columns (raw
value + 6 contextual variants). ``resp_breath_rate_bpm`` is computed from the
AIRFLOW channel (features.py), so it is excluded from both matched cells even
though ``feature_groups`` files it under physio.

Column order follows the physio_only list, so cells are order-preserving
subsequences of it.

The cells are matched by *modality*, not by information content. The declared
differences between each cell and its DL config are in ``INPUT_MATCH_CAVEATS``
(returned per cell by ``caveats_for`` and written into every run record); the
pre-registration and the results text must carry them. In particular the belts
are label-proximal (SHHS MOP 6.610), so ecg_belt and every belt contrast carry
that caveat.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

CONTEXT_SUFFIXES: tuple[str, ...] = (
    "_lag1", "_lead1", "_roll5_mean", "_roll5_std", "_roll11_mean", "_roll11_std",
)
N_VARIANTS_PER_BASE = 1 + len(CONTEXT_SUFFIXES)

CELLS: tuple[str, ...] = ("physio_only_repro", "ecg_only", "ecg_belt")

ECG_BASE_PREFIXES: tuple[str, ...] = ("hr_", "hrv_")
BELT_BASES: tuple[str, ...] = ("resp_thor_rms", "resp_abdo_rms", "resp_thor_abdo_corr")

# Never allowed in a matched cell: airflow-derived, SpO2, EEG and position inputs.
FORBIDDEN_BASES: tuple[str, ...] = ("resp_breath_rate_bpm",)
FORBIDDEN_PREFIXES: tuple[str, ...] = ("eeg_", "position_", "spo2_", "resp_airflow")

EXPECTED_N_COLUMNS = {"physio_only_repro": 217, "ecg_only": 56, "ecg_belt": 77}
EXPECTED_ECG_BASES: tuple[str, ...] = (
    "hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
    "hrv_lf_power", "hrv_hf_power", "hrv_lf_hf_ratio", "hrv_total_power_freq",
)

# Declared input-match differences between the LightGBM cells and the DL configs
# (review of 2026-09-30). None is a code defect; each must be stated wherever the
# matched-cell contrasts are reported.
INPUT_MATCH_CAVEATS: dict[str, str] = {
    "edr_no_engineered_counterpart": (
        "EDR (a DL input to M and P4) has no engineered counterpart in the LightGBM cells."
    ),
    "absolute_scale": (
        "The DL configs apply per-window SoftMinMax ((x - q05) / (q95 - q05) over each 5-min window) "
        "to RR, EDR, THOR and ABDO, after stage B has already removed each subject's median RR and EDR "
        "level, so M and P4 cannot see absolute heart rate, absolute RR variability or absolute belt "
        "amplitude within a window. The LightGBM cells see hr_mean in bpm, SDNN/RMSSD in ms, LF/HF/"
        "total power in ms^2 and belt RMS in raw units. The cells are matched by modality, not by "
        "information content: M < ecg_only must not be read as 'engineered ECG features beat learned RR'."
    ),
    "temporal_context": (
        "LightGBM contextual variants (roll11 = epochs i-5..i+5) give each epoch 150 s of raw signal "
        "per side, about 195 s for the hrv_lf/hf/lf_hf/total_power bases whose base value already uses "
        "a centred 120-s window. The DL 5-min window scores each epoch at a fixed centre position "
        "p = 0..5 with 60+30p s before and 60+30(5-p) s after it, so edge epochs see only 60 s on one "
        "side. Both arms see wake context across stage boundaries."
    ),
    "rr_cleaning": (
        "RR cleaning differs. LightGBM: R peaks at the native 125 Hz (8-ms RR resolution), de Chazal "
        "robust-RR merge/subdivide, RR kept only in [400, 1500] ms (40-150 bpm) with successive ratio "
        "in (0.6, 1.6), epochs with < 3 valid RR set to NaN; the frequency-domain bases instead gate "
        "each 2-min window on median RR in [400, 1500] ms. DL: the same NeuroKit2 detector after "
        "resampling to 256 Hz, Kubios fixpeaks per gap-split segment, RR clipped to [0.3, 2.0] s "
        "(30-200 bpm). Epochs outside 40-150 bpm are NaN for the LightGBM ECG bases but present to M. "
        "Report the per-subject hr_mean NaN fraction (train/val, qc_hr_mean_nan_trainval.parquet) "
        "next to the DL RR QC gap fraction."
    ),
    "belt_label_proximity": (
        "Belts are label-proximal: under SHHS MOP 6.610 a hypopnoea is a >= 30 % amplitude drop in any "
        "respiratory signal, the THOR/ABDO belts included, for > 10 s. This applies to P4, to ecg_belt, "
        "to ecg_belt - ecg_only and to the (P4 - M) - (ecg_belt - ecg_only) contrast; frame those "
        "results against the rule that generated the labels, not as 'physio' evidence."
    ),
}
_CELL_CAVEATS: dict[str, tuple[str, ...]] = {
    "physio_only_repro": (),
    "ecg_only": ("edr_no_engineered_counterpart", "absolute_scale", "temporal_context", "rr_cleaning"),
    "ecg_belt": ("edr_no_engineered_counterpart", "absolute_scale", "temporal_context", "rr_cleaning",
                 "belt_label_proximity"),
}


def caveats_for(cell: str) -> dict[str, str]:
    """Declared input-match caveats that apply to ``cell`` (empty for the reproduction cell)."""
    if cell not in _CELL_CAVEATS:
        raise ValueError(f"unknown cell {cell!r} (expected one of {CELLS})")
    return {k: INPUT_MATCH_CAVEATS[k] for k in _CELL_CAVEATS[cell]}


def base_feature(col: str) -> str:
    """Strip one contextual suffix: ``hrv_sdnn_roll5_std`` -> ``hrv_sdnn``."""
    for s in CONTEXT_SUFFIXES:
        if col.endswith(s):
            return col[: -len(s)]
    return col


def _check_physio_list(physio_cols: Sequence[str]) -> None:
    cols = list(physio_cols)
    assert len(cols) == len(set(cols)), "physio_only list has duplicate columns"
    assert len(cols) == EXPECTED_N_COLUMNS["physio_only_repro"], (
        f"physio_only list has {len(cols)} columns, expected 217")
    per_base = Counter(base_feature(c) for c in cols)
    bad = {b: n for b, n in per_base.items() if n != N_VARIANTS_PER_BASE}
    assert not bad, f"bases without exactly {N_VARIANTS_PER_BASE} variants: {bad}"


def cell_columns(cell: str, physio_cols: Sequence[str]) -> list[str]:
    """Columns of ``cell`` as an order-preserving subset of the 217-column physio_only list."""
    _check_physio_list(physio_cols)
    cols = list(physio_cols)
    if cell == "physio_only_repro":
        out = cols
    elif cell == "ecg_only":
        out = [c for c in cols if base_feature(c).startswith(ECG_BASE_PREFIXES)]
    elif cell == "ecg_belt":
        out = [c for c in cols
               if base_feature(c).startswith(ECG_BASE_PREFIXES) or base_feature(c) in BELT_BASES]
    else:
        raise ValueError(f"unknown cell {cell!r} (expected one of {CELLS})")
    check_cell_columns(cell, out, physio_cols)
    return out


def check_cell_columns(cell: str, cols: Iterable[str], physio_cols: Sequence[str]) -> None:
    """Assert the column-set contract for ``cell``; raises AssertionError on any violation."""
    cols = list(cols)
    physio = list(physio_cols)
    assert len(cols) == EXPECTED_N_COLUMNS[cell], (
        f"{cell}: {len(cols)} columns, expected {EXPECTED_N_COLUMNS[cell]}")
    assert len(cols) == len(set(cols)), f"{cell}: duplicate columns"
    pos = {c: i for i, c in enumerate(physio)}
    missing = [c for c in cols if c not in pos]
    assert not missing, f"{cell}: columns not in physio_only list: {missing[:5]}"
    order = [pos[c] for c in cols]
    assert order == sorted(order), f"{cell}: column order differs from the physio_only list"
    if cell == "physio_only_repro":
        assert cols == physio, "physio_only_repro must equal the physio_only list exactly"
        return

    bases = [base_feature(c) for c in cols]
    forbidden = [c for c, b in zip(cols, bases)
                 if b in FORBIDDEN_BASES or b.startswith(FORBIDDEN_PREFIXES)]
    assert not forbidden, f"{cell}: forbidden (airflow/SpO2/EEG/position) columns: {forbidden}"
    ecg_bases = sorted({b for b in bases if b.startswith(ECG_BASE_PREFIXES)})
    assert ecg_bases == sorted(EXPECTED_ECG_BASES), f"{cell}: ECG bases {ecg_bases}"
    belt = sorted({b for b in bases if b in BELT_BASES})
    other = sorted({b for b in bases if not b.startswith(ECG_BASE_PREFIXES) and b not in BELT_BASES})
    assert not other, f"{cell}: unexpected bases {other}"
    if cell == "ecg_only":
        assert not belt, f"ecg_only contains belt features {belt}"
    elif cell == "ecg_belt":
        assert belt == sorted(BELT_BASES), f"ecg_belt belt bases {belt}"
    else:
        raise ValueError(f"unknown cell {cell!r}")
    per_base = Counter(bases)
    assert all(n == N_VARIANTS_PER_BASE for n in per_base.values()), (
        f"{cell}: bases without all {N_VARIANTS_PER_BASE} contextual variants: {per_base}")
