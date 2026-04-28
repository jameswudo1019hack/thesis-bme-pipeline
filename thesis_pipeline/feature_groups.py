"""Feature taxonomy for the Aim 2 ablation: AASM-rule-defining vs physiological.

Background
----------
The AASM apnoea/hypopnoea label is *defined* by:
  1. ≥90% airflow drop for ≥10 s (apnoea), OR
  2. ≥30% airflow drop with **≥3% SpO₂ desat** OR arousal (hypopnoea).

Some hand-crafted features in our pipeline (e.g. ``hypoxic_burden_epoch``,
``odi3_count``, ``desat_depth``, ``resp_airflow``) directly measure one of
those defining conditions — they're not "predicting" the label, they're
re-implementing the scoring rule. We call those **AASM-rule features**.

Other features (HRV, EEG, position, statistical SpO₂ moments) measure
autonomic / cortical / postural *responses* to apnoeic events. They're
predictive without being part of the label definition. We call those
**physiological features**.

This module classifies each column so we can run a 3-way ablation:
  full         — all features (headline number, caveated)
  physio_only  — drop AASM-rule features (defensible novel SOTA)
  aasm_only    — only AASM-rule features (sanity check: rule predicts itself)

Categorisation rule
-------------------
A column is in a class if it starts with any prefix from that class.
Contextual derivatives (``_lag1``, ``_lead1``, ``_roll5_mean``, etc.) inherit
their base feature's class.

Unknown columns default to **physiological** (conservative: under-flagging
AASM-rule features would inflate the physio result, but over-flagging
physiological features as AASM would deflate it — we'd rather err on the
side of a stricter physio number being achievable, then explicitly review
the unknown list).
"""
from __future__ import annotations

from typing import Iterable

# Features that directly measure an AASM-rule defining condition
AASM_RULE_PREFIXES: tuple[str, ...] = (
    "hypoxic_burden",   # continuous integration of ≥3% SpO₂ desat
    "odi3_count",       # count of 3% desats per epoch
    "odi4_count",       # count of 4% desats per epoch
    "desat_depth",      # magnitude of SpO₂ drops
    "resp_airflow",     # airflow drop = primary apnoea/hypopnoea criterion
    "spo2_min",         # bottom of desat = threshold-related
    "spo2_max",         # ceiling of desat baseline = threshold-related
)

# Features that measure responses to apnoeic events (autonomic / cortical / postural)
PHYSIOLOGICAL_PREFIXES: tuple[str, ...] = (
    "hr_",              # heart rate
    "hrv_",             # heart rate variability (time + freq domain) — incl hrv_sampen (Phase 1 Exp 1)
    "eeg_",             # EEG band power, complexity
    "ecg_",             # raw ECG-derived features that aren't HRV
    "position_",        # body position
    "spo2_mean",        # statistical, not threshold-based
    "spo2_std",         # statistical
    "spo2_sampen",      # Phase 1 Exp 1 — SpO2 sample entropy (complexity, not threshold)
    "spo2_psd",         # Phase 1 Exp 2 — SpO2 Welch PSD (frequency content; apnea_band/total/ratio)
    "spo2_range",       # statistical (range = max - min, not threshold)
    "resp_chest",       # chest belt — effort, not airflow drop
    "resp_abdo",        # abdominal belt — effort, not airflow drop
)


def classify_feature(col_name: str) -> str:
    """Return 'aasm', 'physio', or 'unknown' for a single feature column.

    Lowercased prefix match. Contextual suffixes (lag/lead/roll) inherit class.
    """
    name = col_name.lower()
    for p in AASM_RULE_PREFIXES:
        if name.startswith(p):
            return "aasm"
    for p in PHYSIOLOGICAL_PREFIXES:
        if name.startswith(p):
            return "physio"
    return "unknown"


def split_features(columns: Iterable[str]) -> dict[str, list[str]]:
    """Split a list of column names into the three classes.

    Returns dict with keys: 'aasm', 'physio', 'unknown'. Order preserved.
    """
    out: dict[str, list[str]] = {"aasm": [], "physio": [], "unknown": []}
    for c in columns:
        out[classify_feature(c)].append(c)
    return out


def feature_subset(columns: Iterable[str], mode: str) -> list[str]:
    """Return the feature subset for the given ablation mode.

    Modes:
      'full'        — all input columns
      'physio_only' — drop AASM-rule columns; keep physio + unknown
      'aasm_only'   — keep only AASM-rule columns
    """
    cols = list(columns)
    if mode == "full":
        return cols
    groups = split_features(cols)
    if mode == "physio_only":
        # Keep physiological + unknown (unknown defaults to physio per the
        # conservative rule above).
        return groups["physio"] + groups["unknown"]
    if mode == "aasm_only":
        return groups["aasm"]
    raise ValueError(f"unknown mode: {mode!r} (expected 'full', 'physio_only', 'aasm_only')")
