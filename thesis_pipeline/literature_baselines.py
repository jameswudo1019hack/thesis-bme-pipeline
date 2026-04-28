"""Literature-comparable feature subsets for cross-paper validation.

Why this exists
---------------
Our audit-v6 schema has 273 features including aggressive feature engineering
(`hypoxic_burden_epoch` with trailing baseline, contextual lag/lead/roll
derivatives across 6 horizons, frequency HRV with degraded-ECG gates, body
position, etc.). Published SHHS apnoea-detection papers in the 2018–2020
era typically used 20–40 hand-crafted features per epoch with no contextual
rolling. Their reported AUCs land at ~0.85–0.90.

To validate our pipeline against the literature, we need a feature subset
that approximates "what a median 2018–2020 SHHS paper used." This module
defines that subset.

This is NOT an exact replication of any single paper — it's a fair
approximation of the *common feature set* across the cited works (Mostafa
2019, Olsen 2020, ElMoaqet 2020, Bahrami 2022, Wang 2023). Exact replication
would require re-implementing each paper's specific feature definitions,
which is out of scope. The goal is to answer: *if we use literature-style
feature engineering with our protocol, does our cohort give literature-style
AUC?* If yes, our protocol is sound and our higher AUCs in v6/v8/v8.5 reflect
genuine extra feature engineering (and the AASM-rule-recapitulation issue
documented in `feature_groups.py`).

The Lit-2019 feature set
------------------------

| Class | Features | Approx count |
|---|---|---|
| SpO2 stats | spo2_mean, spo2_std, spo2_min, spo2_max | 4 |
| Desaturation | odi3_count | 1 |
| HRV time-domain | hr_mean, hrv_sdnn, hrv_rmssd, hrv_pnn50 | 4 |
| HRV freq-domain | hrv_lf_power, hrv_hf_power, hrv_total_power_freq | 3 |
| EEG band power | eeg_delta_power, eeg_theta_power, eeg_alpha_power, eeg_beta_power, eeg_sigma_power, eeg_total_power | 6 |
| Respiratory | resp_airflow_rms, resp_airflow_std | 2 |

Total: ~20 base features. NO contextual derivatives. NO hypoxic_burden, NO
position channels, NO audit-v3 trailing-baseline desat_depth, NO Sprint 1
extras. This is the "what 2019-era feature engineering looked like" subset.

Note: this feature set still includes some AASM-rule features (`spo2_min`,
`odi3_count`, `resp_airflow_*`) — that's faithful to the literature, which
also did not separate rule-features from physiological-features. Comparing
this Lit-2019 baseline to our 8.5-tax (full) tells us how much extra AUC our
audit-v6 + Sprint 1 + contextual machinery added on top of literature
feature engineering.
"""
from __future__ import annotations

from typing import Iterable

# Base feature names (no _lag1, _lead1, _roll5_*, _roll11_* suffixes).
LIT_2019_BASE_FEATURES: tuple[str, ...] = (
    # SpO2 statistical moments
    "spo2_mean",
    "spo2_std",
    "spo2_min",
    "spo2_max",
    # Desaturation event count (3% threshold)
    "odi3_count",
    # HRV time-domain
    "hr_mean",
    "hrv_sdnn",
    "hrv_rmssd",
    "hrv_pnn50",
    # HRV frequency-domain
    "hrv_lf_power",
    "hrv_hf_power",
    "hrv_total_power_freq",
    # EEG band power (5 main bands + total)
    "eeg_delta_power",
    "eeg_theta_power",
    "eeg_alpha_power",
    "eeg_beta_power",
    "eeg_sigma_power",
    "eeg_total_power",
    # Respiratory effort / airflow
    "resp_airflow_rms",
    "resp_airflow_std",
)


def lit_2019_subset(columns: Iterable[str]) -> list[str]:
    """Return columns from `columns` that match the Lit-2019 base feature set.

    Strict: no contextual derivatives. Returns only columns that match a
    base name in LIT_2019_BASE_FEATURES (no suffix).
    """
    cols = set(columns)
    found = [f for f in LIT_2019_BASE_FEATURES if f in cols]
    return found


def lit_2019_missing(columns: Iterable[str]) -> list[str]:
    """Return Lit-2019 base features NOT present in `columns`.

    Useful for diagnostics — if too many are missing, the current schema
    can't faithfully replicate the Lit-2019 baseline.
    """
    cols = set(columns)
    return [f for f in LIT_2019_BASE_FEATURES if f not in cols]
