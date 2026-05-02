"""Regression tests for sleep_mask / wake_mask / subject_metadata semantics.

These tests guard against the 2026-05-02 filter-inversion bug where
wake_mask returned True for sleep stages despite its name, causing every
consumer of df[~wake_mask(...)] to keep wake epochs and train apnoea
models on the wrong cohort. The cascading bug in subject_metadata
silently inverted clinical TST/AHI denominators.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python -m pytest tests/` from the Code/ directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis_pipeline.epochs import sleep_mask, wake_mask, subject_metadata  # noqa: E402


def test_sleep_mask_returns_true_for_sleep():
    s = np.array(["W", "N1", "N2", "N3", "REM", "?"])
    m = sleep_mask(s)
    assert m.tolist() == [False, True, True, True, True, False]


def test_wake_mask_is_alias_for_sleep_mask():
    """wake_mask is misleadingly named but kept for backwards compatibility.

    It MUST return identical values to sleep_mask (both True for sleep).
    """
    s = np.array(["W", "N1", "N2", "N3", "REM", "?"])
    assert np.array_equal(sleep_mask(s), wake_mask(s))


def test_subject_metadata_mixed_stages():
    df = pd.DataFrame({
        "sleep_stage": ["W", "N1", "N2", "REM", "W"],
        "apnoea_label": [0, 1, 0, 1, 0],
    })
    m = subject_metadata(df)
    assert m["tst_min"] == 1.5, f"TST wrong: {m['tst_min']}"
    assert m["tib_min"] == 2.5
    assert abs(m["sleep_efficiency"] - 0.6) < 1e-9
    assert m["n_apnoea_epochs"] == 2.0  # both apnoeas during sleep
    assert abs(m["ahi_proxy_per_hr"] - 80.0) < 1e-9  # 2 events / 1.5 min × 60 = 80/h


def test_subject_metadata_all_wake_zero_tst():
    df = pd.DataFrame({
        "sleep_stage": ["W", "W", "W"],
        "apnoea_label": [0, 0, 0],
    })
    m = subject_metadata(df)
    assert m["tst_min"] == 0.0
    # sleep_efficiency = TST / TIB = 0 / 1.5 = 0.0 (not NaN — TIB > 0, ratio is valid)
    assert m["sleep_efficiency"] == 0.0
    # ahi_proxy_per_hr = events / TST; TST = 0 so undefined → NaN
    assert np.isnan(m["ahi_proxy_per_hr"])


def test_subject_metadata_all_sleep_full_tst():
    df = pd.DataFrame({
        "sleep_stage": ["N2"] * 10,
        "apnoea_label": [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    })
    m = subject_metadata(df)
    assert m["tst_min"] == 5.0  # 10 × 30s = 300s = 5 min
    assert m["sleep_efficiency"] == 1.0
    assert m["n_apnoea_epochs"] == 3.0
    assert abs(m["ahi_proxy_per_hr"] - 36.0) < 1e-9  # 3 events / 5 min × 60
