"""Framingham (D'Agostino 2008) implementation against the paper's worked examples."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis_pipeline.cvd_covariates import event_heart_rate, frs_linear_predictor, frs_risk10, self_reported_cvd  # noqa: E402


def test_dagostino_case1_woman():
    # 61-year-old woman, TC 180, HDL 47, SBP 124 untreated, smoker, no diabetes:
    # published sum beta*x = 26.9653, 10-year risk 10.48 %.
    lp = frs_linear_predictor([61], [180], [47], [124], [0], [1], [0], [0])
    assert lp[0] + 26.1931 == pytest.approx(26.9653, abs=5e-4)
    assert frs_risk10(lp, [0])[0] == pytest.approx(0.1048, abs=5e-4)


def test_dagostino_case2_man():
    # 53-year-old man, TC 161, HDL 55, SBP 125 treated, non-smoker, diabetic: 10-year risk 15.6 %.
    lp = frs_linear_predictor([53], [161], [55], [125], [1], [0], [1], [1])
    assert frs_risk10(lp, [1])[0] == pytest.approx(0.156, abs=1e-3)


def test_event_hr_weighting_and_quality_gate():
    h = pd.DataFrame({
        "savbnbh": [60, 60], "nremepbp": [100, 100],
        "savbnoh": [70, np.nan], "nremepop": [300, 0],
        "savbrbh": [80, 60], "remepbp": [100, 0],
        "savbroh": [np.nan, 60], "remepop": [0, 0],
        "hrqual": [4, 1],
    })
    hr = event_heart_rate(h)
    assert hr.iloc[0] == pytest.approx((60 * 100 + 70 * 300 + 80 * 100) / 500)
    assert np.isnan(hr.iloc[1])


def test_self_reported_cvd_missing_handling():
    h = pd.DataFrame({"mi15": [1, 0, np.nan, 8], "stroke15": [0, 0, np.nan, 0], "hf15": [np.nan, 0, np.nan, 0],
                      "cabg15": [0, 0, np.nan, 0], "ca15": [0, 0, np.nan, 0], "angina15": [0, np.nan, np.nan, 0]})
    assert self_reported_cvd(h).tolist()[:2] == [1.0, 0.0]
    assert np.isnan(self_reported_cvd(h).iloc[2])
    assert self_reported_cvd(h).iloc[3] == 0.0  # 8 = don't know is not a history
