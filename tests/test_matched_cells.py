"""Column-set contract for the Aim 2 matched-input LightGBM cells (critique #1).

ecg_only and ecg_belt must be strict, order-preserving subsets of the canonical
217-column physio_only list, and must never contain airflow-derived, SpO2, EEG
or position inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis_pipeline.matched_cells import (  # noqa: E402
    BELT_BASES,
    CELLS,
    CONTEXT_SUFFIXES,
    EXPECTED_ECG_BASES,
    INPUT_MATCH_CAVEATS,
    base_feature,
    caveats_for,
    cell_columns,
    check_cell_columns,
)

CANONICAL_LIST = (
    ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only" / "feature_list.json"
)


@pytest.fixture(scope="module")
def physio() -> list[str]:
    if not CANONICAL_LIST.exists():
        pytest.skip(f"canonical feature list not present: {CANONICAL_LIST}")
    return json.loads(CANONICAL_LIST.read_text())


def _is_forbidden(col: str) -> bool:
    b = base_feature(col)
    return b == "resp_breath_rate_bpm" or b.startswith(("eeg_", "position_", "spo2_", "resp_airflow"))


def test_base_feature_strips_one_context_suffix():
    assert base_feature("hr_mean") == "hr_mean"
    assert base_feature("hrv_sdnn_lag1") == "hrv_sdnn"
    assert base_feature("spo2_std_roll5_std") == "spo2_std"
    assert base_feature("resp_thor_abdo_corr_roll11_mean") == "resp_thor_abdo_corr"
    assert len(CONTEXT_SUFFIXES) == 6


def test_canonical_list_is_217_columns_of_31_bases(physio):
    assert len(physio) == 217 == len(set(physio))
    assert len({base_feature(c) for c in physio}) == 31
    assert "resp_breath_rate_bpm" in physio  # the airflow feature the matched cells must drop


def test_physio_only_repro_is_the_canonical_list_unchanged(physio):
    assert cell_columns("physio_only_repro", physio) == physio


def test_ecg_only_is_hr_hrv_bases_only(physio):
    cols = cell_columns("ecg_only", physio)
    assert len(cols) == 56
    bases = {base_feature(c) for c in cols}
    assert bases == set(EXPECTED_ECG_BASES)
    assert all(base_feature(c).startswith(("hr_", "hrv_")) for c in cols)
    # every hr_/hrv_ column of the physio list is included, with all 7 variants
    assert set(cols) == {c for c in physio if base_feature(c).startswith(("hr_", "hrv_"))}


def test_ecg_belt_is_ecg_only_plus_three_belt_bases(physio):
    ecg = cell_columns("ecg_only", physio)
    belt = cell_columns("ecg_belt", physio)
    assert len(belt) == 77
    extra = set(belt) - set(ecg)
    assert set(ecg) < set(belt)
    assert {base_feature(c) for c in extra} == set(BELT_BASES)
    assert len(extra) == 21


@pytest.mark.parametrize("cell", ["ecg_only", "ecg_belt"])
def test_matched_cells_exclude_airflow_spo2_eeg_position(physio, cell):
    cols = cell_columns(cell, physio)
    assert not [c for c in cols if _is_forbidden(c)]
    assert not [c for c in cols if c.startswith("resp_breath_rate_bpm")]
    assert not [c for c in cols if c.startswith(("eeg_", "position_", "spo2_", "resp_airflow"))]


@pytest.mark.parametrize("cell", CELLS)
def test_cells_are_order_preserving_subsets(physio, cell):
    cols = cell_columns(cell, physio)
    pos = [physio.index(c) for c in cols]
    assert pos == sorted(pos)
    assert set(cols) <= set(physio)


def test_checker_rejects_injected_airflow_feature(physio):
    ecg = cell_columns("ecg_only", physio)
    # swap one hrv column for an airflow-derived one: same count, still a physio subset
    bad = sorted(ecg[:-1] + ["resp_breath_rate_bpm"], key=physio.index)
    with pytest.raises(AssertionError, match="forbidden"):
        check_cell_columns("ecg_only", bad, physio)


def test_checker_rejects_missing_context_variant(physio):
    belt = cell_columns("ecg_belt", physio)
    dropped = [c for c in belt if c != "resp_thor_rms_lag1"]
    with pytest.raises(AssertionError):
        check_cell_columns("ecg_belt", dropped, physio)


def test_unknown_cell_raises(physio):
    with pytest.raises(ValueError):
        cell_columns("full", physio)


def test_physio_list_contract_rejects_wrong_length(physio):
    with pytest.raises(AssertionError, match="217"):
        cell_columns("ecg_only", physio[:-7])


def test_input_match_caveats_per_cell():
    ecg, belt = caveats_for("ecg_only"), caveats_for("ecg_belt")
    # declared for both cells (review 2026-09-30): scale, context and RR-cleaning mismatches
    for k in ("edr_no_engineered_counterpart", "absolute_scale", "temporal_context", "rr_cleaning"):
        assert k in ecg and k in belt
    # the belts are label-defining under SHHS MOP 6.610: only the belt cell carries that caveat
    assert "belt_label_proximity" in belt and "belt_label_proximity" not in ecg
    assert "MOP 6.610" in belt["belt_label_proximity"]
    assert caveats_for("physio_only_repro") == {}
    assert set(belt) <= set(INPUT_MATCH_CAVEATS)
    with pytest.raises(ValueError):
        caveats_for("full")
