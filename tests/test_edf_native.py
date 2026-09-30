"""thesis_pipeline.edf_native: header parsing, digital reads, scaling sign, rates, airflow picker.

Synthetic EDFs (tests/_synth_edf.py) cover the logic; the mne cross-check
runs on three real, already-materialised SHHS-1 EDFs and skips if they are
not local (it never triggers a download: placeholders are detected first).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _synth_edf import Sig, make_subject, write_edf  # noqa: E402
from thesis_pipeline import edf_native  # noqa: E402
from thesis_pipeline.edf_native import (  # noqa: E402
    AirflowPick,
    DatalessFileError,
    RateDeviation,
    airflow_candidates,
    assert_rates,
    check_rates,
    pick_airflow,
    read_digital,
    read_header,
    to_physical,
)


def _sig(label, fs, n_rec, lo=-128, hi=128, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return Sig(label, fs, rng.integers(lo, hi, n_rec * fs).astype(np.int16), **kw)


def test_header_and_digital_roundtrip_multirate(tmp_path):
    n = 95
    sigs = [
        _sig("SaO2", 1, n, -32768, 32767, 1, pmin=0.0, pmax=100.0, dmin=-32768, dmax=32767),
        _sig("ECG", 125, n, seed=2, pmin=-1.25, pmax=1.25, unit="mV"),
        _sig("THOR RES", 10, n, seed=3, pmin=1.0, pmax=-1.0),
        _sig("AIRFLOW", 10, n, seed=4, pmin=1.0, pmax=-1.0),
        _sig("AIRFLOW", 10, n, seed=5, pmin=1.0, pmax=-1.0),  # duplicated label
        _sig("POSITION", 1, n, 0, 4, 6, pmin=0.0, pmax=3.0, dmin=0, dmax=3),
    ]
    p = write_edf(tmp_path / "x.edf", sigs, n)
    hdr = read_header(p)
    assert hdr.n_signals == 6 and hdr.n_records == 95 and hdr.record_duration == 1.0
    assert hdr.labels == ["SaO2", "ECG", "THOR RES", "AIRFLOW", "AIRFLOW", "POSITION"]
    assert [s.fs for s in hdr.signals] == [1, 125, 10, 10, 10, 1]
    assert hdr.signals[1].unit == "mV" and hdr.signals[1].pmin == -1.25
    assert hdr.indices("airflow") == [3, 4]
    assert hdr.find("ECG") == [1] and hdr.find("SPO2") == [0] and hdr.find("AIRFLOW") == [3, 4]
    assert hdr.find("OXSTAT") == []
    assert hdr.notes == ()
    dig = read_digital(p, hdr=hdr)
    for i, s in enumerate(sigs):
        assert dig[i].dtype == np.int16
        assert np.array_equal(dig[i], s.digital), s.label
    sub = read_digital(p, [4, 1])
    assert sorted(sub) == [1, 4] and np.array_equal(sub[4], sigs[4].digital)


def test_physical_scaling_keeps_sign(tmp_path):
    n = 3
    thor = Sig("THOR RES", 10, np.array([-128, 127, 0] * 10 * n, dtype=np.int16)[: 10 * n], 1.0, -1.0)
    ecg = Sig("ECG", 125, np.full(125 * n, 127, dtype=np.int16), -1.25, 1.25, unit="mV")
    p = write_edf(tmp_path / "s.edf", [thor, ecg], n)
    hdr = read_header(p)
    t = hdr.signals[0]
    assert t.gain < 0 and t.sign == -1
    assert hdr.signals[1].sign == 1
    phys = to_physical(np.array([-128, 127], dtype=np.int16), t)
    assert phys.dtype == np.float32
    assert np.allclose(phys, [1.0, -1.0])
    assert np.isclose(to_physical(np.array([127], dtype=np.int16), hdr.signals[1])[0], 1.25)
    # digital * gain + offset == physical
    d = np.arange(-128, 128)
    assert np.allclose(d * t.gain + t.offset, to_physical(d.astype(np.int16), t), atol=1e-6)


def test_truncated_file_uses_complete_records(tmp_path):
    n = 40
    p = write_edf(tmp_path / "t.edf", [_sig("ECG", 125, n)], n, truncate_records=31)
    hdr = read_header(p)
    assert hdr.n_records_header == 40 and hdr.n_records == 31
    assert any("truncated" in s for s in hdr.notes)
    assert read_digital(p, [0], hdr=hdr)[0].size == 31 * 125


def test_unknown_record_count_is_derived(tmp_path):
    p = write_edf(tmp_path / "u.edf", [_sig("ECG", 125, 7)], 7, header_n_records=-1)
    hdr = read_header(p)
    assert hdr.n_records == 7 and any("n_records=-1" in s for s in hdr.notes)


def test_dataless_is_refused(tmp_path, monkeypatch):
    p = write_edf(tmp_path / "d.edf", [_sig("ECG", 125, 2)], 2)
    monkeypatch.setattr(edf_native, "_is_dataless", lambda path: True)
    with pytest.raises(DatalessFileError):
        read_header(p)
    with pytest.raises(DatalessFileError):
        read_digital(p)


def test_rate_checks(tmp_path):
    n = 4
    sigs = [_sig("ECG", 250, n), _sig("THOR RES", 10, n), _sig("NEW AIR", 5, n)]
    hdr = read_header(write_edf(tmp_path / "r.edf", sigs, n))
    dev = check_rates(hdr, {"ECG": hdr.find("ECG"), "THOR": hdr.find("THOR"), "AIRFLOW": hdr.find("AIRFLOW")})
    assert len(dev) == 2 and "ECG" in dev[0] and "fs=250" in dev[0] and "AIRFLOW" in dev[1]
    with pytest.raises(RateDeviation):
        assert_rates(hdr, {"ECG": hdr.find("ECG")})
    assert_rates(hdr, {"THOR": hdr.find("THOR")})  # no raise


def test_record_duration_deviation(tmp_path):
    hdr = read_header(write_edf(tmp_path / "rd.edf", [_sig("ECG", 125, 4)], 2, record_duration=2.0))
    assert hdr.signals[0].fs == 125
    dev = check_rates(hdr, {"ECG": [0]})
    assert dev == ["record_duration=2s (expected 1s)"]


# --------------------------------------------------------------------------- airflow picker


@pytest.mark.parametrize("label", ["AIRFLOW", "NEW AIR", "NEWAIR", "New A/F", "AUX", "New Air", "New AIR"])
def test_airflow_label_spellings_recognised(tmp_path, label):
    hdr = read_header(write_edf(tmp_path / "a.edf", [_sig(label, 10, 3), _sig("SOUND", 10, 3)], 3))
    assert airflow_candidates(hdr) == [0]


def test_non_airflow_labels_ignored(tmp_path):
    sigs = [_sig(x, 10, 3) for x in ("SOUND", "nasal", "THOR RES", "AIR")]
    hdr = read_header(write_edf(tmp_path / "n.edf", sigs, 3))
    assert airflow_candidates(hdr) == []
    pick = pick_airflow(hdr, {})
    assert pick == AirflowPick(index=None, label=None, std_lsb=0.0, ok=False, candidates=())


def test_live_channel_beats_dead_airflow_first_in_header(tmp_path):
    n = 60
    dead = _sig("AIRFLOW", 10, n, -2, 3, seed=1, pmin=1.0, pmax=-1.0)
    live = _sig("New Air", 10, n, -90, 90, seed=2, pmin=-125.0, pmax=125.0, unit="uV")
    p = write_edf(tmp_path / "l.edf", [_sig("ECG", 125, n), dead, live], n)
    hdr = read_header(p)
    dig = read_digital(p, hdr=hdr)
    pick = pick_airflow(hdr, dig)
    assert pick.index == 2 and pick.label == "New Air" and pick.ok
    assert pick.std_lsb == pytest.approx(np.std(live.digital.astype(float)))
    assert [c.index for c in pick.rejected] == [1]
    assert pick.rejected[0].std_lsb < edf_native.AIRFLOW_LIVE_STD_LSB


def test_duplicate_airflow_labels_both_considered(tmp_path):
    n = 30
    a0 = _sig("AIRFLOW", 10, n, -1, 2, seed=1)
    a1 = _sig("AIRFLOW", 10, n, -50, 50, seed=2)
    p = write_edf(tmp_path / "dup.edf", [a0, a1], n)
    hdr = read_header(p)
    pick = pick_airflow(hdr, read_digital(p, hdr=hdr))
    assert pick.index == 1 and len(pick.candidates) == 2 and pick.ok


def test_all_dead_is_flagged_and_ties_go_to_lowest_index(tmp_path):
    n = 30
    flat = np.zeros(10 * n, dtype=np.int16)
    p = write_edf(tmp_path / "dead.edf", [Sig("AUX", 10, flat), Sig("NEW AIR", 10, flat.copy())], n)
    hdr = read_header(p)
    pick = pick_airflow(hdr, read_digital(p, hdr=hdr))
    assert pick.index == 0 and pick.std_lsb == 0.0 and not pick.ok


def test_std_threshold_is_strict(tmp_path):
    n = 10
    x = np.tile(np.array([-5, 5], dtype=np.int16), 5 * n)  # std exactly 5 LSB
    p = write_edf(tmp_path / "thr.edf", [Sig("AUX", 10, x)], n)
    hdr = read_header(p)
    pick = pick_airflow(hdr, read_digital(p, hdr=hdr))
    assert pick.std_lsb == pytest.approx(5.0) and not pick.ok


def test_pick_restricted_to_stored_segment(tmp_path):
    n = 10
    x = np.zeros(10 * n, dtype=np.int16)
    x[80:] = 100  # activity only in the tail
    p = write_edf(tmp_path / "seg.edf", [Sig("AUX", 10, x)], n)
    hdr = read_header(p)
    d = read_digital(p, hdr=hdr)
    assert pick_airflow(hdr, d).ok
    assert not pick_airflow(hdr, d, n_samples=80).ok


def test_synthetic_subject_helper_reads_back(tmp_path):
    gt = make_subject(tmp_path, 900001)
    hdr = read_header(gt["edf"])
    dig = read_digital(gt["edf"], hdr=hdr)
    assert np.array_equal(dig[hdr.find("ECG")[0]], gt["ecg"])
    assert pick_airflow(hdr, dig).label == "NEW AIR"


# --------------------------------------------------------------------------- real files vs mne

SHHS_ROOT = Path(
    "/Users/jameswu/Library/CloudStorage/OneDrive-SharedLibraries-TheUniversityofSydney(Staff)/"
    "Philip de Chazal - SHHS"
)
# 200001: NEW AIR; 202500: NEWAIR; 203265: AIRFLOW + AUX (inverted-range AIRFLOW).
REAL_IDS = (200001, 202500, 203265)
UNIT_SCALE = {"uV": 1e-6, "µV": 1e-6, "mV": 1e-3, "V": 1.0, "": 1.0}


def _local(p: Path) -> bool:
    try:
        return not (os.stat(p).st_flags & edf_native.UF_DATALESS)
    except FileNotFoundError:
        return False


@pytest.mark.parametrize("sid", REAL_IDS)
def test_matches_mne_on_real_edf(sid):
    mne = pytest.importorskip("mne")
    p = SHHS_ROOT / f"shhs1-{sid}.edf"
    if not _local(p):
        pytest.skip(f"{p.name} not materialised locally (never downloaded by tests)")
    hdr = read_header(p)
    dig = read_digital(p, hdr=hdr)
    groups: dict[float, list[int]] = {}
    for s in hdr.signals:
        groups.setdefault(s.fs, []).append(s.index)
    wanted = {i for c in ("ECG", "THOR", "ABDO", "SPO2", "HR", "POSITION", "AIRFLOW") for i in hdr.find(c)}
    checked = 0
    for fs, idxs in groups.items():
        idxs = [i for i in idxs if i in wanted]
        labels = [hdr.signals[i].label for i in idxs]
        if not idxs or len(set(labels)) != len(labels):
            continue
        raw = mne.io.read_raw_edf(p, include=labels, preload=True, verbose="ERROR")
        assert raw.info["sfreq"] == fs
        for i in idxs:
            s = hdr.signals[i]
            ours = (dig[i].astype(np.float64) - s.dmin) * s.gain + s.pmin
            theirs = raw.get_data(picks=[s.label])[0] / UNIT_SCALE.get(s.unit, 1.0)
            assert theirs.shape == ours.shape, s.label
            tol = abs(s.gain) * 1e-6
            assert np.max(np.abs(theirs - ours)) <= tol, s.label
            # and our float32 converter agrees to float32 precision
            assert np.allclose(to_physical(dig[i], s), ours, rtol=1e-6, atol=abs(s.gain) * 1e-3)
            checked += 1
    assert checked >= 7
    # rates the stage-A extractor asserts
    assert_rates(hdr, {"ECG": hdr.find("ECG"), "THOR": hdr.find("THOR"), "ABDO": hdr.find("ABDO"),
                       "SPO2": hdr.find("SPO2"), "HR": hdr.find("HR"), "POSITION": hdr.find("POSITION"),
                       "AIRFLOW": hdr.find("AIRFLOW")})
