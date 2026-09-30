"""Tests for the DL front-end (dl_signals) and stage-A -> stage-B derivation (dl_stage_b)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl_synthetic import synth_ecg, write_stage_a  # noqa: E402
from thesis_pipeline.dl_signals import (  # noqa: E402
    ecg_to_rr_edr,
    edr_beats,
    hr_agreement,
    interp_runs,
    resp_to_4hz,
    spo2_to_4hz,
)
from thesis_pipeline.dl_stage_b import (  # noqa: E402
    CHANNELS,
    StageAError,
    assert_not_under_desktop,
    derive_subject,
    load_stage_a,
    sec_event_from_events,
    subject_layout,
)


_NO_FIX = {"lastiter_ectopic": 0, "lastiter_missed": 0, "lastiter_extra": 0, "lastiter_longshort": 0,
           "removed_or_moved": 0, "added_or_moved_to": 0, "n_segments": 1, "n_fix_failed": 0}


def _true_rr_on_grid(beats: np.ndarray, n_out: int, fs_out: int = 4) -> np.ndarray:
    rr = np.diff(beats)
    return np.interp(np.arange(n_out) / fs_out, beats[1:], rr)


# ---------------------------------------------------------------- RR / EDR


def test_rr_recovers_known_intervals():
    n_sec = 600
    ecg, beats, _ = synth_ecg(n_sec, hr_bpm=65, seed=1)
    rr4, edr4, qc = ecg_to_rr_edr(ecg, 125, n_sec, edr="psa")
    assert qc["ok"] and qc["detector"] == "neurokit"
    assert rr4.shape == (n_sec * 4,) and edr4.shape == (n_sec * 4,)
    truth = _true_rr_on_grid(beats, n_sec * 4)
    inner = slice(4 * 5, 4 * (n_sec - 5))
    assert np.max(np.abs(rr4[inner] - truth[inner])) < 0.01  # < 10 ms everywhere
    assert qc["rr_gap_frac"] < 0.01
    assert "kubios_artifact_frac" not in qc and qc["kubios_removed_or_moved_frac"] < 0.01
    assert qc["frontend"] == "dl-frontend-1.1"


def test_rr_gap_is_linear_and_flagged():
    n_sec = 600
    ecg, beats, _ = synth_ecg(n_sec, hr_bpm=60, gaps=((200.0, 230.0),), seed=2)
    rr4, _, qc = ecg_to_rr_edr(ecg, 125, n_sec)
    assert qc["ok"] and qc["n_gaps"] >= 1
    # ~30 s of 600 s is a gap
    assert 0.04 < qc["rr_gap_frac"] < 0.07
    # inside the gap RR is a straight line between the neighbouring valid samples
    g = rr4[4 * 205:4 * 225].astype(np.float64)
    assert np.allclose(np.diff(g, 2), 0.0, atol=1e-4)
    assert np.all((rr4 >= 0.3) & (rr4 <= 2.0))


def test_rr_is_clipped_to_physiological_range(monkeypatch):
    import thesis_pipeline.dl_signals as ds

    # controlled peak train at 256 Hz: 0.2 s intervals, one 2.5 s pause, then 0.8 s
    t = np.concatenate([np.arange(1.0, 5.0, 0.2), [7.5], np.arange(8.3, 100.0, 0.8)])
    peaks = np.round(t * 256).astype(np.int64)
    monkeypatch.setattr(ds, "_detect_peaks", lambda x, fs: (peaks, "stub"))
    monkeypatch.setattr(ds, "_fixpeaks_segments", lambda p, fs, g: (p, _NO_FIX))
    ecg, _, _ = synth_ecg(100, seed=0)
    rr4, _, qc = ds.ecg_to_rr_edr(ecg, 125, 100)
    assert qc["ok"] and qc["n_gaps"] == 0 and qc["rr_clip_frac"] > 0.1
    assert rr4.min() >= 0.3 and rr4.max() <= 2.0
    assert np.allclose(rr4[4 * 2:4 * 4], 0.3, atol=1e-4)  # 0.2 s intervals clipped up
    assert np.isclose(rr4[4 * 7 + 2], 2.0, atol=1e-4)  # the 2.5 s pause clipped down


def test_interp_runs_spline_inside_linear_across_hold_outside():
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 20.0, 21.0, 22.0, 23.0, 24.0])
    v = np.array([1, 2, 1, 2, 1, 5, 6, 5, 6, 5], float)
    seg = np.array([0] * 5 + [1] * 5)
    t_out = np.arange(0, 30, 0.25)
    out, gap = interp_runs(t, v, seg, t_out)
    assert np.all(out[t_out < 1.0] == 1.0) and np.all(out[t_out > 24.0] == 5.0)  # hold
    mid = (t_out > 5.0) & (t_out < 20.0)
    assert np.allclose(out[mid], np.interp(t_out[mid], [5.0, 20.0], [1.0, 5.0]))  # linear
    assert gap[mid].all() and not gap[(t_out >= 1.0) & (t_out <= 5.0)].any()


def test_interp_runs_pchip_does_not_overshoot():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.4, 8.4, 9.4, 10.4])
    v = np.array([1.0, 1.0, 40.0, 80.0, 60.0, 2.4, 1.0, 1.0, 1.0, 1.0])
    t_out = np.arange(0, 11, 0.25)
    cubic, _ = interp_runs(t, v, np.zeros(len(t), int), t_out, method="cubic")
    pchip, _ = interp_runs(t, v, np.zeros(len(t), int), t_out, method="pchip")
    assert cubic.min() < 0  # the frontend-1.0 overshoot
    assert pchip.min() >= 1.0 - 1e-12 and pchip.max() <= 80.0 + 1e-12
    on_knots = np.isin(t_out, t)
    assert np.allclose(pchip[on_knots], np.interp(t_out[on_knots], t, v))
    with pytest.raises(ValueError, match="interpolation"):
        interp_runs(t, v, np.zeros(len(t), int), t_out, method="akima")


def test_edr_stays_nonnegative_next_to_an_outlier_cluster(monkeypatch):
    """A 3-beat outlier cluster survives the width-5 median correction; frontend 1.0's
    cubic spline then dipped to impossible negative areas (78 of 80 pilot subjects)."""
    import thesis_pipeline.dl_signals as ds

    tb = np.concatenate([np.arange(1.0, 31.0), [31.0, 32.0, 33.0, 34.0, 36.4], np.arange(37.4, 99.0)])
    area = 1.0 + 0.05 * np.sin(tb)
    i = int(np.searchsorted(tb, 31.0))
    area[i:i + 4] = [40.0, 80.0, 60.0, 2.4]  # cluster, then a 2.4 s beat gap (< GAP_S) before the next run
    peaks = np.round(tb * 256).astype(np.int64)
    monkeypatch.setattr(ds, "_detect_peaks", lambda x, fs: (peaks, "stub"))
    monkeypatch.setattr(ds, "_fixpeaks_segments", lambda p, fs, g: (p, _NO_FIX))
    monkeypatch.setattr(ds, "edr_beats", lambda x, p, fs, method="psa": area[np.searchsorted(peaks, p)])
    corrected, n_corr = ds._median5_correct(area)
    assert n_corr == 0 and corrected[i + 1] == 80.0  # the cluster survives the median correction
    ecg, _, _ = synth_ecg(100, seed=0)
    _, edr4, qc = ds.ecg_to_rr_edr(ecg, 125, 100, edr="psa")
    assert qc["ok"] and edr4.min() >= 0.0 and edr4.max() <= 80.0 + 1e-3
    cubic, _ = interp_runs(tb, corrected, np.zeros(len(tb), int), np.arange(400) / 4.0, method="cubic")
    assert cubic.min() < 0  # same beats through the old cubic path go negative
    assert qc["edr_beat_max_over_median"] == pytest.approx(80.0 / np.median(area))


def test_kubios_qc_measures_the_correction_extent():
    """NeuroKit's iterative info lists only the last iteration; QC must count every change."""
    import thesis_pipeline.dl_signals as ds

    fs = 256
    rng = np.random.default_rng(2)
    t = 0.6 + np.cumsum(1.0 + 0.05 * np.sin(np.arange(320) / 7) + rng.normal(0, 0.01, 320))
    t = t[t < 300]
    keep = np.ones(len(t), bool)
    keep[rng.choice(len(t), 10, replace=False)] = False  # 10 missed beats
    t = np.sort(np.concatenate([t[keep], rng.uniform(5, 295, 10)]))  # 10 extra detections
    peaks = np.unique(np.round(t * fs).astype(np.int64))
    out, fx = ds._fixpeaks_segments(peaks, fs, 3.0)
    assert fx["removed_or_moved"] == len(np.setdiff1d(peaks, out))
    assert fx["added_or_moved_to"] == len(np.setdiff1d(out, peaks))
    last = sum(fx[f"lastiter_{k}"] for k in ds.KUBIOS_TYPES)
    assert fx["removed_or_moved"] >= 10 and last < fx["removed_or_moved"] / 2


def test_edr_psa_tracks_respiratory_amplitude():
    n_sec = 300
    ecg, beats, amp = synth_ecg(n_sec, hr_bpm=70, amp_mod=0.3, resp_period_s=5.0, seed=3)
    _, edr4, qc = ecg_to_rr_edr(ecg, 125, n_sec, edr="psa")
    at_beats = edr4[np.clip((beats * 4).astype(int), 0, len(edr4) - 1)]
    r = np.corrcoef(at_beats, amp)[0, 1]
    assert r > 0.9, r
    _, edr_r, _ = ecg_to_rr_edr(ecg, 125, n_sec, edr="ramp")
    r2 = np.corrcoef(edr_r[np.clip((beats * 4).astype(int), 0, len(edr_r) - 1)], amp)[0, 1]
    assert r2 > 0.9, r2


def test_psa_area_is_the_shoelace_area_and_translation_invariant():
    fs = 256
    x = np.zeros(1000)
    k = np.arange(-10, 11)
    x[500 + k] = np.exp(-0.5 * (k / 2.5) ** 2)
    a1 = edr_beats(x, np.array([500]), fs, "psa")[0]
    a2 = edr_beats(x + 3.0, np.array([500]), fs, "psa")[0]
    a3 = edr_beats(2 * x, np.array([500]), fs, "psa")[0]
    w = x[490:511]
    X, Y = w[:-2], w[2:]
    ref = 0.5 * abs(np.sum(X * np.roll(Y, -1) - np.roll(X, -1) * Y))
    assert a1 > 0 and np.isclose(a1, ref) and np.isclose(a1, a2) and np.isclose(a3, 4 * a1)
    assert np.isnan(edr_beats(x, np.array([3]), fs, "psa")[0])  # window leaves the signal


def test_flat_ecg_returns_zeros_not_ok():
    rr4, edr4, qc = ecg_to_rr_edr(np.zeros(125 * 600), 125, 600)
    assert not qc["ok"] and qc["reason"] == "flat_or_short_ecg"
    assert not rr4.any() and not edr4.any() and len(rr4) == 2400


def test_hr_agreement():
    rr4 = np.full(4 * 300, 1.0)
    qc = hr_agreement(rr4, np.full(300, 61.0))
    assert qc["hr_check_median_abs_bpm"] == pytest.approx(1.0) and qc["hr_check_frac_within_5bpm"] == 1.0


# ---------------------------------------------------------------- belts / SpO2


def test_resp_to_4hz_length_and_alignment():
    x = np.zeros(10 * 100)
    x[10 * 40] = 1.0  # impulse at t = 40 s
    y = resp_to_4hz(x, 10, 400)
    assert y.shape == (400,) and y.dtype == np.float32
    assert int(np.argmax(y)) == 160  # t = 40 s at 4 Hz


def test_spo2_short_gap_interpolated_long_gap_zeroed():
    pct = np.full(600, 96.0)
    pct[100:130] = 0.0  # 30 s dropout -> interpolated
    pct[300:400] = 0.0  # 100 s dropout -> zeroed
    x4, qc = spo2_to_4hz(pct, 2400)
    assert qc["ok"] and qc["spo2_n_long_gaps"] == 1
    assert np.allclose(x4[4 * 105:4 * 125], (96 - 95) / 5)
    assert np.allclose(x4[4 * 310:4 * 390], 0.0)
    assert qc["spo2_long_gap_frac"] == pytest.approx(100 / 600)


def test_spo2_all_dropout_not_ok():
    x4, qc = spo2_to_4hz(np.zeros(100), 400)
    assert not qc["ok"] and not x4.any()


# ---------------------------------------------------------------- stage-A adapter + derivation


def test_sec_event_rule_is_s_plus_half():
    m = sec_event_from_events([{"start": 10.4, "duration": 10.2}], 40)
    # centres 10.5 .. 20.5 inside [10.4, 20.6) -> seconds 10 .. 20
    assert m.nonzero()[0].tolist() == list(range(10, 21))


def test_load_stage_a_design_format_and_signs(tmp_path):
    write_stage_a(tmp_path, 300001, n_epochs=6, thor_spikes_s=(45.0,))
    sa = load_stage_a(tmp_path / "shhs1-300001.npz")
    assert sa.n_epochs == 6 and sa.fs == {"ECG": 125, "THOR": 10, "ABDO": 10, "AIRFLOW": 10, "SPO2": 1, "HR": 1}
    assert sa.signals["ECG"].shape == (6 * 30 * 125,)
    # THOR physical range is inverted: digital -100 spike becomes the positive maximum
    assert int(np.argmax(sa.signals["THOR"])) == 450
    assert np.allclose(sa.signals["SPO2"], 96.09375)  # 123 * 100 / 128
    assert sa.airflow_ok and not sa.flags


def test_load_stage_a_rejects_unknown_stage_codes(tmp_path):
    stages = np.array([0, 1, 2, 4, 2, 2], np.int8)  # 4 is not in the default map
    write_stage_a(tmp_path, 300002, n_epochs=6, stages=stages, flat_ecg=True)
    with pytest.raises(StageAError, match="stage codes"):
        load_stage_a(tmp_path / "shhs1-300002.npz")
    meta = json.loads((tmp_path / "shhs1-300002.json").read_text())
    meta["stage_codes"] = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
    (tmp_path / "shhs1-300002.json").write_text(json.dumps(meta))
    sa = load_stage_a(tmp_path / "shhs1-300002.npz")
    assert sa.sleep_epoch.tolist() == [False, True, True, True, True, True]


def test_load_stage_a_flat_prototype_format(tmp_path):
    n = 4
    np.savez(tmp_path / "shhs1-300003.npz",
             ecg=np.zeros(n * 3750, np.int8), thor=np.zeros(n * 300, np.int8),
             sleep_stage=np.array([0, 2, 2, 5], np.int8), apnoea_label=np.zeros(n, np.int8),
             event_sec=np.zeros(n * 30, np.uint8), spo2=np.full(n * 30, 123, np.uint8))
    (tmp_path / "shhs1-300003.json").write_text(json.dumps(
        {"subject_id": 300003, "n_epochs": n, "fs": {"ecg": 125, "thor": 10, "spo2": 1},
         "scale": {"ecg_mV_per_lsb": 2.5 / 255, "spo2_pct_per_code": 0.78125}}))
    sa = load_stage_a(tmp_path / "shhs1-300003.npz")
    assert sa.fs["ECG"] == 125 and sa.sleep_epoch.tolist() == [False, True, True, True]
    assert "THOR_sign_default" in sa.flags and "missing_AIRFLOW" in sa.flags


def test_derive_subject_shapes_targets_and_ok_flags(tmp_path):
    ev = [(95.0, 20.0)]
    write_stage_a(tmp_path, 300004, n_epochs=24, events=ev, seed=4)
    res = derive_subject(load_stage_a(tmp_path / "shhs1-300004.npz"))
    n_sec = 24 * 30
    assert res["signals"].shape == (len(CHANNELS), n_sec * 4) and res["signals"].dtype == np.float16
    assert res["channel_ok"].tolist() == [True] * 6
    assert res["sec_target"].nonzero()[0].tolist() == list(range(95, 115))
    assert res["sec_sleep"][:60].sum() == 0 and res["sec_sleep"][60:].all()
    assert res["apnoea_label"].tolist()[3] == 1  # 95-115 s overlaps epoch 3 by 15 s
    # RR centred per subject: median ~ 0 ms; EDR relative: median ~ 0
    assert abs(float(np.median(res["signals"][0].astype(np.float32)))) < 1.0
    assert abs(float(np.median(res["signals"][1].astype(np.float32)))) < 0.05
    assert res["qc"]["hr_check_frac_within_5bpm"] > 0.9


def test_derive_subject_flat_ecg_and_airflow_not_ok(tmp_path):
    write_stage_a(tmp_path, 300005, n_epochs=12, flat_ecg=True, airflow_ok=False)
    res = derive_subject(load_stage_a(tmp_path / "shhs1-300005.npz"))
    ok = dict(zip(CHANNELS, res["channel_ok"]))
    assert not ok["RR"] and not ok["EDR"] and not ok["AIRFLOW"] and ok["THOR"] and ok["SPO2"]
    assert not res["signals"][0].any() and not res["signals"][4].any()


def test_subject_layout_chunks_hold_whole_subjects():
    lay = subject_layout(np.array([10, 7, 13, 6]), max_chunk_bytes=1000 * 4 * 2)
    assert (lay["n_epochs_pad"] % 6 == 0).all() and (lay["n_epochs_pad"] >= lay["n_epochs"]).all()
    assert lay["n_windows"].tolist() == [2, 2, 3, 1]
    for _, g in lay.groupby("chunk"):
        starts = g["block_start1"].to_numpy()
        ends = starts + g["block_s"].to_numpy()
        assert starts[0] == 0 and np.all(starts[1:] == ends[:-1])
    assert lay["chunk"].nunique() > 1


def test_refuses_desktop_output():
    with pytest.raises(ValueError, match="Desktop"):
        assert_not_under_desktop(Path.home() / "Desktop" / "dl_cache")
