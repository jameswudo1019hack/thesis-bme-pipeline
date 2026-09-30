"""Stage-A DL cache: labels, extraction, lossless encodings, atomic IO, ledger, retries, CLI.

Everything runs on synthetic EDF/XML files written to tmp_path, except the
optional integration test at the bottom, which checks label parity on the
real cache (~/thesis_dl_cache/raw_v1) if it exists. No test reads a
placeholder or triggers a download.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts"))

from _synth_edf import make_subject, spo2_digital  # noqa: E402
from thesis_pipeline import dl_cache, dl_labels, edf_native, splits  # noqa: E402
from thesis_pipeline.epochs import apnoea_labels, build_epoch_frame  # noqa: E402
from thesis_pipeline.io import RespiratoryEvent, read_nsrr_xml  # noqa: E402

# --------------------------------------------------------------------------- dl_labels


def _ev(start, dur, kind="Hypopnea"):
    return RespiratoryEvent(start_sec=start, duration_sec=dur, kind=kind)


def test_sec_event_midpoint_rule():
    m = dl_labels.sec_event_mask([_ev(35.0, 16.0, "Obstructive apnea")], 100)
    assert np.flatnonzero(m).tolist() == list(range(35, 51))  # 16 s
    m = dl_labels.sec_event_mask([_ev(35.5, 10.0)], 100)
    assert np.flatnonzero(m).tolist() == list(range(35, 45))  # 35.5 in [35.5,45.5); 45.5 not
    m = dl_labels.sec_event_mask([_ev(35.6, 10.0)], 100)
    assert np.flatnonzero(m).tolist() == list(range(36, 46))  # 35.5 < 35.6 -> s=35 out; 45.5 < 45.6 -> s=45 in
    m = dl_labels.sec_event_mask([_ev(10.2, 0.2)], 100)
    assert m.sum() == 0  # no midpoint inside
    m = dl_labels.sec_event_mask([_ev(10.4, 0.2)], 100)
    assert np.flatnonzero(m).tolist() == [10]


def test_sec_event_bits_overlap_and_clip():
    evs = [_ev(0, 10, "Obstructive apnea"), _ev(5, 10, "Central apnea"), _ev(8, 4, "Mixed apnea"),
           _ev(95, 20, "Hypopnea"), _ev(-3, 5, "Hypopnea"), _ev(500, 5, "Hypopnea")]
    m = dl_labels.sec_event_mask(evs, 100)
    assert m.dtype == np.uint8 and m.size == 100
    assert m[0] == 1 | 8 and m[1] == 1 | 8 and m[2] == 1  # (-3, 2) -> seconds 0..1
    assert m[6] == 1 | 2 and m[9] == 1 | 2 | 4 and m[12] == 2
    assert m[95:].tolist() == [8] * 5
    assert dl_labels.ANY_EVENT == 15


def test_sec_event_unknown_kind_raises():
    with pytest.raises(ValueError):
        dl_labels.sec_event_mask([_ev(0, 10, "Snore")], 30)


def test_sec_sleep_and_stage_codes():
    stages = np.array(["W", "N1", "N2", "N3", "REM", "?"], dtype=object)
    codes = dl_labels.encode_stages(stages)
    assert codes.dtype == np.int8 and codes.tolist() == [0, 1, 2, 3, 5, 9]
    s = dl_labels.sec_sleep(codes)
    assert s.dtype == np.uint8 and s.size == 180
    assert s.reshape(6, 30).all(axis=1).tolist() == [False, True, True, True, True, False]
    assert s.reshape(6, 30).any(axis=1).tolist() == [False, True, True, True, True, False]
    assert dl_labels.sec_sleep(codes, 200)[180:].sum() == 0
    with pytest.raises(ValueError):
        dl_labels.encode_stages(np.array(["N4"], dtype=object))


def test_epoch_labels_use_build_epoch_frame(tmp_path):
    gt = make_subject(tmp_path, 900001)
    hyp, evs = read_nsrr_xml(gt["xml"])
    stage, ap, frame = dl_labels.epoch_labels(900001, 12, hyp, evs)
    ref = build_epoch_frame(900001, "shhs1", 12, hyp, evs)
    assert np.array_equal(ap, ref["apnoea_label"].to_numpy().astype(np.int8))
    assert np.array_equal(ap, apnoea_labels(12, evs))
    assert frame.equals(ref)
    assert stage.tolist() == [dl_labels.STAGE_CODES[x] for x in gt["stages"]]


def test_epoch_labels_differ_from_naive_per_second_rule():
    """Why epoch labels never come from the mask: two short events in one epoch."""
    evs = [_ev(0, 6), _ev(15, 6)]  # 12 s of events in epoch 0, but no single event >= 10 s
    assert apnoea_labels(1, evs).tolist() == [0]
    assert dl_labels.sec_event_mask(evs, 30).astype(bool).sum() >= 10


# --------------------------------------------------------------------------- encoders


def test_spo2_integer_code_is_lossless_on_shhs_lattice():
    d = spo2_digital(np.array([0, 50, 85, 94, 95, 96, 99, 100]))
    sig = edf_native.EdfSignal(0, "SaO2", "", "", 0.0, 100.0, -32768, 32767, "", 1, 1.0)
    arr, meta = dl_cache.encode_spo2(d, sig)
    assert arr.dtype == np.uint8 and meta["encoding"] == "uint8_pct_lut"
    assert arr.tolist() == [0, 50, 85, 94, 95, 96, 99, 100]
    assert np.array_equal(dl_cache.decode_digital(arr, meta), d)
    assert meta["max_abs_code_minus_pct"] < 0.4
    # values seen in shhs1-203443: p=79 -> 19199, p=84 -> 22527, p=97 -> 30975
    assert spo2_digital(np.array([79, 84, 97])).tolist() == [19199, 22527, 30975]


def test_spo2_falls_back_to_uint16_when_codes_collide():
    d = np.array([10000, 10100, 10200], dtype=np.int16)  # ~65.4 %, 65.5 %, 65.7 % -> codes 65, 65/66 collide
    sig = edf_native.EdfSignal(0, "SaO2", "", "", 0.0, 100.0, -32768, 32767, "", 1, 1.0)
    arr, meta = dl_cache.encode_spo2(d, sig)
    assert arr.dtype == np.uint16 and meta["encoding"] == "uint16_offset" and "fallback_reason" in meta
    assert np.array_equal(dl_cache.decode_digital(arr, meta), d)


def test_rank_encoding_and_fallback():
    rng = np.random.default_rng(0)
    d = rng.choice(np.array([-32767, -20000, -1638, 613, 32767], dtype=np.int16), 300).reshape(10, 30)
    arr, meta = dl_cache.encode_rank(d)
    assert arr.dtype == np.uint8 and meta["encoding"] == "uint8_rank_lut" and len(meta["lut"]) == 5
    assert np.array_equal(dl_cache.decode_digital(arr, meta), d)
    many = np.arange(-300, 300, dtype=np.int16).reshape(20, 30)
    arr2, meta2 = dl_cache.encode_rank(many)
    assert arr2.dtype == np.uint16 and np.array_equal(dl_cache.decode_digital(arr2, meta2), many)


def test_int8_and_uint8_fallbacks():
    a, m = dl_cache.encode_int8(np.array([-128, 0, 127], dtype=np.int16))
    assert a.dtype == np.int8 and m["encoding"] == "int8_digital"
    a, m = dl_cache.encode_int8(np.array([-129, 0, 127], dtype=np.int16))
    assert a.dtype == np.int16 and np.array_equal(dl_cache.decode_digital(a, m), [-129, 0, 127])
    a, m = dl_cache.encode_uint8(np.array([0, 3], dtype=np.int16))
    assert a.dtype == np.uint8
    a, m = dl_cache.encode_uint8(np.array([-1, 3], dtype=np.int16))
    assert a.dtype == np.uint16 and np.array_equal(dl_cache.decode_digital(a, m), [-1, 3])


# --------------------------------------------------------------------------- extraction


def test_extract_subject_is_lossless_and_aligned(tmp_path):
    gt = make_subject(tmp_path, 900001, n_epochs=12, tail_s=17)
    arrays, meta = dl_cache.extract_subject(gt["edf"], gt["xml"], 900001, split="train")
    n = 12
    assert meta["n_epochs"] == n and meta["n_seconds"] == 360 and meta["tail_seconds_dropped"] == 17
    exp = {"ecg": (np.int8, (n, 3750)), "thor": (np.int8, (n, 300)), "abdo": (np.int8, (n, 300)),
           "airflow": (np.int8, (n, 300)), "airflow_rej0": (np.int8, (n, 300)), "spo2": (np.uint8, (n, 30)),
           "hr": (np.uint8, (n, 30)), "position": (np.uint8, (n, 30)), "oxstat": (np.uint8, (n, 30)),
           "epoch_stage": (np.int8, (n,)), "apnoea_label": (np.int8, (n,)), "sec_event": (np.uint8, (n, 30))}
    assert set(arrays) == set(exp)
    for k, (dt, shape) in exp.items():
        assert arrays[k].dtype == dt and arrays[k].shape == shape, k
    ch = meta["channels"]
    # lossless: stored segment == EDF digital samples of the first n*30 seconds
    assert np.array_equal(dl_cache.decode_digital(arrays["ecg"], ch["ecg"]).ravel(), gt["ecg"][: 360 * 125])
    assert np.array_equal(arrays["thor"].ravel(), gt["thor"][:3600])
    assert np.array_equal(arrays["abdo"].ravel(), gt["abdo"][:3600])
    assert np.array_equal(dl_cache.decode_digital(arrays["spo2"], ch["spo2"]).ravel(), gt["spo2"][:360])
    assert np.array_equal(dl_cache.decode_digital(arrays["hr"], ch["hr"]).ravel(), gt["hr"][:360])
    assert np.array_equal(arrays["position"].ravel(), gt["pos"][:360])
    # airflow: live NEW AIR chosen over dead AIRFLOW; the rejected one is kept
    af = meta["airflow"]
    assert af["chosen_label"] == "NEW AIR" and af["ok"] and len(af["candidates"]) == 2
    assert np.array_equal(arrays["airflow"].ravel(), gt["air"]["NEW AIR"][0][:3600])
    assert np.array_equal(arrays["airflow_rej0"].ravel(), gt["air"]["AIRFLOW"][0][:3600])
    assert af["rejected_arrays"] == {"airflow_rej0": ch["airflow_rej0"]["index"]}
    # scaling sign recorded
    assert ch["thor"]["scaling"]["sign"] == -1 and ch["ecg"]["scaling"]["sign"] == 1
    assert ch["airflow"]["scaling"]["sign"] == 1 and ch["airflow_rej0"]["scaling"]["sign"] == -1
    # physical decode matches the EDF scaling
    phys = dl_cache.decode_physical(arrays["thor"], ch["thor"])
    assert np.allclose(phys.ravel(), gt["thor"][:3600] * (-2 / 255) + (1 - (-128) * (-2 / 255)), atol=1e-6)
    # SpO2 code == integer percent, 0 = dropout
    assert arrays["spo2"].ravel()[5:8].tolist() == [0, 0, 0]
    # labels
    hyp, evs = read_nsrr_xml(gt["xml"])
    ref = build_epoch_frame(900001, "shhs1", n, hyp, evs)
    assert np.array_equal(arrays["apnoea_label"], ref["apnoea_label"].to_numpy().astype(np.int8))
    assert np.array_equal(arrays["sec_event"].ravel(), dl_labels.sec_event_mask(evs, 360))
    sec = arrays["sec_event"].ravel()
    assert sec[35:51].tolist() == [1] * 16 and sec[34] == 0 and sec[51] == 0
    assert sec[100] & 2 and sec[100] & 8  # central + hypopnoea overlap
    lab = meta["labels"]
    sleep = np.isin(arrays["epoch_stage"], dl_labels.SLEEP_CODES)
    assert lab["n_sleep_epochs"] == int(sleep.sum())
    assert lab["n_apnoea_epochs_sleep"] == int(arrays["apnoea_label"][sleep].sum())
    assert lab["events"][0] == [35.0, 16.0, 1] and lab["n_events"] == 4
    assert meta["qc"]["missing_channels"] == []


def test_extract_flags_missing_and_dead_channels(tmp_path):
    gt = make_subject(tmp_path, 900002, airflow_labels=("AIRFLOW",), live_airflow="none", with_oxstat=False)
    arrays, meta = dl_cache.extract_subject(gt["edf"], gt["xml"], 900002)
    assert meta["airflow"]["chosen_label"] == "AIRFLOW" and not meta["airflow"]["ok"]
    assert "airflow_rej0" not in arrays
    assert meta["channels"]["oxstat"]["encoding"] == "missing" and arrays["oxstat"].sum() == 0
    assert meta["qc"]["missing_channels"] == ["OXSTAT"]
    gt3 = make_subject(tmp_path, 900003, airflow_labels=())
    arrays3, meta3 = dl_cache.extract_subject(gt3["edf"], gt3["xml"], 900003)
    assert meta3["airflow"]["chosen_index"] is None and not meta3["airflow"]["ok"]
    assert arrays3["airflow"].shape == (12, 300) and not arrays3["airflow"].any()
    assert dl_cache._flags(meta3)[0] == "airflow_missing"


def test_xml_warning_flag_ignores_routine_spo2_concepts():
    base = {"airflow": {"ok": True, "chosen_index": 1}, "channels": {"spo2": {"encoding": "x"},
            "hr": {"encoding": "x"}}}
    w1 = "read_nsrr_xml: ignored 3 respiratory-typed event(s) with non-canonical concepts: 'SpO2 artifact'×1, " \
         "'SpO2 desaturation'×2"
    w2 = w1 + ", 'Unsure'×1"
    assert dl_cache._flags({**base, "qc": {"xml_warnings": [w1]}}) == []
    assert dl_cache._flags({**base, "qc": {"xml_warnings": [w2]}}) == ["xml_noncanonical_resp"]


def test_extract_refuses_rate_deviation(tmp_path):
    gt = make_subject(tmp_path, 900004, ecg_fs=250)
    with pytest.raises(edf_native.RateDeviation, match="ECG"):
        dl_cache.extract_subject(gt["edf"], gt["xml"], 900004)
    for lab, canon in (("THOR RES", "THOR"), ("ABDO RES", "ABDO")):  # the other M / P4 inputs
        g = make_subject(tmp_path, 900010, fs_override={lab: 25})
        with pytest.raises(edf_native.RateDeviation, match=canon):
            dl_cache.extract_subject(g["edf"], g["xml"], 900010)


@pytest.mark.parametrize("case, kw, canon, key", [
    ("position_2hz", {"fs_override": {"POSITION": 2}}, "POSITION", "position"),
    ("hr_2hz", {"fs_override": {"H.R.": 2}}, "HR", "hr"),
    ("spo2_2hz", {"fs_override": {"SaO2": 2}}, "SPO2", "spo2"),
    ("oxstat_2hz", {"fs_override": {"OX stat": 2}}, "OXSTAT", "oxstat"),
    ("stray_aux_25hz", {"extra_signals": (("AUX", 25),)}, "AIRFLOW", None),
    ("picked_airflow_20hz", {"fs_override": {"NEW AIR": 20}}, "AIRFLOW", None),
])
def test_off_rate_non_model_channel_is_dropped_not_fatal(tmp_path, case, kw, canon, key):
    """Reviewer repro (split='test'): these used to fail the whole subject with RateDeviation."""
    sid = 900020
    gt = make_subject(tmp_path, sid, **kw)
    rec = dl_cache.build_one({"subject_id": sid, "split": "test", "edf": str(gt["edf"]), "xml": str(gt["xml"]),
                              "out_dir": str(tmp_path / "c"), "backoff": []})
    assert rec["status"] == "ok", rec
    assert f"rate_dropped:{canon}" in rec["flags"]
    arrays, meta = dl_cache.load_raw(tmp_path / "c", sid, verify=True)
    (rd,) = meta["qc"]["rate_dropped"]
    assert rd["canon"] == canon and rd["fs"] != rd["expected_fs"]
    assert any(n.startswith(f"{canon} dropped: ") for n in meta["qc"]["notes"])
    # raw samples kept at the native rate, exactly (no re-download ever needed)
    sig = gt["signals"][rd["index"]]
    off = arrays[rd["array"]]
    assert off.ndim == 1 and meta["channels"][rd["array"]]["rate_dropped"]
    assert np.array_equal(dl_cache.decode_digital(off, meta["channels"][rd["array"]]),
                          sig.digital[: meta["n_seconds"] * sig.fs])
    # the M / P4 inputs are untouched
    assert np.array_equal(arrays["ecg"].ravel(), gt["ecg"][: meta["n_seconds"] * 125])
    assert np.array_equal(arrays["thor"].ravel(), gt["thor"][: meta["n_seconds"] * 10])
    ph = dl_cache.load_physical(tmp_path / "c", sid)
    if key is not None:  # the regular array is zero-filled, encoding "missing"
        assert meta["channels"][key]["encoding"] == "missing" and not arrays[key].any()
        assert ph["signals"][canon] is None and canon in meta["qc"]["missing_channels"]
    elif case == "stray_aux_25hz":  # excluded from the pick; the live NEW AIR still wins
        assert meta["airflow"]["chosen_label"] == "NEW AIR" and meta["airflow"]["ok"]
        assert "AUX" not in [c["label"] for c in meta["airflow"]["candidates"]]
        assert meta["airflow"]["rate_dropped_candidates"][0]["label"] == "AUX"
    else:  # the live candidate is off-rate: the dead on-rate AIRFLOW is picked, airflow_ok False
        assert meta["airflow"]["chosen_label"] == "AIRFLOW" and not meta["airflow"]["ok"]
        assert "airflow_dead" in rec["flags"] and "airflow_rej0" not in arrays


def test_on_rate_sidecar_has_no_rate_dropped_key(tmp_path):
    """Sidecars of on-rate subjects are unchanged by the drop path (no new keys)."""
    gt = make_subject(tmp_path, 900021)
    arrays, meta = dl_cache.extract_subject(gt["edf"], gt["xml"], 900021)
    assert "rate_dropped" not in meta["qc"] and "rate_dropped_candidates" not in meta["airflow"]
    assert not any(k.endswith("_offrate") or "offrate" in k for k in arrays)


def test_extract_refuses_dataless_xml(tmp_path, monkeypatch):
    gt = make_subject(tmp_path, 900005)
    real_stat = os.stat

    class FakeStat:
        def __init__(self, st):
            self._st = st

        def __getattr__(self, k):
            return getattr(self._st, k)

        @property
        def st_flags(self):
            return edf_native.UF_DATALESS

    monkeypatch.setattr(dl_cache.os, "stat",
                        lambda p, *a, **k: FakeStat(real_stat(p)) if str(p).endswith(".xml") else real_stat(p))
    with pytest.raises(edf_native.DatalessFileError):
        dl_cache.extract_subject(gt["edf"], gt["xml"], 900005)
    rec = dl_cache.build_one({"subject_id": 900005, "edf": str(gt["edf"]), "xml": str(gt["xml"]),
                              "out_dir": str(tmp_path / "c"), "fetch_xml": False})
    assert rec["status"] == "failed" and "fetch_xml is off" in rec["error"]


# --------------------------------------------------------------------------- write / load / ledger


def test_write_load_roundtrip_and_integrity(tmp_path):
    gt = make_subject(tmp_path, 900001)
    arrays, meta = dl_cache.extract_subject(gt["edf"], gt["xml"], 900001, split="val")
    out = tmp_path / "cache"
    side = dl_cache.write_raw(out, 900001, arrays, meta, provenance={"git": {"commit": "abc"}})
    assert sorted(p.name for p in out.iterdir()) == ["shhs1-900001.json", "shhs1-900001.npz"]  # no tmp left
    assert side["npz"]["sha256"] == dl_cache.sha256_file(out / "shhs1-900001.npz")
    assert dl_cache.is_complete(out, 900001)
    back, m2 = dl_cache.load_raw(out, 900001, verify=True)
    assert set(back) == set(arrays)
    for k in arrays:
        assert back[k].dtype == arrays[k].dtype and np.array_equal(back[k], arrays[k])
    assert m2["split"] == "val" and m2["provenance"]["git"]["commit"] == "abc"
    assert m2["arrays"]["ecg"]["sha256"] == dl_cache.sha256_array(arrays["ecg"])
    sub, _ = dl_cache.load_raw(out, 900001, keys=("apnoea_label",))
    assert list(sub) == ["apnoea_label"]
    # corrupt one byte -> not complete, verify fails
    p = out / "shhs1-900001.npz"
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))
    assert not dl_cache.is_complete(out, 900001)
    assert dl_cache.is_complete(out, 900001, verify=False)  # size unchanged
    with pytest.raises(Exception):
        dl_cache.load_raw(out, 900001, verify=True)
    # missing sidecar -> incomplete
    (out / "shhs1-900001.json").unlink()
    assert not dl_cache.is_complete(out, 900001, verify=False)


def test_ledger_latest_and_torn_line(tmp_path):
    led = dl_cache.Ledger(tmp_path / "ledger.jsonl")
    led.append({"subject_id": 1, "status": "failed"})
    led.append({"subject_id": 2, "status": "ok"})
    led.append({"subject_id": 1, "status": "ok"})
    with open(led.path, "a") as fh:
        fh.write('{"subject_id": 3, "sta')  # killed mid-write
    latest = led.latest()
    assert latest[1]["status"] == "ok" and latest[2]["status"] == "ok" and 3 not in latest
    assert all("ts" in r for r in led.records())


def test_ledger_append_after_torn_line_keeps_the_new_record(tmp_path):
    """Reviewer repro: before the fix, record 3 was glued onto the fragment and lost."""
    led = dl_cache.Ledger(tmp_path / "ledger.jsonl")
    led.append({"subject_id": 1, "status": "ok"})
    with open(led.path, "a") as fh:
        fh.write('{"subject_id": 2, "sta')  # torn (ENOSPC / power loss)
    led.append({"subject_id": 3, "status": "failed"})
    led.append({"subject_id": 4, "status": "ok"})
    lines = led.path.read_text().splitlines()
    assert lines[1] == '{"subject_id": 2, "sta' and len(lines) == 4
    assert sorted(led.latest()) == [1, 3, 4] and led.latest()[3]["status"] == "failed"
    empty = dl_cache.Ledger(tmp_path / "new.jsonl")
    empty.append({"subject_id": 5})
    assert empty.path.read_text().count("\n") == 1 and not empty.path.read_text().startswith("\n")


def test_ledger_flag_drift_and_reflag(tmp_path):
    gt = make_subject(tmp_path, 900009)
    out = tmp_path / "c"
    rec = dl_cache.build_one({"subject_id": 900009, "edf": str(gt["edf"]), "xml": str(gt["xml"]),
                              "out_dir": str(out)})
    led = dl_cache.Ledger(out / "ledger.jsonl")
    led.append(rec)
    assert rec["status"] == "ok" and dl_cache.ledger_flag_drift(out) == []
    stale = {**rec, "flags": ["xml_warnings"]}  # what the smoke run's older _flags wrote
    led.append(stale)
    drift = dl_cache.ledger_flag_drift(out)
    assert [d["subject_id"] for d in drift] == [900009] and drift[0]["sidecar_flags"] == rec["flags"]
    fixed = dl_cache.reflag_ledger(out)
    assert len(fixed) == 1 and dl_cache.ledger_flag_drift(out) == []
    last = led.latest()[900009]
    assert last["flags"] == rec["flags"] and last["reflag"]["stale_flags"] == ["xml_warnings"]
    assert last["n_epochs"] == rec["n_epochs"] and len(led.records()) == 3  # append-only


# --------------------------------------------------------------------------- download helpers


def test_with_backoff_retries_then_succeeds():
    calls, slept, log = [], [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("Operation not permitted")
        return "ok"

    assert dl_cache.with_backoff(fn, delays=(30, 60, 120), sleep=slept.append, log=log) == "ok"
    assert slept == [30, 60] and len(log) == 2


def test_with_backoff_exhausts_and_does_not_retry_other_errors():
    slept = []
    with pytest.raises(TimeoutError):
        dl_cache.with_backoff(lambda: (_ for _ in ()).throw(TimeoutError("t")), delays=(1, 2),
                              sleep=slept.append)
    assert slept == [1, 2]
    slept.clear()
    with pytest.raises(ValueError):
        dl_cache.with_backoff(lambda: (_ for _ in ()).throw(ValueError("x")), delays=(1, 2), sleep=slept.append)
    assert slept == []


def test_default_backoff_is_5_retries_30s_to_8min():
    slept = []
    with pytest.raises(PermissionError):
        dl_cache.with_backoff(lambda: (_ for _ in ()).throw(PermissionError("p")), sleep=slept.append)
    assert slept == [30, 60, 120, 240, 480]


def test_materialise_errors(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 1000)
    assert dl_cache.materialise(f, timeout_s=10) >= 0
    with pytest.raises(PermissionError):
        dl_cache.materialise(f, 10, reader=("/bin/sh", "-c", "echo 'cat: x: Operation not permitted' >&2; exit 1",
                                            "sh"))
    with pytest.raises(TimeoutError):
        dl_cache.materialise(f, 0.3, reader=("/bin/sh", "-c", "sleep 5", "sh"))
    with pytest.raises(OSError):
        dl_cache.materialise(f, 10, reader=("/bin/sh", "-c", "echo boom >&2; exit 3", "sh"))


@pytest.mark.parametrize("msg, cls", [
    ("Operation not permitted", PermissionError),
    ("Permission denied", PermissionError),
    ("Operation timed out", TimeoutError),  # errno 60, os.strerror(60) on macOS
    ("Resource temporarily unavailable", BlockingIOError),
    ("Interrupted system call", InterruptedError),
    ("Resource deadlock avoided", dl_cache.MaterialiseError),
    ("Input/output error", dl_cache.MaterialiseError),
])
def test_materialise_reader_errors_are_classified_and_retried(tmp_path, msg, cls):
    """Every non-zero reader exit is retried under the default backoff (reviewer repro: 0 retries before)."""
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 10)
    reader = ("/bin/sh", "-c", f"echo 'cat: x.edf: {msg}' >&2; exit 1", "sh")
    slept, log = [], []
    with pytest.raises(cls) as ei:
        dl_cache.with_backoff(lambda: dl_cache.materialise(f, 10, reader=reader), sleep=slept.append, log=log)
    assert type(ei.value) is cls and isinstance(ei.value, OSError) and isinstance(ei.value, dl_cache.RETRYABLE)
    assert slept == [30, 60, 120, 240, 480] and len(log) == 6
    assert dl_cache.is_transient_failure({"status": "failed", "error": f"{cls.__name__}: {ei.value}"})


def test_build_one_retries_timeout_on_read(tmp_path, monkeypatch):
    """An in-process ETIMEDOUT (Python raises TimeoutError for errno 60) on the extraction read is retried."""
    gt = make_subject(tmp_path, 900008)
    real = dl_cache.extract_subject
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(60, "Operation timed out")  # -> TimeoutError
        return real(*a, **k)

    monkeypatch.setattr(dl_cache, "extract_subject", flaky)
    monkeypatch.setattr(dl_cache.time, "sleep", lambda s: None)
    rec = dl_cache.build_one({"subject_id": 900008, "edf": str(gt["edf"]), "xml": str(gt["xml"]),
                              "out_dir": str(tmp_path / "c"), "backoff": [0.0]})
    assert rec["status"] == "ok" and len(calls) == 2 and rec["retries"][0][1].startswith("TimeoutError")


def test_transient_failure_classification():
    assert dl_cache.is_transient_failure({"status": "failed", "error_type": "StillDataless"})
    assert dl_cache.is_transient_failure({"status": "failed", "error": "DatalessFileError: x is a placeholder"})
    assert not dl_cache.is_transient_failure({"status": "failed", "error": "RateDeviation: ECG [3:ECG] fs=250"})
    assert not dl_cache.is_transient_failure({"status": "failed", "error_type": "ExtractionError"})
    assert not dl_cache.is_transient_failure({"status": "ok", "error_type": "TimeoutError"})


# --------------------------------------------------------------------------- label parity


def _meta_and_key(tmp_path, sids):
    rows, keys = [], []
    for sid in sids:
        arrays, meta = dl_cache.load_raw(tmp_path / "cache", sid, keys=("epoch_stage", "apnoea_label"))
        st, ap = arrays["epoch_stage"], arrays["apnoea_label"]
        sl = np.isin(st, dl_labels.SLEEP_CODES)
        rows.append({"sid": sid, "n_epochs": float(st.size), "tst_min": sl.sum() * 0.5,
                     "n_apnoea_epochs": float(ap[sl].sum())})
        idx = np.flatnonzero(sl)
        keys.append(pd.DataFrame({"subject_id": np.int32(sid), "epoch_idx": idx.astype(np.int32),
                                  "apnoea_label": ap[idx].astype(np.int8)}))
    return pd.DataFrame(rows).set_index("sid", drop=False), pd.concat(keys, ignore_index=True)


def test_label_parity_detects_mismatch(tmp_path):
    for sid in (900001, 900002):
        gt = make_subject(tmp_path, sid, seed=sid)
        dl_cache.write_raw(tmp_path / "cache", sid, *dl_cache.extract_subject(gt["edf"], gt["xml"], sid))
    meta, key = _meta_and_key(tmp_path, (900001, 900002))
    rep = dl_cache.label_parity(tmp_path / "cache", [900001, 900002], meta, key[key.subject_id == 900001])
    assert rep[["n_epochs_ok", "n_sleep_ok", "apnoea_sum_ok"]].all().all()
    assert rep.set_index("subject_id").loc[900001, "key_rows_ok"] == True  # noqa: E712
    assert pd.isna(rep.set_index("subject_id").loc[900002, "key_rows_ok"])
    bad_meta = meta.copy()
    bad_meta.loc[900002, "n_apnoea_epochs"] += 1
    bad_key = key.copy()
    bad_key.loc[bad_key.index[0], "apnoea_label"] = 1 - bad_key.loc[bad_key.index[0], "apnoea_label"]
    rep2 = dl_cache.label_parity(tmp_path / "cache", [900001, 900002], bad_meta, bad_key)
    r = rep2.set_index("subject_id")
    assert not r.loc[900002, "apnoea_sum_ok"] and not r.loc[900001, "key_rows_ok"]


# --------------------------------------------------------------------------- CLI on a synthetic root


def _synthetic_root(tmp_path):
    root = tmp_path / "shhs"
    root.mkdir()
    ids = list(range(900001, 900013))
    for sid in ids:
        make_subject(root, sid, seed=sid)
    s = splits.aim2_split(ids)
    rec = splits.split_record(s, ids_source="synthetic", n_ids=len(ids))
    sp = tmp_path / "split.json"
    splits.write_split(sp, rec)
    return root, sp, s


def test_order_subjects_val_test_then_seeded_train():
    import build_dl_cache as b

    s = {"train": np.arange(100, 120), "val": np.array([5, 6]), "test": np.array([1, 2, 3])}
    o = b.order_subjects(s, ["val", "test", "train"], 42)
    assert [x for x, _ in o[:5]] == [5, 6, 1, 2, 3]
    train = [x for x, n in o if n == "train"]
    assert sorted(train) == list(range(100, 120)) and train != sorted(train)
    assert train == [x for x, n in b.order_subjects(s, ["train"], 42)]
    assert train != [x for x, n in b.order_subjects(s, ["train"], 43)]


def test_cli_only_local_end_to_end(tmp_path):
    import build_dl_cache as b

    root, sp, s = _synthetic_root(tmp_path)
    # break one subject's EDF (ECG at 250 Hz) so it fails
    bad = int(s["train"][0])
    make_subject(root, bad, seed=bad, ecg_fs=250)
    out = tmp_path / "cache"
    args = ["--only-local", "--out", str(out), "--split", str(sp), "--shhs-root", str(root),
            "--workers", "2", "--min-free-gb", "0", "--backoff", "0"]
    r = CliRunner().invoke(b.main, args + ["--dry-run"])
    assert r.exit_code == 0, r.output
    assert "todo" in r.output and not out.exists()
    r = CliRunner().invoke(b.main, args)
    assert r.exit_code == 0, r.output
    led = dl_cache.Ledger(out / "ledger.jsonl").latest()
    assert len(led) == 12 and led[bad]["status"] == "failed" and "RateDeviation" in led[bad]["error"]
    assert sum(v["status"] == "ok" for v in led.values()) == 11
    assert (out / "failures.jsonl").exists() and (out / "run_config.json").exists()
    for sid in s["val"]:
        m = dl_cache.load_meta(out, int(sid))
        assert m["split"] == "val" and m["provenance"]["split_sha256"] == splits.file_sha256(sp)
    # rerun: everything cached or previously failed -> nothing to do
    n_lines = len((out / "ledger.jsonl").read_text().splitlines())
    r = CliRunner().invoke(b.main, args)
    assert r.exit_code == 0 and "cached" in r.output and "failed_before_skipped" in r.output
    assert len((out / "ledger.jsonl").read_text().splitlines()) == n_lines
    # --retry-failed retries it (fixed file now succeeds)
    make_subject(root, bad, seed=bad)
    r = CliRunner().invoke(b.main, args + ["--retry-failed"])
    assert r.exit_code == 0, r.output
    assert dl_cache.Ledger(out / "ledger.jsonl").latest()[bad]["status"] == "ok"
    rep = dl_cache.label_parity(out, list(range(900001, 900013)),
                                _meta_and_key(tmp_path, list(range(900001, 900013)))[0])
    assert rep[["n_epochs_ok", "n_sleep_ok", "apnoea_sum_ok"]].all().all()


def test_cli_stops_at_disk_floor(tmp_path):
    import build_dl_cache as b

    root, sp, _ = _synthetic_root(tmp_path)
    out = tmp_path / "cache"
    r = CliRunner().invoke(b.main, ["--only-local", "--out", str(out), "--split", str(sp),
                                    "--shhs-root", str(root), "--min-free-gb", "1e9"])
    assert r.exit_code == 2 and "below --min-free-gb" in r.output
    assert not any(out.glob("*.npz"))


def test_cli_download_mode_on_local_synthetic_root(tmp_path):
    """Exercises the download code path on files that are already local (no network)."""
    import build_dl_cache as b

    root, sp, s = _synthetic_root(tmp_path)
    out = tmp_path / "cache"
    r = CliRunner().invoke(b.main, ["--download", "--order", "val,test", "--out", str(out), "--split", str(sp),
                                    "--shhs-root", str(root), "--min-free-gb", "0", "--workers", "1"])
    assert r.exit_code == 0, r.output
    led = dl_cache.Ledger(out / "ledger.jsonl").records()
    assert [x["subject_id"] for x in led] == [int(i) for i in s["val"]] + [int(i) for i in s["test"]]
    assert all(x["status"] == "ok" and x["mode"] == "download" for x in led)


def test_cli_refuses_desktop_and_requires_mode(tmp_path):
    import build_dl_cache as b

    root, sp, _ = _synthetic_root(tmp_path)
    r = CliRunner().invoke(b.main, ["--only-local", "--out", str(Path.home() / "Desktop" / "x"),
                                    "--split", str(sp), "--shhs-root", str(root), "--dry-run"])
    assert r.exit_code != 0 and "Desktop" in r.output
    r = CliRunner().invoke(b.main, ["--split", str(sp), "--shhs-root", str(root), "--dry-run"])
    assert r.exit_code != 0
    # reviewer repro: ~/Documents (iCloud Desktop & Documents sync) used to pass the guard
    probe = Path.home() / "Documents" / "__dlcache_guard_probe"
    r = CliRunner().invoke(b.main, ["--only-local", "--out", str(probe), "--split", str(sp),
                                    "--shhs-root", str(root), "--dry-run"])
    assert r.exit_code == 2 and "cloud-synced" in r.output and not probe.exists()


def test_cloud_synced_root_on_a_fake_home(tmp_path):
    home = tmp_path / "home"
    for d in ("Desktop", "Desktop2", "Documents", "Library/Mobile Documents/com~apple~CloudDocs",
              "Library/CloudStorage/GoogleDrive-x/My Drive", "Library/CloudStorage/OneDrive-Y", "thesis_dl_cache"):
        (home / d).mkdir(parents=True)
    (home / "OneDrive - Y").symlink_to(home / "Library/CloudStorage/OneDrive-Y")
    (home / "Library/Mobile Documents/com~apple~CloudDocs/Desktop").symlink_to(home / "Desktop")
    f = dl_cache.cloud_synced_root
    assert f(home / "Desktop" / "x", home) == home / "Desktop"
    assert f(home / "Documents" / "cache", home) == home / "Documents"
    assert f(home / "Library/CloudStorage/GoogleDrive-x/My Drive/c", home) == home / "Library/CloudStorage"
    assert f(home / "OneDrive - Y" / "c", home) is not None  # symlink into CloudStorage
    assert f(home / "Library/Mobile Documents/com~apple~CloudDocs/Desktop/c", home) is not None
    assert f(home / "Desktop2" / "x", home) is None  # no string-prefix false positive
    assert f(home / "thesis_dl_cache" / "raw_v1", home) is None
    assert f(tmp_path / "elsewhere", home) is None


def _write_verify_inputs(tmp_path, out, s, ids):
    """subject_metadata-like parquet + key parquet for the cached ids (as test_verify_cli_on_synthetic_cache)."""
    assert out == tmp_path / "cache"  # _meta_and_key reads tmp_path / "cache"
    meta, key = _meta_and_key(tmp_path, [i for i in ids if dl_cache.is_complete(out, i)])
    mpath, kpath = tmp_path / "meta.parquet", tmp_path / "key.parquet"
    meta.assign(subject_id=[f"shhs1-{i}" for i in meta["sid"]]).drop(columns="sid").reset_index(drop=True) \
        .to_parquet(mpath)
    key[key.subject_id.isin(s["test"])].to_parquet(kpath)
    return mpath, kpath


def test_gate_failed_test_subject_exits_3_and_verify_require_complete(tmp_path):
    """Reviewer repro: a failed TEST subject gave build exit 0 and verify ALL CHECKS PASSED."""
    import build_dl_cache as b
    import verify_dl_cache as v

    root, sp, s = _synthetic_root(tmp_path)
    bad = int(s["test"][0])
    make_subject(root, bad, seed=bad, ecg_fs=250)  # permanent failure (RateDeviation on ECG)
    out = tmp_path / "cache"
    args = ["--only-local", "--out", str(out), "--split", str(sp), "--shhs-root", str(root),
            "--min-free-gb", "0", "--workers", "1", "--backoff", "0"]
    r = CliRunner().invoke(b.main, args)
    assert r.exit_code == 3, r.output
    assert "GATE test: 1 subject(s)" in r.output and str(bad) in r.output
    r = CliRunner().invoke(b.main, args)  # resume: failed_before_skipped still blocks
    assert r.exit_code == 3 and "failed_before_skipped" in r.output
    r = CliRunner().invoke(b.main, args + ["--dry-run"])  # a dry run only reports
    assert r.exit_code == 0 and "GATE test" in r.output
    # a failed TRAIN subject never changes the exit code (test_cli_only_local_end_to_end)
    ids = list(range(900001, 900013))
    mpath, kpath = _write_verify_inputs(tmp_path, out, s, ids)
    vargs = ["--cache", str(out), "--split", str(sp), "--metadata", str(mpath), "--key", str(kpath)]
    r = CliRunner().invoke(v.main, vargs)
    n_test = len(s["test"])
    assert r.exit_code == 0 and f"test  {n_test - 1:5d}/{n_test:<5d}" in r.output and "ALL CHECKS PASSED" in r.output
    r = CliRunner().invoke(v.main, vargs + ["--require-complete", "val,test"])
    assert r.exit_code == 1 and "required split(s) incomplete: ['test']" in r.output and "CHECKS FAILED" in r.output
    r = CliRunner().invoke(v.main, vargs + ["--require-complete", "val"])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(v.main, vargs + ["--require-complete", "nope"])
    assert r.exit_code == 2
    # fixed file + --retry-failed -> complete -> build 0, verify --require-complete 0
    make_subject(root, bad, seed=bad)
    r = CliRunner().invoke(b.main, args + ["--retry-failed"])
    assert r.exit_code == 0 and "GATE" not in r.output, r.output
    mpath, kpath = _write_verify_inputs(tmp_path, out, s, ids)
    r = CliRunner().invoke(v.main, vargs + ["--require-complete", "val,test"])
    assert r.exit_code == 0 and "ALL CHECKS PASSED" in r.output, r.output


def test_plan_auto_retries_transient_val_test_failures_only(tmp_path):
    import build_dl_cache as b

    root, sp, s = _synthetic_root(tmp_path)
    out = tmp_path / "cache"
    led = dl_cache.Ledger(out / "ledger.jsonl")
    t, v_, tr = int(s["test"][0]), int(s["val"][0]), int(s["train"][0])
    t2 = int(s["test"][1])
    led.append({"subject_id": t, "status": "failed", "error": "TimeoutError: x", "error_type": "TimeoutError"})
    led.append({"subject_id": v_, "status": "failed", "error": "MaterialiseError: x.edf: reader exit 1: EIO"})
    led.append({"subject_id": tr, "status": "failed", "error_type": "PermissionError", "error": "PermissionError: p"})
    led.append({"subject_id": t2, "status": "failed", "error_type": "RateDeviation", "error": "RateDeviation: ECG"})
    subjects = b.order_subjects(splits.load_split(sp), ["val", "test", "train"], 42)
    todo, states, blocked = b.plan(subjects, root, out, led, "download", retry_failed=False, fetch_xml=True,
                                   verify_existing=True)
    ids = {x["subject_id"] for x in todo}
    assert t in ids and v_ in ids and tr not in ids and t2 not in ids
    assert states[("test", "todo_transient_auto_retry")] == 1 and states[("val", "todo_transient_auto_retry")] == 1
    assert blocked == {"test": [(t2, "failed_before_skipped")], "train": [(tr, "failed_before_skipped")]}
    t3 = int(s["test"][2])
    (root / f"shhs1-{t3}.edf").unlink()  # a missing file blocks too
    _, _, blocked = b.plan(subjects, root, out, led, "download", False, True, True)
    assert (t3, "file_missing") in blocked["test"]


def test_run_configs_jsonl_keeps_every_run(tmp_path):
    import build_dl_cache as b

    root, sp, s = _synthetic_root(tmp_path)
    out = tmp_path / "cache"
    base = ["--only-local", "--out", str(out), "--split", str(sp), "--shhs-root", str(root),
            "--min-free-gb", "0", "--workers", "1"]
    assert CliRunner().invoke(b.main, base + ["--limit", "2"]).exit_code == 0
    assert CliRunner().invoke(b.main, base).exit_code == 0
    runs = dl_cache.Ledger(out / "run_configs.jsonl").records()
    assert len(runs) == 2 and runs[0]["n_todo"] == 2 and runs[1]["n_todo"] == 10 and all("ts" in x for x in runs)
    assert json.loads((out / "run_config.json").read_text())["n_todo"] == 10


# --------------------------------------------------------------------------- real cache (optional)

REAL_CACHE = Path.home() / "thesis_dl_cache" / "raw_v1"
METADATA = ROOT / "features" / "subject_metadata.parquet"
KEY = ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only" / "test_predictions.parquet"


@pytest.mark.skipif(not (REAL_CACHE.exists() and METADATA.exists() and KEY.exists()),
                    reason="real stage-A cache not built")
def test_real_cache_label_parity():
    sids = sorted(int(p.stem.split("-")[1]) for p in REAL_CACHE.glob("shhs1-*.json"))
    if not sids:
        pytest.skip("empty cache")
    meta = splits.read_subject_metadata(METADATA)
    key = pd.read_parquet(KEY, columns=["subject_id", "epoch_idx", "apnoea_label"])
    rep = dl_cache.label_parity(REAL_CACHE, sids, meta, key)
    assert rep["n_epochs_ok"].all() and rep["n_sleep_ok"].all() and rep["apnoea_sum_ok"].all()
    t = rep[rep["split"] == "test"]
    if len(t):
        assert t["key_rows_ok"].eq(True).all()
        # test subjects' sleep rows cover exactly their key rows
        assert int(t["n_key_rows"].sum()) == int(key[key.subject_id.isin(t.subject_id)].shape[0])
    # and no test subject is missing from the key / no non-test subject is in it
    assert set(rep.loc[rep["split"] != "test", "subject_id"]).isdisjoint(set(key.subject_id.unique()))
    raw = json.loads((REAL_CACHE / f"shhs1-{sids[0]}.json").read_text())
    assert raw["extractor_version"] == dl_cache.EXTRACTOR_VERSION


def test_verify_cli_on_synthetic_cache(tmp_path):
    import build_dl_cache as b
    import verify_dl_cache as v

    root, sp, s = _synthetic_root(tmp_path)
    out = tmp_path / "cache"
    r = CliRunner().invoke(b.main, ["--only-local", "--out", str(out), "--split", str(sp), "--shhs-root", str(root),
                                    "--min-free-gb", "0", "--workers", "1"])
    assert r.exit_code == 0, r.output
    ids = list(range(900001, 900013))
    meta, key = _meta_and_key(tmp_path, ids)
    mpath, kpath = tmp_path / "meta.parquet", tmp_path / "key.parquet"
    meta.assign(subject_id=[f"shhs1-{i}" for i in meta["sid"]]).drop(columns="sid").reset_index(drop=True) \
        .to_parquet(mpath)
    key[key.subject_id.isin(s["test"])].to_parquet(kpath)
    args = ["--cache", str(out), "--split", str(sp), "--metadata", str(mpath), "--key", str(kpath)]
    r = CliRunner().invoke(v.main, args + ["--report", str(tmp_path / "rep.csv")])
    assert r.exit_code == 0, r.output
    assert "ALL CHECKS PASSED" in r.output and (tmp_path / "rep.csv").exists()
    # a label mismatch in one test subject's key rows must fail the run
    k = pd.read_parquet(kpath)
    k.loc[k.index[0], "apnoea_label"] = 1 - k.loc[k.index[0], "apnoea_label"]
    k.to_parquet(kpath)
    r = CliRunner().invoke(v.main, args)
    assert r.exit_code == 1 and "CHECKS FAILED" in r.output


def test_build_one_retries_eperm_on_read(tmp_path, monkeypatch):
    gt = make_subject(tmp_path, 900006)
    real = dl_cache.extract_subject
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("[Errno 1] Operation not permitted")
        return real(*a, **k)

    monkeypatch.setattr(dl_cache, "extract_subject", flaky)
    monkeypatch.setattr(dl_cache.time, "sleep", lambda s: None)
    rec = dl_cache.build_one({"subject_id": 900006, "edf": str(gt["edf"]), "xml": str(gt["xml"]),
                              "out_dir": str(tmp_path / "c"), "backoff": [0.0]})
    assert rec["status"] == "ok" and len(calls) == 2 and len(rec["retries"]) == 1


def test_load_physical_decodes_every_encoding(tmp_path):
    from thesis_pipeline.edf_native import read_header, to_physical

    gt = make_subject(tmp_path, 900007, with_oxstat=False)
    out = tmp_path / "c"
    dl_cache.write_raw(out, 900007, *dl_cache.extract_subject(gt["edf"], gt["xml"], 900007))
    ph = dl_cache.load_physical(out, 900007)
    hdr = read_header(gt["edf"])
    n = ph["n_sec"]
    assert n == 360 and ph["fs"] == {"ECG": 125, "THOR": 10, "ABDO": 10, "AIRFLOW": 10, "SPO2": 1, "HR": 1,
                                     "POSITION": 1, "OXSTAT": 1}
    for canon, lab in (("ECG", "ECG"), ("THOR", "THOR RES"), ("ABDO", "ABDO RES"), ("AIRFLOW", "NEW AIR"),
                       ("SPO2", "SaO2"), ("HR", "H.R."), ("POSITION", "POSITION")):
        sig = hdr.signals[hdr.indices(lab)[0]]
        raw = next(s.digital for s in gt["signals"] if s.label == lab)
        want = to_physical(raw[: n * ph["fs"][canon]], sig)
        assert ph["signals"][canon].dtype == np.float32
        assert np.allclose(ph["signals"][canon], want, rtol=1e-6, atol=1e-5), canon
    assert ph["signals"]["THOR"][0] == pytest.approx(-(gt["thor"][0] + 128) * 2 / 255 + 1, abs=1e-6)  # sign kept
    assert ph["signals"]["OXSTAT"] is None and not ph["ok"]["OXSTAT"] and ph["ok"]["AIRFLOW"]
    assert ph["units"]["ECG"] == "mV" and ph["sec_event"].shape == (360,)
    assert ph["sleep_epoch"].tolist() == np.isin(ph["epoch_stage"], dl_labels.SLEEP_CODES).tolist()
    # SpO2 uint16 fallback path decodes to the same physical values
    ch = ph["meta"]["channels"]["spo2"]
    a16, m16 = dl_cache.encode_uint16(dl_cache.decode_digital(
        dl_cache.load_raw(out, 900007, keys=("spo2",))[0]["spo2"], ch))
    m16["scaling"] = ch["scaling"]
    assert np.allclose(dl_cache.decode_physical(a16, m16).ravel(), ph["signals"]["SPO2"])
    with pytest.raises(KeyError):
        dl_cache.load_physical(out, 900007, channels=("EEG",))
