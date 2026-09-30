"""Stage-A raw-signal cache for the Aim 2 DL arm (lossless, native rate, one npz per subject).

Layout (scripts/build_dl_cache.py writes to ``~/thesis_dl_cache/raw_v1``, outside iCloud):

    shhs1-<id>.npz    arrays below (np.savez_compressed)
    shhs1-<id>.json   sidecar; written LAST, so its presence marks a complete subject
    ledger.jsonl      one line per attempt (written by scripts/build_dl_cache.py)

Arrays (n = n_epochs = floor(n_records * record_duration / 30); the < 30 s tail is dropped):

    ecg            int8   (n, 3750)  digital, 125 Hz
    thor, abdo     int8   (n, 300)   digital, 10 Hz (physical range inverted: gain < 0)
    airflow        int8   (n, 300)   digital, the picked candidate (edf_native.pick_airflow)
    airflow_rej<k> int8   (n, 300)   every rejected airflow candidate, so a later picker
                                     change needs no re-download
    spo2           uint8  (n, 30)    integer-percent code (see encode_spo2); uint16 fallback
    hr             uint8  (n, 30)    oximeter H.R., rank code into a per-subject LUT; uint16 fallback
    position       uint8  (n, 30)    digital
    oxstat         uint8  (n, 30)    digital (all 0 + encoding "missing" when absent)
    epoch_stage    int8   (n,)       dl_labels.STAGE_CODES, via epochs.build_epoch_frame
    apnoea_label   int8   (n,)       epochs.build_epoch_frame (the Aim 2 label code path)
    sec_event      uint8  (n, 30)    per-second 4-kind bitmask (dl_labels.sec_event_mask)

Only when a channel's native rate deviates (never seen in 1,760 local headers):
    <key>_offrate  int8/16 (n*30*fs,) raw digital samples at the deviating rate, 1-D
    airflow_offrate<k>                likewise for an off-rate airflow candidate
    (the regular array is then zero-filled, encoding "missing"; see extract_subject)

Every channel's sidecar entry carries its header scaling (gain with sign),
its encoding and, for LUT encodings, the code -> digital table, so
``decode_digital`` recovers the EDF digital samples exactly.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import io as _io
import json
import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import dl_labels, edf_native
from .edf_native import EdfHeader, EdfSignal
from .io import read_nsrr_xml

EXTRACTOR_VERSION = "raw_v1.0"
EPOCH_S = 30

# canonical name -> (array key, samples per 30-s epoch)
CHANNEL_LAYOUT: dict[str, tuple[str, int]] = {
    "ECG": ("ecg", 125 * EPOCH_S),
    "THOR": ("thor", 10 * EPOCH_S),
    "ABDO": ("abdo", 10 * EPOCH_S),
    "AIRFLOW": ("airflow", 10 * EPOCH_S),
    "SPO2": ("spo2", EPOCH_S),
    "HR": ("hr", EPOCH_S),
    "POSITION": ("position", EPOCH_S),
    "OXSTAT": ("oxstat", EPOCH_S),
}
# Channels whose rate deviation fails the subject: the M / P4 model inputs (plus any
# record-duration deviation). A rate deviation on any other channel (RATE_DROPPABLE and each
# airflow candidate) drops only that channel, so a test subject is never lost over a channel
# M / P4 do not use (design A7): its array is zero-filled with encoding "missing", its raw
# samples are kept at the native rate as "<key>_offrate", and the ledger gets rate_dropped:<canon>.
RATE_REQUIRED = ("ECG", "THOR", "ABDO")
RATE_DROPPABLE = ("SPO2", "HR", "POSITION", "OXSTAT")
UINT16_OFFSET = 32768


class ExtractionError(RuntimeError):
    pass


# --------------------------------------------------------------------------- hashing


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path | str) -> dict:
    """{'commit': HEAD sha or None, 'dirty': bool or None} for provenance."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, timeout=30
        ).stdout
        return {"commit": head, "dirty": bool(status.strip())}
    except Exception:  # noqa: BLE001 - provenance must never break extraction
        return {"commit": None, "dirty": None}


# --------------------------------------------------------------------------- encoders


def _enc(arr: np.ndarray, encoding: str, **extra) -> tuple[np.ndarray, dict]:
    return arr, {"encoding": encoding, "dtype": str(arr.dtype), **extra}


def encode_int8(d: np.ndarray) -> tuple[np.ndarray, dict]:
    """Digital samples as int8 when they fit (SHHS 8-bit channels), else int16."""
    if d.size == 0 or (d.min() >= -128 and d.max() <= 127):
        return _enc(d.astype(np.int8), "int8_digital")
    return _enc(d.astype(np.int16), "int16_digital")


def encode_uint8(d: np.ndarray) -> tuple[np.ndarray, dict]:
    """Digital samples as uint8 when 0..255 (POSITION, OX stat), else uint16 + 32768."""
    if d.size == 0 or (d.min() >= 0 and d.max() <= 255):
        return _enc(d.astype(np.uint8), "uint8_digital")
    return encode_uint16(d)


def encode_uint16(d: np.ndarray) -> tuple[np.ndarray, dict]:
    """Lossless for any int16 input: stored = digital + 32768."""
    return _enc((d.astype(np.int32) + UINT16_OFFSET).astype(np.uint16), "uint16_offset", offset=UINT16_OFFSET)


def encode_lut(d: np.ndarray, codes: np.ndarray, encoding: str) -> tuple[np.ndarray, dict] | None:
    """uint8 ``codes`` + a code -> digital table, if that table reproduces ``d`` exactly.

    Returns None (caller falls back to uint16) if any code maps to two digital
    values, a code is outside 0..255, or the round trip is not exact.
    """
    codes = np.asarray(codes)
    if codes.size and (codes.min() < 0 or codes.max() > 255):
        return None
    codes8 = codes.astype(np.uint8)
    lut = np.full(256, np.iinfo(np.int32).min, dtype=np.int64)
    lut[codes8] = d  # last write wins; the check below catches collisions
    if not np.array_equal(lut[codes8], d.astype(np.int64)):
        return None
    used = np.unique(codes8)
    table = [[int(c), int(lut[c])] for c in used]
    return _enc(codes8, encoding, lut=table)


def encode_spo2(d: np.ndarray, sig: EdfSignal) -> tuple[np.ndarray, dict]:
    """SaO2 as uint8 integer-percent code with an exact code -> digital LUT, else uint16.

    SHHS-1 SaO2 is stored 16-bit, but on local files the digital values sit
    on a 256-step lattice, j = ceil(p * 2.56) for integer SpO2 p, so the
    header percentage j * 100/256 lies in [p, p + 0.39) and
    ``code = round(header pct)`` recovers p (0 = dropout, digital -32767).
    That is lossless when each code maps to one digital value, which is
    checked here per subject, not assumed; otherwise uint16.
    """
    pct = edf_native.to_physical(d, sig).astype(np.float64)
    codes = np.rint(pct)
    enc = encode_lut(d, codes, "uint8_pct_lut")
    if enc is None:
        arr, meta = encode_uint16(d)
        meta["fallback_reason"] = "code -> digital not one-to-one or out of 0..255"
        return arr, meta
    arr, meta = enc
    meta["max_abs_code_minus_pct"] = float(np.max(np.abs(codes - pct))) if pct.size else 0.0
    return arr, meta


def encode_rank(d: np.ndarray) -> tuple[np.ndarray, dict]:
    """uint8 rank of each sample among the subject's sorted unique values (<= 256), else uint16.

    Used for the oximeter H.R.: its digital lattice (~204.8 LSB per step, i.e.
    0.78 bpm under the header scaling) would collide if rounded to integer
    bpm; the rank LUT is exact by construction and checked anyway. The header
    scaling itself is right: header-scaled H.R. matched 60 / median RR within
    ~1 bpm on 9 ten-minute stretches of 3 subjects (2026-09-30).
    """
    uniq, inv = np.unique(d, return_inverse=True)
    if uniq.size <= 256:
        enc = encode_lut(d, inv.reshape(d.shape), "uint8_rank_lut")
        if enc is not None:
            return enc
    arr, meta = encode_uint16(d)
    meta["fallback_reason"] = f"{uniq.size} unique values > 256"
    return arr, meta


def decode_digital(arr: np.ndarray, ch: dict) -> np.ndarray:
    """Stored array -> EDF digital samples (int32), using the sidecar channel entry."""
    enc = ch["encoding"]
    if enc in ("int8_digital", "int16_digital", "uint8_digital"):
        return arr.astype(np.int32)
    if enc == "uint16_offset":
        return arr.astype(np.int32) - int(ch.get("offset", UINT16_OFFSET))
    if enc in ("uint8_pct_lut", "uint8_rank_lut"):
        lut = np.zeros(256, dtype=np.int32)
        for code, dig in ch["lut"]:
            lut[code] = dig
        return lut[arr.astype(np.int64)]
    if enc == "missing":
        raise ValueError(f"channel {ch.get('label')!r} is missing in this recording")
    raise ValueError(f"unknown encoding {enc!r}")


def decode_physical(arr: np.ndarray, ch: dict) -> np.ndarray:
    """Stored array -> physical header units (float32, sign kept)."""
    s = ch["scaling"]
    return ((decode_digital(arr, ch).astype(np.float64) - s["dmin"]) * s["gain"] + s["pmin"]).astype(np.float32)


# --------------------------------------------------------------------------- extraction


def _segment(d: np.ndarray, n_epochs: int, per_epoch: int) -> np.ndarray:
    need = n_epochs * per_epoch
    if d.size < need:
        raise ExtractionError(f"signal has {d.size} samples, need {need}")
    return d[:need].reshape(n_epochs, per_epoch)


def _chan_meta(hdr: EdfHeader, idx: int, enc_meta: dict, shape) -> dict:
    s = hdr.signals[idx]
    return {"label": s.label, "index": idx, "fs": s.fs, "shape": list(shape), "scaling": s.scaling(), **enc_meta}


def extract_subject(
    edf_path: Path | str,
    xml_path: Path | str,
    subject_id: int,
    split: str | None = None,
    cohort: str = "shhs1",
) -> tuple[dict[str, np.ndarray], dict]:
    """Read one subject's EDF + NSRR XML into stage-A arrays and a sidecar dict.

    Both files must already be local (never triggers a download: the EDF
    reader refuses placeholders and so does this function for the XML).
    Raises ``edf_native.RateDeviation`` (record duration, ECG, THOR or ABDO
    off-rate) / ``ExtractionError`` on anything it will not guess about. Any
    other off-rate channel is dropped, not fatal (see RATE_REQUIRED).
    """
    edf_path, xml_path = Path(edf_path), Path(xml_path)
    if os.stat(xml_path).st_flags & edf_native.UF_DATALESS:
        raise edf_native.DatalessFileError(f"placeholder, refusing to read: {xml_path}")
    hdr = edf_native.read_header(edf_path)

    picks: dict[str, int | None] = {}
    notes: list[str] = list(hdr.notes)
    for canon in ("ECG", "THOR", "ABDO", "SPO2", "HR", "POSITION", "OXSTAT"):
        hits = hdr.find(canon)
        picks[canon] = hits[0] if hits else None
        if len(hits) > 1:
            notes.append(f"{canon}: {len(hits)} signals match, used index {hits[0]}")

    edf_native.assert_rates(hdr, {c: [picks[c]] for c in RATE_REQUIRED if picks[c] is not None})
    rate_dropped: list[dict] = []

    def off_rate(canon: str, idx: int) -> bool:
        dev = edf_native.check_rates(hdr, {canon: [idx]})
        if dev:
            s = hdr.signals[idx]
            notes.append(f"{canon} dropped: {dev[0]}")
            rate_dropped.append({"canon": canon, "index": idx, "label": s.label, "fs": s.fs,
                                 "expected_fs": edf_native.EXPECTED_FS[canon]})
        return bool(dev)

    for canon in RATE_DROPPABLE:
        if picks[canon] is not None and off_rate(canon, picks[canon]):
            picks[canon] = None
    air_idx = [i for i in edf_native.airflow_candidates(hdr) if not off_rate("AIRFLOW", i)]

    n_epochs = int(hdr.n_records * hdr.record_duration // EPOCH_S)
    if n_epochs <= 0:
        raise ExtractionError(f"recording shorter than one epoch ({hdr.duration_s} s)")
    n_sec = n_epochs * EPOCH_S

    want = [i for i in picks.values() if i is not None] + air_idx + [r["index"] for r in rate_dropped]
    dig = edf_native.read_digital(edf_path, want, hdr=hdr)

    arrays: dict[str, np.ndarray] = {}
    channels: dict[str, dict] = {}
    missing: list[str] = []

    def put(canon: str, idx: int | None, encoder: Callable[[np.ndarray], tuple[np.ndarray, dict]],
            key: str | None = None) -> None:
        k, per = CHANNEL_LAYOUT[canon]
        k = key or k
        if idx is None:
            fill_dtype = np.int8 if per > EPOCH_S else np.uint8
            arrays[k] = np.zeros((n_epochs, per), dtype=fill_dtype)
            channels[k] = {"label": None, "index": None, "encoding": "missing", "dtype": str(arrays[k].dtype),
                           "shape": [n_epochs, per]}
            missing.append(canon)
            return
        seg = _segment(dig[idx], n_epochs, per)
        arr, meta = encoder(seg)
        arrays[k] = arr
        channels[k] = _chan_meta(hdr, idx, meta, arr.shape)

    put("ECG", picks["ECG"], encode_int8)
    put("THOR", picks["THOR"], encode_int8)
    put("ABDO", picks["ABDO"], encode_int8)
    pick = edf_native.pick_airflow(hdr, dig, n_samples=n_sec * 10, candidates=air_idx)
    put("AIRFLOW", pick.index, encode_int8)
    for j, rej in enumerate(pick.rejected):
        put("AIRFLOW", rej.index, encode_int8, key=f"airflow_rej{j}")
    if picks["SPO2"] is not None:
        sig = hdr.signals[picks["SPO2"]]
        put("SPO2", picks["SPO2"], lambda x: encode_spo2(x, sig))
    else:
        put("SPO2", None, encode_uint8)
    put("HR", picks["HR"], encode_rank)
    put("POSITION", picks["POSITION"], encode_uint8)
    put("OXSTAT", picks["OXSTAT"], encode_uint8)
    # off-rate channels: keep the raw digital samples at their native rate (record duration
    # is asserted 1 s, so samples_per_record == fs), so no re-download is ever needed.
    n_air_off = 0
    for r in rate_dropped:
        if r["canon"] == "AIRFLOW":
            k, n_air_off = f"airflow_offrate{n_air_off}", n_air_off + 1
        else:
            k = f"{CHANNEL_LAYOUT[r['canon']][0]}_offrate"
        arr, m = encode_int8(dig[r["index"]][: n_sec * hdr.signals[r["index"]].samples_per_record])
        arrays[k] = arr
        channels[k] = _chan_meta(hdr, r["index"], {**m, "rate_dropped": True}, arr.shape)
        r["array"] = k

    # ---- labels: the Aim 2 code path only
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hypno, events = read_nsrr_xml(xml_path)
    xml_warnings = [str(w.message) for w in caught]
    stage, apnoea, _frame = dl_labels.epoch_labels(subject_id, n_epochs, hypno, events, cohort=cohort)
    arrays["epoch_stage"] = stage
    arrays["apnoea_label"] = apnoea
    arrays["sec_event"] = dl_labels.sec_event_mask(events, n_sec).reshape(n_epochs, EPOCH_S)

    sleep = dl_labels.sleep_epochs(stage)
    qc = _qc(arrays, channels, hdr, picks)
    qc.update(missing_channels=missing, notes=notes, xml_warnings=xml_warnings)
    if rate_dropped:  # only when present, so sidecars of on-rate subjects are unchanged
        qc["rate_dropped"] = rate_dropped

    meta = {
        "subject_id": int(subject_id),
        "cohort": cohort,
        "split": split,
        "extractor_version": EXTRACTOR_VERSION,
        "edf": {
            "file": edf_path.name,
            "bytes": hdr.file_size,
            "n_signals": hdr.n_signals,
            "labels": hdr.labels,
            "n_records": hdr.n_records,
            "n_records_header": hdr.n_records_header,
            "record_duration": hdr.record_duration,
            "start_time": hdr.start_time,
        },
        "xml": {"file": xml_path.name, "bytes": os.stat(xml_path).st_size},
        "n_epochs": n_epochs,
        "n_seconds": n_sec,
        "tail_seconds_dropped": float(hdr.duration_s - n_sec),
        "channels": channels,
        "airflow": {
            "rule": "highest whole-night std (LSB) among labels "
            f"{sorted(edf_native.AIRFLOW_CANDIDATE_LABELS)}; ok = std > {edf_native.AIRFLOW_LIVE_STD_LSB:g} LSB",
            "chosen_index": pick.index,
            "chosen_label": pick.label,
            "std_lsb": pick.std_lsb,
            "ok": pick.ok,
            "candidates": [c.as_dict() for c in pick.candidates],
            "rejected_arrays": {f"airflow_rej{j}": r.index for j, r in enumerate(pick.rejected)},
        },
        "labels": {
            "stage_codes": dl_labels.STAGE_CODES,
            "event_bits": dl_labels.EVENT_BITS,
            "sec_event_rule": "bit set iff s + 0.5 in [start, start + duration)",
            "epoch_label_rule": "epochs.build_epoch_frame (>= 10 s overlap with one event)",
            "n_sleep_epochs": int(sleep.sum()),
            "n_apnoea_epochs_sleep": int(apnoea[sleep].sum()),
            "n_apnoea_epochs_all": int(apnoea.sum()),
            "n_hypnogram_epochs": int(len(hypno.stages)),
            "n_events": len(events),
            "events": dl_labels.events_table(events),
        },
        "qc": qc,
    }
    air_off = [r for r in rate_dropped if r["canon"] == "AIRFLOW"]
    if air_off:
        meta["airflow"]["rate_dropped_candidates"] = air_off  # excluded from the pick
    return arrays, meta


def _qc(arrays: dict, channels: dict, hdr: EdfHeader, picks: dict) -> dict:
    out: dict = {}
    for k in ("ecg", "thor", "abdo", "airflow"):
        ch = channels[k]
        if ch["encoding"] == "missing":
            out[f"{k}_std_lsb"] = None
            continue
        a = arrays[k]
        out[f"{k}_std_lsb"] = float(a.astype(np.float64).std())
        if k == "ecg":
            s = ch["scaling"]
            out["ecg_clip_frac"] = float(np.mean((a == s["dmin"]) | (a == s["dmax"])))
            out["ecg_low_amplitude"] = bool(out["ecg_std_lsb"] < 5.0)
    if channels["spo2"]["encoding"] != "missing":
        pct = decode_physical(arrays["spo2"], channels["spo2"])
        out["spo2_lt50_frac"] = float(np.mean(pct < 50))
        out["spo2_zero_frac"] = float(np.mean(pct < 0.5))
    out["hr_n_unique"] = len(channels["hr"].get("lut", [])) or None
    return out


# --------------------------------------------------------------------------- canonical decoder

PHYSICAL_KEYS = {"ECG": "ecg", "THOR": "thor", "ABDO": "abdo", "AIRFLOW": "airflow", "SPO2": "spo2",
                 "HR": "hr", "POSITION": "position", "OXSTAT": "oxstat"}


def load_physical(out_dir: Path | str, subject_id: int,
                  channels: Iterable[str] = tuple(PHYSICAL_KEYS), cohort: str = "shhs1") -> dict:
    """The canonical way to read stage A: native-rate, 1-D, physical-unit float32 signals.

    Decodes every encoding (int8 / int16 digital, uint16 offset, SpO2 pct LUT,
    H.R. rank LUT) through the sidecar, applies the header gain WITH its sign
    (THOR / ABDO / label-AIRFLOW have inverted ranges), and returns::

        {"subject_id", "n_epochs", "n_sec",
         "signals": {CANON: float32 (n_sec * fs,) or None if missing},
         "fs":      {CANON: fs},
         "units":   {CANON: header unit string},
         "ok":      {CANON: bool}   # False if missing; AIRFLOW: the picker's airflow_ok
         "epoch_stage", "apnoea_label" (n_epochs,), "sec_event" (n_sec,) uint8 bitmask,
         "sleep_epoch" bool (n_epochs,), "airflow": sidecar airflow block, "meta": sidecar}

    Units are the EDF header's: ECG mV; SpO2 %; H.R. bpm; THOR / ABDO a.u.;
    AIRFLOW a.u. for label ``AIRFLOW`` but uV for ``NEW AIR`` etc. (so only
    scale-free use across subjects; polarity across labels is unverified).
    SpO2 = header percentage (integer SpO2 p maps to [p, p + 0.39)); the
    stored uint8 code is round(pct) = p.
    """
    want = [c.upper() for c in channels]
    unknown = [c for c in want if c not in PHYSICAL_KEYS]
    if unknown:
        raise KeyError(f"unknown channel(s) {unknown}; choose from {sorted(PHYSICAL_KEYS)}")
    keys = [PHYSICAL_KEYS[c] for c in want] + ["epoch_stage", "apnoea_label", "sec_event"]
    arrays, meta = load_raw(out_dir, subject_id, keys=keys, cohort=cohort)
    n_epochs = int(meta["n_epochs"])
    out: dict = {"subject_id": int(subject_id), "n_epochs": n_epochs, "n_sec": n_epochs * EPOCH_S,
                 "signals": {}, "fs": {}, "units": {}, "ok": {}}
    for c in want:
        k = PHYSICAL_KEYS[c]
        ch = meta["channels"][k]
        per = CHANNEL_LAYOUT[c][1]
        out["fs"][c] = per // EPOCH_S
        if ch["encoding"] == "missing":
            out["signals"][c], out["units"][c], out["ok"][c] = None, None, False
            continue
        out["signals"][c] = decode_physical(arrays[k], ch).reshape(-1)
        out["units"][c] = ch["scaling"]["unit"]
        out["ok"][c] = bool(meta["airflow"]["ok"]) if c == "AIRFLOW" else True
    out["epoch_stage"] = arrays["epoch_stage"]
    out["apnoea_label"] = arrays["apnoea_label"]
    out["sec_event"] = arrays["sec_event"].reshape(-1)
    out["sleep_epoch"] = dl_labels.sleep_epochs(arrays["epoch_stage"])
    out["airflow"] = meta["airflow"]
    out["meta"] = meta
    return out


# --------------------------------------------------------------------------- write / load


# Cloud-synced (and evictable) folders under the home directory. Stage A holds test-subject
# signals, which must never reach Drive / Colab, and iCloud may evict anything under
# Desktop & Documents. Symlinks (e.g. '~/OneDrive - X' -> Library/CloudStorage) are resolved.
CLOUD_SYNCED_HOME_DIRS = ("Desktop", "Documents", "Library/Mobile Documents", "Library/CloudStorage")
CLOUD_SYNCED_HOME_GLOBS = ("OneDrive*", "Dropbox*", "Google Drive*", "Creative Cloud Files*")


def cloud_synced_root(path: Path | str, home: Path | str | None = None) -> Path | None:
    """The cloud-synced folder that ``path`` lies in (after resolving symlinks), else None.

    Uses ``Path.is_relative_to`` (not a string prefix, so ``~/Desktop2`` is not
    mistaken for ``~/Desktop``).
    """
    home = Path(home) if home is not None else Path.home()
    roots = [home / d for d in CLOUD_SYNCED_HOME_DIRS]
    for pattern in CLOUD_SYNCED_HOME_GLOBS:
        roots += sorted(home.glob(pattern))
    p = Path(path).expanduser().resolve()
    for r in roots:
        for cand in dict.fromkeys((r.absolute(), r.resolve())):
            if p.is_relative_to(cand):
                return r
    return None


def paths(out_dir: Path | str, subject_id: int, cohort: str = "shhs1") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    return out_dir / f"{cohort}-{subject_id}.npz", out_dir / f"{cohort}-{subject_id}.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_raw(out_dir: Path | str, subject_id: int, arrays: dict[str, np.ndarray], meta: dict,
              provenance: dict | None = None) -> dict:
    """Atomically write ``shhs1-<id>.npz`` then ``shhs1-<id>.json``. Returns the final sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = paths(out_dir, subject_id, meta.get("cohort", "shhs1"))
    buf = _io.BytesIO()
    np.savez_compressed(buf, **arrays)
    blob = buf.getvalue()
    _atomic_write_bytes(npz_path, blob)
    meta = dict(meta)
    meta["arrays"] = {
        k: {"dtype": str(a.dtype), "shape": list(a.shape), "sha256": sha256_array(a)} for k, a in arrays.items()
    }
    meta["npz"] = {"file": npz_path.name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    meta["created_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    if provenance:
        meta["provenance"] = provenance
    _atomic_write_bytes(json_path, (json.dumps(meta, separators=(",", ":"), default=_json_default) + "\n").encode())
    return meta


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def load_meta(out_dir: Path | str, subject_id: int, cohort: str = "shhs1") -> dict:
    return json.loads(paths(out_dir, subject_id, cohort)[1].read_text())


def load_raw(out_dir: Path | str, subject_id: int, keys: Iterable[str] | None = None,
             verify: bool = False, cohort: str = "shhs1") -> tuple[dict[str, np.ndarray], dict]:
    """Load arrays (all, or ``keys``) and the sidecar. ``verify`` checks file and array sha256s."""
    npz_path, _ = paths(out_dir, subject_id, cohort)
    meta = load_meta(out_dir, subject_id, cohort)
    if verify and sha256_file(npz_path) != meta["npz"]["sha256"]:
        raise ValueError(f"{npz_path}: file sha256 mismatch")
    with np.load(npz_path, allow_pickle=False) as z:
        names = list(z.files) if keys is None else list(keys)
        arrays = {k: z[k] for k in names}
    if verify:
        for k, a in arrays.items():
            want = meta["arrays"][k]
            if sha256_array(a) != want["sha256"] or str(a.dtype) != want["dtype"] or list(a.shape) != want["shape"]:
                raise ValueError(f"{npz_path}:{k} content mismatch")
    return arrays, meta


def is_complete(out_dir: Path | str, subject_id: int, verify: bool = True, cohort: str = "shhs1") -> bool:
    """True if sidecar + npz exist and the npz matches the sidecar's size (and sha256 if ``verify``)."""
    npz_path, json_path = paths(out_dir, subject_id, cohort)
    if not (json_path.exists() and npz_path.exists()):
        return False
    try:
        meta = json.loads(json_path.read_text())
        if npz_path.stat().st_size != meta["npz"]["bytes"]:
            return False
        if meta.get("extractor_version") != EXTRACTOR_VERSION:
            return False
        return (not verify) or sha256_file(npz_path) == meta["npz"]["sha256"]
    except (OSError, ValueError, KeyError):
        return False


# --------------------------------------------------------------------------- ledger


class Ledger:
    """Append-only JSONL of per-subject attempts. Single writer (the driver process)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def append(self, rec: dict) -> None:
        """Append one record. If a killed run left a torn last line (no trailing newline),
        a newline is written first, so the fragment stays on its own (skipped) line
        instead of swallowing this record."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"), **rec}
        line = (json.dumps(rec, default=_json_default) + "\n").encode()
        with open(self.path, "a+b") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() > 0:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":
                    line = b"\n" + line
            fh.write(line)  # "a" mode: always lands at the end, whatever the read position
            fh.flush()
            os.fsync(fh.fileno())

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn last line from a killed run
        return out

    def latest(self) -> dict[int, dict]:
        last: dict[int, dict] = {}
        for r in self.records():
            if "subject_id" in r:
                last[int(r["subject_id"])] = r
        return last


def ledger_flag_drift(out_dir: Path | str, ledger: Ledger | None = None, cohort: str = "shhs1") -> list[dict]:
    """Latest 'ok' ledger records whose ``flags`` differ from ``_flags`` recomputed from the sidecar.

    The sidecar is the source of truth; drift means the record was written by an
    older ``_flags`` (e.g. the 2026-09-30 smoke run's 'xml_warnings' flag).
    """
    out_dir = Path(out_dir)
    ledger = ledger or Ledger(out_dir / "ledger.jsonl")
    drift = []
    for sid, rec in sorted(ledger.latest().items()):
        if rec.get("status") != "ok" or not paths(out_dir, sid, cohort)[1].exists():
            continue
        want = _flags(load_meta(out_dir, sid, cohort))
        if rec.get("flags") != want:
            drift.append({"subject_id": sid, "ledger_flags": rec.get("flags"), "sidecar_flags": want, "record": rec})
    return drift


def reflag_ledger(out_dir: Path | str, ledger: Ledger | None = None, cohort: str = "shhs1") -> list[dict]:
    """Append a corrected copy of every drifted record (flags recomputed; ``reflag`` says from what).

    Append-only: the stale record stays in the file; ``latest()`` now returns the
    corrected one. Returns the drift list that was fixed.
    """
    out_dir = Path(out_dir)
    ledger = ledger or Ledger(out_dir / "ledger.jsonl")
    drift = ledger_flag_drift(out_dir, ledger, cohort)
    for d in drift:
        rec = {k: v for k, v in d["record"].items() if k not in ("ts", "reflag")}
        rec["flags"] = d["sidecar_flags"]
        rec["reflag"] = {"stale_flags": d["ledger_flags"], "stale_ts": d["record"].get("ts"),
                         "note": "not an extraction attempt: flags recomputed from the sidecar by the current _flags"}
        ledger.append(rec)
    return drift


# --------------------------------------------------------------------------- download helpers


class StillDataless(OSError):
    """The file is still a placeholder after a full read."""


class MaterialiseError(OSError):
    """The reader child exited non-zero for a reason not classified in READER_ERRORS.

    Retried like the classified errors: the child's errno arrives only as stderr
    text, and FileProvider errors other than EPERM (EIO, EDEADLK, ...) may be
    transient. A permanent one costs at most the backoff (15.5 min by default).
    """


def is_dataless(path: Path | str) -> bool:
    return bool(os.stat(path).st_flags & edf_native.UF_DATALESS)


# Reader stderr (lower-cased substring) -> the exception Python itself maps that errno to
# (EPERM / EACCES, ETIMEDOUT = errno 60 'Operation timed out', EAGAIN, EINTR).
READER_ERRORS: tuple[tuple[str, type[OSError]], ...] = (
    ("not permitted", PermissionError),
    ("eperm", PermissionError),
    ("permission denied", PermissionError),
    ("timed out", TimeoutError),
    ("resource temporarily unavailable", BlockingIOError),
    ("interrupted system call", InterruptedError),
)


def materialise(path: Path | str, timeout_s: float, reader: tuple[str, ...] = ("/bin/cat",)) -> float:
    """Force a FileProvider placeholder to download by reading it fully in a killable child.

    Returns seconds taken. Raises TimeoutError (the child ran past ``timeout_s``
    or reported ETIMEDOUT), PermissionError (EPERM, the OneDrive throttling
    failure, or EACCES), BlockingIOError, InterruptedError, StillDataless, or
    MaterialiseError (any other non-zero exit). All of these are in RETRYABLE.
    """
    path = Path(path)
    t0 = time.time()
    try:
        proc = subprocess.run([*reader, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"reading {path.name} exceeded {timeout_s:.0f}s") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip()
        low = err.lower()
        for needle, cls in READER_ERRORS:
            if needle in low:
                raise cls(f"{path.name}: {err}")
        raise MaterialiseError(f"{path.name}: reader exit {proc.returncode}: {err}")
    if is_dataless(path):
        raise StillDataless(f"{path.name} still dataless after a full read")
    return time.time() - t0


RETRYABLE = (TimeoutError, PermissionError, StillDataless, MaterialiseError, InterruptedError, BlockingIOError)
# The in-process read of an already-materialised file (extract_subject): the errno classes
# Python raises for EPERM / EACCES, ETIMEDOUT, EINTR and EAGAIN.
RETRYABLE_READ = (PermissionError, TimeoutError, InterruptedError, BlockingIOError)
# Failures a later run retries on its own for val / test subjects (no --retry-failed needed):
# every retryable class plus a placeholder met mid-run (evicted between plan and read).
TRANSIENT_ERROR_TYPES = frozenset(c.__name__ for c in RETRYABLE) | {"DatalessFileError"}


def is_transient_failure(rec: dict) -> bool:
    """True if a ledger record is a failure of a transient kind (see TRANSIENT_ERROR_TYPES)."""
    if rec.get("status") != "failed":
        return False
    name = rec.get("error_type") or str(rec.get("error", "")).split(":", 1)[0].strip()
    return name in TRANSIENT_ERROR_TYPES


def with_backoff(fn: Callable[[], object], delays: Iterable[float] = (30, 60, 120, 240, 480),
                 retry_on: tuple[type[BaseException], ...] = RETRYABLE,
                 sleep: Callable[[float], None] = time.sleep, log: list | None = None):
    """Call ``fn``; on a retryable error sleep through ``delays`` (exponential) and retry.

    With the default delays that is 1 try + 5 retries, 30 s -> 8 min. The
    last error is re-raised. ``log`` (if given) collects ``(attempt, error)``.
    """
    delays = list(delays)
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except retry_on as exc:
            if log is not None:
                log.append((attempt, f"{type(exc).__name__}: {exc}"))
            if attempt == len(delays):
                raise
            sleep(delays[attempt])
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- one unit of work


def build_one(task: dict) -> dict:
    """Extract + write one subject; never raises. Returns a ledger record.

    ``task`` keys: subject_id, split, edf, xml, out_dir, provenance,
    fetch_xml (bool), fetch_edf (bool), timeout_xml_s, timeout_edf_s,
    backoff (list of delays).
    """
    sid = int(task["subject_id"])
    rec: dict = {"subject_id": sid, "split": task.get("split"), "mode": task.get("mode")}
    t0 = time.time()
    retries: list = []
    try:
        edf, xml = Path(task["edf"]), Path(task["xml"])
        delays = task.get("backoff", (30, 60, 120, 240, 480))
        if is_dataless(xml):
            if not task.get("fetch_xml"):
                raise edf_native.DatalessFileError(f"{xml.name} is a placeholder and fetch_xml is off")
            rec["xml_fetch_s"] = with_backoff(lambda: materialise(xml, task.get("timeout_xml_s", 120)),
                                              delays, log=retries)
        if is_dataless(edf):
            if not task.get("fetch_edf"):
                raise edf_native.DatalessFileError(f"{edf.name} is a placeholder and fetch_edf is off")
            rec["edf_fetch_s"] = with_backoff(lambda: materialise(edf, task.get("timeout_edf_s", 900)),
                                              delays, log=retries)
            rec["edf_bytes_fetched"] = os.stat(edf).st_size
        t1 = time.time()
        # OneDrive can also refuse the read itself with EPERM (2026-04-29 run) or time out
        # (errno 60, scripts/scale_process_parallel.py:146): retry those too.
        arrays, meta = with_backoff(lambda: extract_subject(edf, xml, sid, split=task.get("split")),
                                    delays, retry_on=RETRYABLE_READ, log=retries)
        t2 = time.time()
        meta = write_raw(task["out_dir"], sid, arrays, meta, provenance=task.get("provenance"))
        rec.update(
            status="ok",
            extract_s=round(t2 - t1, 3),
            write_s=round(time.time() - t2, 3),
            npz_bytes=meta["npz"]["bytes"],
            n_epochs=meta["n_epochs"],
            airflow=meta["airflow"]["chosen_label"],
            airflow_ok=meta["airflow"]["ok"],
            airflow_std_lsb=round(meta["airflow"]["std_lsb"], 3),
            n_airflow_candidates=len(meta["airflow"]["candidates"]),
            spo2_encoding=meta["channels"]["spo2"]["encoding"],
            hr_encoding=meta["channels"]["hr"]["encoding"],
            flags=_flags(meta),
        )
    except Exception as exc:  # noqa: BLE001 - one bad subject must not stop the run
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__)
    rec["seconds"] = round(time.time() - t0, 3)
    if retries:
        rec["retries"] = retries
    return rec


def _flags(meta: dict) -> list[str]:
    f = []
    q = meta["qc"]
    if not meta["airflow"]["ok"]:
        f.append("airflow_dead" if meta["airflow"]["chosen_index"] is not None else "airflow_missing")
    if q.get("missing_channels"):
        f.append("missing:" + ",".join(q["missing_channels"]))
    if q.get("rate_dropped"):
        f.append("rate_dropped:" + ",".join(dict.fromkeys(r["canon"] for r in q["rate_dropped"])))
    if q.get("ecg_low_amplitude"):
        f.append("ecg_low_amplitude")
    if (q.get("ecg_clip_frac") or 0) > 0.01:
        f.append("ecg_clipped")
    for k in ("spo2", "hr"):
        if meta["channels"][k]["encoding"] == "uint16_offset":
            f.append(f"{k}_uint16")
    if meta["qc"].get("notes"):
        f.append("notes")
    # read_nsrr_xml warns about every ignored respiratory-typed concept; SpO2
    # artifact / desaturation are in nearly every file, so only flag the rest.
    routine = ("'SpO2 artifact'", "'SpO2 desaturation'")
    for w in meta["qc"].get("xml_warnings", []):
        tail = w.split("concepts:", 1)[-1]
        if any(c.strip().split("×")[0] not in routine for c in tail.split(",")):
            f.append("xml_noncanonical_resp")
            break
    return f


# --------------------------------------------------------------------------- label parity


def label_parity(
    out_dir: Path | str,
    subject_ids: Iterable[int],
    metadata: pd.DataFrame,
    key: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-subject label checks of the cache against the Aim 2 artefacts (no scoring).

    * ``n_epochs_ok``: cache n_epochs == subject_metadata.n_epochs
    * ``n_sleep_ok``: cache sleep-epoch count == subject_metadata.tst_min * 2
    * ``apnoea_sum_ok``: sleep-epoch apnoea label sum == subject_metadata.n_apnoea_epochs
    * ``key_rows_ok`` (subjects present in ``key``): the cache's sleep rows
      ``(epoch_idx, apnoea_label)`` equal the key file's rows exactly, in order.

    ``metadata`` must be indexed by integer subject id (splits.read_subject_metadata).
    """
    key_groups = None
    if key is not None:
        k = key[["subject_id", "epoch_idx", "apnoea_label"]]
        key_groups = {int(s): g for s, g in k.groupby("subject_id", sort=False)}
    rows = []
    for sid in subject_ids:
        sid = int(sid)
        arrays, meta = load_raw(out_dir, sid, keys=("epoch_stage", "apnoea_label"))
        st, ap = arrays["epoch_stage"], arrays["apnoea_label"]
        sleep = dl_labels.sleep_epochs(st)
        m = metadata.loc[sid]
        r = {
            "subject_id": sid,
            "split": meta.get("split"),
            "n_epochs": int(st.size),
            "n_epochs_ok": int(st.size) == int(m["n_epochs"]),
            "n_sleep_ok": int(sleep.sum()) == int(round(m["tst_min"] * 2)),
            "apnoea_sum_ok": int(ap[sleep].sum()) == int(round(m["n_apnoea_epochs"])),
            "key_rows_ok": np.nan,
        }
        if key_groups is not None and sid in key_groups:
            g = key_groups[sid]
            idx = np.flatnonzero(sleep).astype(np.int32)
            r["key_rows_ok"] = bool(
                np.array_equal(g["epoch_idx"].to_numpy(), idx)
                and np.array_equal(g["apnoea_label"].to_numpy().astype(np.int8), ap[idx])
            )
            r["n_key_rows"] = int(len(g))
        rows.append(r)
    return pd.DataFrame(rows)
