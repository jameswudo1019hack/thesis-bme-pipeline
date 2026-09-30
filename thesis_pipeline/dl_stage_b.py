"""Stage B of the Aim 2 DL cache: stage-A native signals -> packed 4 Hz model arrays.

Stage A (built by ``dl_cache.py`` / ``build_dl_cache.py``, design section 2.2) is one
``shhs1-<id>.npz`` plus ``shhs1-<id>.json`` per subject with lossless native-rate
channels and labels. Files written by ``dl_cache.extract_subject`` (sidecar with per-channel
``encoding``) are decoded with the canonical ``dl_cache.load_physical`` (LUT / offset
encodings, header gain with sign, picker ``airflow_ok``). For any other stage-A file
``load_stage_a`` is a tolerant adapter: it accepts per-epoch 2-D
arrays ((n_epochs, 3750) for ECG, (n_epochs, 300) for the 10 Hz channels, ...) or flat
1-D arrays, lower- or upper-case keys and the aliases listed in ``_ALIASES``. It also
reads the scouts' prototype npz format.

Stage B layout (design section 2.2, ``pack_split``), one folder per split::

    <out>/<split>/
        signals_<split>_<CH>_<k>.npy   fp16 (n_samples_k,) per channel CH and chunk k
        sec_target_<split>_<k>.npy     uint8 (n_seconds_k,) per-second SDB target
        sec_sleep_<split>_<k>.npy      uint8 (n_seconds_k,) 1 = sleep second (N1-REM)
        index_<split>.parquet          one row per subject: chunk, off1 (second offset of
                                       epoch 0 in the chunk), n_epochs, n_epochs_pad,
                                       n_windows, ok_<CH>, QC columns
        epochs_<split>.parquet         subject_id, epoch_idx, stage, sleep, apnoea_label
        qc_<split>.jsonl               per-subject QC
        MANIFEST.json                  sha256 of every file, channel order, fs, pad, split sha256

Each channel is stored in its own file so the M config ([RR, EDR]) can be copied to
Colab without the belts. Each subject is laid out as
``[60 s zeros | night | zeros to a multiple of 6 epochs + 60 s]`` so no 5-min window
crosses a subject boundary, and chunks (<= 1.5 GB per file) hold whole subjects.

Stored values (SoftMinMax is affine-invariant per window, so these per-subject
shifts and scales do not change the model input; they only protect fp16 precision):
    RR   = 1000 * (rr_s - median_rr_s)          [ms, centred per subject]
    EDR  = edr / median(edr) - 1                 [relative]
    THOR, ABDO, AIRFLOW = physical-sign digital units (gain sign applied)
    SPO2 = (pct - 95) / 5                        [fixed affine, bypasses SoftMinMax]
Missing / flat channels are zero with ok_<CH> = False.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .dl_signals import FRONTEND_VERSION, ecg_to_rr_edr, hr_agreement, resp_to_4hz, spo2_to_4hz

CHANNELS: tuple[str, ...] = ("RR", "EDR", "THOR", "ABDO", "AIRFLOW", "SPO2")
FS = 4
PAD_S = 60
EPOCH_S = 30
WIN_EPOCHS = 6
WIN_S = 300
CENTRE_S = 180
MAX_CHUNK_BYTES = int(1.5e9)
STAGE_B_VERSION = "stage-b-1.0"

STAGE_CODES_DEFAULT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 5, "?": 9}
SLEEP_STAGES = ("N1", "N2", "N3", "REM")

_ALIASES: dict[str, tuple[str, ...]] = {
    "ECG": ("ecg", "ekg"),
    "THOR": ("thor", "thor_res", "thor res"),
    "ABDO": ("abdo", "abdo_res", "abdo res"),
    "AIRFLOW": ("airflow", "air", "flow"),
    "SPO2": ("spo2", "sao2"),
    "HR": ("hr", "h.r.", "pulse", "ox_hr", "oxhr"),
    "POSITION": ("position",),
    "OXSTAT": ("oxstat", "ox_stat", "ox stat"),
    "STAGE": ("epoch_stage", "sleep_stage", "stage"),
    "APNOEA": ("apnoea_label", "apnea_label"),
    "SEC_EVENT": ("sec_event", "event_sec"),
    "AIRFLOW_OK": ("airflow_ok",),
}
# Physical sign when the sidecar carries no header scaling (SHHS-1 header facts,
# design section 2.1: THOR/ABDO physical range is inverted [1, -1]). AIRFLOW has no
# default because its convention differs by label (critique #6).
_DEFAULT_SIGN = {"ECG": 1.0, "THOR": -1.0, "ABDO": -1.0}
_NATIVE_FS = {"ECG": 125, "THOR": 10, "ABDO": 10, "AIRFLOW": 10, "SPO2": 1, "HR": 1}


class StageAError(ValueError):
    """Stage-A file cannot be interpreted safely."""


# ---------------------------------------------------------------------------
# stage-A adapter
# ---------------------------------------------------------------------------


@dataclass
class StageA:
    subject_id: int
    n_epochs: int
    epoch_stage: np.ndarray  # int codes, (n_epochs,)
    sleep_epoch: np.ndarray  # bool, (n_epochs,)
    apnoea_label: np.ndarray  # int8, (n_epochs,)
    sec_event: np.ndarray | None  # uint8 (n_sec,), > 0 = inside a scored event
    signals: dict[str, np.ndarray] = field(default_factory=dict)  # float, physical sign
    fs: dict[str, int] = field(default_factory=dict)
    airflow_ok: bool = False
    meta: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def n_sec(self) -> int:
        return self.n_epochs * EPOCH_S


def _find_key(keys: Iterable[str], canon: str) -> str | None:
    low = {k.lower(): k for k in keys}
    for a in (canon.lower(),) + _ALIASES.get(canon, ()):
        if a in low:
            return low[a]
    return None


def _walk(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _scale_for(meta: dict, canon: str) -> tuple[float, float] | None:
    """(gain, offset) for digital -> physical from any sidecar dict describing ``canon``."""
    names = {canon.lower(), *(_ALIASES.get(canon, ()))}
    for d in _walk(meta):
        for k, v in d.items():
            if not isinstance(v, dict) or str(k).lower() not in names:
                continue
            low = {str(a).lower(): b for a, b in v.items()}
            if "gain" in low and isinstance(low["gain"], (int, float)):
                return float(low["gain"]), float(low.get("offset", 0.0) or 0.0)
            pmin = next((low[a] for a in ("physical_min", "phys_min", "pmin") if a in low), None)
            pmax = next((low[a] for a in ("physical_max", "phys_max", "pmax") if a in low), None)
            dmin = next((low[a] for a in ("digital_min", "dig_min", "dmin") if a in low), None)
            dmax = next((low[a] for a in ("digital_max", "dig_max", "dmax") if a in low), None)
            if None not in (pmin, pmax, dmin, dmax) and float(dmax) != float(dmin):
                gain = (float(pmax) - float(pmin)) / (float(dmax) - float(dmin))
                return gain, float(pmin) - gain * float(dmin)
    return None


def _meta_value(meta: dict, *names: str, default=None):
    for d in _walk(meta):
        for n in names:
            if n in d and not isinstance(d[n], (dict, list)):
                return d[n]
    return default


def load_stage_a(
    npz_path: str | Path, json_path: str | Path | None = None, labels_only: bool = False
) -> StageA:
    """Read one stage-A subject into canonical float arrays (see module docstring).

    ``labels_only`` skips the signal arrays (fast; used for the epochs table)."""
    npz_path = Path(npz_path)
    json_path = Path(json_path) if json_path else npz_path.with_suffix(".json")
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    if _is_track_a(meta):
        return _load_track_a(npz_path, meta, labels_only)
    flags: list[str] = []
    with np.load(npz_path, allow_pickle=False) as z:
        keys = list(z.files)

        def get(canon: str) -> np.ndarray | None:
            k = _find_key(keys, canon)
            return None if k is None else np.asarray(z[k])

        stage = get("STAGE")
        if stage is None:
            raise StageAError(f"{npz_path.name}: no epoch stage array")
        stage = stage.reshape(-1)
        n_epochs = int(meta.get("n_epochs", len(stage)))
        if len(stage) != n_epochs:
            raise StageAError(f"{npz_path.name}: stage length {len(stage)} != n_epochs {n_epochs}")
        codes = meta.get("stage_codes") or STAGE_CODES_DEFAULT
        if isinstance(codes, dict) and all(isinstance(v, str) for v in codes.values()):
            codes = {v: int(k) for k, v in codes.items()}  # accept {code: name}
        known = set(int(v) for v in codes.values())
        seen = set(np.unique(stage).astype(int).tolist())
        if not seen <= known:
            raise StageAError(
                f"{npz_path.name}: stage codes {sorted(seen - known)} not in mapping {codes}; "
                "add 'stage_codes' to the sidecar"
            )
        sleep_codes = [int(codes[s]) for s in SLEEP_STAGES if s in codes]
        sleep_epoch = np.isin(stage.astype(int), sleep_codes)

        apn = get("APNOEA")
        if apn is None:
            raise StageAError(f"{npz_path.name}: no apnoea_label array")
        apn = apn.reshape(-1).astype(np.int8)
        if len(apn) != n_epochs:
            raise StageAError(f"{npz_path.name}: apnoea_label length mismatch")
        n_sec = n_epochs * EPOCH_S

        sec_event = get("SEC_EVENT")
        if sec_event is not None:
            sec_event = sec_event.reshape(-1)[:n_sec].astype(np.uint8)
            if len(sec_event) != n_sec:
                raise StageAError(f"{npz_path.name}: sec_event length {len(sec_event)} != {n_sec}")
        elif meta.get("events") is not None:
            sec_event = sec_event_from_events(meta["events"], n_sec)
            flags.append("sec_event_from_sidecar_events")

        sig: dict[str, np.ndarray] = {}
        fs: dict[str, int] = {}
        for canon in ("ECG", "THOR", "ABDO", "AIRFLOW", "SPO2", "HR"):
            if labels_only:
                break
            a = get(canon)
            if a is None:
                flags.append(f"missing_{canon}")
                continue
            flat = a.reshape(-1)
            if a.ndim == 2 and a.shape[0] == n_epochs:
                f = a.shape[1] // EPOCH_S
            else:
                f = int(_meta_value(meta, f"fs_{canon.lower()}", default=0) or 0)
                fs_meta = meta.get("fs", {}) if isinstance(meta.get("fs"), dict) else {}
                f = int(fs_meta.get(canon.lower(), fs_meta.get(canon, f)) or 0) or (len(flat) // n_sec)
            if f <= 0 or len(flat) < f * n_sec:
                raise StageAError(f"{npz_path.name}: {canon} has {len(flat)} samples for fs={f}")
            if f != _NATIVE_FS[canon]:
                flags.append(f"fs_{canon}_{f}")
            flat = flat[: f * n_sec]
            fs[canon] = f
            sig[canon] = _to_physical(flat, canon, meta, flags)
        airflow_ok_arr = get("AIRFLOW_OK")
    if airflow_ok_arr is not None:
        airflow_ok = bool(np.asarray(airflow_ok_arr).reshape(-1)[0])
    else:
        airflow_ok = bool(meta.get("airflow_ok", False)) if "AIRFLOW" in sig else False
        if "AIRFLOW" in sig and "airflow_ok" not in meta:
            flags.append("airflow_ok_unknown")
    sid = int(meta.get("subject_id", _sid_from_name(npz_path.name)))
    return StageA(
        subject_id=sid,
        n_epochs=n_epochs,
        epoch_stage=stage.astype(np.int16),
        sleep_epoch=sleep_epoch,
        apnoea_label=apn,
        sec_event=sec_event,
        signals=sig,
        fs=fs,
        airflow_ok=airflow_ok,
        meta=meta,
        flags=flags,
    )


def _is_track_a(meta: dict) -> bool:
    """Sidecar written by ``dl_cache.extract_subject`` (per-channel ``encoding`` entries)."""
    ch = meta.get("channels")
    return bool(meta.get("extractor_version")) and isinstance(ch, dict) and all(
        isinstance(v, dict) and "encoding" in v for v in ch.values()
    )


def _check_stage_codes(stage: np.ndarray, codes: dict, name: str) -> np.ndarray:
    known = {int(v) for v in codes.values()}
    seen = set(np.unique(stage).astype(int).tolist())
    if not seen <= known:
        raise StageAError(f"{name}: stage codes {sorted(seen - known)} not in mapping {codes}")
    return np.isin(stage.astype(int), [int(codes[s]) for s in SLEEP_STAGES if s in codes])


def _load_track_a(npz_path: Path, meta: dict, labels_only: bool) -> StageA:
    """Decode a stage-A subject through ``dl_cache.load_physical`` (LUT / offset encodings,
    header gain with sign, picker ``airflow_ok``)."""
    from . import dl_cache

    sid = int(meta["subject_id"])
    cohort = npz_path.name.split("-")[0]
    chans = () if labels_only else ("ECG", "THOR", "ABDO", "AIRFLOW", "SPO2", "HR")
    d = dl_cache.load_physical(npz_path.parent, sid, channels=chans, cohort=cohort)
    n_epochs = int(d["n_epochs"])
    stage = np.asarray(d["epoch_stage"]).reshape(-1)
    codes = meta.get("labels", {}).get("stage_codes") or STAGE_CODES_DEFAULT
    sleep = _check_stage_codes(stage, codes, npz_path.name)
    if not np.array_equal(sleep, np.asarray(d["sleep_epoch"], dtype=bool)):
        raise StageAError(f"{npz_path.name}: sleep mask disagrees with dl_labels.sleep_epochs")
    flags = [f"missing_{c}" for c in chans if d["signals"].get(c) is None]
    flags += [f"qc_note:{n}" for n in meta.get("qc", {}).get("notes", [])]
    sig = {c: np.asarray(v, dtype=np.float64) for c, v in d["signals"].items() if v is not None}
    fs = {c: int(round(float(d["fs"][c]))) for c in sig}
    for c, f in fs.items():
        if f != _NATIVE_FS[c]:
            flags.append(f"fs_{c}_{f}")
    return StageA(
        subject_id=sid,
        n_epochs=n_epochs,
        epoch_stage=stage.astype(np.int16),
        sleep_epoch=sleep,
        apnoea_label=np.asarray(d["apnoea_label"]).reshape(-1).astype(np.int8),
        sec_event=np.asarray(d["sec_event"]).reshape(-1).astype(np.uint8),
        signals=sig,
        fs=fs,
        airflow_ok=bool(d["ok"].get("AIRFLOW", False)),
        meta=meta,
        flags=flags,
    )


def _sid_from_name(name: str) -> int:
    digits = "".join(ch for ch in name.split("-")[-1] if ch.isdigit())
    return int(digits) if digits else -1


def _to_physical(a: np.ndarray, canon: str, meta: dict, flags: list[str]) -> np.ndarray:
    """Digital -> physical using sidecar scaling; else a documented default."""
    x = a.astype(np.float64)
    if canon == "SPO2":
        if a.dtype == np.uint8:
            per = _meta_value(meta, "spo2_pct_per_code", default=100.0 / 128.0)
            return x * float(per)
        sc = _scale_for(meta, canon)
        if sc is None:
            flags.append("spo2_unscaled")
            return x
        return x * sc[0] + sc[1]
    if canon == "ECG":
        per = _meta_value(meta, "ecg_mV_per_lsb")
        if per is not None:
            return x * float(per)
    sc = _scale_for(meta, canon)
    if sc is not None:
        return x * sc[0] + sc[1]
    if canon in _DEFAULT_SIGN:
        flags.append(f"{canon}_sign_default")
        return x * _DEFAULT_SIGN[canon]
    flags.append(f"{canon}_unscaled")
    return x


def sec_event_from_events(events: Sequence, n_sec: int) -> np.ndarray:
    """Per-second mask: second s is inside an event if s + 0.5 lies in [start, start + dur)."""
    out = np.zeros(n_sec, dtype=np.uint8)
    centres = np.arange(n_sec) + 0.5
    for ev in events:
        if isinstance(ev, dict):
            start = float(ev.get("start", ev.get("start_sec")))
            dur = float(ev.get("duration", ev.get("duration_sec")))
        else:
            start, dur = float(ev[0]), float(ev[1])
        a = int(np.searchsorted(centres, start, side="left"))
        b = int(np.searchsorted(centres, start + dur, side="left"))
        out[a:b] = 1
    return out


# ---------------------------------------------------------------------------
# per-subject derivation
# ---------------------------------------------------------------------------


def derive_subject(sa: StageA, edr: str = "psa") -> dict:
    """Stage A -> 4 Hz channels in CHANNELS order plus 1 Hz target / sleep masks."""
    t0 = time.perf_counter()
    n_sec = sa.n_sec
    n4 = n_sec * FS
    signals = np.zeros((len(CHANNELS), n4), dtype=np.float32)
    ok = np.zeros(len(CHANNELS), dtype=bool)
    qc: dict[str, Any] = {"subject_id": sa.subject_id, "n_epochs": sa.n_epochs, "flags": list(sa.flags)}

    if "ECG" in sa.signals:
        rr4, edr4, q = ecg_to_rr_edr(sa.signals["ECG"], sa.fs["ECG"], n_sec, edr=edr)
        qc.update({f"ecg_{k}": v for k, v in q.items()})
        if q["ok"]:
            med_rr = float(np.median(rr4))
            med_edr = float(np.median(edr4))
            signals[0] = 1000.0 * (rr4 - med_rr)
            ok[0] = True
            qc["rr_median_s"] = med_rr
            if med_edr > 0 and np.isfinite(med_edr):
                signals[1] = edr4 / med_edr - 1.0
                ok[1] = True
            else:
                qc["edr_reason"] = "nonpositive_median"
            qc["edr_median"] = med_edr
            if "HR" in sa.signals:
                qc.update(hr_agreement(rr4, sa.signals["HR"]))
    for ci, canon in ((2, "THOR"), (3, "ABDO"), (4, "AIRFLOW")):
        if canon not in sa.signals:
            continue
        if canon == "AIRFLOW" and not sa.airflow_ok:
            qc["airflow_reason"] = "airflow_ok_false"
            continue
        x4 = resp_to_4hz(sa.signals[canon], sa.fs[canon], n4)
        sd = float(np.std(x4))
        qc[f"{canon.lower()}_std"] = sd
        if sd > 1e-6:
            signals[ci] = x4
            ok[ci] = True
    if "SPO2" in sa.signals:
        s4, q = spo2_to_4hz(sa.signals["SPO2"], n4, fs_in=sa.fs["SPO2"])
        qc.update({(k if k.startswith("spo2_") else f"spo2_{k}"): v for k, v in q.items()})
        if q["ok"]:
            signals[5] = s4
            ok[5] = True

    sec_sleep = np.repeat(sa.sleep_epoch.astype(np.uint8), EPOCH_S)
    if sa.sec_event is None:
        raise StageAError(f"subject {sa.subject_id}: no per-second event mask or event list")
    sec_target = (sa.sec_event > 0).astype(np.uint8)
    if not np.isfinite(signals).all():
        qc["nonfinite_replaced"] = int((~np.isfinite(signals)).sum())
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    qc["t_derive_s"] = round(time.perf_counter() - t0, 3)
    qc.update({f"ok_{c}": bool(o) for c, o in zip(CHANNELS, ok)})
    qc["sec_target_prev_sleep"] = (
        float(sec_target[sec_sleep == 1].mean()) if sec_sleep.any() else float("nan")
    )
    return {
        "subject_id": sa.subject_id,
        "n_epochs": sa.n_epochs,
        "signals": signals.astype(np.float16),
        "channel_ok": ok,
        "sec_target": sec_target,
        "sec_sleep": sec_sleep,
        "epoch_stage": sa.epoch_stage,
        "sleep_epoch": sa.sleep_epoch,
        "apnoea_label": sa.apnoea_label,
        "qc": qc,
    }


def _derive_worker(args: tuple[str, str]) -> dict:
    npz, edr = args
    try:
        return derive_subject(load_stage_a(npz), edr=edr)
    except Exception as exc:  # noqa: BLE001 - reported per subject in the ledger
        return {"error": f"{type(exc).__name__}: {exc}", "npz": npz}


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, bufsize: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit(repo: Path | None = None) -> str:
    repo = repo or Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all",
             "--", "thesis_pipeline", "scripts", "splits"],
            capture_output=True, text=True, timeout=10,
        )
        c = out.stdout.strip() or "unknown"
        # tracked edits or untracked code under thesis_pipeline/, scripts/, splits/ mean the
        # commit hash does not describe the code that ran
        code_dirty = [ln for ln in dirty.stdout.splitlines()
                      if ln[:2] != "??" or ln.rstrip().endswith((".py", ".json"))]
        return c + ("-dirty" if code_dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def assert_not_under_desktop(path: str | Path) -> None:
    """Large DL data must never be written under ~/Desktop (iCloud-synced)."""
    p = Path(path).expanduser().resolve()
    desk = (Path.home() / "Desktop").resolve()
    if p == desk or desk in p.parents:
        raise ValueError(f"refusing to write large DL data under {desk}: {p}")


def subject_layout(n_epochs: np.ndarray, max_chunk_bytes: int = MAX_CHUNK_BYTES) -> pd.DataFrame:
    """Chunk assignment and per-second offsets for subjects in the given order."""
    n_epochs = np.asarray(n_epochs, dtype=np.int64)
    n_pad = -(-n_epochs // WIN_EPOCHS) * WIN_EPOCHS
    block_s = PAD_S + n_pad * EPOCH_S + PAD_S
    per_ch_bytes = block_s * FS * 2
    chunk = np.zeros(len(n_epochs), dtype=np.int32)
    start = np.zeros(len(n_epochs), dtype=np.int64)
    k, used = 0, 0
    for i, b in enumerate(per_ch_bytes):
        if used > 0 and used + b > max_chunk_bytes:
            k, used = k + 1, 0
        chunk[i] = k
        start[i] = used // (FS * 2)
        used += int(b)
    return pd.DataFrame({
        "chunk": chunk,
        "block_start1": start,
        "off1": start + PAD_S,
        "n_epochs": n_epochs.astype(np.int32),
        "n_epochs_pad": n_pad.astype(np.int32),
        "n_windows": (n_pad // WIN_EPOCHS).astype(np.int32),
        "block_s": block_s,
    })


def chunk_lengths(layout: pd.DataFrame) -> dict[int, int]:
    """Seconds per chunk."""
    return {int(k): int(g["block_s"].sum()) for k, g in layout.groupby("chunk")}


def _file_names(split: str, k: int) -> dict[str, str]:
    d = {c: f"signals_{split}_{c}_{k:03d}.npy" for c in CHANNELS}
    d["sec_target"] = f"sec_target_{split}_{k:03d}.npy"
    d["sec_sleep"] = f"sec_sleep_{split}_{k:03d}.npy"
    return d


def pack_split(
    split: str,
    ids: Sequence[int],
    raw_dir: str | Path,
    out_root: str | Path,
    split_json: str | Path | None = None,
    workers: int = 6,
    edr: str = "psa",
    max_chunk_bytes: int = MAX_CHUNK_BYTES,
    allow_missing: bool = False,
    log=print,
) -> dict:
    """Derive stage B for ``ids`` (sorted) and pack them into ``out_root/split``.

    Resumable: subjects already listed as done in ``qc_<split>.jsonl`` with a matching
    ``layout_<split>.json`` are skipped. Returns the manifest dict.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")
    out_root = Path(out_root).expanduser()
    assert_not_under_desktop(out_root)
    out = out_root / split
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir).expanduser()
    ids = sorted(int(i) for i in ids)

    present, missing = [], []
    n_ep = []
    for sid in ids:
        npz = raw_dir / f"shhs1-{sid}.npz"
        if not npz.exists():
            missing.append(sid)
            continue
        js = npz.with_suffix(".json")
        n = None
        if js.exists():
            n = json.loads(js.read_text()).get("n_epochs")
        if n is None:
            with np.load(npz, allow_pickle=False) as z:
                k = _find_key(z.files, "STAGE")
                n = int(np.asarray(z[k]).reshape(-1).shape[0])
        present.append(sid)
        n_ep.append(int(n))
    if missing and not allow_missing:
        raise FileNotFoundError(f"{len(missing)} {split} subjects have no stage-A npz, e.g. {missing[:5]}")

    layout = subject_layout(np.asarray(n_ep), max_chunk_bytes)
    layout.insert(0, "subject_id", np.asarray(present, dtype=np.int32))
    lens = chunk_lengths(layout)
    layout_json = out / f"layout_{split}.json"
    ledger = out / f"qc_{split}.jsonl"
    layout_sig = {
        "ids": present, "n_epochs": n_ep, "max_chunk_bytes": max_chunk_bytes,
        "edr": edr, "frontend": FRONTEND_VERSION, "stage_b": STAGE_B_VERSION,
    }
    resume = layout_json.exists() and json.loads(layout_json.read_text()) == layout_sig
    done: dict[int, dict] = {}
    if resume and ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "done":
                    done[int(r["subject_id"])] = r
    else:
        if ledger.exists():
            ledger.unlink()
        layout_json.write_text(json.dumps(layout_sig))

    mode = "r+" if resume else "w+"
    mm: dict[tuple[int, str], np.memmap] = {}
    for k, n1 in lens.items():
        for key, name in _file_names(split, k).items():
            path = out / name
            if key in CHANNELS:
                shape, dt = (n1 * FS,), np.float16
            else:
                shape, dt = (n1,), np.uint8
            if mode == "r+" and path.exists():
                mm[(k, key)] = np.lib.format.open_memmap(path, mode="r+")
            else:
                mm[(k, key)] = np.lib.format.open_memmap(path, mode="w+", dtype=dt, shape=shape)

    row_of = {int(s): i for i, s in enumerate(layout["subject_id"])}
    todo = [s for s in present if s not in done]
    log(f"[{split}] {len(present)} subjects ({len(done)} already packed, {len(todo)} to derive), "
        f"{len(lens)} chunk(s), workers={workers}")
    args = [(str(raw_dir / f"shhs1-{s}.npz"), edr) for s in todo]
    t0 = time.time()
    n_err = 0

    def _consume(res: dict) -> None:
        nonlocal n_err
        if "error" in res:
            n_err += 1
            sid = _sid_from_name(Path(res["npz"]).name)
            rec = {"subject_id": sid, "status": "error", "error": res["error"]}
        else:
            sid = int(res["subject_id"])
            r = layout.iloc[row_of[sid]]
            if int(r["n_epochs"]) != int(res["n_epochs"]):
                raise StageAError(f"{sid}: n_epochs changed between layout and derive")
            k, o1 = int(r["chunk"]), int(r["off1"])
            n_sec = int(res["n_epochs"]) * EPOCH_S
            for ci, c in enumerate(CHANNELS):
                mm[(k, c)][o1 * FS:(o1 + n_sec) * FS] = res["signals"][ci]
            mm[(k, "sec_target")][o1:o1 + n_sec] = res["sec_target"]
            mm[(k, "sec_sleep")][o1:o1 + n_sec] = res["sec_sleep"]
            rec = {"status": "done", **_jsonable(res["qc"])}
        with open(ledger, "a") as f:
            f.write(json.dumps(rec) + "\n")

    if workers > 1 and len(args) > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_derive_worker, args, chunksize=1), 1):
                _consume(res)
                if i % 50 == 0 or i == len(args):
                    log(f"[{split}] {i}/{len(args)} derived ({time.time() - t0:.0f} s, {n_err} errors)")
    else:
        for i, a in enumerate(args, 1):
            _consume(_derive_worker(a))
            if i % 50 == 0 or i == len(args):
                log(f"[{split}] {i}/{len(args)} derived ({time.time() - t0:.0f} s, {n_err} errors)")
    for m in mm.values():
        m.flush()
    mm.clear()  # release the memmaps before hashing

    # final QC table (last record per subject wins)
    recs: dict[int, dict] = {}
    for line in ledger.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[int(r["subject_id"])] = r
    errors = {s: r["error"] for s, r in recs.items() if r.get("status") != "done"}
    if errors:
        raise RuntimeError(f"[{split}] {len(errors)} subjects failed derivation, e.g. "
                           f"{list(errors.items())[:3]}; fix and re-run (resumes)")
    qc_df = pd.DataFrame([recs[s] for s in present])
    idx = layout.drop(columns=["block_s"]).copy()
    for c in CHANNELS:
        idx[f"ok_{c}"] = qc_df[f"ok_{c}"].astype(bool).to_numpy()
    keep_qc = [c for c in qc_df.columns if c not in idx.columns and c not in ("status", "flags")
               and qc_df[c].map(lambda v: isinstance(v, (int, float, bool, np.number)) or v is None).all()]
    for c in keep_qc:
        idx[f"qc_{c}"] = pd.to_numeric(qc_df[c], errors="coerce").to_numpy()
    idx.to_parquet(out / f"index_{split}.parquet", index=False)

    ep_frames = []
    for sid in present:
        sa = load_stage_a(raw_dir / f"shhs1-{sid}.npz", labels_only=True)
        ep_frames.append(pd.DataFrame({
            "subject_id": np.full(sa.n_epochs, sid, dtype=np.int32),
            "epoch_idx": np.arange(sa.n_epochs, dtype=np.int32),
            "stage": sa.epoch_stage.astype(np.int8),
            "sleep": sa.sleep_epoch.astype(bool),
            "apnoea_label": sa.apnoea_label.astype(np.int8),
        }))
    epochs = pd.concat(ep_frames, ignore_index=True)
    epochs.to_parquet(out / f"epochs_{split}.parquet", index=False)

    files = sorted(p.name for p in out.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    manifest = {
        "split": split,
        "stage_b_version": STAGE_B_VERSION,
        "frontend_version": FRONTEND_VERSION,
        "edr_method": edr,
        "channels": list(CHANNELS),
        "fs": FS,
        "pad_s": PAD_S,
        "epoch_s": EPOCH_S,
        "win_epochs": WIN_EPOCHS,
        "n_subjects": len(present),
        "n_epochs": int(len(epochs)),
        "n_sleep_epochs": int(epochs["sleep"].sum()),
        "missing_subjects": missing,
        "chunks": {str(k): v for k, v in lens.items()},
        "split_json": str(split_json) if split_json else None,
        "split_sha256": sha256_file(split_json) if split_json else None,
        "git_commit": git_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stored_values": {
            "RR": "1000 * (rr_s - subject median rr_s)",
            "EDR": "edr / subject median edr - 1",
            "THOR/ABDO/AIRFLOW": "physical-sign units",
            "SPO2": "(pct - 95) / 5",
        },
        "airflow_rule": "stage-A pick (highest whole-night std among AIRFLOW/NEW AIR/NEWAIR/NEW A/F/AUX); "
                        "zero with ok_AIRFLOW = False unless airflow_ok",
        "sha256": {name: sha256_file(out / name) for name in files},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    log(f"[{split}] packed {len(present)} subjects -> {out} ({time.time() - t0:.0f} s)")
    return manifest


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        elif isinstance(v, float) and not np.isfinite(v):
            v = None
        out[k] = v
    return out


def verify_manifest(split_dir: str | Path, files: Iterable[str] | None = None) -> list[str]:
    """Return names whose sha256 does not match MANIFEST.json (empty list = OK)."""
    split_dir = Path(split_dir)
    man = json.loads((split_dir / "MANIFEST.json").read_text())
    names = list(files) if files is not None else list(man["sha256"])
    bad = []
    for n in names:
        p = split_dir / n
        if not p.exists() or sha256_file(p) != man["sha256"].get(n):
            bad.append(n)
    return bad


def env_workers(default: int = 6) -> int:
    try:
        return max(1, int(os.environ.get("DL_DERIVE_WORKERS", default)))
    except ValueError:
        return default
