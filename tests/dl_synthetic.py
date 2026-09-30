"""Synthetic stage-A subjects (design section 2.2 format) for the DL-arm tests.

Not a test module. Writes ``shhs1-<id>.npz`` + ``shhs1-<id>.json`` with per-epoch 2-D
arrays: ecg (n, 3750) int8, thor / abdo / airflow (n, 300) int8, spo2 (n, 30) uint8
codes, hr (n, 30) uint8, epoch_stage (n,) int8, apnoea_label (n,) int8,
sec_event (n, 30) uint8, and a sidecar with EDF-style physical / digital ranges.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ECG_FS = 125
RESP_FS = 10
ECG_LSB_MV = 2.5 / 255.0


def synth_ecg(
    n_sec: int,
    fs: int = ECG_FS,
    hr_bpm: float = 60.0,
    rr_mod: float = 0.05,
    resp_period_s: float = 4.0,
    amp_mod: float = 0.25,
    gaps: tuple[tuple[float, float], ...] = (),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECG in mV with Gaussian R and T waves. Returns (ecg_mV, beat_times_s, beat_amp)."""
    rng = np.random.default_rng(seed)
    beats, t = [], 0.6
    while t < n_sec - 0.6:
        beats.append(t)
        t += 60.0 / hr_bpm * (1.0 + rr_mod * np.sin(2 * np.pi * t / 23.0))
    beats = np.asarray(beats)
    amp = 1.0 + amp_mod * np.sin(2 * np.pi * beats / resp_period_s)
    n = n_sec * fs
    x = np.zeros(n)
    half = int(0.08 * fs)
    k = np.arange(-half, half + 1)
    for tb, a in zip(beats, amp):
        c = int(round(tb * fs))
        idx = c + k
        ok = (idx >= 0) & (idx < n)
        x[idx[ok]] += a * np.exp(-0.5 * (k[ok] / (0.010 * fs)) ** 2)
        ct = c + int(0.25 * fs)
        idx = ct + k
        ok = (idx >= 0) & (idx < n)
        x[idx[ok]] += 0.15 * np.exp(-0.5 * (k[ok] / (0.04 * fs)) ** 2)
    x += 0.01 * rng.standard_normal(n)
    for a, b in gaps:
        x[int(a * fs):int(b * fs)] = 0.0
    keep = np.ones(len(beats), bool)
    for a, b in gaps:
        keep &= ~((beats >= a) & (beats < b))
    return x, beats[keep], amp[keep]


def write_stage_a(
    raw_dir: Path,
    sid: int,
    n_epochs: int = 24,
    stages: np.ndarray | None = None,
    events: list[tuple[float, float]] | None = None,
    thor_spikes_s: tuple[float, ...] = (),
    ecg: np.ndarray | None = None,
    flat_ecg: bool = False,
    airflow_ok: bool = True,
    seed: int = 0,
    spo2_pct: np.ndarray | None = None,
) -> Path:
    """Write one synthetic stage-A subject. ``stages`` are codes (W0 N1 1 N2 2 N3 3 REM 5)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    n_sec = n_epochs * 30
    rng = np.random.default_rng(seed)
    if stages is None:
        stages = np.full(n_epochs, 2, dtype=np.int8)
        stages[:2] = 0
    stages = np.asarray(stages, dtype=np.int8)
    events = events or []
    # labels via the Aim 2 epoch rule
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from thesis_pipeline.epochs import apnoea_labels
    from thesis_pipeline.io import RespiratoryEvent

    apn = apnoea_labels(n_epochs, [RespiratoryEvent(a, d, "Hypopnea") for a, d in events]).astype(np.int8)
    sec_event = np.zeros(n_sec, dtype=np.uint8)
    centres = np.arange(n_sec) + 0.5
    for a, d in events:
        sec_event[(centres >= a) & (centres < a + d)] = 8  # hypopnoea bit
    if ecg is None:
        if flat_ecg:
            ecg = np.zeros(n_sec * ECG_FS)
        else:
            ecg, _, _ = synth_ecg(n_sec, seed=seed)
    ecg_dig = np.clip(np.round(ecg / ECG_LSB_MV), -128, 127).astype(np.int8)
    t10 = np.arange(n_sec * RESP_FS) / RESP_FS
    thor = np.round(-40 * np.sin(2 * np.pi * t10 / 4.0) + rng.normal(0, 1, len(t10)))
    thor_dig = np.zeros_like(thor) if thor_spikes_s else thor
    for s in thor_spikes_s:
        thor_dig[int(round(s * RESP_FS))] = -100  # physical sign is inverted -> +100 physical
    abdo_dig = np.round(-30 * np.sin(2 * np.pi * t10 / 4.0 + 0.3))
    air_dig = np.round(20 * np.sin(2 * np.pi * t10 / 4.0 + 0.1))
    if spo2_pct is None:
        spo2_pct = np.full(n_sec, 96.0)
    spo2_code = np.clip(np.round(np.asarray(spo2_pct) * 128 / 100), 0, 255).astype(np.uint8)
    hr = np.full(n_sec, 60, dtype=np.uint8)
    arrays = {
        "ecg": ecg_dig.reshape(n_epochs, 30 * ECG_FS),
        "thor": np.clip(thor_dig, -128, 127).astype(np.int8).reshape(n_epochs, 300),
        "abdo": np.clip(abdo_dig, -128, 127).astype(np.int8).reshape(n_epochs, 300),
        "airflow": np.clip(air_dig, -128, 127).astype(np.int8).reshape(n_epochs, 300),
        "spo2": spo2_code.reshape(n_epochs, 30),
        "hr": hr.reshape(n_epochs, 30),
        "position": np.zeros((n_epochs, 30), np.uint8),
        "oxstat": np.zeros((n_epochs, 30), np.uint8),
        "epoch_stage": stages,
        "apnoea_label": apn,
        "sec_event": sec_event.reshape(n_epochs, 30),
    }
    np.savez_compressed(raw_dir / f"shhs1-{sid}.npz", **arrays)
    meta = {
        "subject_id": sid,
        "n_epochs": n_epochs,
        "airflow_ok": airflow_ok,
        "channels": {
            "ECG": {"physical_min": -1.25, "physical_max": 1.25, "digital_min": -128, "digital_max": 127},
            "THOR": {"physical_min": 1.0, "physical_max": -1.0, "digital_min": -128, "digital_max": 127},
            "ABDO": {"physical_min": 1.0, "physical_max": -1.0, "digital_min": -128, "digital_max": 127},
            "AIRFLOW": {"physical_min": -125.0, "physical_max": 125.0, "digital_min": -128, "digital_max": 127},
            "HR": {"physical_min": 0, "physical_max": 255, "digital_min": 0, "digital_max": 255},
        },
        "events": [{"start": a, "duration": d, "kind": "Hypopnea"} for a, d in events],
    }
    (raw_dir / f"shhs1-{sid}.json").write_text(json.dumps(meta))
    return raw_dir / f"shhs1-{sid}.npz"


def write_split_json(path: Path, train, val, test) -> Path:
    path.write_text(json.dumps({"train": list(map(int, train)), "val": list(map(int, val)),
                                "test": list(map(int, test))}))
    return path


# subject id -> (split, n_epochs, extras)
SPIKE_EPOCHS = (0, 7, 17)
SUBJECTS = {
    400001: ("train", 20, {"thor_spikes_s": tuple(30 * i + 15.0 for i in SPIKE_EPOCHS),
                            "events": [(30 * i + 15.0, 1.0) for i in SPIKE_EPOCHS]}),
    400002: ("train", 25, {"events": [(95.0, 20.0), (400.0, 35.0)]}),
    400003: ("train", 31, {"events": [(200.0, 15.0), (610.0, 12.0)],
                            "stages": np.array([0] * 3 + [2] * 10 + [0] * 4 + [5] * 14, np.int8)}),
    400004: ("val", 19, {"events": [(150.0, 25.0)]}),
    400005: ("val", 24, {"events": [(300.0, 18.0), (500.0, 30.0)]}),
    400006: ("test", 22, {"events": [(120.0, 20.0)]}),
}


def build_packed_dataset(root: Path, max_chunk_bytes: int = 1000 * 8, workers: int = 1) -> dict:
    """Write synthetic stage A for SUBJECTS, a split JSON, and pack train / val / test."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from thesis_pipeline.dl_stage_b import pack_split

    root = Path(root)
    raw = root / "raw"
    for sid, (_, n_ep, extra) in SUBJECTS.items():
        write_stage_a(raw, sid, n_epochs=n_ep, seed=sid % 97, **extra)
    split = {s: [i for i, v in SUBJECTS.items() if v[0] == s] for s in ("train", "val", "test")}
    sj = write_split_json(root / "split.json", split["train"], split["val"], split["test"])
    out = root / "model"
    for s in ("train", "val", "test"):
        pack_split(s, split[s], raw, out, split_json=sj, workers=workers,
                   max_chunk_bytes=max_chunk_bytes, log=lambda *a: None)
    return {"raw": raw, "out": out, "split_json": sj, "split": split}
