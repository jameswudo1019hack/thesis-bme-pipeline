"""Windows, batches and split guards for the Aim 2 DL arm (design sections 2.4, 4, 5).

Data path (critique #9): the packed stage-B arrays stay on the host as read-only
memory maps (``np.load(mmap_mode="r")``); each batch is gathered on the CPU
(``start[:, None] + arange``) and only the batch is moved to the device. Several
training processes on one machine therefore share the OS page cache instead of each
holding a copy of the training array in GPU memory.

Window geometry (4 Hz input, 1 Hz output):
    * Input window = 300 s = 1200 samples; output = 300 per-second logits; output
      second t <-> input samples 4t .. 4t+3 <-> absolute second s0 + t.
    * Evaluation grid (fixed, anchored at EDF t = 0): window j has input seconds
      [180 j - 60, 180 j + 240) and its central 180 s are epochs 6j .. 6j+5.
      J = ceil(n_epochs / 6) windows per subject; a window is kept when its centre
      holds at least one sleep epoch, so every sleep epoch is scored exactly once.
    * Training windows: the same grid shifted by a per-subject random phase of
      0-179 whole seconds, drawn from ``default_rng([model_seed, pass, 0x5EED])``; a
      window is kept when its centre holds at least one sleep second and its input
      stays inside the subject's padded block.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

from .dl_stage_b import CENTRE_S, CHANNELS, EPOCH_S, FS, PAD_S, WIN_EPOCHS, WIN_S

WIN_SAMPLES = WIN_S * FS  # 1200
CTX_S = (WIN_S - CENTRE_S) // 2  # 60


class TestDataRefused(RuntimeError):
    """Raised when test data reaches a code path that must never see it."""

    __test__ = False  # not a pytest class


# ---------------------------------------------------------------------------
# split file + guards
# ---------------------------------------------------------------------------


def load_split(path: str | Path) -> dict:
    """Load a frozen split JSON: {"train": [...], "val": [...], "test": [...]}.

    Returns {"train","val","test": np.ndarray[int64], "sha256": str, "path": str}.
    Asserts the three sets are pairwise disjoint.
    """
    path = Path(path)
    raw = path.read_bytes()
    d = json.loads(raw)
    src = d.get("splits", d)
    out = {k: np.asarray(sorted(int(i) for i in src[k]), dtype=np.int64) for k in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = np.intersect1d(out[a], out[b])
        if len(inter):
            raise ValueError(f"split {path} has {len(inter)} ids in both {a} and {b}")
    out["sha256"] = hashlib.sha256(raw).hexdigest()
    out["path"] = str(path)
    return out


def assert_ids_in_split(ids: Sequence[int], split: dict, which: str) -> None:
    """``ids`` must be a subset of split[which] and share nothing with split['test']
    unless which == 'test'."""
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    if which != "test":
        leak = np.intersect1d(ids, split["test"])
        if len(leak):
            raise TestDataRefused(f"{len(leak)} test subject(s) in the {which} data, e.g. {leak[:5].tolist()}")
    extra = np.setdiff1d(ids, split[which])
    if len(extra):
        raise ValueError(f"{len(extra)} {which} subject(s) are not in split[{which!r}], e.g. {extra[:5].tolist()}")


# ---------------------------------------------------------------------------
# packed split reader
# ---------------------------------------------------------------------------


class PackedSplit:
    """Read-only view of one packed stage-B split folder.

    Parameters
    ----------
    root : folder containing MANIFEST.json (e.g. ``~/thesis_dl_cache/model_v1/train``)
    channels : channel names to open, in model order (subset of CHANNELS)
    allow_test : must be True to open a split whose manifest says "test"
    """

    def __init__(self, root: str | Path, channels: Sequence[str], allow_test: bool = False) -> None:
        self.root = Path(root).expanduser()
        self.manifest = json.loads((self.root / "MANIFEST.json").read_text())
        self.split = self.manifest["split"]
        if self.split == "test" and not allow_test:
            raise TestDataRefused(f"{self.root} is a test split; refused")
        bad = [c for c in channels if c not in CHANNELS]
        if bad:
            raise ValueError(f"unknown channels {bad}")
        self.channels = tuple(channels)
        self.ch_idx = [CHANNELS.index(c) for c in channels]
        if int(self.manifest["fs"]) != FS or int(self.manifest["pad_s"]) != PAD_S:
            raise ValueError("manifest fs / pad_s do not match this code")
        s = self.split
        self.index = pd.read_parquet(self.root / f"index_{s}.parquet").reset_index(drop=True)
        self.epochs = pd.read_parquet(self.root / f"epochs_{s}.parquet")
        self.subject_ids = self.index["subject_id"].to_numpy(np.int64)
        chunks = sorted(int(k) for k in self.manifest["chunks"])
        self.sig = {
            c: {k: np.load(self.root / f"signals_{s}_{c}_{k:03d}.npy", mmap_mode="r") for k in chunks}
            for c in self.channels
        }
        self.target = {k: np.load(self.root / f"sec_target_{s}_{k:03d}.npy", mmap_mode="r") for k in chunks}
        self.sleep = {k: np.load(self.root / f"sec_sleep_{s}_{k:03d}.npy", mmap_mode="r") for k in chunks}
        # epoch offsets for vectorised sleep look-ups
        ep = self.epochs.sort_values(["subject_id", "epoch_idx"], kind="stable")
        counts = ep.groupby("subject_id", sort=True).size()
        if not np.array_equal(counts.index.to_numpy(np.int64), np.sort(self.subject_ids)) or not np.array_equal(
            counts.loc[self.subject_ids].to_numpy(), self.index["n_epochs"].to_numpy()
        ):
            raise ValueError("epochs table does not match index n_epochs")
        order = {int(s): i for i, s in enumerate(counts.index)}
        starts = np.concatenate([[0], np.cumsum(counts.to_numpy())])
        self._ep_start = np.array([starts[order[int(s)]] for s in self.subject_ids], dtype=np.int64)
        self._ep_sorted = ep.reset_index(drop=True)
        self._sleep_ep = ep["sleep"].to_numpy(bool)
        self._sleep_cum = np.concatenate([[0], np.cumsum(self._sleep_ep)]).astype(np.int64)

    # -- per-subject arrays ------------------------------------------------
    @property
    def n_subjects(self) -> int:
        return len(self.index)

    def channel_ok(self) -> np.ndarray:
        return self.index[[f"ok_{c}" for c in self.channels]].to_numpy(bool)

    def sleep_seconds(self, row: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
        """Sleep seconds in [t0, t1) (seconds relative to epoch 0) for subject rows."""
        row = np.asarray(row)
        n = self.index["n_epochs"].to_numpy(np.int64)[row]
        e0 = self._ep_start[row]

        def S(t):
            t = np.clip(np.asarray(t, dtype=np.int64), 0, n * EPOCH_S)
            e = np.minimum(t // EPOCH_S, n)
            rem = t - e * EPOCH_S
            full = self._sleep_cum[e0 + e] - self._sleep_cum[e0]
            inside = e < n
            part = np.zeros_like(t)
            part[inside] = self._sleep_ep[(e0 + e)[inside]] * rem[inside]
            return full * EPOCH_S + part

        return S(t1) - S(t0)

    # -- gather --------------------------------------------------------------
    def gather(self, win: "Windows", sel: np.ndarray) -> dict:
        """CPU gather of windows ``sel`` -> numpy batch dict.

        x (B, 1200, C) float16, valid (B, 1200) bool (inside the night, by index),
        y (B, 300) uint8 per-second target, mask (B, 300) bool (sleep and inside night).
        """
        sel = np.asarray(sel)
        B = len(sel)
        C = len(self.channels)
        x = np.empty((B, WIN_SAMPLES, C), dtype=np.float16)
        y = np.empty((B, WIN_S), dtype=np.uint8)
        slp = np.empty((B, WIN_S), dtype=np.uint8)
        chunk = win.chunk[sel]
        s0 = win.s0[sel]
        ar4 = np.arange(WIN_SAMPLES)
        ar1 = np.arange(WIN_S)
        for k in np.unique(chunk):
            m = np.flatnonzero(chunk == k)
            i4 = (s0[m] * FS)[:, None] + ar4
            i1 = s0[m][:, None] + ar1
            for ci, c in enumerate(self.channels):
                x[m, :, ci] = self.sig[c][int(k)][i4]
            y[m] = self.target[int(k)][i1]
            slp[m] = self.sleep[int(k)][i1]
        off1 = self.index["off1"].to_numpy(np.int64)[win.row[sel]]
        n_sec = self.index["n_epochs"].to_numpy(np.int64)[win.row[sel]] * EPOCH_S
        rel1 = s0[:, None] + ar1 - off1[:, None]
        inside1 = (rel1 >= 0) & (rel1 < n_sec[:, None])
        rel4 = (s0[:, None] * FS + ar4) - (off1 * FS)[:, None]
        valid = (rel4 >= 0) & (rel4 < (n_sec * FS)[:, None])
        return {"x": x, "valid": valid, "y": y, "mask": (slp > 0) & inside1}


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


@dataclass
class Windows:
    row: np.ndarray  # index row per window
    chunk: np.ndarray  # chunk id per window
    s0: np.ndarray  # absolute input start second within the chunk
    j: np.ndarray  # grid index (eval) or shifted-grid index (train)
    phase: np.ndarray  # phase shift in seconds (0 for eval)

    def __len__(self) -> int:
        return len(self.row)

    def take(self, idx: np.ndarray) -> "Windows":
        return Windows(self.row[idx], self.chunk[idx], self.s0[idx], self.j[idx], self.phase[idx])


def _expand(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """rows repeated by counts and a within-row arange."""
    rows = np.repeat(np.arange(len(counts)), counts)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    within = np.arange(counts.sum()) - np.repeat(starts, counts)
    return rows, within


def eval_windows(ps: PackedSplit) -> Windows:
    """Fixed grid (phase 0); keep windows whose centre holds >= 1 sleep epoch."""
    J = ps.index["n_windows"].to_numpy(np.int64)
    rows, j = _expand(J)
    c0 = WIN_EPOCHS * j * EPOCH_S  # centre start, seconds rel. epoch 0
    keep = ps.sleep_seconds(rows, c0, c0 + CENTRE_S) > 0
    rows, j = rows[keep], j[keep]
    off1 = ps.index["off1"].to_numpy(np.int64)[rows]
    chunk = ps.index["chunk"].to_numpy(np.int64)[rows]
    return Windows(rows, chunk, off1 + CENTRE_S * j - CTX_S, j, np.zeros_like(j))


def train_window_starts(
    ps: PackedSplit, seed: int, pass_idx: int, subsample: float = 1.0
) -> Windows:
    """Phase-shifted training windows for one pass (deterministic in seed, pass_idx)."""
    rng = np.random.default_rng([int(seed), int(pass_idx), 0x5EED])
    n = ps.n_subjects
    phase = rng.integers(0, CENTRE_S, size=n)
    J = ps.index["n_windows"].to_numpy(np.int64)
    # centre [phase + 180 j, + 180) must lie inside the padded night [0, 180 J)
    J_eff = np.where(phase > 0, J - 1, J)
    rows, j = _expand(np.maximum(J_eff, 0))
    ph = phase[rows]
    c0 = ph + CENTRE_S * j
    keep = ps.sleep_seconds(rows, c0, c0 + CENTRE_S) > 0
    rows, j, ph, c0 = rows[keep], j[keep], ph[keep], c0[keep]
    off1 = ps.index["off1"].to_numpy(np.int64)[rows]
    chunk = ps.index["chunk"].to_numpy(np.int64)[rows]
    win = Windows(rows, chunk, off1 + c0 - CTX_S, j, ph)
    if subsample < 1.0:
        k = int(np.floor(subsample * len(win)))
        pick = np.sort(rng.permutation(len(win))[:k])
        win = win.take(pick)
    return win


def pass_order(n: int, seed: int, pass_idx: int) -> np.ndarray:
    """Shuffled window order for one pass (deterministic)."""
    return np.random.default_rng([int(seed), int(pass_idx), 0x0DE5]).permutation(n)


def iter_batches(
    ps: PackedSplit,
    win: Windows,
    batch: int,
    order: np.ndarray | None = None,
    start_batch: int = 0,
    prefetch: int = 2,
    drop_singletons: bool = False,
) -> Iterator[tuple[int, np.ndarray, dict]]:
    """Yield (batch_index, window_indices, batch_dict). A background thread gathers
    up to ``prefetch`` batches ahead on the CPU."""
    idx_all = np.arange(len(win)) if order is None else np.asarray(order)
    nb = -(-len(idx_all) // batch)
    todo = list(range(start_batch, nb))

    def make(b: int):
        sel = idx_all[b * batch:(b + 1) * batch]
        return b, sel, ps.gather(win, sel)

    if prefetch <= 0:
        for b in todo:
            out = make(b)
            if drop_singletons and len(out[1]) < 2:
                continue
            yield out
        return

    q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop = threading.Event()
    _END = object()

    def worker():
        try:
            for b in todo:
                if stop.is_set():
                    return
                q.put(make(b))
            q.put(_END)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer
            q.put(exc)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                break
            if isinstance(item, BaseException):
                raise item
            if drop_singletons and len(item[1]) < 2:
                continue
            yield item
    finally:
        stop.set()
        while th.is_alive():
            try:
                q.get_nowait()
            except queue.Empty:
                th.join(timeout=0.05)


# ---------------------------------------------------------------------------
# per-second -> epoch mapping for evaluation windows
# ---------------------------------------------------------------------------


def centre_epochs(ps: PackedSplit, win: Windows, sec_probs: np.ndarray) -> pd.DataFrame:
    """Map centre seconds of eval windows to epochs.

    ``sec_probs`` is (W, 300). Returns one row per real epoch in the window centres:
    subject_id, epoch_idx, sleep, apnoea_label, stage, and ``sec`` (n, 30) as an
    object-free companion array in ``df.attrs['sec']``.
    """
    sec_probs = np.asarray(sec_probs)
    if sec_probs.shape != (len(win), WIN_S):
        raise ValueError(f"sec_probs must be ({len(win)}, {WIN_S}); got {sec_probs.shape}")
    cen = sec_probs[:, CTX_S:CTX_S + CENTRE_S].reshape(len(win), WIN_EPOCHS, EPOCH_S)
    n_ep = ps.index["n_epochs"].to_numpy(np.int64)[win.row]
    ep_idx = win.j[:, None] * WIN_EPOCHS + np.arange(WIN_EPOCHS)[None, :]
    real = ep_idx < n_ep[:, None]
    w_i, e_i = np.nonzero(real)
    epoch_idx = ep_idx[w_i, e_i]
    rows = win.row[w_i]
    gidx = ps._ep_start[rows] + epoch_idx
    ep = ps._ep_sorted
    df = pd.DataFrame({
        "subject_id": ps.subject_ids[rows].astype(np.int32),
        "epoch_idx": epoch_idx.astype(np.int32),
        "sleep": ep["sleep"].to_numpy(bool)[gidx],
        "stage": ep["stage"].to_numpy()[gidx],
        "apnoea_label": ep["apnoea_label"].to_numpy(np.int8)[gidx],
    })
    assert (ep["epoch_idx"].to_numpy()[gidx] == epoch_idx).all()
    sec = cen[w_i, e_i, :]
    order = np.lexsort((df["epoch_idx"].to_numpy(), df["subject_id"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)
    df.attrs["sec"] = sec[order]
    return df


def sleep_epoch_key(ps: PackedSplit) -> pd.DataFrame:
    """(subject_id int32, epoch_idx int32, apnoea_label int8) for sleep epochs, sorted."""
    ep = ps.epochs[ps.epochs["sleep"]].sort_values(["subject_id", "epoch_idx"], kind="stable")
    return pd.DataFrame({
        "subject_id": ep["subject_id"].to_numpy(np.int32),
        "epoch_idx": ep["epoch_idx"].to_numpy(np.int32),
        "apnoea_label": ep["apnoea_label"].to_numpy(np.int8),
    })
