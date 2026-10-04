"""Trainer for the Aim 2 Olsen BiGRU (the vault pre-registration note governs).

Recipe A (default): AdamW lr 1e-3, weight_decay 1e-4, batch 128, unweighted per-second
BCEWithLogits masked to sleep seconds inside the night, ReduceLROnPlateau(mode="max",
factor 0.1, patience 3, abs threshold 1e-4, no LR floor) on validation epoch AUC with the
``mean`` aggregator (the same AUC selects the checkpoint and drives early stopping), early
stop on the 6th consecutive pass without improvement, at most 25 passes. torch cuts the LR
when the count of non-improving passes EXCEEDS the patience, so recipe A cuts on the 4th
consecutive non-improving pass. If no gain follows the cut, 2 passes run at the reduced LR
before the stop; a gain > 1e-4 resets both counters, so training continues and further
x0.1 cuts are possible (``schedule_facts``, ``SCHEDULE_NOTE``). Recipes B and C are the
Colab-gate fallbacks (``scripts/bench_bigru.gate_recipe``).

Pre-registration freeze: ``Trainer.run()`` refuses to train unless the frozen
pre-registration record (``prereg.AIM2_DL_FREEZE``) exists, except for pilot runs with
``TrainSettings.allow_unfrozen`` (never reportable; ``predict_aim2_dl.py`` refuses them for
test). The freeze summary at the start of the run (pass 0, step 0) is kept in
``state["prereg_freeze_first"]`` and never overwritten on resume; every checkpoint and
metrics.json also carry the current summary (``prereg_freeze``), ``allow_unfrozen`` and the
cache ``data_versions`` (front-end, stage-B and EDR versions of train and val). A production
invocation (no ``allow_unfrozen``) refuses to resume a checkpoint that can never be reported
(it did not start under the freeze, or a pilot invocation resumed it), before any pass is
spent; that run needs fresh checkpoint folders.

Code commits: every invocation of ``Trainer.run()`` (the first and every resume) appends its
code commit (``git_commit()``, "-dirty" for modified code) to ``state["git_commits"]``, which
is never cleared; checkpoints and metrics.json carry the list (and ``git_commit``, the commit
of the invocation that saved them), so a run trained partly at another commit is visible
(``predict_aim2_dl.py`` and ``evaluate_aim2_dl.py`` refuse it for test). A resumed invocation
that adds a commit (or is the first with ``allow_unfrozen``) re-saves last.pt before any pass,
so an invocation that only re-finalises a finished run is recorded in last.pt as well.

Test protection: the trainer only opens packed folders whose MANIFEST split is
"train" / "val", checks every subject against the frozen split JSON (no test id may
appear) and checks the cache was packed from that same split file (sha256).

Checkpoints: ``last.pt`` after every pass (and every ``ckpt_every_steps`` batches if
set) with model, optimiser, scheduler, GradScaler, loop state, all RNG states and a
copy of the best weights so far (``best_model``), so the one file is self-consistent;
``best.pt`` on every validation improvement. ``finalize`` checks best.pt against the
best weights in last.pt and rewrites it from last.pt if they differ (a stale Drive copy
or an interrupted save). Writes are atomic (tmp + os.replace) and optionally mirrored
to a second folder (Google Drive on Colab). A failed mirror copy (Drive I/O error) is
logged and retried on the next save instead of killing the run; after finalising, a
mirror that still fails raises. A new invocation resumes from the most advanced of the
local and mirrored ``last.pt``; a mirror written by a different configuration (another
seed's folder) is refused before anything is copied. Window phases and order depend
only on (model_seed, pass), so a resumed run replays exactly.

Non-finite steps: without AMP (CPU, or the pre-declared ``--no-amp`` fallback) a step
whose loss or gradients are non-finite is skipped (no optimiser update) and counted
(``nonfinite``); with AMP the GradScaler skips steps whose scaled gradients overflow (the
loss may be finite) and those are counted separately (``scaler_skipped``: the step after
which the GradScaler scale fell). Both counts go to history.csv and metrics.json. Every
skipped step (either kind) is a non-finite training step: 20 consecutive ones (no applied
step in between) raise FloatingPointError, the NaN fallback, with or without AMP; isolated
ones trigger nothing (healthy fp16 training never skips 20 in a row: the scale would have to
fall by 2^20, e.g. from the initial 2^16 to 2^-4). A checkpoint with non-finite weights is
never written.

Outputs in ``out_dir``: history.csv, val_predictions.parquet (best checkpoint, all
aggregators), postproc.json (aggregator + threshold chosen on validation) and
metrics.json (``scale_pos_weight`` = 1.0 because the loss is unweighted).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .dl_aggregate import AGGREGATORS, choose_aggregator, sec_to_epoch
from .dl_data import (
    PackedSplit,
    TestDataRefused,
    Windows,
    assert_ids_in_split,
    centre_epochs,
    eval_windows,
    iter_batches,
    load_split,
    pass_order,
    sleep_epoch_key,
    train_window_starts,
)
from . import prereg
from .dl_eval import THRESHOLDS, threshold_f1max
from .dl_models import OlsenBiGRU, count_parameters
from .dl_stage_b import CHANNELS, git_commit, sha256_file
from .prereg import PreregMismatch

CONFIGS: dict[str, tuple[str, ...]] = {
    "M": ("RR", "EDR"),
    "P4": ("RR", "EDR", "THOR", "ABDO"),
    "F6": CHANNELS,
}
DEFERRED_CONFIGS = {
    "F6": "deferred until the Aim 1 airflow picker defect is fixed (decision 2026-09-30)",
}


@dataclass(frozen=True)
class Recipe:
    name: str
    batch: int
    lr_patience: int
    stop_patience: int
    max_passes: int
    subsample: float


RECIPES = {
    "A": Recipe("A", 128, 3, 6, 25, 1.0),
    "B": Recipe("B", 256, 3, 6, 20, 1.0),
    "C": Recipe("C", 256, 2, 4, 15, 0.5),
}

SCHEDULE_NOTE = (
    "Olsen 2020 (olsen_raw.txt L382-387): LR/10 when evaluation accuracy 'plateaued for more "
    "than 10 epochs, i.e. one iteration through the training set', stop after a further "
    "20-epoch plateau. The wording is ambiguous. Reading (a): 10 'epochs' = 1 pass, so "
    "patience is about 1 pass (LR) and 2 passes (stop). Reading (b): 'epoch' = a full pass, "
    "so patience is 10 / 20 passes. This build: torch ReduceLROnPlateau(mode='max', factor 0.1, "
    "patience P, abs threshold 1e-4, no min_lr) on validation epoch AUC (mean aggregator; the same "
    "AUC selects the checkpoint and drives early stopping). torch cuts the LR when the count of "
    "consecutive passes without a gain > 1e-4 EXCEEDS P, i.e. on the (P+1)-th such pass (Keras, "
    "Olsen's framework, cuts on the P-th); early stopping fires on the S-th consecutive such pass. "
    "Recipes A/B (P = 3, S = 6): LR cut on the 4th non-improving pass, stop on the 6th; recipe C "
    "(P = 2, S = 4): cut on the 3rd, stop on the 4th. If no gain follows a cut, S-P-1 passes run at "
    "the reduced LR before the stop (2 for A/B, 1 for C). A gain > 1e-4 resets both counters, so "
    "training can continue at the reduced LR and further x0.1 cuts (no floor) are possible. "
    "Caps 25 / 20 / 15 passes. More patient than reading (a), less than (b)."
)


def schedule_facts(r: Recipe) -> dict:
    """When the LR cut and the early stop fire for recipe ``r`` (torch semantics).

    ``passes_at_reduced_lr_before_stop_if_no_gain`` holds only when no gain follows the cut;
    a gain resets both counters and further cuts are possible (``further_cuts_possible``).
    """
    return {
        "torch_lr_patience": r.lr_patience,
        "lr_cut_on_consecutive_nonimproving_pass": r.lr_patience + 1,
        "stop_on_consecutive_nonimproving_pass": r.stop_patience,
        "passes_at_reduced_lr_before_stop_if_no_gain": max(r.stop_patience - r.lr_patience - 1, 0),
        "gain_resets_both_counters": True,
        "further_cuts_possible": True,
        "min_lr": 0.0,
        "max_passes": r.max_passes,
        "min_delta_abs": 1e-4,
    }


@dataclass
class TrainSettings:
    config: str
    model_seed: int
    recipe: str = "A"
    amp: bool = True
    device: str = "auto"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    min_delta: float = 1e-4
    hidden: int = 128
    eval_batch: int = 512
    prefetch: int = 2
    ckpt_every_steps: int = 0
    stop_after_passes: int | None = None
    max_passes: int | None = None
    max_batches_per_pass: int | None = None
    aggregator: str = "auto"
    allow_deferred: bool = False
    num_threads: int | None = None
    split_seed: int = 42
    allow_unfrozen: bool = False  # PILOT ONLY: train without the freeze record (never reportable)

    def run_fields(self) -> dict:
        """Fields that change the trained weights (hashed; resume requires equality).

        ``allow_unfrozen`` is deliberately absent: it does not change the weights, and the
        checkpoint records whether the run STARTED under the freeze (prereg_freeze_first).
        """
        keys = ("config", "model_seed", "recipe", "amp", "lr", "weight_decay", "min_delta",
                "hidden", "max_passes", "max_batches_per_pass", "split_seed")
        return {k: getattr(self, k) for k in keys}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_model(channels: tuple[str, ...], hidden: int = 128) -> OlsenBiGRU:
    skip = [i for i, c in enumerate(channels) if c == "SPO2"]
    return OlsenBiGRU(c_in=len(channels), hidden=hidden, skip_norm_channels=skip)


def _to_device(bd: dict, dev: torch.device) -> tuple[torch.Tensor, ...]:
    nb = dev.type == "cuda"
    x = torch.from_numpy(bd["x"]).to(dev, non_blocking=nb)
    valid = torch.from_numpy(bd["valid"]).to(dev, non_blocking=nb)
    y = torch.from_numpy(bd["y"]).to(dev, non_blocking=nb).float()
    m = torch.from_numpy(bd["mask"]).to(dev, non_blocking=nb)
    return x, valid, y, m


def predict_seconds(
    model: OlsenBiGRU, ps: PackedSplit, win: Windows, device: torch.device, amp: bool,
    batch: int = 512, prefetch: int = 2,
) -> np.ndarray:
    """Per-second probabilities (W, 300) float32 for windows ``win``."""
    model.eval()
    out = np.empty((len(win), 300), dtype=np.float32)
    with torch.no_grad():
        for _, sel, bd in iter_batches(ps, win, batch, None, prefetch=prefetch):
            x, valid, _, _ = _to_device(bd, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x, valid)
            out[sel] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def score_split(
    model: OlsenBiGRU, ps: PackedSplit, device: torch.device, amp: bool, batch: int = 512,
    prefetch: int = 2, return_seconds: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """Sleep-epoch scores for every aggregator on the fixed evaluation grid.

    Returns subject_id, epoch_idx, apnoea_label, stage, p_mean, p_max, p_k10, p_c10,
    sorted by (subject_id, epoch_idx); asserts every sleep epoch is scored exactly once.
    With ``return_seconds`` returns ``(df, sec30)``: the per-second probabilities, float32
    (n_sleep_epochs, 30), row-aligned with ``df``.
    """
    win = eval_windows(ps)
    sec = predict_seconds(model, ps, win, device, amp, batch, prefetch)
    df = centre_epochs(ps, win, sec)
    sec30 = df.attrs.pop("sec")
    keep = df["sleep"].to_numpy(bool)
    df = df[keep].drop(columns=["sleep"]).reset_index(drop=True)
    sec30 = sec30[keep]
    for r in AGGREGATORS:
        df[f"p_{r}"] = sec_to_epoch(sec30, r)
    key = sleep_epoch_key(ps)
    if not (
        np.array_equal(df["subject_id"].to_numpy(np.int64), key["subject_id"].to_numpy(np.int64))
        and np.array_equal(df["epoch_idx"].to_numpy(np.int64), key["epoch_idx"].to_numpy(np.int64))
    ):
        raise AssertionError("evaluation grid did not score every sleep epoch exactly once")
    if return_seconds:
        sec30 = np.ascontiguousarray(sec30, dtype=np.float32)
        if sec30.shape != (len(df), 30):
            raise AssertionError(f"per-second array {sec30.shape} is not row-aligned with {len(df)} epochs")
        return df, sec30
    return df


def load_model_from_ckpt(path: str | Path, device: torch.device) -> tuple[OlsenBiGRU, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["settings"]
    model = build_model(tuple(ck["channels"]), hidden=int(st["hidden"]))
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _all_finite(tensors) -> bool:
    ts = [t for t in tensors if t is not None and torch.is_floating_point(t)]
    return bool(torch.stack([torch.isfinite(t).all() for t in ts]).all()) if ts else True


DATA_VERSION_KEYS = ("frontend_version", "stage_b_version", "edr_method")


def _data_versions(ps: PackedSplit) -> dict:
    """Front-end / stage-B / EDR versions from a packed split's MANIFEST."""
    return {k: ps.manifest.get(k) for k in DATA_VERSION_KEYS}


UNFROZEN_MSG = ("production training needs the frozen pre-registration record "
                "prereg/aim2_dl_olsen_v1_freeze.json; pass --allow-unfrozen for pilot runs (never reportable)")


class Trainer:
    """See module docstring. ``val_dir`` may be None only for ``smoke()``."""

    def __init__(
        self,
        settings: TrainSettings,
        train_dir: str | Path,
        val_dir: str | Path | None,
        split_json: str | Path,
        ckpt_dir: str | Path,
        out_dir: str | Path,
        mirror_dir: str | Path | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        s = settings
        if s.config not in CONFIGS:
            raise ValueError(f"config must be one of {sorted(CONFIGS)}")
        if s.config in DEFERRED_CONFIGS and not s.allow_deferred:
            raise ValueError(f"config {s.config} is {DEFERRED_CONFIGS[s.config]}")
        if s.recipe not in RECIPES:
            raise ValueError(f"recipe must be one of {sorted(RECIPES)}")
        if s.aggregator != "auto" and s.aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be 'auto' or one of {AGGREGATORS}")
        self.s = s
        self.log = log
        self.channels = CONFIGS[s.config]
        self.recipe = RECIPES[s.recipe]
        self.max_passes = int(s.max_passes or self.recipe.max_passes)
        self.split = load_split(split_json)
        self.train = PackedSplit(train_dir, self.channels)
        if self.train.split != "train":
            raise TestDataRefused(f"train_dir manifest split is {self.train.split!r}, expected 'train'")
        assert_ids_in_split(self.train.subject_ids, self.split, "train")
        self.val = None
        if val_dir is not None:
            self.val = PackedSplit(val_dir, self.channels)
            if self.val.split != "val":
                raise TestDataRefused(f"val_dir manifest split is {self.val.split!r}, expected 'val'")
            assert_ids_in_split(self.val.subject_ids, self.split, "val")
            if np.intersect1d(self.train.subject_ids, self.val.subject_ids).size:
                raise ValueError("train and val caches overlap")
        for ps in filter(None, (self.train, self.val)):
            ms = ps.manifest.get("split_sha256")
            if ms is not None and ms != self.split["sha256"]:
                raise ValueError(f"{ps.root} was packed from a different split file")
        self.ckpt_dir = Path(ckpt_dir)
        self.out_dir = Path(out_dir)
        self.mirror_dir = Path(mirror_dir) if mirror_dir else None
        for d in filter(None, (self.ckpt_dir, self.out_dir, self.mirror_dir)):
            d.mkdir(parents=True, exist_ok=True)

        self.device = resolve_device(s.device)
        self.amp = bool(s.amp and self.device.type in ("cuda", "mps"))
        if s.num_threads:
            torch.set_num_threads(int(s.num_threads))
        torch.manual_seed(s.model_seed)
        random.seed(s.model_seed)
        np.random.seed(s.model_seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
        self.model = build_model(self.channels, s.hidden).to(self.device)
        self.n_params = count_parameters(self.model)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=s.lr, weight_decay=s.weight_decay)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="max", factor=0.1, patience=self.recipe.lr_patience,
            threshold=s.min_delta, threshold_mode="abs",
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)

        data_sig = {
            "train": {k: v for k, v in self.train.manifest["sha256"].items()
                      if not k.startswith("signals_") or any(f"_{c}_" in k for c in self.channels)},
            "val": ({k: v for k, v in self.val.manifest["sha256"].items()
                     if not k.startswith("signals_") or any(f"_{c}_" in k for c in self.channels)}
                    if self.val else None),
            "split_sha256": self.split["sha256"],
        }
        payload = json.dumps({"run": s.run_fields(), "data": data_sig}, sort_keys=True)
        self.config_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.data_versions = {"train": _data_versions(self.train),
                              "val": _data_versions(self.val) if self.val else None}
        self.prereg_freeze: dict | None = None  # current freeze summary; set by run()
        self.state = {
            "pass_idx": 0, "step_in_pass": 0, "best_auc": -np.inf, "best_pass": -1,
            "bad_passes": 0, "history": [], "done": False, "stop_reason": "",
            "loss_sum": 0.0, "n_loss": 0, "nonfinite": 0, "consec_nonfinite": 0, "scaler_skipped": 0,
            "pass_t_train": 0.0, "pass_windows": 0, "git_commits": [],
        }
        self.last_step_scaler_skipped = False  # set by _step under AMP
        self.best_model: dict | None = None  # CPU copy of the best weights; saved inside last.pt
        self._mirror_pending: dict[str, Path] = {}
        self.mirror_retry_wait_s = 10.0
        self.git = git_commit()

    # -- RNG / checkpoint ----------------------------------------------------
    def _rng_state(self) -> dict:
        st = {"torch": torch.get_rng_state(), "python": random.getstate(), "numpy": np.random.get_state()}
        if torch.cuda.is_available():
            st["cuda"] = torch.cuda.get_rng_state_all()
        if self.device.type == "mps":
            st["mps"] = torch.mps.get_rng_state()
        return st

    def _set_rng_state(self, st: dict) -> None:
        torch.set_rng_state(st["torch"])
        random.setstate(st["python"])
        np.random.set_state(st["numpy"])
        if "cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda"])
        if "mps" in st and self.device.type == "mps":
            torch.mps.set_rng_state(st["mps"])

    def _payload(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scheduler": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": json.loads(json.dumps(self.state, default=float)),
            "rng": self._rng_state(),
            "config_hash": self.config_hash,
            "best_model": self.best_model,
            "settings": asdict(self.s),
            "channels": list(self.channels),
            "n_params": self.n_params,
            "git_commit": self.git,
            "git_commits": list(self.state.get("git_commits") or []),
            "prereg_freeze": self.prereg_freeze,
            "allow_unfrozen": self._unfrozen(),
            "data_versions": self.data_versions,
            "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # -- pre-registration freeze -----------------------------------------------
    def _unfrozen(self) -> bool:
        """True if this or any earlier invocation of the run passed ``allow_unfrozen``."""
        return bool(self.s.allow_unfrozen or self.state.get("allow_unfrozen_ever"))

    def _check_freeze(self) -> dict | None:
        """Current freeze summary. Refuses without one unless ``allow_unfrozen`` (then None)."""
        try:
            return prereg.freeze_summary()
        except PreregMismatch as exc:
            if self.s.allow_unfrozen:
                self.log(f"WARNING: PILOT run without the pre-registration freeze ({exc}); never reportable")
                return None
            raise PreregMismatch(f"{UNFROZEN_MSG} ({exc})") from exc

    def _save(self, name: str, payload: dict) -> None:
        if not _all_finite(payload["model"].values()):
            raise FloatingPointError(
                f"non-finite model weights; {name} not written (the previous checkpoint is kept). "
                "Per the pre-declared rule rerun ALL seeds of this config with --no-amp.")
        dst = self.ckpt_dir / name
        tmp = dst.with_name(dst.name + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, dst)
        self._mirror(dst)

    def _mirror(self, src: Path) -> None:
        """Copy ``src`` to the mirror; on an I/O error log it and retry on the next save."""
        if self.mirror_dir is None:
            return
        try:
            if self.mirror_dir.resolve() == src.parent.resolve():
                return
        except OSError:
            pass
        self._mirror_pending[src.name] = src
        self._flush_mirror()

    def _flush_mirror(self) -> bool:
        for name, src in list(self._mirror_pending.items()):
            try:
                _copy_atomic(src, self.mirror_dir / name)
                del self._mirror_pending[name]
            except OSError as exc:
                self.log(f"WARNING: mirroring {name} to {self.mirror_dir} failed ({exc!r}); "
                         "the local copy is intact, retrying on the next save")
        return not self._mirror_pending

    @staticmethod
    def _progress_of(ck: dict) -> tuple:
        st = ck["state"]
        return (bool(st["done"]), int(st["pass_idx"]), int(st["step_in_pass"]))

    @classmethod
    def _progress(cls, path: Path) -> tuple:
        return cls._progress_of(torch.load(path, map_location="cpu", weights_only=False))

    def _sync_from_mirror(self) -> None:
        if self.mirror_dir is None:
            return
        m, loc = self.mirror_dir / "last.pt", self.ckpt_dir / "last.pt"
        if not m.exists():
            return
        mck = torch.load(m, map_location="cpu", weights_only=False)
        if mck["config_hash"] != self.config_hash:
            ms = mck.get("settings", {})
            raise ValueError(
                f"mirror {m} has config_hash {mck['config_hash']} != {self.config_hash} (it belongs to "
                f"config {ms.get('config')} model_seed {ms.get('model_seed')}); wrong --mirror-dir? "
                "Nothing was copied.")
        if not loc.exists() or self._progress_of(mck) > self._progress(loc):
            for name in ("last.pt", "best.pt"):
                if (self.mirror_dir / name).exists():
                    _copy_atomic(self.mirror_dir / name, self.ckpt_dir / name)
            self.log(f"resume: copied checkpoints from mirror {self.mirror_dir}")

    def maybe_resume(self) -> bool:
        self._sync_from_mirror()
        path = self.ckpt_dir / "last.pt"
        if not path.exists():
            return False
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["config_hash"] != self.config_hash:
            raise ValueError(
                f"{path} has config_hash {ck['config_hash']} != {self.config_hash}; "
                "use a new --ckpt-dir for a different configuration or data"
            )
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        self.sched.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.state = ck["state"]
        if not isinstance(self.state.get("git_commits"), list):  # a checkpoint from before the commit history
            self.state["git_commits"] = [str(ck.get("git_commit") or "unknown")]
        self.best_model = ck.get("best_model")
        self._set_rng_state(ck["rng"])
        self.log(f"resumed from {path}: pass {self.state['pass_idx']}, step {self.state['step_in_pass']}, "
                 f"best AUC {self.state['best_auc']:.4f}")
        return True

    # -- training ---------------------------------------------------------------
    def _step(self, bd: dict) -> float:
        """One optimiser step. Returns the loss; a non-finite value means the step was skipped."""
        x, valid, y, m = _to_device(bd, self.device)
        self.model.train()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            logits = self.model(x, valid)
        le = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
        mf = m.float()
        loss = (le * mf).sum() / mf.sum().clamp(min=1.0)
        self.opt.zero_grad(set_to_none=True)
        if self.amp:  # GradScaler skips the step itself when the scaled gradients are inf / NaN
            scale_before = self.scaler.get_scale()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            # update() lowers the scale exactly when it found inf / NaN gradients, i.e. skipped the step
            self.last_step_scaler_skipped = bool(self.scaler.get_scale() < scale_before)
            return float(loss.detach().item())
        loss_v = float(loss.detach().item())
        if not np.isfinite(loss_v):
            return loss_v  # no backward, no step: the weights are untouched
        loss.backward()
        if not _all_finite(p.grad for p in self.model.parameters()):
            self.opt.zero_grad(set_to_none=True)
            return float("nan")  # fp32 gradient overflow: step skipped, counted as non-finite
        self.opt.step()
        return loss_v

    def _account(self, loss: float, scaler_skipped: bool = False) -> None:
        """Count one training step. It is a non-finite training step when its loss is non-finite or
        no update was applied because of non-finite gradients (without AMP ``_step`` then returns
        NaN; with AMP the GradScaler skipped it). Only an applied step resets the consecutive count;
        20 in a row raise the NaN fallback. The totals ``nonfinite`` and ``scaler_skipped`` stay separate."""
        st = self.state
        if scaler_skipped:
            st["scaler_skipped"] = int(st.get("scaler_skipped", 0)) + 1
        finite = bool(np.isfinite(loss))
        if finite:
            st["loss_sum"] += loss
            st["n_loss"] += 1
        else:
            st["nonfinite"] += 1  # a skipped step (loss or, without AMP, gradients non-finite)
        if finite and not scaler_skipped:  # the optimiser step was applied
            st["consec_nonfinite"] = 0
            return
        st["consec_nonfinite"] += 1
        if st["consec_nonfinite"] >= 20:
            raise FloatingPointError(
                "20 consecutive non-finite training steps (non-finite loss, or non-finite gradients: the step was "
                "skipped by the GradScaler or, without AMP, by the fp32 check); per the pre-declared rule rerun "
                "ALL seeds of this config with --no-amp (same precision for every seed)"
            )

    def train_pass(self) -> None:
        s, st = self.s, self.state
        p = int(st["pass_idx"])
        win = train_window_starts(self.train, s.model_seed, p, self.recipe.subsample)
        order = pass_order(len(win), s.model_seed, p)
        batch = self.recipe.batch
        limit = s.max_batches_per_pass
        t0 = time.perf_counter()
        for b, sel, bd in iter_batches(self.train, win, batch, order, start_batch=int(st["step_in_pass"]),
                                       prefetch=s.prefetch, drop_singletons=True):
            if limit is not None and b >= limit:
                break
            self.last_step_scaler_skipped = False
            loss = self._step(bd)
            self._account(loss, scaler_skipped=self.last_step_scaler_skipped)
            st["step_in_pass"] = b + 1
            st["pass_windows"] += len(sel)
            if s.ckpt_every_steps and (b + 1) % s.ckpt_every_steps == 0:
                st["pass_t_train"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                self._save("last.pt", self._payload())
        st["pass_t_train"] += time.perf_counter() - t0
        st["_n_windows_pass"] = int(len(win))

    def validate(self) -> tuple[float, dict]:
        if self.val is None:
            raise RuntimeError("no validation split")
        df = score_split(self.model, self.val, self.device, self.amp, self.s.eval_batch, self.s.prefetch)
        y = df["apnoea_label"].to_numpy()
        aucs = {r: float(roc_auc_score(y, df[f"p_{r}"])) for r in AGGREGATORS}
        return aucs["mean"], aucs

    def end_pass(self) -> None:
        st = self.state
        p = int(st["pass_idx"])
        t0 = time.perf_counter()
        auc, aucs = self.validate()
        t_val = time.perf_counter() - t0
        lr_before = self.opt.param_groups[0]["lr"]
        self.sched.step(auc)
        improved = auc > st["best_auc"] + self.s.min_delta
        if improved:
            st["best_auc"], st["best_pass"], st["bad_passes"] = float(auc), p, 0
            self.best_model = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}
        else:
            st["bad_passes"] += 1
        row = {
            "pass": p,
            "lr": lr_before,
            "train_loss": st["loss_sum"] / max(st["n_loss"], 1),
            "n_batches": st["n_loss"],
            "n_windows_pass": st.get("_n_windows_pass"),
            "n_windows_seen": st["pass_windows"],
            "nonfinite_total": st["nonfinite"],
            "scaler_skipped_total": int(st.get("scaler_skipped", 0)),
            **{f"val_auc_{r}": v for r, v in aucs.items()},
            "improved": bool(improved),
            "best_auc": st["best_auc"],
            "t_train_s": round(st["pass_t_train"], 2),
            "t_val_s": round(t_val, 2),
            "train_windows_per_s": round(st["pass_windows"] / max(st["pass_t_train"], 1e-9), 2),
        }
        st["history"].append(row)
        self.log(f"pass {p}: loss {row['train_loss']:.4f}  val AUC(mean) {auc:.4f}  "
                 f"best {st['best_auc']:.4f}@{st['best_pass']}  lr {lr_before:.1e}  "
                 f"{row['train_windows_per_s']:.0f} win/s")
        st.update({"pass_idx": p + 1, "step_in_pass": 0, "loss_sum": 0.0, "n_loss": 0,
                   "pass_t_train": 0.0, "pass_windows": 0})
        st.pop("_n_windows_pass", None)
        if st["bad_passes"] >= self.recipe.stop_patience:
            st["done"], st["stop_reason"] = True, f"early stop: {st['bad_passes']} passes without improvement"
        elif st["pass_idx"] >= self.max_passes:
            st["done"], st["stop_reason"] = True, f"pass cap {self.max_passes}"
        self._save("last.pt", self._payload())  # carries best_model, so it is self-consistent
        if improved:
            self._save("best.pt", self._payload())
        _atomic_write_text(self.out_dir / "history.csv", pd.DataFrame(st["history"]).to_csv(index=False))
        self._mirror(self.out_dir / "history.csv")

    def run(self) -> str:
        """Train (resuming if possible) and finalise. Returns 'done' or 'paused'."""
        if self.val is None:
            raise RuntimeError("run() needs a validation split")
        self.prereg_freeze = self._check_freeze()  # refuses before any checkpoint is read or written
        resumed = self.maybe_resume()
        st = self.state
        if not resumed:  # a run starting from pass 0 / step 0 records the freeze it started under
            st["prereg_freeze_first"] = self.prereg_freeze
        else:  # never overwritten on resume; a checkpoint without the record counts as unfrozen
            st.setdefault("prereg_freeze_first", None)
            first, cur = st["prereg_freeze_first"], self.prereg_freeze
            if not self.s.allow_unfrozen and (first is None or st.get("allow_unfrozen_ever")):
                why = ("did not start under the frozen pre-registration (an --allow-unfrozen pilot or a run "
                       "started before the freeze)" if first is None else "was resumed by an --allow-unfrozen run")
                raise PreregMismatch(
                    f"{self.ckpt_dir / 'last.pt'} {why}: this checkpoint can never be reported "
                    "(predict_aim2_dl.py refuses it for test), so production training will not continue it. "
                    "Use a fresh --ckpt-dir and --mirror-dir for the production run, or pass --allow-unfrozen "
                    "to continue it as a pilot")
            if first is not None and cur is not None and first.get("body_sha256") != cur.get("body_sha256"):
                self.log(f"WARNING: this run started under frozen body {first.get('body_sha256', '')[:12]} but "
                         f"the freeze record now says {cur.get('body_sha256', '')[:12]}")
        new_record = False
        if self.s.allow_unfrozen and not st.get("allow_unfrozen_ever"):
            st["allow_unfrozen_ever"] = True
            new_record = True
        commits = st.setdefault("git_commits", [])
        if self.git not in commits:  # every invocation's code commit; never cleared or overwritten
            commits.append(self.git)
            new_record = True
        if resumed and new_record:
            # Saved now, while the model still holds last.pt's weights: an invocation that runs no pass (a
            # re-finalise) would otherwise record its commit / flag only in metrics.json, and a later
            # re-finalise from last.pt would erase it. Same weights, optimiser, scheduler and RNG states.
            self._save("last.pt", self._payload())
        n_this = 0
        while not self.state["done"]:
            self.train_pass()
            self.end_pass()
            n_this += 1
            if self.s.stop_after_passes is not None and n_this >= self.s.stop_after_passes and not self.state["done"]:
                self.log(f"paused after {n_this} pass(es) this invocation; re-run to resume")
                return "paused"
        self.finalize()
        return "done"

    # -- post-processing ------------------------------------------------------
    def _ensure_best_ckpt(self) -> tuple[Path, bool]:
        """best.pt must hold the best weights recorded in last.pt; rewrite it if not.

        best.pt and last.pt are separate files mirrored seconds apart, so after a crash or
        a partial Drive upload best.pt can be stale. Returns (path, rewritten).
        """
        best = self.ckpt_dir / "best.pt"
        st = self.state
        ck = torch.load(best, map_location="cpu", weights_only=False) if best.exists() else None
        same_state = ck is not None and (int(ck["state"]["best_pass"]) == int(st["best_pass"])
                                         and float(ck["state"]["best_auc"]) == float(st["best_auc"]))
        if self.best_model is None:  # last.pt from a trainer that did not store best_model
            if not same_state:
                raise RuntimeError(f"{best} does not match the loop state (best pass {st['best_pass']}) and "
                                   "last.pt holds no best weights; cannot finalise safely")
            return best, False
        same_weights = same_state and ck["model"].keys() == self.best_model.keys() and all(
            torch.equal(ck["model"][k].cpu(), v) for k, v in self.best_model.items())
        if same_weights:
            return best, False
        self.log(f"WARNING: {best} does not hold the best weights recorded in last.pt (best pass "
                 f"{st['best_pass']}, found {ck['state']['best_pass'] if ck else 'no file'}); "
                 "rewriting best.pt from last.pt")
        self._save("best.pt", {**self._payload(), "model": self.best_model, "rewritten_from_last": True})
        return best, True

    def finalize(self) -> dict:
        best, rewritten = self._ensure_best_ckpt()
        _atomic_write_text(self.out_dir / "history.csv", pd.DataFrame(self.state["history"]).to_csv(index=False))
        ck = torch.load(best, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        df = score_split(self.model, self.val, self.device, self.amp, self.s.eval_batch, self.s.prefetch)
        y = df["apnoea_label"].to_numpy()
        aucs = {r: float(roc_auc_score(y, df[f"p_{r}"])) for r in AGGREGATORS}
        recheck = abs(aucs["mean"] - float(self.state["best_auc"]))
        if recheck > 1e-3:
            self.log(f"WARNING: best.pt scores val AUC(mean) {aucs['mean']:.5f} but the loop recorded "
                     f"{self.state['best_auc']:.5f} (|diff| {recheck:.2e})")
        rule = choose_aggregator(aucs)
        agg = rule["chosen"] if self.s.aggregator == "auto" else self.s.aggregator
        thr, f1 = threshold_f1max(y, df[f"p_{agg}"].to_numpy())
        p = df[f"p_{agg}"].to_numpy(np.float64)
        val_pred = pd.DataFrame({
            "subject_id": df["subject_id"].to_numpy(np.int32),
            "epoch_idx": df["epoch_idx"].to_numpy(np.int32),
            "apnoea_label": df["apnoea_label"].to_numpy(np.int8),
            "pred_prob": p,
            "pred_label": (p > thr).astype(np.int64),
            **{f"p_{r}": df[f"p_{r}"].to_numpy(np.float64) for r in AGGREGATORS},
        })
        val_pred.to_parquet(self.out_dir / "val_predictions.parquet", index=False)
        best_sha = sha256_file(best)
        postproc = {
            "aggregator": agg,
            "aggregator_source": "pre-declared rule on this run's validation predictions"
            if self.s.aggregator == "auto" else "frozen choice passed with --aggregator",
            "aggregator_rule": rule,
            "threshold": thr,
            "threshold_rule": "F1-max over np.linspace(0.05, 0.95, 91), strict >, first max",
            "threshold_grid": [float(THRESHOLDS[0]), float(THRESHOLDS[-1]), len(THRESHOLDS)],
            "val_f1_at_threshold": f1,
            "fitted_on": "validation sleep epochs",
            "device": str(self.device),
            "amp": self.amp,
            "best_ckpt_sha256": best_sha,
        }
        _atomic_write_text(self.out_dir / "postproc.json", json.dumps(postproc, indent=2))
        tr_mask = self.train.epochs["sleep"].to_numpy(bool)
        st = self.state
        metrics = {
            "model": "OlsenBiGRU",
            "config": self.s.config,
            "channels": list(self.channels),
            "model_seed": self.s.model_seed,
            "split_seed": self.s.split_seed,
            "recipe": asdict(self.recipe),
            "optimizer": {"name": "AdamW", "lr": self.s.lr, "weight_decay": self.s.weight_decay},
            "loss": "per-second BCEWithLogits, unweighted, masked to sleep seconds inside the night",
            "scale_pos_weight": 1.0,
            "schedule": "ReduceLROnPlateau(max, factor 0.1) on val epoch AUC (mean aggregator)",
            "schedule_facts": {**schedule_facts(self.recipe), "max_passes": self.max_passes},
            "schedule_note": SCHEDULE_NOTE,
            "n_params": self.n_params,
            "passes_run": int(st["pass_idx"]),
            "best_pass": int(st["best_pass"]),
            "best_val_auc_mean_agg": float(st["best_auc"]),
            "stop_reason": st["stop_reason"],
            "val_auc_by_aggregator_best_ckpt": aucs,
            "aggregator": agg,
            "threshold": thr,
            "val_f1_at_threshold": f1,
            "n_train_subjects": int(self.train.n_subjects),
            "n_train_sleep_epochs": int(tr_mask.sum()),
            "n_val_subjects": int(self.val.n_subjects),
            "n_val_sleep_epochs": int(len(df)),
            "train_epoch_prevalence_sleep": float(self.train.epochs["apnoea_label"].to_numpy()[tr_mask].mean()),
            "nonfinite_losses": int(st["nonfinite"]),
            "scaler_skipped_steps": int(st.get("scaler_skipped", 0)),
            "skipped_steps_note": "nonfinite_losses: steps skipped because the loss (or, without AMP, a gradient) "
                                  "was non-finite; scaler_skipped_steps: AMP steps the GradScaler skipped because "
                                  "the scaled gradients overflowed (the loss may be finite; a non-finite loss "
                                  "under AMP counts in both). Isolated skipped steps trigger nothing; the NaN "
                                  "fallback is triggered by 20 consecutive non-finite training steps (steps of "
                                  "either kind, with no applied step in between).",
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else None,
            "amp": self.amp,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "determinism_note": "cuDNN GRU kernels are not bitwise deterministic; CPU resume is (tested).",
            "train_seconds_total": float(sum(h["t_train_s"] for h in st["history"])),
            "val_seconds_total": float(sum(h["t_val_s"] for h in st["history"])),
            "config_hash": self.config_hash,
            "split_json": self.split["path"],
            "split_sha256": self.split["sha256"],
            "train_manifest_split_sha256": self.train.manifest.get("split_sha256"),
            "frontend_version": self.train.manifest.get("frontend_version"),
            "stage_b_version": self.train.manifest.get("stage_b_version"),
            "edr_method": self.train.manifest.get("edr_method"),
            "data_versions": self.data_versions,
            "prereg_freeze": self.prereg_freeze,
            "prereg_freeze_first": st.get("prereg_freeze_first"),
            "allow_unfrozen": self._unfrozen(),
            "git_commit": self.git,
            "git_commits": list(st.get("git_commits") or []),
            "best_ckpt_sha256": best_sha,
            "best_ckpt_rewritten_from_last": rewritten,
            "best_auc_recheck_abs_diff": recheck,
            "settings": asdict(self.s),
        }
        _atomic_write_text(self.out_dir / "metrics.json", json.dumps(metrics, indent=2, default=float))
        # the final set, including files a failed mirror of an earlier invocation left behind
        for src in (self.ckpt_dir / "last.pt", best, *(self.out_dir / n for n in (
                "history.csv", "val_predictions.parquet", "postproc.json", "metrics.json"))):
            if src.exists():
                self._mirror(src)
        for _ in range(3):
            if self._flush_mirror():
                break
            time.sleep(self.mirror_retry_wait_s)
        if self._mirror_pending:
            raise RuntimeError(
                f"finalised locally (ckpt {self.ckpt_dir}, outputs {self.out_dir}) but mirroring "
                f"{sorted(self._mirror_pending)} to {self.mirror_dir} still fails; re-run to re-mirror")
        self.log(f"final: val AUC {aucs}; aggregator {agg}; threshold {thr:.2f} (F1 {f1:.4f})")
        return metrics

    # -- smoke ----------------------------------------------------------------------
    def smoke(self, n_batches: int = 50) -> dict:
        """Train ``n_batches`` steps from scratch (no checkpoints); report loss trend."""
        losses, t_steps = [], []
        p = 0
        while len(losses) < n_batches:
            win = train_window_starts(self.train, self.s.model_seed, p, 1.0)
            order = pass_order(len(win), self.s.model_seed, p)
            for _, _, bd in iter_batches(self.train, win, self.recipe.batch, order,
                                         prefetch=self.s.prefetch, drop_singletons=True):
                t0 = time.perf_counter()
                losses.append(self._step(bd))
                if self.device.type == "mps":
                    torch.mps.synchronize()
                t_steps.append(time.perf_counter() - t0)
                if len(losses) >= n_batches:
                    break
            p += 1
        k = max(1, min(10, n_batches // 5))
        first, last = float(np.mean(losses[:k])), float(np.mean(losses[-k:]))
        steady = t_steps[3:] if len(t_steps) > 6 else t_steps
        return {
            "device": str(self.device), "amp": self.amp, "batch": self.recipe.batch,
            "n_batches": len(losses), "losses": losses,
            f"loss_first{k}_mean": first, f"loss_last{k}_mean": last, "loss_fell": last < first,
            "sec_per_batch_median": float(np.median(steady)),
            "train_windows_per_s": float(self.recipe.batch / np.median(steady)),
            "passes_touched": p, "n_params": self.n_params,
        }
