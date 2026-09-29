"""Aim 3 — LightGBM 5-class sleep-staging baseline (task T5).

Classical-ML arm of Aim 3. Labels every 30-s epoch W / N1 / N2 / N3 / REM
from the per-epoch features already stored in the SHHS-1 parquets, under
three modality configurations:

  eeg_only         EEG spectral features only (12 base x 7 context = 84 cols)
  cardioresp_only  everything except EEG (SpO2, HR/HRV, ECG bands, CPC proxy,
                   respiratory effort/airflow, position)
  multimodal       union of the two

Stage-label leakage
-------------------
Four feature families encode the stage label and are dropped from EVERY
config, together with their contextual derivatives:

  odi3_count, odi4_count  forced to 0 on non-sleep epochs (spo2_features sleep_mask)
  desat_depth             forced to NaN on non-sleep epochs (same mask)
  hypoxic_burden          built from technician-scored respiratory events,
                          which are only scored during sleep

Keeping them would let the model find wake from the missing-value pattern.
Time-of-night (epoch_idx, epoch_start_sec) is also excluded.

Protocol
--------
* Epoch window per subject: 30 min before first sleep -> 30 min after last
  sleep (the Sleep-EDF in-bed convention of Supratak et al. 2017, applied to
  SHHS). Unscored '?' epochs dropped. Training and early stopping use this
  window only. The same model is ALSO evaluated on every scored epoch of the
  test subjects ("whole_night" in metrics.json), which is the protocol of
  whole-night SHHS studies; report both.
* Outer split: GroupShuffleSplit(test_size=0.2, random_state=42) over the
  sorted unique subject IDs. GroupShuffleSplit depends only on the unique
  group list, so the test subjects are IDENTICAL to every Aim 2 run; this is
  asserted against an Aim 2 recovery test_predictions file.
* Inner 95/5 subject holdout for early stopping. Test set touched once.
* Fixed LightGBM hyperparameters (same as Aim 2 FIXED_PARAMS), multiclass.
* Metrics: accuracy, Cohen's kappa, macro-F1, per-class F1, confusion
  matrix, subject-level bootstrap 95% CIs, per-subject kappa distribution.

Outputs: models/aim3_staging_v1/<config>/{metrics.json, test_predictions.parquet,
         bootstrap_subject.npz, feature_list.json, feature_importance.json}
         models/aim3_staging_v1/summary.json

Usage:
  python scripts/fit_aim3_staging_baseline.py                      # all configs
  python scripts/fit_aim3_staging_baseline.py --configs eeg_only
  python scripts/fit_aim3_staging_baseline.py --max-subjects 200   # smoke test
"""
from __future__ import annotations

import datetime
import io
import json
import os
import platform
import subprocess
import re
import sys
import tarfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

import click
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.model_selection import GroupShuffleSplit

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

FEATURES_DIR = CODE_ROOT / "features"
OUT_ROOT = CODE_ROOT / "models" / "aim3_staging_v1"
AIM2_REFERENCE_PREDS = (
    CODE_ROOT / "models" / "recovery_2026-05-03" / "aim2_v6_past_only" / "test_predictions.parquet"
)

STAGES = ("W", "N1", "N2", "N3", "REM")
STAGE_TO_INT = {s: i for i, s in enumerate(STAGES)}
N_CLASSES = len(STAGES)
TRIM_EPOCHS = 60  # 30 min either side of the sleep period

NON_FEATURE = {
    "subject_id", "cohort", "epoch_idx", "epoch_start_sec",
    "sleep_stage", "apnoea_label", "features_version",
}
# Features whose values are a function of the stage label (see module docstring).
STAGE_LEAK_PREFIXES = ("odi3_count", "odi4_count", "desat_depth", "hypoxic_burden")

CONFIGS = ("eeg_only", "cardioresp_only", "multimodal")

# Same as Aim 2 FIXED_PARAMS, with the objective switched to multiclass.
FIXED_PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "learning_rate": 0.1,
    "num_leaves": 48,
    "min_child_samples": 100,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
FIXED_N_ROUNDS = 800
FIXED_EARLY_STOPPING = 50


# --------------------------------------------------------------------------- features

def select_features(all_cols: list[str], config: str) -> list[str]:
    candidates = [
        c for c in all_cols
        if c not in NON_FEATURE and not c.startswith(STAGE_LEAK_PREFIXES)
    ]
    if config == "eeg_only":
        cols = [c for c in candidates if c.startswith("eeg_")]
    elif config == "cardioresp_only":
        cols = [c for c in candidates if not c.startswith("eeg_")]
    elif config == "multimodal":
        cols = candidates
    else:
        raise ValueError(f"unknown config {config!r}")
    leaked = [c for c in cols if c.startswith(STAGE_LEAK_PREFIXES)]
    assert not leaked, f"stage-leaking features selected: {leaked}"
    return cols


# --------------------------------------------------------------------------- feature source

class TarSource:
    """Random access to per-subject parquets inside an UNCOMPRESSED tar of the features dir.

    Each read opens its own file handle at the member's byte offset, so it is
    thread-safe. Used when iCloud has evicted the local parquets; the tar is
    the same features_version snapshot the Colab runs extract.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.members: dict[int, tuple[int, int]] = {}
        with tarfile.open(self.path) as tf:
            for m in tf:
                hit = re.fullmatch(r"(?:.*/)?shhs1-(\d+)\.parquet", m.name)
                if m.isfile() and hit:
                    self.members[int(hit.group(1))] = (m.offset_data, m.size)

    def data(self, sid: int) -> bytes:
        off, size = self.members[sid]
        with open(self.path, "rb") as fh:
            fh.seek(off)
            return fh.read(size)


_TAR: TarSource | None = None  # set in main() when --features-tar is given


def _read(key, columns: list[str]):
    """Read columns for one subject; key is a parquet Path (dir mode) or a subject id (tar mode)."""
    if _TAR is None:
        return pq.read_table(key, columns=columns)
    return pq.read_table(io.BytesIO(_TAR.data(int(key))), columns=columns)


def _schema_names(key) -> list[str]:
    if _TAR is None:
        return pq.read_schema(key).names
    return pq.ParquetFile(io.BytesIO(_TAR.data(int(key)))).schema_arrow.names


def _key_id(key) -> int:
    return int(key) if _TAR is not None else int(Path(key).stem.split("-")[1])


# --------------------------------------------------------------------------- cohort index

def build_index(files: list[Path], version: str) -> list[dict]:
    """Pass 1: per subject, which rows to keep and their stage labels."""
    def one(f: Path) -> dict | None:
        t = _read(f, ["subject_id", "sleep_stage", "features_version"]).to_pandas()
        if t.empty or t["features_version"].iloc[0] != version:
            return None
        st = t["sleep_stage"].astype(str).values
        sleep_idx = np.where(np.isin(st, ["N1", "N2", "N3", "REM"]))[0]
        if sleep_idx.size == 0:
            return None
        lo = max(0, sleep_idx[0] - TRIM_EPOCHS)
        hi = min(len(st), sleep_idx[-1] + TRIM_EPOCHS + 1)
        window = np.arange(lo, hi)
        keep = window[np.isin(st[window], STAGES)]
        keep_all = np.where(np.isin(st, STAGES))[0]
        return {
            "file": f,
            "subject_id": int(t["subject_id"].iloc[0]),
            "rows": keep,
            "y": np.array([STAGE_TO_INT[s] for s in st[keep]], dtype=np.int8),
            "rows_all": keep_all,
            "y_all": np.array([STAGE_TO_INT[s] for s in st[keep_all]], dtype=np.int8),
        }

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = [r for r in ex.map(one, files) if r is not None]
    out.sort(key=lambda r: r["subject_id"])
    return out


def split_subjects(subject_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same outer + inner splits as the Aim 2 scripts, computed on unique subject IDs."""
    subj = np.unique(subject_ids)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_rel, te_rel = next(outer.split(np.zeros(len(subj)), groups=subj))
    tv = subj[tv_rel]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    tr_rel, va_rel = next(inner.split(np.zeros(len(tv)), groups=tv))
    return tv[tr_rel], tv[va_rel], subj[te_rel]


def load_matrix(index_by_subject: dict, ordered_subjects: list[int], cols: list[str],
                rows_key: str = "rows", y_key: str = "y"):
    """Pass 2: fill one preallocated float32 matrix in the given subject order."""
    sizes = np.array([len(index_by_subject[s][rows_key]) for s in ordered_subjects])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    n = int(offsets[-1])
    gb = n * len(cols) * 4 / 1e9
    print(f"  allocating X: {n:,} epochs x {len(cols)} features = {gb:.1f} GB (float32)")
    X = np.empty((n, len(cols)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    sid = np.empty(n, dtype=np.int32)
    eidx = np.empty(n, dtype=np.int32)

    def fill(k: int) -> None:
        r = index_by_subject[ordered_subjects[k]]
        a, b = offsets[k], offsets[k + 1]
        df = _read(r["file"], cols + ["epoch_idx"]).to_pandas()
        X[a:b] = df[cols].to_numpy(dtype=np.float32)[r[rows_key]]
        eidx[a:b] = df["epoch_idx"].to_numpy()[r[rows_key]]
        y[a:b] = r[y_key]
        sid[a:b] = r["subject_id"]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(fill, range(len(ordered_subjects))))
    print(f"  loaded in {time.time()-t0:.0f}s")
    return X, y, sid, eidx


# --------------------------------------------------------------------------- metrics

def cm_metrics(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    n = cm.sum()
    rows, cols = cm.sum(1), cm.sum(0)
    po = np.trace(cm) / n
    pe = float((rows * cols).sum() / n**2)
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    denom = rows + cols
    f1 = np.where(denom > 0, 2 * np.diag(cm) / np.where(denom > 0, denom, 1), np.nan)
    return {"accuracy": float(po), "kappa": float(kappa), "macro_f1": float(np.nanmean(f1)),
            "f1": {s: float(v) for s, v in zip(STAGES, f1)}}


def per_subject_cms(y_true: np.ndarray, y_pred: np.ndarray, sid: np.ndarray):
    subjects, inv = np.unique(sid, return_inverse=True)
    flat = inv * N_CLASSES * N_CLASSES + y_true.astype(np.int64) * N_CLASSES + y_pred
    cms = np.bincount(flat, minlength=len(subjects) * N_CLASSES**2)
    return subjects, cms.reshape(len(subjects), N_CLASSES, N_CLASSES)


def subject_bootstrap(cms: np.ndarray, n_resamples: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n_subj = cms.shape[0]
    out = {k: np.empty(n_resamples) for k in ("accuracy", "kappa", "macro_f1", *[f"f1_{s}" for s in STAGES])}
    for i in range(n_resamples):
        counts = np.bincount(rng.integers(0, n_subj, n_subj), minlength=n_subj)
        m = cm_metrics(np.tensordot(counts, cms, axes=1))
        out["accuracy"][i], out["kappa"][i], out["macro_f1"][i] = m["accuracy"], m["kappa"], m["macro_f1"]
        for s in STAGES:
            out[f"f1_{s}"][i] = m["f1"][s]
    return out


# --------------------------------------------------------------------------- fit

def evaluate(y_true: np.ndarray, probs: np.ndarray, sid: np.ndarray) -> dict:
    """Pooled metrics, subject-bootstrap CIs, per-subject kappa, confusion matrix."""
    pred = probs.argmax(1).astype(np.int8)
    _, cms = per_subject_cms(y_true, pred, sid)
    overall = cm_metrics(cms.sum(0))
    boot = subject_bootstrap(cms)
    subj_kappa = np.array([cm_metrics(c)["kappa"] for c in cms])
    return {
        "test": overall,
        "test_ci95_subject_bootstrap": {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                                        for k, v in boot.items()},
        "per_subject_kappa": {"median": float(np.nanmedian(subj_kappa)),
                              "q25": float(np.nanpercentile(subj_kappa, 25)),
                              "q75": float(np.nanpercentile(subj_kappa, 75))},
        "confusion_matrix": {"labels": list(STAGES), "rows_true_cols_pred": cms.sum(0).tolist()},
        "n_epochs": int(len(y_true)),
        "_boot": boot, "_pred": pred,
    }


def fit_config(config, index_by_subject, all_cols, tr_s, va_s, te_s, out_dir: Path, env: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = select_features(all_cols, config)
    print(f"\n{'='*78}\n=== Config: {config}  ({len(cols)} features)\n{'='*78}")

    order = list(tr_s) + list(va_s) + list(te_s)
    X, y, sid, eidx = load_matrix(index_by_subject, order, cols)
    n_tr = sum(len(index_by_subject[s]["rows"]) for s in tr_s)
    n_va = sum(len(index_by_subject[s]["rows"]) for s in va_s)
    te = slice(n_tr + n_va, len(y))
    print(f"  train {n_tr:,} epochs / {len(tr_s)} subj | val {n_va:,} / {len(va_s)} | "
          f"test {len(y)-n_tr-n_va:,} / {len(te_s)}")
    print("  train stage mix: " + ", ".join(
        f"{s} {100*np.mean(y[:n_tr]==i):.1f}%" for i, s in enumerate(STAGES)))

    dtrain = lgb.Dataset(X[:n_tr], y[:n_tr], feature_name=cols, free_raw_data=True)
    dval = lgb.Dataset(X[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va], reference=dtrain)
    t0 = time.time()
    booster = lgb.train(
        FIXED_PARAMS, dtrain, num_boost_round=FIXED_N_ROUNDS, valid_sets=[dval],
        callbacks=[lgb.early_stopping(FIXED_EARLY_STOPPING, verbose=False), lgb.log_evaluation(50)],
    )
    fit_s = time.time() - t0
    best_iter = int(booster.best_iteration or FIXED_N_ROUNDS)
    print(f"  fit in {fit_s/60:.1f} min (best_iter={best_iter})")

    probs = booster.predict(X[te], num_iteration=best_iter).astype(np.float32)
    y_te, sid_te, eidx_te = y[te], sid[te], eidx[te]
    del X, dtrain, dval
    ev = evaluate(y_te, probs, sid_te)
    pred = ev.pop("_pred")
    np.savez(out_dir / "bootstrap_subject.npz", **ev.pop("_boot"))
    ci = ev["test_ci95_subject_bootstrap"]
    overall = ev["test"]
    print(f"  [trimmed] acc {overall['accuracy']:.4f} {[round(v, 4) for v in ci['accuracy']]}  "
          f"kappa {overall['kappa']:.4f} {[round(v, 4) for v in ci['kappa']]}  "
          f"macro-F1 {overall['macro_f1']:.4f} {[round(v, 4) for v in ci['macro_f1']]}")
    print("  per-class F1: " + ", ".join(f"{s} {overall['f1'][s]:.3f}" for s in STAGES))

    pd.DataFrame({
        "subject_id": sid_te, "epoch_idx": eidx_te,
        "true_stage": np.array(STAGES)[y_te], "pred_stage": np.array(STAGES)[pred],
        **{f"prob_{s}": probs[:, i] for i, s in enumerate(STAGES)},
    }).to_parquet(out_dir / "test_predictions.parquet", index=False)

    # Whole-night sensitivity evaluation: same booster, every scored epoch of the test subjects.
    Xw, yw, sidw, eidxw = load_matrix(index_by_subject, list(te_s), cols, rows_key="rows_all", y_key="y_all")
    probs_w = booster.predict(Xw, num_iteration=best_iter).astype(np.float32)
    del Xw
    ev_w = evaluate(yw, probs_w, sidw)
    pred_w = ev_w.pop("_pred")
    np.savez(out_dir / "bootstrap_subject_whole_night.npz", **ev_w.pop("_boot"))
    pd.DataFrame({
        "subject_id": sidw, "epoch_idx": eidxw,
        "true_stage": np.array(STAGES)[yw], "pred_stage": np.array(STAGES)[pred_w],
        **{f"prob_{s}": probs_w[:, i] for i, s in enumerate(STAGES)},
    }).to_parquet(out_dir / "test_predictions_whole_night.parquet", index=False)
    tw = ev_w["test"]
    print(f"  [whole-night] acc {tw['accuracy']:.4f}  kappa {tw['kappa']:.4f}  macro-F1 {tw['macro_f1']:.4f}  "
          f"({ev_w['n_epochs']:,} epochs)")

    (out_dir / "feature_list.json").write_text(json.dumps(cols, indent=2))
    gain = booster.feature_importance(importance_type="gain")
    top = sorted(zip(cols, gain.tolist()), key=lambda t: -t[1])[:40]
    (out_dir / "feature_importance.json").write_text(json.dumps(top, indent=2))

    metrics = {
        "config": config,
        "model": "lightgbm-multiclass",
        "task": "5-class staging W/N1/N2/N3/REM",
        "epoch_window": f"first sleep - {TRIM_EPOCHS} epochs to last sleep + {TRIM_EPOCHS} epochs; '?' dropped",
        "excluded_stage_leak_prefixes": list(STAGE_LEAK_PREFIXES),
        "n_features": len(cols),
        "n_train_subjects": len(tr_s), "n_val_subjects": len(va_s), "n_test_subjects": len(te_s),
        "n_train_epochs": int(n_tr), "n_test_epochs": int(len(y_te)),
        "best_iter": best_iter, "fit_seconds": fit_s, "fixed_params": FIXED_PARAMS,
        **{k: v for k, v in ev.items() if k != "n_epochs"},
        "whole_night": {"description": "same model, every scored epoch of the test subjects (no wake trim)", **ev_w},
        "env": env,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  saved -> {out_dir}")
    return metrics


def run_environment() -> dict:
    def _git(*args):
        try:
            return subprocess.run(["git", "-C", str(CODE_ROOT), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "lightgbm": lgb.__version__, "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@click.command()
@click.option("--features-version", default="2026-05-01-phase1batch-v1", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True,
              help="Outer-split seed. MUST be 42 to share the Aim 2 test subjects.")
@click.option("--configs", default=",".join(CONFIGS), show_default=True,
              help="Comma-separated subset of: " + ", ".join(CONFIGS))
@click.option("--max-subjects", type=int, default=None,
              help="Smoke test on N subjects drawn only from the canonical train/val pool "
                   "(never held-out test subjects). Not for reporting.")
@click.option("--out-root", type=click.Path(path_type=Path), default=OUT_ROOT, show_default=True)
@click.option("--aim2-reference", type=click.Path(path_type=Path), default=AIM2_REFERENCE_PREDS,
              show_default=True,
              help="Any Aim 2 test_predictions.parquet; its subject set must equal this run's test "
                   "subjects. On Colab, point at the Drive copy under results/.")
@click.option("--features-tar", type=click.Path(exists=True, path_type=Path), default=None,
              help="Read per-subject parquets from an uncompressed tar of the features dir instead of "
                   "features/ (e.g. Code/features-phase1batch-v1.tar when iCloud has evicted files).")
def main(features_version, seed, configs, max_subjects, out_root, aim2_reference, features_tar) -> None:
    global _TAR
    configs = [c.strip() for c in configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            raise click.BadParameter(f"unknown config {c!r}")
    out_root.mkdir(parents=True, exist_ok=True)

    if features_tar is not None:
        _TAR = TarSource(features_tar)
        files = sorted(_TAR.members)
        source = f"tar {features_tar.name}"
    else:
        files = [f for f in sorted(FEATURES_DIR.glob("*.parquet"))
                 if f.name != "subject_metadata.parquet" and not f.name.startswith("._")]
        source = str(FEATURES_DIR)
    canon_te: set[int] = set()
    if max_subjects:
        # Smoke subjects come only from the canonical train/val pool, never the held-out test set.
        all_ids = np.array([_key_id(f) for f in files])
        canon_te = {int(x) for x in split_subjects(all_ids, seed)[2]}
        files = [f for f in files if _key_id(f) not in canon_te][:max_subjects]
    print(f"=== Aim 3 staging baseline ===\n  {len(files)} subjects from {source}, version {features_version}")

    t0 = time.time()
    index = build_index(files, features_version)
    index_by_subject = {r["subject_id"]: r for r in index}
    print(f"  indexed {len(index)} subjects in {time.time()-t0:.0f}s; "
          f"{sum(len(r['rows']) for r in index):,} epochs in window")

    tr_s, va_s, te_s = split_subjects(np.array(list(index_by_subject)), seed)
    if max_subjects is None:
        ref = set(pd.read_parquet(aim2_reference, columns=["subject_id"])["subject_id"].unique().tolist())
        assert set(int(s) for s in te_s) == ref, (
            f"test subjects differ from Aim 2 reference ({len(te_s)} vs {len(ref)})")
        print(f"  outer split matches Aim 2 test subjects exactly ({len(ref)} subjects)")
    else:
        smoke = {int(x) for x in np.concatenate([tr_s, va_s, te_s])}
        assert not (smoke & canon_te), "smoke subset contains canonical held-out test subjects"
        print(f"  SMOKE MODE: {len(smoke)} subjects, all from the canonical train/val pool (not for reporting)")

    all_cols = _schema_names(index[0]["file"])
    env = run_environment()
    print("  env: " + json.dumps(env))
    summary = {"features_version": features_version, "seed": seed, "env": env, "configs": {}}
    for c in configs:
        summary["configs"][c] = fit_config(c, index_by_subject, all_cols, tr_s, va_s, te_s, out_root / c, env)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*78}\n{'config':16s} {'feat':>5s} {'acc':>7s} {'kappa':>7s} {'mF1':>7s}  "
          + " ".join(f"{s:>5s}" for s in STAGES) + "   kappa(whole-night)")
    for c, m in summary["configs"].items():
        t = m["test"]
        print(f"{c:16s} {m['n_features']:>5d} {t['accuracy']:>7.4f} {t['kappa']:>7.4f} {t['macro_f1']:>7.4f}  "
              + " ".join(f"{t['f1'][s]:>5.3f}" for s in STAGES)
              + f"   {m['whole_night']['test']['kappa']:.4f}")


if __name__ == "__main__":
    main()
