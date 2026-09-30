"""Aim 2 — matched-input LightGBM cells for the raw-signal DL comparison (critique #1, design D4).

The Olsen-style DL configs see only some raw channels (M = RR + EDR from the
ECG; P4 = M + thoracic/abdominal belts). Their primary LightGBM comparators are
cells whose inputs are strict subsets of the canonical 217-column physio_only
list (see thesis_pipeline/matched_cells.py):

  physio_only_repro  all 217 columns — reproduction check against the canonical
                     fit (models/recovery_2026-05-03/aim2_v85_taxonomy/physio_only,
                     test AUC 0.8818)
  ecg_only           hr_/hrv_ bases (56 columns)             -> comparator for M
  ecg_belt           ecg_only + thoracic/abdominal belt RMS
                     and their correlation (77 columns)      -> comparator for P4

The cells are matched by modality, not by information content, and the belts are
label-proximal; the declared differences (matched_cells.INPUT_MATCH_CAVEATS) are
written into every run record and must be carried into the pre-registration.

Protocol — identical to scripts/fit_aim2_v85_taxonomy_ablation.py (FIXED_PARAMS,
n_estimators and early stopping and subject_bootstrap are imported from it):
  * features_version 2026-05-01-phase1batch-v1; sleep epochs only (sleep_mask)
  * outer GroupShuffleSplit(test_size=0.2, random_state=42) over subjects;
    inner GroupShuffleSplit(test_size=0.05, random_state=42) of the TV pool for
    early stopping (the 232-subject inner validation set, which is also the DL
    validation set). Frozen in splits/aim2_seed42.json (sha256-verified).
  * rows ordered by subject id then epoch, exactly as the canonical concatenation
  * scale_pos_weight = neg/pos on inner train; threshold = F1-max on inner val
    over np.linspace(0.05, 0.95, 91) with strict '>'
  * test: AUC, AUC-PR, tuned F1/P/R, 1000-resample subject bootstrap (seed 42),
    metrics_extended.json via write_extended_metrics

Feature parquets are read from the uncompressed features tar (TarSource), never
from features/ (iCloud-evicted files hang). The NSRR clinical AHI for the
extended metrics comes from the tar's features/subject_metadata.parquet,
filtered to the subjects being scored.

Test protection
---------------
  --train-only   takes the train / val / test subject ids from the sha256-verified
                 split file and removes the test subjects from the tar member
                 table before any member is read, so no test subject's parquet
                 member is opened (hypnogram, features and labels alike). What
                 is still read that mentions test subjects: their ids (split
                 file, canonical test_predictions.parquet subject_id column, tar
                 header names) to check the split, and the cohort-level
                 subject_metadata table (subject_id, ahi_a0h3a), which is filtered
                 to the validation subjects before use. No test metric is
                 computed. Everything is written to <cell>/val/.
  full mode      physio_only_repro may be scored on test (it re-derives an
                 already-reported number with an unchanged model). ecg_only /
                 ecg_belt refuse to score test without --prereg <vault note>
                 (sha256 logged) AND a saved --train-only run in <cell>/val/.
                 Before any test row is loaded, the refit must reproduce that
                 run: validation predictions within 1e-6 and a byte-identical
                 model file (sha256 of <cell>/test/model_full.txt ==
                 metrics_trainonly.json model_sha256 == sha256 of val/model.txt).
                 A second full run for a cell with outputs in <cell>/test/ is
                 refused unless --rerun-reason is given; the earlier outputs are
                 then moved to <cell>/test/superseded_<UTC>/ and the reason is
                 recorded there and in the new metrics.json.

Outputs: models/aim2_matched_cells_v1/<cell>/ — one split per folder, so a
bootstrap array or metrics_extended.json can never be paired with the wrong split
(scripts/evaluate_aim2_dl.py --ref NAME=<cell>/val with --split val, or
NAME=<cell>/test with --split test):
  val/   (train-only) val_predictions.parquet, bootstrap_aucs_subject.npy,
         bootstrap_auprs_subject.npy, metrics_extended.json (validation subjects),
         metrics_trainonly.json, model.txt, feature_list.json,
         qc_hr_mean_nan_trainval.parquet (per-subject hr_mean NaN fraction)
  test/  (full)       test_predictions.parquet, bootstrap_aucs_subject.npy,
         bootstrap_auprs_subject.npy, metrics_extended.json, metrics.json,
         model_full.txt, feature_list.json
Every run record carries code_sha256 of the protocol files (the git sha alone
does not identify untracked code).

Usage:
  python scripts/fit_aim2_matched_cells.py --cell physio_only_repro
  python scripts/fit_aim2_matched_cells.py --cell ecg_only --train-only
  python scripts/fit_aim2_matched_cells.py --cell ecg_belt --train-only
  python scripts/fit_aim2_matched_cells.py --cell ecg_only --prereg '<vault note>'   # once, after the freeze
  python scripts/fit_aim2_matched_cells.py --cell ecg_only --train-only --smoke 200 \\
      --out-root /private/tmp/claude-501/dl_build/trackC/smoke
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import subprocess
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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

from fit_aim2_v85_taxonomy_ablation import (  # noqa: E402
    FIXED_EARLY_STOPPING,
    FIXED_N_ESTIMATORS,
    FIXED_PARAMS,
    subject_bootstrap,
)
from fit_aim3_staging_baseline import TarSource, run_environment  # noqa: E402
from thesis_pipeline.epochs import sleep_mask  # noqa: E402
from thesis_pipeline.extended_metrics import write_extended_metrics  # noqa: E402
from thesis_pipeline.matched_cells import CELLS, caveats_for, cell_columns  # noqa: E402
from thesis_pipeline.splits import load_split  # noqa: E402

DEFAULT_TAR = CODE_ROOT / "features-phase1batch-v1.tar"
DEFAULT_SPLIT = CODE_ROOT / "splits" / "aim2_seed42.json"
OUT_ROOT = CODE_ROOT / "models" / "aim2_matched_cells_v1"
CANONICAL_DIR = CODE_ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only"
METADATA_MEMBER = "features/subject_metadata.parquet"
THRESHOLDS = np.linspace(0.05, 0.95, 91)  # fit_aim2_v85_taxonomy_ablation.py:273
AUC_TOL = 1e-4
PROB_TOL = 1e-6
REPRO_CELL = "physio_only_repro"
# Code whose bytes define the protocol; hashed into every run record.
PROTOCOL_FILES: tuple[str, ...] = (
    "scripts/fit_aim2_matched_cells.py",
    "thesis_pipeline/matched_cells.py",
    "scripts/fit_aim2_v85_taxonomy_ablation.py",
    "scripts/fit_aim3_staging_baseline.py",
    "thesis_pipeline/extended_metrics.py",
    "thesis_pipeline/epochs.py",
    "thesis_pipeline/splits.py",
)
PARTIAL_MODEL = ".model_full.txt.partial"


# --------------------------------------------------------------------------- data

def read_tar_member(tar_path: Path, name: str) -> bytes:
    with tarfile.open(tar_path) as tf:
        m = tf.getmember(name)
        off, size = m.offset_data, m.size
    with open(tar_path, "rb") as fh:
        fh.seek(off)
        return fh.read(size)


def nsrr_ahi(tar_path: Path, subjects) -> pd.DataFrame:
    """NSRR ahi_a0h3a (int subject_id) from the tar's subject_metadata, for ``subjects`` only."""
    sm = pd.read_parquet(io.BytesIO(read_tar_member(tar_path, METADATA_MEMBER)),
                         columns=["subject_id", "ahi_a0h3a"])
    sid = pd.to_numeric(sm["subject_id"].astype(str).str.extract(r"(\d+)$", expand=False),
                        errors="coerce")
    sm = sm.assign(subject_id=sid).dropna(subset=["subject_id"])
    sm["subject_id"] = sm["subject_id"].astype(np.int64)
    return sm[sm["subject_id"].isin({int(s) for s in subjects})].reset_index(drop=True)


def build_index(src: TarSource, version: str, subjects=None) -> dict[int, np.ndarray]:
    """Per subject: parquet row positions kept by the canonical filters.

    Canonical load_cohort keeps rows with features_version == version, then
    sleep_mask (N1/N2/N3/REM). Only hypnogram and version columns are read.
    ``subjects`` limits the members opened (default: every member of the tar).
    """
    def one(sid: int):
        t = pq.read_table(io.BytesIO(src.data(sid)),
                          columns=["subject_id", "sleep_stage", "features_version"]).to_pandas()
        assert (t["subject_id"].to_numpy() == sid).all(), f"subject_id mismatch in member {sid}"
        keep = (t["features_version"].to_numpy() == version) & sleep_mask(t["sleep_stage"].values)
        return sid, np.flatnonzero(keep)

    todo = sorted(src.members) if subjects is None else sorted(int(s) for s in subjects)
    with ThreadPoolExecutor(max_workers=8) as ex:
        out = dict(ex.map(one, todo))
    return {sid: rows for sid, rows in out.items() if rows.size}


def split_subjects(subject_ids, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outer + inner split of fit_aim2_v85_taxonomy_ablation.py (lines 237, 378).

    GroupShuffleSplit depends only on np.unique(groups), and returns sorted row
    indices, so computing it on the unique subject list gives the same subjects
    and, with rows in (subject, epoch) order, the same row partitions.
    """
    subj = np.unique(np.asarray(subject_ids))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_rel, te_rel = next(outer.split(np.zeros(len(subj)), groups=subj))
    tv = subj[tv_rel]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    tr_rel, va_rel = next(inner.split(np.zeros(len(tv)), groups=tv))
    return tv[tr_rel], tv[va_rel], subj[te_rel]


def load_rows(src: TarSource, index: dict, subjects, cols: list[str]):
    """Feature matrix (float32, like load_cohort's float64->float32 cast), labels, keys.

    Subjects in ascending id order, rows in parquet order: identical to the
    canonical concatenation followed by the sleep filter and row-index split.
    """
    subjects = [int(s) for s in sorted(subjects)]
    sizes = np.array([len(index[s]) for s in subjects])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    n = int(offsets[-1])
    X = np.empty((n, len(cols)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    sid = np.empty(n, dtype=np.int32)
    eidx = np.empty(n, dtype=np.int32)

    def fill(k: int) -> None:
        s = subjects[k]
        a, b = offsets[k], offsets[k + 1]
        rows = index[s]
        df = pq.read_table(io.BytesIO(src.data(s)), columns=cols + ["epoch_idx", "apnoea_label"]).to_pandas()
        X[a:b] = df[cols].to_numpy(dtype=np.float32)[rows]
        y[a:b] = df["apnoea_label"].to_numpy().astype(np.int8)[rows]
        eidx[a:b] = df["epoch_idx"].to_numpy().astype(np.int32)[rows]
        sid[a:b] = s

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(fill, range(len(subjects))))
    return X, y, sid, eidx


# --------------------------------------------------------------------------- helpers

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_provenance() -> dict:
    """sha256 of every protocol file, and which of them git does not track.

    run_environment()'s git_sha / git_dirty ignore untracked files, so they do not
    identify uncommitted protocol code; these hashes do.
    """
    shas = {rel: sha256_file(CODE_ROOT / rel) for rel in PROTOCOL_FILES}
    try:
        out = subprocess.run(["git", "-C", str(CODE_ROOT), "ls-files", "--", *PROTOCOL_FILES],
                             capture_output=True, text=True, check=True).stdout.split()
        untracked = [f for f in PROTOCOL_FILES if f not in set(out)]
    except Exception:
        untracked = None
    return {"code_sha256": shas, "code_untracked_in_git": untracked}


def pred_frame(sid, eidx, y, probs, thresh) -> pd.DataFrame:
    """Same columns and dtypes as the canonical test_predictions.parquet."""
    return pd.DataFrame({
        "subject_id": sid.astype(np.int32),
        "epoch_idx": eidx.astype(np.int32),
        "apnoea_label": y.astype(np.int8),
        "pred_prob": probs.astype(np.float64),
        "pred_label": (probs > thresh).astype(int),
    })


def hr_nan_qc(parts: dict[str, tuple[np.ndarray, np.ndarray]], cols: list[str],
              col: str = "hr_mean") -> tuple[pd.DataFrame | None, dict | None]:
    """Per-subject NaN fraction of ``col`` over sleep epochs, per partition (critique: RR cleaning)."""
    if col not in cols:
        return None, None
    j = cols.index(col)
    frames, summary = [], {}
    for name, (X, sid) in parts.items():
        isnan = np.isnan(X[:, j])
        g = pd.DataFrame({"subject_id": sid, "nan": isnan}).groupby("subject_id")["nan"]
        per = pd.DataFrame({"n_sleep_epochs": g.size(), "nan_frac": g.mean()}).reset_index()
        per.insert(0, "partition", name)
        frames.append(per)
        f = per["nan_frac"].to_numpy()
        summary[name] = {
            "n_subjects": int(len(per)),
            "epoch_nan_frac": float(isnan.mean()) if isnan.size else None,
            "subject_nan_frac_mean": float(f.mean()) if f.size else None,
            "subject_nan_frac_median": float(np.median(f)) if f.size else None,
            "subject_nan_frac_p90": float(np.percentile(f, 90)) if f.size else None,
            "subject_nan_frac_max": float(f.max()) if f.size else None,
            "n_subjects_nan_frac_gt_0.2": int((f > 0.2).sum()),
            "n_subjects_nan_frac_gt_0.5": int((f > 0.5).sum()),
        }
    return pd.concat(frames, ignore_index=True), {"column": col, **summary}


def existing_test_outputs(test_dir: Path) -> list[Path]:
    """Entries in ``test_dir`` other than earlier superseded_* archives."""
    if not test_dir.exists():
        return []
    return sorted(p for p in test_dir.iterdir() if not p.name.startswith("superseded_"))


def archive_test_outputs(test_dir: Path, reason: str) -> Path:
    """Move every current test output into test_dir/superseded_<UTC>/ with the rerun reason."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = test_dir / f"superseded_{stamp}"
    dest.mkdir()
    moved = []
    for p in existing_test_outputs(test_dir):
        p.rename(dest / p.name)
        moved.append(p.name)
    (dest / "superseded_reason.json").write_text(json.dumps(
        {"archived_utc": stamp, "rerun_reason": reason, "moved": moved}, indent=2))
    return dest


def check_against_trainonly(val_dir: Path, va_df: pd.DataFrame, refit_model: Path) -> dict:
    """Compare a refit with the saved --train-only run in ``val_dir`` (predictions and model bytes)."""
    saved = json.loads((val_dir / "metrics_trainonly.json").read_text())
    prev = pd.read_parquet(val_dir / "val_predictions.parquet")
    same_keys = bool(prev[["subject_id", "epoch_idx"]].reset_index(drop=True).equals(
        va_df[["subject_id", "epoch_idx"]].reset_index(drop=True)))
    diff = (float(np.abs(prev["pred_prob"].to_numpy() - va_df["pred_prob"].to_numpy()).max())
            if same_keys else float("inf"))
    refit_sha = sha256_file(refit_model)
    file_sha = sha256_file(val_dir / "model.txt")
    out = {
        "saved_run": str(val_dir),
        "val_keys_equal": same_keys,
        "val_pred_max_abs_diff": diff,
        "prob_tol": PROB_TOL,
        "trainonly_model_sha256_recorded": saved.get("model_sha256"),
        "trainonly_model_sha256_file": file_sha,
        "refit_model_sha256": refit_sha,
    }
    out["passed"] = bool(same_keys and diff <= PROB_TOL
                         and refit_sha == file_sha == saved.get("model_sha256"))
    return out


def reproduction_check(out_dir: Path, test_df: pd.DataFrame, metrics: dict, canonical_dir: Path) -> dict:
    """Compare a physio_only_repro test run with the canonical physio_only artefacts."""
    canon = pd.read_parquet(canonical_dir / "test_predictions.parquet")
    canon_m = json.loads((canonical_dir / "metrics.json").read_text())
    keys = ["subject_id", "epoch_idx", "apnoea_label"]
    keys_equal = bool(canon[keys].equals(test_df[keys]))
    dtypes_equal = bool((canon.dtypes == test_df.dtypes).all())
    diff = np.abs(canon["pred_prob"].to_numpy() - test_df["pred_prob"].to_numpy()) if keys_equal else None
    boot = {}
    for name in ("bootstrap_aucs_subject.npy", "bootstrap_auprs_subject.npy"):
        a, b = np.load(canonical_dir / name), np.load(out_dir / name)
        boot[name] = float(np.nanmax(np.abs(a - b)))
    ext = {}
    ce, re_ = canonical_dir / "metrics_extended.json", out_dir / "metrics_extended.json"
    if ce.exists() and re_.exists():
        a, b = json.loads(ce.read_text()), json.loads(re_.read_text())
        for k, v in a.items():
            if isinstance(v, (int, float)) and isinstance(b.get(k), (int, float)):
                ext[k] = float(abs(v - b[k]))
    auc_diff = abs(metrics["test_auc_roc"] - canon_m["test_auc_roc"])
    out = {
        "canonical_dir": str(canonical_dir),
        "canonical_test_auc_roc": canon_m["test_auc_roc"],
        "repro_test_auc_roc": metrics["test_auc_roc"],
        "test_auc_abs_diff": auc_diff,
        "auc_within_tol": bool(auc_diff <= AUC_TOL),
        "auc_tol": AUC_TOL,
        "key_columns_equal": keys_equal,
        "dtypes_equal": dtypes_equal,
        "pred_prob_max_abs_diff": None if diff is None else float(diff.max()),
        "pred_prob_mean_abs_diff": None if diff is None else float(diff.mean()),
        "pred_prob_within_tol": bool(diff is not None and diff.max() <= PROB_TOL),
        "prob_tol": PROB_TOL,
        "pred_label_mismatches": None if not keys_equal else int(
            (canon["pred_label"].to_numpy() != test_df["pred_label"].to_numpy()).sum()),
        "best_iter": {"canonical": canon_m["best_iter"], "repro": metrics["best_iter"]},
        "best_threshold": {"canonical": canon_m["best_threshold"], "repro": metrics["best_threshold"]},
        "scale_pos_weight_equal": bool(canon_m["scale_pos_weight"] == metrics["scale_pos_weight"]),
        "bootstrap_max_abs_diff": boot,
        "metrics_extended_abs_diff": ext,
    }
    out["passed"] = bool(out["auc_within_tol"] and out["pred_prob_within_tol"] and keys_equal)
    return out


def _assert_boot_equal(out_dir: Path, aucs: np.ndarray, auprs: np.ndarray) -> None:
    """write_extended_metrics must reproduce subject_bootstrap exactly (the pairing premise)."""
    for name, arr in (("bootstrap_aucs_subject.npy", aucs), ("bootstrap_auprs_subject.npy", auprs)):
        saved = np.load(out_dir / name)
        assert np.array_equal(saved, arr, equal_nan=True), f"{out_dir / name} differs from subject_bootstrap"


# --------------------------------------------------------------------------- main

@click.command()
@click.option("--cell", type=click.Choice(CELLS), required=True)
@click.option("--train-only", is_flag=True,
              help="Fit and report inner-validation metrics only (to <cell>/val/). Subject ids come "
                   "from the split file and test subjects' tar members are never opened.")
@click.option("--features-tar", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=DEFAULT_TAR, show_default=True)
@click.option("--features-version", default="2026-05-01-phase1batch-v1", show_default=True)
@click.option("--split-file", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=DEFAULT_SPLIT, show_default=True,
              help="Frozen Aim 2 split (sha256 sidecar verified).")
@click.option("--seed", type=int, default=42, show_default=True,
              help="Outer split seed. MUST be 42 to share the canonical Aim 2 test subjects.")
@click.option("--canonical-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=CANONICAL_DIR, show_default=True,
              help="Canonical physio_only run: source of the 217-column list, test subjects, "
                   "scale_pos_weight and the reproduction reference.")
@click.option("--out-root", type=click.Path(path_type=Path), default=OUT_ROOT, show_default=True)
@click.option("--prereg", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Pre-registration note. Required to score ecg_only / ecg_belt on test.")
@click.option("--rerun-reason", default=None,
              help="Required to score test again when <cell>/test/ already holds outputs; the old "
                   "outputs are moved to <cell>/test/superseded_<UTC>/.")
@click.option("--smoke", type=int, default=None,
              help="Pipeline smoke test on the first N inner-train subjects (and N//5 val "
                   "subjects); forces --train-only; needs --out-root. Not for reporting.")
def main(cell, train_only, features_tar, features_version, split_file, seed, canonical_dir, out_root,
         prereg, rerun_reason, smoke):
    t_start = time.time()
    if seed != 42:
        raise click.BadParameter("--seed must be 42 (canonical Aim 2 test subjects)")
    if smoke:
        train_only = True
        if out_root == OUT_ROOT:
            raise click.UsageError("--smoke needs an explicit --out-root outside models/")
    mode = "train_only" if train_only else "full"
    cell_dir = out_root / cell
    val_dir, test_dir = cell_dir / "val", cell_dir / "test"

    # ---- full-mode preflight: nothing is read until these pass
    prev_outputs: list[Path] = []
    if not train_only:
        if cell != REPRO_CELL and prereg is None:
            raise click.UsageError(
                f"Scoring {cell} on the held-out test set requires --prereg <vault note>. "
                "Use --train-only for inner-validation metrics.")
        if cell != REPRO_CELL and not ((val_dir / "metrics_trainonly.json").exists()
                                       and (val_dir / "val_predictions.parquet").exists()
                                       and (val_dir / "model.txt").exists()):
            raise click.UsageError(
                f"no saved --train-only run in {val_dir}; refusing to score {cell} on test. "
                f"Run --cell {cell} --train-only with the same --out-root first.")
        prev_outputs = existing_test_outputs(test_dir)
        if prev_outputs and not rerun_reason:
            raise click.UsageError(
                f"{test_dir} already holds test outputs ({', '.join(p.name for p in prev_outputs)}); "
                "the test set is scored once. Pass --rerun-reason '<logged bug reason>' to re-score; "
                "the old outputs are then kept under superseded_<UTC>/.")
        if rerun_reason and not prev_outputs:
            raise click.UsageError(f"--rerun-reason given but {test_dir} holds no earlier test outputs")
    elif rerun_reason:
        raise click.UsageError("--rerun-reason only applies to full (test-scoring) mode")

    physio_cols = json.loads((canonical_dir / "feature_list.json").read_text())
    cols = cell_columns(cell, physio_cols)
    canon_m = json.loads((canonical_dir / "metrics.json").read_text())
    canon_test_ids = set(pd.read_parquet(canonical_dir / "test_predictions.parquet",
                                         columns=["subject_id"])["subject_id"].unique().tolist())
    canon_summary_path = canonical_dir.parent / "summary.json"
    canon_cohort = json.loads(canon_summary_path.read_text())["cohort"] if canon_summary_path.exists() else None
    split = load_split(split_file)  # verifies the .sha256 sidecar and disjointness
    assert {int(s) for s in split["test"]} == canon_test_ids, \
        "split-file test subjects differ from the canonical physio_only run"

    print(f"=== Aim 2 matched cell: {cell}  ({len(cols)} features)  mode={mode}"
          f"{'  SMOKE' if smoke else ''} ===")
    print(f"  tar: {features_tar}\n  split: {split_file} (sha256 {split['sha256'][:12]})\n"
          f"  out: {val_dir if train_only else test_dir}")

    # ---- cohort index (hypnogram + version only) and split
    t0 = time.time()
    src = TarSource(features_tar)  # member table from the tar headers only
    missing = ({int(s) for k in ("train", "val", "test") for s in split[k]} - set(src.members))
    assert not missing, f"{len(missing)} split subjects have no tar member, e.g. {sorted(missing)[:5]}"
    if train_only:
        for s in split["test"]:
            del src.members[int(s)]  # any read of a test member now raises KeyError
        tr_s, va_s, te_s = split["train"], split["val"], split["test"]
        if smoke:
            tr_s, va_s = tr_s[:smoke], va_s[: max(smoke // 5, 10)]
        index = build_index(src, features_version, subjects=np.concatenate([tr_s, va_s]))
        assert set(index) == {int(s) for s in tr_s} | {int(s) for s in va_s}, \
            "some train/val subjects have no sleep epochs"
        n_sleep = sum(len(r) for r in index.values())
        t_index = time.time() - t0
        print(f"  indexed {len(index)} train+val subjects / {n_sleep:,} sleep epochs in {t_index:.0f}s "
              f"(test members not opened)")
        if canon_cohort is not None and not smoke:
            assert len(tr_s) + len(va_s) + len(te_s) == canon_cohort["n_subjects_total"]
            assert n_sleep == canon_cohort["n_epochs_total"] - canon_cohort["n_test_epochs"], (
                n_sleep, canon_cohort["n_epochs_total"] - canon_cohort["n_test_epochs"])
    else:
        index = build_index(src, features_version)
        n_sleep = sum(len(r) for r in index.values())
        tr_s, va_s, te_s = split_subjects(list(index), seed)
        t_index = time.time() - t0
        print(f"  indexed {len(index)} subjects / {n_sleep:,} sleep epochs in {t_index:.0f}s")
        for k, arr in (("train", tr_s), ("val", va_s), ("test", te_s)):
            assert np.array_equal(np.sort(np.asarray(arr, dtype=np.int64)), split[k]), \
                f"rebuilt {k} subjects differ from {split_file}"
        if canon_cohort is not None:
            assert len(index) == canon_cohort["n_subjects_total"], (len(index), canon_cohort["n_subjects_total"])
            assert n_sleep == canon_cohort["n_epochs_total"], (n_sleep, canon_cohort["n_epochs_total"])
            assert sum(len(index[s]) for s in te_s) == canon_cohort["n_test_epochs"]
    print(f"  split: train {len(tr_s)} / val {len(va_s)} / test {len(te_s)} subjects")
    assert not (set(tr_s) | set(va_s)) & set(te_s)
    print("  cohort and test subjects match the canonical run")

    # ---- load train + val (test rows only in full mode, after the guards below)
    t0 = time.time()
    Xtr, ytr, sid_tr, _ = load_rows(src, index, tr_s, cols)
    Xva, yva, sid_va, eidx_va = load_rows(src, index, va_s, cols)
    t_load = time.time() - t0
    print(f"  loaded train {len(ytr):,} / val {len(yva):,} epochs x {len(cols)} features in {t_load:.0f}s "
          f"({(Xtr.nbytes + Xva.nbytes) / 1e9:.2f} GB float32)")
    qc_df, qc_summary = (hr_nan_qc({"train": (Xtr, sid_tr), "val": (Xva, sid_va)}, cols)
                         if train_only else (None, None))

    pos = float(np.sum(ytr))
    neg = float(len(ytr) - pos)
    spw = neg / max(pos, 1.0)
    print(f"  inner train: {len(ytr):,} epochs / {len(np.unique(sid_tr))} subj ({ytr.mean()*100:.1f}% positive)")
    print(f"  inner val:   {len(yva):,} epochs / {len(np.unique(sid_va))} subj")
    print(f"  scale_pos_weight: {spw!r}")
    if not smoke:
        assert spw == canon_m["scale_pos_weight"], (
            f"inner-train set differs from canonical: spw {spw!r} vs {canon_m['scale_pos_weight']!r}")

    # ---- fit (fit_aim2_v85_taxonomy_ablation.py:251-265)
    t0 = time.time()
    model = lgb.LGBMClassifier(n_estimators=FIXED_N_ESTIMATORS, scale_pos_weight=spw, **FIXED_PARAMS)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(FIXED_EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
    )
    fit_time = time.time() - t0
    best_iter = int(model.best_iteration_ or FIXED_N_ESTIMATORS)
    print(f"  fit in {fit_time:.0f}s  (best_iter={best_iter})")
    del Xtr

    # ---- inner validation (early-stopping set; also where the threshold is chosen)
    va_probs = model.predict_proba(Xva)[:, 1]
    va_f1s = [f1_score(yva, (va_probs > t).astype(int), zero_division=0) for t in THRESHOLDS]
    best_thresh = float(THRESHOLDS[int(np.argmax(va_f1s))])
    va_pred = (va_probs > best_thresh).astype(int)
    va_df = pred_frame(sid_va, eidx_va, yva, va_probs, best_thresh)
    va_aucs, va_auprs, va_ci = subject_bootstrap(va_df[["subject_id", "epoch_idx", "apnoea_label"]],
                                                 va_probs, n_resamples=1000, seed=42)
    inner_val = {
        "note": "inner-validation subjects = early-stopping + threshold set (and the DL validation set); "
                "AUC is optimistic by the early-stopping selection",
        "n_val_subjects": int(len(np.unique(sid_va))),
        "n_val_epochs": int(len(yva)),
        "val_auc_roc": float(roc_auc_score(yva, va_probs)),
        "val_auc_pr": float(average_precision_score(yva, va_probs)),
        "val_auc_ci_low": va_ci["auc_ci_low"], "val_auc_ci_high": va_ci["auc_ci_high"],
        "val_aupr_ci_low": va_ci["aupr_ci_low"], "val_aupr_ci_high": va_ci["aupr_ci_high"],
        "val_f1_tuned": float(f1_score(yva, va_pred, zero_division=0)),
        "val_precision_tuned": float(precision_score(yva, va_pred, zero_division=0)),
        "val_recall_tuned": float(recall_score(yva, va_pred, zero_division=0)),
        "best_threshold": best_thresh,
    }
    print(f"  inner-val AUC {inner_val['val_auc_roc']:.4f} [{va_ci['auc_ci_low']:.4f}, {va_ci['auc_ci_high']:.4f}]"
          f"  AUC-PR {inner_val['val_auc_pr']:.4f}  F1@{best_thresh:.2f} {inner_val['val_f1_tuned']:.4f}")

    env = run_environment()
    common = {
        "cell": cell,
        "mode": mode,
        "smoke": smoke,
        "filter_used": "sleep-only",
        "features_version": features_version,
        "features_tar": str(features_tar),
        "features_tar_bytes": features_tar.stat().st_size,
        "split_file": str(split_file),
        "split_file_sha256": split["sha256"],
        "n_features": len(cols),
        "n_train_subjects": int(len(tr_s)),
        "n_val_subjects": int(len(va_s)),
        "n_train_epochs": int(len(ytr)),
        "outer_split_seed": seed,
        "inner_split_seed": 42,
        "best_iter": best_iter,
        "fixed_params": FIXED_PARAMS,
        "n_estimators": FIXED_N_ESTIMATORS,
        "early_stopping_rounds": FIXED_EARLY_STOPPING,
        "scale_pos_weight": spw,
        "best_threshold": best_thresh,
        "inner_val": inner_val,
        "input_match_caveats": caveats_for(cell),
        "protocol_source": "scripts/fit_aim2_v85_taxonomy_ablation.py (FIXED_PARAMS, subject_bootstrap imported)",
        "column_source": str(canonical_dir / "feature_list.json"),
        **code_provenance(),
        "env": env,
    }

    if train_only:
        val_dir.mkdir(parents=True, exist_ok=True)
        model_path = val_dir / "model.txt"
        model.booster_.save_model(str(model_path))
        va_df.to_parquet(val_dir / "val_predictions.parquet", index=False)
        # validation bootstrap arrays + extended metrics (same RNG and subjects as the DL val eval)
        t0 = time.time()
        va_ext = write_extended_metrics(val_dir, va_df, subject_metadata=nsrr_ahi(features_tar, va_s))
        t_ext = time.time() - t0
        _assert_boot_equal(val_dir, va_aucs, va_auprs)
        assert va_ext["n_subjects_with_nsrr_ahi"] == inner_val["n_val_subjects"], va_ext["n_subjects_with_nsrr_ahi"]
        (val_dir / "feature_list.json").write_text(json.dumps(cols, indent=2))
        if qc_df is not None:
            qc_df.to_parquet(val_dir / "qc_hr_mean_nan_trainval.parquet", index=False)
        runtime = {"index_seconds": t_index, "load_seconds": t_load, "fit_seconds": fit_time,
                   "extended_metrics_seconds": t_ext, "total_seconds": time.time() - t_start}
        metrics = {
            **common,
            "split": "val",
            "test_set": ("train-only mode: test subjects' tar members were removed from the member table "
                         "before any member was read, so no test parquet (hypnogram, features or labels) "
                         "was opened. Test ids were read from the split file, the canonical "
                         "test_predictions.parquet subject_id column and the tar header names, to check "
                         "the split; the cohort-level subject_metadata table (subject_id, ahi_a0h3a) was "
                         "read and filtered to the validation subjects before use. No test metric computed."),
            "nsrr_ahi_source": f"{features_tar}:{METADATA_MEMBER} (filtered to the validation subjects)",
            "hr_mean_nan_qc": qc_summary,
            "model_sha256": sha256_file(model_path),
            "runtime": runtime,
        }
        (val_dir / "metrics_trainonly.json").write_text(json.dumps(metrics, indent=2))
        print(f"  runtime: {json.dumps({k: round(v, 1) for k, v in runtime.items()})}")
        print(f"  saved -> {val_dir}  (train-only; no test member opened, nothing about test written)")
        return

    # ---- full mode: the refit must reproduce the saved train-only run BEFORE any test row is read
    prereg_info = {"path": str(prereg), "sha256": sha256_file(prereg)} if prereg is not None else None
    cell_dir.mkdir(parents=True, exist_ok=True)
    partial = cell_dir / PARTIAL_MODEL
    model.booster_.save_model(str(partial))
    guard = None
    if (val_dir / "metrics_trainonly.json").exists():  # mandatory for ecg cells (preflight above)
        guard = check_against_trainonly(val_dir, va_df, partial)
        print(f"  refit vs saved train-only run: val max |diff| {guard['val_pred_max_abs_diff']:.3g}, "
              f"model sha256 {'equal' if guard['passed'] else 'DIFFERENT'} ({guard['refit_model_sha256'][:12]})")
        if not guard["passed"]:
            partial.unlink()
            raise SystemExit(f"refit does not reproduce the saved train-only run in {val_dir}; "
                             f"not scoring test. {json.dumps(guard)}")
    elif cell != REPRO_CELL:  # preflight makes this unreachable; kept as a hard stop
        partial.unlink()
        raise SystemExit(f"no saved train-only run in {val_dir}; not scoring test")

    rerun = None
    if prev_outputs:
        archived = archive_test_outputs(test_dir, rerun_reason)
        rerun = {"reason": rerun_reason, "superseded_dir": str(archived),
                 "superseded": [p.name for p in prev_outputs]}
        print(f"  rerun: earlier test outputs moved to {archived}")
    test_dir.mkdir(parents=True, exist_ok=True)
    model_path = test_dir / "model_full.txt"
    os.replace(partial, model_path)

    t0 = time.time()
    Xte, yte, sid_te, eidx_te = load_rows(src, index, te_s, cols)
    t_load_test = time.time() - t0
    print(f"  test:        {len(yte):,} epochs / {len(np.unique(sid_te))} subj (loaded in {t_load_test:.0f}s)")

    # fit_aim2_v85_taxonomy_ablation.py:267-304
    probs = model.predict_proba(Xte)[:, 1]
    test_auc = float(roc_auc_score(yte, probs))
    test_aupr = float(average_precision_score(yte, probs))
    test_df = pred_frame(sid_te, eidx_te, yte, probs, best_thresh)
    preds = test_df["pred_label"].to_numpy()
    test_f1 = float(f1_score(yte, preds, zero_division=0))
    test_p = float(precision_score(yte, preds, zero_division=0))
    test_r = float(recall_score(yte, preds, zero_division=0))
    print(f"  Test AUC: {test_auc:.4f}   AUC-PR: {test_aupr:.4f}")
    print(f"  F1@{best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    t0 = time.time()
    aucs, auprs, ci = subject_bootstrap(test_df[["subject_id", "epoch_idx", "apnoea_label"]],
                                        probs, n_resamples=1000, seed=42)
    t_boot = time.time() - t0
    print(f"  AUC subject CI:  [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]  (bootstrap {t_boot:.0f}s)")

    test_df.to_parquet(test_dir / "test_predictions.parquet", index=False)
    np.save(test_dir / "bootstrap_aucs_subject.npy", aucs)
    np.save(test_dir / "bootstrap_auprs_subject.npy", auprs)
    (test_dir / "feature_list.json").write_text(json.dumps(cols, indent=2))

    # Extended metrics with NSRR ahi_a0h3a from the tar's subject_metadata (critique #12)
    t0 = time.time()
    ext = write_extended_metrics(test_dir, test_df, subject_metadata=nsrr_ahi(features_tar, te_s))
    t_ext = time.time() - t0
    _assert_boot_equal(test_dir, aucs, auprs)
    n_test_subjects = int(len(np.unique(sid_te)))
    assert ext["n_subjects_with_nsrr_ahi"] == n_test_subjects, (ext["n_subjects_with_nsrr_ahi"], n_test_subjects)

    metrics = {
        **common,
        "split": "test",
        "name": cell,
        "n_test_subjects": n_test_subjects,
        "n_test_epochs": int(len(yte)),
        "test_auc_roc": test_auc,
        "test_auc_pr": test_aupr,
        "test_auc_ci_low": ci["auc_ci_low"],
        "test_auc_ci_high": ci["auc_ci_high"],
        "test_aupr_ci_low": ci["aupr_ci_low"],
        "test_aupr_ci_high": ci["aupr_ci_high"],
        "test_f1_tuned": test_f1,
        "test_precision_tuned": test_p,
        "test_recall_tuned": test_r,
        "fit_seconds": fit_time,
        "prereg": prereg_info,
        "trainonly_guard": guard,
        "trainonly_val_pred_max_abs_diff": None if guard is None else guard["val_pred_max_abs_diff"],
        "rerun": rerun,
        "model_sha256": sha256_file(model_path),
        "nsrr_ahi_source": f"{features_tar}:{METADATA_MEMBER} (filtered to the test subjects)",
        "feature_taxonomy_module": "thesis_pipeline.matched_cells",
    }
    if cell == REPRO_CELL:
        metrics["reproduction_check"] = reproduction_check(test_dir, test_df, metrics, canonical_dir)
    metrics["runtime"] = {"index_seconds": t_index, "load_seconds": t_load, "fit_seconds": fit_time,
                          "load_test_seconds": t_load_test, "bootstrap_seconds": t_boot,
                          "extended_metrics_seconds": t_ext, "total_seconds": time.time() - t_start}
    (test_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  runtime: {json.dumps({k: round(v, 1) for k, v in metrics['runtime'].items()})}")
    if "reproduction_check" in metrics:
        rc = metrics["reproduction_check"]
        print(f"  reproduction: |dAUC| {rc['test_auc_abs_diff']:.2e}  max|dprob| {rc['pred_prob_max_abs_diff']}"
              f"  keys_equal {rc['key_columns_equal']}  passed {rc['passed']}")
        if not rc["passed"]:
            raise SystemExit("physio_only reproduction check FAILED — see metrics.json reproduction_check")
    print(f"  saved -> {test_dir}")


if __name__ == "__main__":
    main()
