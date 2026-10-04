"""Epoch predictions from a trained Aim 2 Olsen BiGRU, in the test_predictions schema.

Pre-registration. ``--prereg`` (the vault note) is required for val and test.
``prereg.check_prereg`` refuses a note whose body (every byte above the line ``## Addenda``)
differs from the frozen body hash committed in ``prereg/aim2_dl_olsen_v1_freeze.json``.
The provenance logs the whole check: whole-file sha256, body sha256 and the freeze
record's sha256. Appending a dated addendum changes only the whole-file hash.

Checks on every run. The postproc belongs to ``--ckpt`` (best_ckpt_sha256). The data
folder's MANIFEST front-end / stage-B / EDR versions equal the training cache's
(``ck["data_versions"]["train"]``; a checkpoint without them is refused). With
``--aggregator-lock`` the lock must come from M seed 42 (``from`` ends in
M/seed42/postproc.json), the pre-declared rule (``choose_aggregator``) applied to the
lock's own validation AUCs must give the locked aggregator, the postproc must use it, and
when the checkpoint is M seed 42 its sha256 must equal the lock's ``best_ckpt_sha256``
(a retrained M seed 42 needs a new lock).

--split val   (allowed at any time) scores validation with the locked aggregator (or the
              postproc's) and runs the PARITY GATE: |validation AUC on this device -
              training-device validation AUC (postproc ``aggregator_rule.aucs``)| must be
              <= PARITY_TOL (0.002). Scored under ``--aggregator-lock`` (with or without
              ``--refit-postproc``) this is the pre-registered gate, so it runs only at the
              pre-registered test precision (MPS: the training precision, i.e. ``--no-amp`` for
              a NaN-fallback config; CPU: fp32) and eval batch (PREREG_EVAL_BATCH = 256);
              without the lock (diagnostics) any precision and batch run and nothing is
              recorded. ``--refit-postproc`` (needs ``--aggregator-lock``) refits the F1-max
              threshold on these predictions and writes postproc_<devtype>.json with the parity
              record, the eval batch and the code commit it ran at (test inference accepts only
              a refit made at the clean freeze commit, at its own eval batch). Also writes
              val_predictions_<devtype>.parquet, val_sec_probs_<devtype>.npy and
              predict_val_<devtype>.json. On a parity failure both json files are still
              written; when scored under ``--aggregator-lock`` (the pre-registered gate) the
              failure is appended to PARITY_FAILED_<devtype>.json in the run folder and (for
              a reportable checkpoint) in DEFAULT_ROOT; then the run exits
              non-zero: the pre-registration then abandons that device for every run and
              re-scores validation and test on CPU in fp32 (if the CPU fp32 fallback fails
              too, it defines no further fallback). Those records are never overwritten:
              a later --refit-postproc on that device type is refused for the run (and for
              every reportable run when the device-wide record exists), so a failure
              cannot be replaced by a passing retry with other settings.
--split test  ONCE per (config, seed), after the code freeze, on the Mac. Refused unless:
              platform.system() == "Darwin"; config M or P4 and model seed 42, 43 or 44;
              the output folder is DEFAULT_ROOT/<config>/seed<k> (the run's training folder
              downloaded from Drive; ``--out-dir`` defaults to it); ``--freeze-tag`` names a
              git tag equal to HEAD on a clean tree; the checkpoint STARTED training under
              the frozen pre-registration body (``state["prereg_freeze_first"]``) and was never
              run with --allow-unfrozen or a PILOT ONLY option (--max-passes,
              --max-batches-per-pass; checked in the checkpoint and in metrics.json);
              ``<out>/metrics.json`` (training metrics) names this checkpoint; every
              invocation of the run (resumes included: ``git_commits`` in the checkpoint and
              in metrics.json) trained at ONE clean commit C_train; the freeze commit is C_train
              or a descendant of it and MODEL_FILES are byte-identical between the two (front-end
              differences are recorded in ``code_lineage``); the lock records M seed 42's
              best_ckpt_sha256; on MPS the precision equals the training precision (fp16
              autocast, fp32 after the NaN fallback); the eval batch is PREREG_EVAL_BATCH (256);
              the postproc was refit on this device type, with this amp and eval batch, at the
              freeze commit, and passed the parity gate; no parity
              failure is recorded for this device type (run folder or DEFAULT_ROOT). It then
              predicts every test subject, checks the key columns against the canonical key
              file, and writes test_predictions.parquet, test_sec_probs.npy (float32
              (n_rows, 30): per-second probabilities row-aligned with the parquet, for the
              event-level secondary endpoint) and predict_test.json. It computes no metrics;
              ``evaluate_aim2_dl.py`` does that.

predict_test.json keys (read by evaluate_aim2_dl.py): ``PREDICT_TEST_KEYS`` below, plus
the full ``check_prereg`` record, front-end QC reasons and the postproc checks.

Once-only guard (test): any earlier test output in the run folder (test_predictions.parquet,
test_sec_probs.npy, predict_test.json, a superseded_* archive, or an attempt that reached
inference in predict_test_attempts.jsonl) means test inference already ran, and a new run
needs ``--rerun-reason``. Every check runs before inference; outputs go to temporary
files, and only after success are the previous parquet, per-second array and
predict_test.json moved together into superseded_<UTC>/ (superseded_<UTC>_<k> if that
folder already exists, e.g. made by evaluate_aim2_dl.py in the same second) with the
reason, along with what evaluate_aim2_dl.py computed from them in the run folder
(metrics_extended.json and the bootstrap arrays), so the original results stay
recoverable. Every attempt is logged.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.dl_aggregate import AGGREGATORS, choose_aggregator  # noqa: E402
from thesis_pipeline.dl_data import PackedSplit, assert_ids_in_split, load_split  # noqa: E402
from thesis_pipeline.dl_eval import (  # noqa: E402
    PARITY_FAILED_FILE,
    PARITY_TOL,  # |this-device - training-device| validation AUC, locked aggregator
    PILOT_ONLY_SETTINGS,
    PRIMARY_CONTRASTS,
    REQUIRED_SEEDS,
    assemble_predictions,
    threshold_f1max,
)
from thesis_pipeline.dl_stage_b import git_commit, sha256_file  # noqa: E402
from thesis_pipeline.dl_train import (  # noqa: E402
    DATA_VERSION_KEYS,
    load_model_from_ckpt,
    resolve_device,
    score_split,
)
from thesis_pipeline.prereg import PreregMismatch, check_prereg  # noqa: E402

DEFAULT_SPLIT = CODE_ROOT / "splits" / "aim2_seed42.json"
DEFAULT_KEY = CODE_ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only" / \
    "test_predictions.parquet"
DEFAULT_ROOT = CODE_ROOT / "models" / "aim2_dl_olsen_v1"  # <config>/seed<k>/ = the downloaded training folder
LOCK_SOURCE = "M/seed42/postproc.json"
TEST_OUTPUTS = ("test_predictions.parquet", "test_sec_probs.npy", "predict_test.json")
# what evaluate_aim2_dl.py --split test writes into the run folder from those predictions; moved with
# them into superseded_<UTC>/ on a rerun, so the original metrics stay next to the original predictions
EVALUATE_OUTPUTS = ("metrics_extended.json", "bootstrap_aucs_subject.npy", "bootstrap_auprs_subject.npy")
# the pre-registered inference batch ('eval batch 256'): test inference and the parity gate run at it
PREREG_EVAL_BATCH = 256
# Pre-registration code pins: between C_train (the checkpoint's training commit) and the freeze
# commit these files must be byte-identical (a change means retraining every run).
MODEL_FILES = tuple(f"thesis_pipeline/{n}" for n in ("dl_models.py", "dl_data.py", "dl_aggregate.py", "dl_train.py"))
# The front-end that turns ECG / belts into model inputs; differences from C_train and from the commit
# that packed train and validation are recorded (a change is allowed only under the test-cache
# contingency, which must reproduce the train and validation data files byte-identically).
FRONTEND_FILES = tuple(f"thesis_pipeline/{n}" for n in ("dl_signals.py", "dl_stage_b.py", "dl_labels.py",
                                                        "features.py"))
TRAIN_VAL_PACK_COMMIT = "bbc0806909f89ac078a67f920d32a7f7be4e10a1"  # packed the train and val caches
PREDICT_TEST_KEYS = (
    "split", "config", "model_seed", "freeze_tag", "freeze_commit", "git_commit", "prereg", "prereg_sha256",
    "prereg_body_sha256", "prereg_freeze_sha256", "ckpt", "ckpt_sha256", "train_git_commit", "train_git_commits",
    "code_lineage", "train_prereg_body_sha256", "postproc_sha256", "postproc_git_commit", "parity", "aggregator",
    "aggregator_locked",
    "aggregator_lock_sha256", "threshold", "device", "amp", "platform", "data_manifest_sha256", "data_versions",
    "key", "key_sha256", "split_sha256", "test_ids_sha256", "test_predictions_sha256", "test_sec_probs_sha256",
    "test_sec_probs_shape", "n_rows", "n_subjects", "frontend_detector_counts", "frontend_failed_subjects",
    "rerun_reason", "replaced_previous", "started", "predict_seconds",
)


def _git_run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(CODE_ROOT), *args], capture_output=True, text=True)


def _git(*args: str) -> str:
    return _git_run(*args).stdout.strip()


def _platform_system() -> str:
    return platform.system()


def _is_sha256(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _is_commit(v) -> bool:
    """A full, clean git commit id (not 'unknown', not '<sha>-dirty')."""
    return isinstance(v, str) and len(v) == 40 and all(c in "0123456789abcdef" for c in v)


def check_freeze(tag: str) -> str:
    """HEAD must equal the tag's commit and the tracked tree must be clean."""
    head = _git("rev-parse", "HEAD")
    tagc = _git("rev-list", "-n", "1", tag)
    if not tagc:
        raise click.ClickException(f"git tag {tag!r} not found")
    if head != tagc:
        raise click.ClickException(f"HEAD {head[:10]} != tag {tag} ({tagc[:10]}); check out the frozen code")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise click.ClickException("tracked files modified; test inference needs a clean frozen tree")
    untracked = [ln for ln in _git("status", "--porcelain", "--untracked-files=all", "--",
                                   "thesis_pipeline", "scripts", "splits", "prereg").splitlines()
                 if ln.startswith("??") and ln.rstrip().endswith((".py", ".json"))]
    if untracked:
        raise click.ClickException(f"untracked code not in the frozen tag: {untracked[:5]}")
    return head


def _changed_files(a: str, b: str, files: tuple[str, ...]) -> list[str]:
    r = _git_run("diff", "--name-only", a, b, "--", *files)
    if r.returncode != 0:
        raise click.ClickException(f"git diff {a[:10]} {b[:10]} failed: {(r.stderr or r.stdout).strip()}")
    return sorted(ln.strip() for ln in r.stdout.splitlines() if ln.strip())


def _commit_exists(c: str) -> bool:
    return _git_run("cat-file", "-e", f"{c}^{{commit}}").returncode == 0


def check_code_lineage(train_commit: str, freeze_commit: str) -> dict:
    """Pre-registration code pins: the freeze commit is C_train (the checkpoint's training commit) or
    a descendant of it, and MODEL_FILES are byte-identical between the two (any change there means
    retraining every run). Front-end files that differ from C_train or from the commit that packed
    train and validation (TRAIN_VAL_PACK_COMMIT) are recorded, not refused: the test-cache
    contingency allows a front-end fix that reproduces the train and validation data files."""
    for c, what in ((train_commit, "training commit (C_train)"), (freeze_commit, "freeze commit")):
        if not _commit_exists(c):
            raise click.ClickException(f"{what} {c!r} is not a commit of {CODE_ROOT}")
    anc = _git_run("merge-base", "--is-ancestor", train_commit, freeze_commit)
    if anc.returncode == 1:
        raise click.ClickException(
            f"the freeze commit {freeze_commit[:10]} does not descend from the checkpoint's training commit "
            f"{train_commit[:10]} (C_train): test inference runs from C_train or a descendant of it")
    if anc.returncode != 0:
        raise click.ClickException(f"git merge-base --is-ancestor failed: {(anc.stderr or anc.stdout).strip()}")
    changed = _changed_files(train_commit, freeze_commit, MODEL_FILES)
    if changed:
        raise click.ClickException(
            f"{changed} differ between the training commit {train_commit[:10]} (C_train) and the freeze commit "
            f"{freeze_commit[:10]}: the pre-registration keeps {list(MODEL_FILES)} byte-identical; any change to "
            "them means retraining every run")
    fe_pack = (_changed_files(TRAIN_VAL_PACK_COMMIT, freeze_commit, FRONTEND_FILES)
               if _commit_exists(TRAIN_VAL_PACK_COMMIT) else None)
    return {"train_commit": train_commit, "freeze_commit": freeze_commit, "train_is_ancestor_of_freeze": True,
            "model_files": list(MODEL_FILES), "model_files_changed": [],
            "frontend_files": list(FRONTEND_FILES),
            "frontend_changed_vs_train_commit": _changed_files(train_commit, freeze_commit, FRONTEND_FILES),
            "train_val_pack_commit": TRAIN_VAL_PACK_COMMIT, "frontend_changed_vs_pack_commit": fe_pack,
            "passed": True}


def train_commit_history(ck: dict, metrics: dict) -> str:
    """The one clean commit every invocation of the run (resumes included) trained at: the checkpoint's
    and the training metrics.json's ``git_commit`` and ``git_commits`` must all name it."""
    hist = {"checkpoint state git_commits": (ck.get("state") or {}).get("git_commits"),
            "metrics.json git_commits": metrics.get("git_commits")}
    missing = [k for k, v in hist.items() if not isinstance(v, list) or not v]
    if missing:
        raise click.ClickException(
            f"no training commit history ({', '.join(missing)}): the run was trained by code older than the "
            "commit history, so it cannot be shown that every invocation ran at one commit")
    t = ck.get("git_commit")
    commits = sorted({str(t), str(metrics.get("git_commit"))} | {str(c) for v in hist.values() for c in v})
    if len(commits) != 1:
        raise click.ClickException(
            f"the run was trained from more than one commit {commits} (checkpoint, metrics.json and every "
            "invocation including resumes): production training uses one commit C_train; never reportable")
    if not _is_commit(t):
        raise click.ClickException(
            f"the run was trained at {t!r}, not a clean commit ('-dirty' = modified or untracked code, or "
            "'unknown'): production training runs at the tagged C_train")
    return t


ATTEMPTS = "predict_test_attempts.jsonl"


def prior_test_outputs(out: Path) -> list[str]:
    """Evidence that test inference already ran in ``out`` (see the module docstring)."""
    found = [n for n in TEST_OUTPUTS if (out / n).exists()]
    found += sorted(p.name for p in out.glob("superseded_*"))
    found += sorted(p.name for p in out.glob("test_predictions.2*.parquet"))  # archives of the first build
    log = out / ATTEMPTS
    if log.exists() and any(json.loads(ln).get("event") == "inference_start"
                            for ln in log.read_text().splitlines() if ln.strip()):
        found.append(f"{ATTEMPTS} (an earlier attempt reached inference)")
    return found


def _log_attempt(out: Path, rec: dict) -> None:
    with open(out / ATTEMPTS, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")


def load_aggregator_lock(path: str, ckpt_sha: str | None = None, config: str | None = None,
                         model_seed: int | None = None, for_test: bool = False) -> dict:
    """Refuse a lock that is not M seed 42's pre-declared choice (see the module docstring)."""
    lock = json.loads(Path(path).read_text())
    agg = lock.get("aggregator")
    if agg not in AGGREGATORS:
        raise click.ClickException(f"{path}: 'aggregator' must be one of {AGGREGATORS}")
    src = str(lock.get("from", "")).replace("\\", "/")
    if not (src == LOCK_SOURCE or src.endswith("/" + LOCK_SOURCE)):
        raise click.ClickException(
            f"{path}: 'from' is {lock.get('from')!r}; the aggregator is locked only from M seed 42 "
            f"validation (a path ending in {LOCK_SOURCE})")
    rule = lock.get("rule")
    aucs = rule.get("aucs") if isinstance(rule, dict) else None
    if not isinstance(aucs, dict):
        raise click.ClickException(f"{path}: no rule.aucs (M seed 42 validation AUC of every candidate)")
    try:
        chosen = choose_aggregator(aucs)["chosen"]
    except (ValueError, TypeError) as exc:
        raise click.ClickException(f"{path}: cannot re-apply the pre-declared rule to rule.aucs ({exc})") from exc
    if chosen != agg:
        raise click.ClickException(
            f"{path}: the pre-declared rule applied to the lock's own AUCs {aucs} chooses {chosen!r}, "
            f"but the lock says {agg!r}")
    sha = lock.get("best_ckpt_sha256")
    if for_test and not _is_sha256(sha):
        raise click.ClickException(
            f"{path}: no best_ckpt_sha256 (sha256 of M seed 42's best.pt); re-make the lock with notebook cell 9")
    if sha is not None and config == "M" and model_seed == 42 and sha != ckpt_sha:
        raise click.ClickException(
            f"--ckpt is M seed 42 but its sha256 {str(ckpt_sha)[:12]} != the lock's best_ckpt_sha256 "
            f"{str(sha)[:12]}: the lock was made from another M seed 42 checkpoint (if M seed 42 was "
            "retrained, the aggregator must be re-chosen and the lock re-made before any test inference)")
    return lock


def check_postproc(pp: dict, ckpt_sha: str, lock: dict | None, test_device: str | None = None,
                   test_amp: bool | None = None, freeze_commit: str | None = None,
                   test_eval_batch: int | None = None) -> None:
    """Refuse a postproc from another checkpoint or aggregator; for test (``test_device``
    given) also one from another device type or amp setting, without a passed parity gate,
    (``freeze_commit`` given) refit by code other than the clean freeze commit, or
    (``test_eval_batch`` given) refit at another eval batch."""
    if pp.get("best_ckpt_sha256") != ckpt_sha:
        raise click.ClickException(
            f"postproc best_ckpt_sha256 {str(pp.get('best_ckpt_sha256'))[:12]} != sha256(--ckpt) {ckpt_sha[:12]}: "
            "this postproc belongs to another checkpoint (seed or run)")
    if lock is not None and pp.get("aggregator") != lock["aggregator"]:
        raise click.ClickException(
            f"postproc aggregator {pp.get('aggregator')!r} != locked {lock['aggregator']!r}; re-finalise the run "
            "with train_aim2_dl.py --aggregator <locked> (no retraining) and refit on this device")
    if test_device is None:
        return
    if str(pp.get("device", "")).split(":")[0] != test_device:
        raise click.ClickException(
            f"postproc was fitted on device {pp.get('device')!r}, test inference runs on {test_device!r}; "
            "use the postproc_<device>.json from --split val --refit-postproc on this device")
    parity = pp.get("parity")
    if not isinstance(parity, dict) or parity.get("passed") is not True:
        raise click.ClickException(
            f"postproc has no passed parity gate (parity = {parity}); run --split val --refit-postproc "
            f"--aggregator-lock ... on this device first; if parity failed, {after_parity_failure(test_device)}")
    if pp.get("amp") != test_amp:
        raise click.ClickException(
            f"postproc was refit with amp={pp.get('amp')!r}, test inference runs with amp={test_amp!r}")
    if test_eval_batch is not None and pp.get("eval_batch") != test_eval_batch:
        raise click.ClickException(
            f"postproc was refit at eval batch {pp.get('eval_batch')!r}, test inference runs at eval batch "
            f"{test_eval_batch}: the parity gate certifies the batch test inference uses")
    if freeze_commit is not None and pp.get("git_commit") != freeze_commit:
        raise click.ClickException(
            f"postproc was refit by code {pp.get('git_commit')!r}, not the clean freeze commit {freeze_commit[:10]}: "
            "the parity gate and the test threshold come from the frozen code; re-run --split val "
            "--refit-postproc --aggregator-lock ... at the freeze tag")


def after_parity_failure(devtype: str) -> str:
    """What the pre-registration says to do once the parity gate failed on ``devtype``."""
    if devtype == "cpu":
        return ("this is the pre-registered CPU fp32 fallback and the pre-registration defines no further "
                "fallback: stop test inference and record the failure in a dated addendum")
    return (f"the pre-registration says abandon {devtype} for every run and re-score validation and test on CPU "
            "in fp32 (--device cpu --no-amp)")


def reportable_ckpt(ck: dict) -> bool:
    """A checkpoint test inference could accept: pre-registered config and seed, training started under
    the freeze, never run with --allow-unfrozen or a PILOT ONLY option."""
    s = ck.get("settings") or {}
    try:
        seed = int(s.get("model_seed"))
    except (TypeError, ValueError):
        return False
    return (s.get("config") in PRIMARY_CONTRASTS and seed in REQUIRED_SEEDS
            and isinstance((ck.get("state") or {}).get("prereg_freeze_first"), dict)
            and ck.get("allow_unfrozen") is not True and not s.get("allow_unfrozen")
            and all(s.get(k) is None for k in PILOT_ONLY_SETTINGS))


def parity_failures(where: Path, devtype: str) -> list[dict]:
    """Failed parity gates on ``devtype`` recorded in folder ``where`` (empty if none)."""
    p = Path(where) / PARITY_FAILED_FILE.format(devtype)
    return json.loads(p.read_text()) if p.exists() else []


def record_parity_failure(where: Path, devtype: str, rec: dict) -> Path:
    """Append ``rec`` to ``where``/PARITY_FAILED_<devtype>.json (never truncated or removed)."""
    where = Path(where)
    where.mkdir(parents=True, exist_ok=True)
    p = where / PARITY_FAILED_FILE.format(devtype)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(parity_failures(where, devtype) + [rec], indent=2))
    os.replace(tmp, p)
    return p


def check_no_parity_failure(out: Path, devtype: str, device_wide: bool) -> None:
    """Refuse a device type on which the parity gate already failed: for this run folder (its
    PARITY_FAILED_<devtype>.json, or a postproc_<devtype>.json whose parity failed) and, with
    ``device_wide`` (a reportable checkpoint), for ANY run (the record in DEFAULT_ROOT). A passing
    retry with other settings can therefore never replace a failure."""
    for where in [Path(out)] + ([Path(DEFAULT_ROOT)] if device_wide else []):
        fails = parity_failures(where, devtype)
        if fails:
            f = fails[0]
            raise click.ClickException(
                f"the parity gate already failed on {devtype} ({where / PARITY_FAILED_FILE.format(devtype)}: "
                f"{f.get('config')} seed {f.get('model_seed')}, |diff| {(f.get('parity') or {}).get('abs_diff')}); "
                f"{after_parity_failure(devtype)}")
    pp_dev = Path(out) / f"postproc_{devtype}.json"
    if pp_dev.exists():
        par = json.loads(pp_dev.read_text()).get("parity")
        if isinstance(par, dict) and par.get("passed") is False:
            raise click.ClickException(
                f"{pp_dev} records a failed parity gate (|diff| {par.get('abs_diff')}); it is never overwritten: "
                f"{after_parity_failure(devtype)}")


def check_data_versions(data_dir: str | Path, ck: dict) -> dict:
    """The packed folder must come from the same front-end / stage-B / EDR as the training cache."""
    want = (ck.get("data_versions") or {}).get("train")
    if not want:
        raise click.ClickException(
            "checkpoint records no data_versions (trained by code older than the pre-registration freeze); "
            f"cannot check that {data_dir} was packed by the same front-end")
    man = json.loads((Path(data_dir) / "MANIFEST.json").read_text())
    got = {k: man.get(k) for k in DATA_VERSION_KEYS}
    if got != {k: want.get(k) for k in DATA_VERSION_KEYS}:
        raise click.ClickException(
            f"{data_dir} MANIFEST {got} != training cache {want}: the cache was packed by a different "
            "front-end / stage-B / EDR version; re-pack it with the frozen code")
    return got


def check_test_run(ck: dict, ckpt_sha: str, out: Path, default_out: Path, pre: dict) -> str:
    """Test-only refusals that need no data (see the module docstring). Returns the training commit."""
    system = _platform_system()
    if system != "Darwin":
        raise click.ClickException(
            f"test inference runs only on the Mac (platform.system() == 'Darwin'); this is {system!r}")
    cfg, seed = ck["settings"]["config"], int(ck["settings"]["model_seed"])
    if cfg not in PRIMARY_CONTRASTS or seed not in REQUIRED_SEEDS:
        raise click.ClickException(
            f"checkpoint is config {cfg!r} seed {seed}; only configs {sorted(PRIMARY_CONTRASTS)} with seeds "
            f"{list(REQUIRED_SEEDS)} are pre-registered for test (F6 is outside the pre-registration)")
    if out.resolve() != default_out.resolve():
        raise click.ClickException(
            f"--out-dir {out} is not {default_out}: test inference for {cfg} seed {seed} writes only into "
            "that run folder (omit --out-dir)")
    first = (ck.get("state") or {}).get("prereg_freeze_first")
    if not isinstance(first, dict):
        raise click.ClickException(
            "the checkpoint did not start training under the frozen pre-registration (prereg_freeze_first is "
            f"{first!r}: an --allow-unfrozen pilot or a run started before the freeze); it is never reportable")
    if first.get("body_sha256") != pre["prereg_body_sha256"]:
        raise click.ClickException(
            f"the checkpoint started training under pre-registration body {str(first.get('body_sha256'))[:12]}, "
            f"but the frozen body is {pre['prereg_body_sha256'][:12]}")
    if ck.get("allow_unfrozen") is True or (ck.get("settings") or {}).get("allow_unfrozen"):
        raise click.ClickException("the checkpoint was trained with --allow-unfrozen (pilot only, never reportable)")
    pilot = {k: (ck.get("settings") or {}).get(k) for k in PILOT_ONLY_SETTINGS}
    if any(v is not None for v in pilot.values()):
        raise click.ClickException(
            f"the checkpoint was trained with PILOT ONLY options {pilot} (--max-passes / --max-batches-per-pass): "
            "the pre-registered recipe's pass cap is never extended and passes are never truncated; never "
            "reportable")
    mpath = out / "metrics.json"
    if not mpath.exists():
        raise click.ClickException(
            f"{mpath} missing: download the run's training folder (Drive RESULTS/{cfg}/seed{seed}/) into {out} "
            "before test inference")
    m = json.loads(mpath.read_text())
    if m.get("best_ckpt_sha256") != ckpt_sha:
        raise click.ClickException(
            f"{mpath} best_ckpt_sha256 {str(m.get('best_ckpt_sha256'))[:12]} != sha256(--ckpt) {ckpt_sha[:12]}: "
            "--ckpt is not the best checkpoint of this run")
    if m.get("config") != cfg or int(m.get("model_seed", -1)) != seed:
        raise click.ClickException(f"{mpath} is for {m.get('config')} seed {m.get('model_seed')}, not {cfg} seed {seed}")
    ms = m.get("settings") or {}
    if m.get("allow_unfrozen") is True or ms.get("allow_unfrozen"):
        raise click.ClickException(f"{mpath} says the run was trained with --allow-unfrozen (never reportable)")
    mpilot = {k: ms.get(k) for k in PILOT_ONLY_SETTINGS if ms.get(k) is not None}
    if mpilot:
        raise click.ClickException(f"{mpath} records PILOT ONLY options {mpilot}; never reportable")
    return train_commit_history(ck, m)


def check_test_precision(ck: dict, devtype: str, amp: bool) -> None:
    """The pre-registered inference precision, for test inference AND the parity-gate refit (the gate
    is measured at the test precision): on MPS the training precision (fp16 autocast, or fp32 for a
    config retrained under the NaN fallback); the CPU fallback is always fp32."""
    train_amp = bool((ck.get("settings") or {}).get("amp", True))
    if devtype == "mps" and amp != train_amp:
        raise click.ClickException(
            f"test inference and the parity-gate refit on MPS run at the training precision (training "
            f"amp={train_amp}: fp16 autocast, or fp32 (--no-amp) for a config retrained under the NaN fallback); "
            f"this run has amp={amp}")
    if devtype == "cpu" and amp:
        raise click.ClickException("test inference and the parity-gate refit on CPU run in fp32 (--no-amp)")


def check_eval_batch(eval_batch: int) -> None:
    """Test inference and the parity gate (any validation scoring under the lock) run at the
    pre-registered eval batch."""
    if eval_batch != PREREG_EVAL_BATCH:
        raise click.ClickException(
            f"test inference and the parity gate (validation scored under --aggregator-lock) run at the "
            f"pre-registered eval batch {PREREG_EVAL_BATCH}; this run has --eval-batch {eval_batch} (score "
            "without --aggregator-lock for diagnostics at another batch)")


def parity_record(pp: dict, agg: str, this_auc: float, device: str, amp: bool) -> dict:
    """Validation AUC on this device vs the training device, for the locked aggregator."""
    try:
        train_auc = float(pp["aggregator_rule"]["aucs"][agg])
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(
            f"postproc has no training-device validation AUC for aggregator {agg!r} (aggregator_rule.aucs)") from exc
    diff = abs(float(this_auc) - train_auc)
    return {"aggregator": agg, "train_device_auc": train_auc, "this_device_auc": float(this_auc),
            "abs_diff": diff, "tol": PARITY_TOL, "passed": bool(diff <= PARITY_TOL), "device": device, "amp": amp,
            "train_device": pp.get("train_device", pp.get("device")),
            "train_amp": pp.get("train_amp", pp.get("amp"))}


def frontend_qc(data_dir: str | Path, which: str) -> dict:
    """Front-end QC of the packed folder (``qc_<split>.jsonl``, last record per subject wins).

    Detector counts come from ``ecg_detector`` (absent when the ECG was flat or too short:
    counted as "none"); failed subjects are those with ``ecg_ok`` False (kept with zero RR/EDR)
    or no detector.
    """
    p = Path(data_dir) / f"qc_{which}.jsonl"
    if not p.exists():
        return {"frontend_qc_file": None, "frontend_detector_counts": None, "frontend_failed_subjects": None,
                "frontend_failed_reasons": None}
    recs: dict[int, dict] = {}
    for ln in p.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            recs[int(r["subject_id"])] = r
    det = {s: (r.get("ecg_detector", r.get("detector")) or "none") for s, r in recs.items()}
    failed = {s: why for s, r in recs.items() if (why := _frontend_failure(r, det[s])) is not None}
    return {"frontend_qc_file": str(p.resolve()), "frontend_qc_sha256": sha256_file(p),
            "frontend_detector_counts": dict(sorted(Counter(det.values()).items())),
            "frontend_failed_subjects": sorted(failed), "frontend_failed_reasons": {str(s): failed[s] for s in sorted(failed)}}


def _frontend_failure(rec: dict, detector: str) -> str | None:
    if rec.get("status") != "done":
        return f"status {rec.get('status')}"
    if rec.get("ecg_ok") is False:
        return rec.get("ecg_reason") or "ecg_ok false"
    if detector == "none":
        return "no R-peak detector ran"
    return None


def _save_npy(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.save(f, arr)


def _rows_aligned(pred: pd.DataFrame, df: pd.DataFrame) -> bool:
    return (np.array_equal(pred["subject_id"].to_numpy(np.int64), df["subject_id"].to_numpy(np.int64))
            and np.array_equal(pred["epoch_idx"].to_numpy(np.int64), df["epoch_idx"].to_numpy(np.int64)))


@click.command()
@click.option("--ckpt", type=click.Path(exists=True, dir_okay=False), required=True, help="best.pt")
@click.option("--postproc", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), required=True,
              help="packed split folder (val or test)")
@click.option("--split", "which", type=click.Choice(["val", "test"]), required=True)
@click.option("--prereg", type=click.Path(exists=True, dir_okay=False), required=True,
              help="pre-registration vault note; its body must match the frozen record (body and whole-file "
                   "sha256 logged)")
@click.option("--out-dir", type=click.Path(file_okay=False), default=None,
              help="default models/aim2_dl_olsen_v1/<config>/seed<k> from the checkpoint; --split test refuses "
                   "any other folder")
@click.option("--split-json", type=click.Path(exists=True, dir_okay=False), default=str(DEFAULT_SPLIT),
              show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--amp/--no-amp", default=True, show_default=True)
@click.option("--eval-batch", type=int, default=256, show_default=True)
@click.option("--refit-postproc", is_flag=True,
              help="val only (needs --aggregator-lock): refit the threshold on these predictions; parity gate")
@click.option("--freeze-tag", default=None, help="test only: git tag of the frozen code (must equal HEAD)")
@click.option("--key", type=click.Path(dir_okay=False), default=str(DEFAULT_KEY), show_default=True,
              help="test only: canonical key parquet (subject_id, epoch_idx, apnoea_label)")
@click.option("--rerun-reason", default=None, help="test only: logged bug reason to replace a prior run")
@click.option("--aggregator-lock", type=click.Path(exists=True, dir_okay=False), default=None,
              help="AGGREGATOR_LOCK.json (required for test and for --refit-postproc; checked for val when given)")
def main(ckpt, postproc, data_dir, which, prereg, out_dir, split_json, device, amp, eval_batch,
         refit_postproc, freeze_tag, key, rerun_reason, aggregator_lock) -> None:
    """Predict epochs for the val or test split (see module docstring)."""
    if which == "val" and rerun_reason:
        raise click.UsageError("--rerun-reason is test-only")
    if refit_postproc and which == "test":
        raise click.UsageError("--refit-postproc is validation-only")
    if refit_postproc and not aggregator_lock:
        raise click.UsageError("--refit-postproc requires --aggregator-lock (the parity gate scores the locked "
                               "aggregator)")
    if which == "test":
        if not freeze_tag:
            raise click.UsageError("--split test requires --freeze-tag (test inference runs after the code freeze)")
        if not aggregator_lock:
            raise click.UsageError("--split test requires --aggregator-lock (the frozen AGGREGATOR_LOCK.json)")
    try:
        pre = check_prereg(prereg)
    except PreregMismatch as exc:
        raise click.ClickException(str(exc)) from exc

    split = load_split(split_json)
    dev = resolve_device(device)
    use_amp = bool(amp and dev.type in ("cuda", "mps"))
    ckpt_sha = sha256_file(ckpt)
    model, ck = load_model_from_ckpt(ckpt, dev)
    cfg, seed = ck["settings"]["config"], int(ck["settings"]["model_seed"])
    default_out = DEFAULT_ROOT / cfg / f"seed{seed}"
    out = Path(out_dir) if out_dir else default_out
    channels = tuple(ck["channels"])
    prov = {
        "split": which,
        "config": cfg,
        "model_seed": seed,
        "channels": list(channels),
        **pre,
        "ckpt": str(Path(ckpt).resolve()),
        "ckpt_sha256": ckpt_sha,
        "train_git_commit": ck.get("git_commit"),
        "train_git_commits": (ck.get("state") or {}).get("git_commits"),
        "train_prereg_body_sha256": ((ck.get("state") or {}).get("prereg_freeze_first") or {}).get("body_sha256"),
        "train_allow_unfrozen": ck.get("allow_unfrozen"),
        "postproc": str(Path(postproc).resolve()),
        "postproc_sha256": sha256_file(postproc),
        "split_json": split["path"],
        "split_sha256": split["sha256"],
        "git_commit": git_commit(),
        "device": str(dev),
        "amp": use_amp,
        "eval_batch": eval_batch,
        "platform": _platform_system(),
        "platform_detail": {"machine": platform.machine(), "release": platform.release(),
                            "python": platform.python_version(), "torch": torch.__version__},
        "rerun_reason": None,
        "replaced_previous": None,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    prior: list[str] = []
    if which == "test":  # every check that does not need data runs first
        train_commit = check_test_run(ck, ckpt_sha, out, default_out, pre)
        check_test_precision(ck, dev.type, use_amp)
        check_eval_batch(eval_batch)
        prov["freeze_commit"] = check_freeze(freeze_tag)
        prov["freeze_tag"] = freeze_tag
        prov["code_lineage"] = check_code_lineage(train_commit, prov["freeze_commit"])
    elif aggregator_lock:  # scored under the lock (refit or not) = the parity gate: test precision and batch
        check_test_precision(ck, dev.type, use_amp)
        check_eval_batch(eval_batch)
    prov["data_dir"] = str(Path(data_dir).resolve())
    prov["data_versions"] = check_data_versions(data_dir, ck)
    prov["data_manifest_sha256"] = sha256_file(Path(data_dir) / "MANIFEST.json")

    pp = json.loads(Path(postproc).read_text())
    lock = (load_aggregator_lock(aggregator_lock, ckpt_sha, cfg, seed, for_test=(which == "test"))
            if aggregator_lock else None)
    if lock is not None:
        prov.update({"aggregator_lock": str(Path(aggregator_lock).resolve()),
                     "aggregator_lock_sha256": sha256_file(aggregator_lock), "aggregator_locked": lock["aggregator"]})
    test_mode = which == "test"
    check_postproc(pp, ckpt_sha, lock, dev.type if test_mode else None, use_amp if test_mode else None,
                   freeze_commit=prov["freeze_commit"] if test_mode else None,
                   test_eval_batch=eval_batch if test_mode else None)
    prov["postproc_git_commit"] = pp.get("git_commit")
    prov["postproc_checks"] = ("ckpt sha256 matched" + (", aggregator == lock (lock from M seed 42, rule "
                                                        "re-applied)" if lock else "")
                               + (", fitted on this device type, amp and eval batch at the freeze commit, "
                                  "parity passed" if test_mode else ""))
    reportable = reportable_ckpt(ck)
    if which == "test" or refit_postproc:  # a failed parity gate on this device is never retried or bypassed
        check_no_parity_failure(out, dev.type, device_wide=(which == "test" or reportable))
    agg = lock["aggregator"] if lock is not None else pp["aggregator"]
    thr = float(pp["threshold"])

    if which == "test":
        prior = prior_test_outputs(out)
        if prior and not rerun_reason:
            raise click.ClickException(
                f"test inference already ran in {out} (found {prior}); it runs once. Pass --rerun-reason "
                "'<logged bug reason>' to replace it (the old outputs are kept under superseded_<UTC>/).")
        if rerun_reason and not prior:
            raise click.UsageError(f"--rerun-reason given but {out} holds no earlier test inference")
        prov["rerun_reason"] = rerun_reason
    else:
        out.mkdir(parents=True, exist_ok=True)

    ps = PackedSplit(data_dir, channels, allow_test=(which == "test"))
    if ps.split != which:
        raise click.ClickException(f"{data_dir} manifest split is {ps.split!r}, not {which!r}")
    assert_ids_in_split(ps.subject_ids, split, which)
    prov.update(frontend_qc(data_dir, which))
    if which == "test":  # every check that does not need predictions runs before inference
        if not np.array_equal(np.sort(ps.subject_ids), split["test"]):
            raise click.ClickException("test cache does not hold exactly the split's test subjects")
        key_df = pd.read_parquet(key, columns=["subject_id", "epoch_idx", "apnoea_label"])
        if not np.array_equal(np.unique(key_df["subject_id"].to_numpy(np.int64)), split["test"]):
            raise click.ClickException(f"key {key} does not hold exactly the split's test subjects")
        _log_attempt(out, {"event": "inference_start", "rerun_reason": rerun_reason, "prior": prior,
                           "freeze_commit": prov["freeze_commit"], "ckpt_sha256": ckpt_sha})
    t0 = time.perf_counter()
    df, sec30 = score_split(model, ps, dev, use_amp, batch=eval_batch, return_seconds=True)
    prov["predict_seconds"] = round(time.perf_counter() - t0, 1)

    if which == "val":
        y = df["apnoea_label"].to_numpy()
        aucs = {c[2:]: float(roc_auc_score(y, df[c])) for c in df.columns if c.startswith("p_")}
        parity = parity_record(pp, agg, aucs[agg], str(dev), use_amp)
        if refit_postproc:
            thr, f1 = threshold_f1max(y, df[f"p_{agg}"].to_numpy())
            new_pp = {**pp, "threshold": thr, "val_f1_at_threshold": f1, "device": str(dev), "amp": use_amp,
                      "eval_batch": eval_batch,  # test inference refuses another batch
                      "train_device": parity["train_device"], "train_amp": parity["train_amp"],
                      "parity": parity,
                      "git_commit": prov["git_commit"],  # test inference refuses a refit not at the freeze commit
                      "refit_from": str(Path(postproc).resolve()),
                      "refit_from_sha256": prov["postproc_sha256"],
                      "aggregator_lock_sha256": prov.get("aggregator_lock_sha256"),
                      "refit_note": "threshold refit on validation predictions recomputed on this device; "
                                    "aggregator kept frozen"}
            (out / f"postproc_{dev.type}.json").write_text(json.dumps(new_pp, indent=2))
        pred = assemble_predictions(df.rename(columns={f"p_{agg}": "pred_prob"}),
                                    df[["subject_id", "epoch_idx", "apnoea_label"]], thr)
        if not _rows_aligned(pred, df):
            raise click.ClickException("validation predictions are not row-aligned with the per-second array")
        pred.to_parquet(out / f"val_predictions_{dev.type}.parquet", index=False)
        sec_path = out / f"val_sec_probs_{dev.type}.npy"
        _save_npy(sec_path, sec30)
        prov.update({"val_auc_by_aggregator": aucs, "aggregator": agg, "threshold": thr, "parity": parity,
                     "val_sec_probs_sha256": sha256_file(sec_path), "val_sec_probs_shape": list(sec30.shape),
                     "val_sec_probs_dtype": str(sec30.dtype),
                     "n_rows": int(len(pred)), "n_subjects": int(pred["subject_id"].nunique())})
        (out / f"predict_val_{dev.type}.json").write_text(json.dumps(prov, indent=2))
        click.echo(json.dumps({k: prov[k] for k in ("val_auc_by_aggregator", "aggregator", "threshold",
                                                   "n_rows", "predict_seconds")}))
        click.echo(f"parity ({agg}): this device {parity['this_device_auc']:.6f} vs training device "
                   f"{parity['train_device_auc']:.6f}, |diff| {parity['abs_diff']:.2e} (tol {PARITY_TOL})")
        if not parity["passed"]:
            rec = {"time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "config": cfg, "model_seed": seed, "ckpt_sha256": ckpt_sha, "run_dir": str(out.resolve()),
                   "refit_postproc": bool(refit_postproc), "eval_batch": eval_batch, "parity": parity}
            if lock is not None:  # the pre-registered measurement (locked aggregator): kept for good
                where = [record_parity_failure(out, dev.type, rec)]
                if reportable:  # device-wide: blocks refits and test inference on this device for every run
                    where.append(record_parity_failure(DEFAULT_ROOT, dev.type, rec))
                note = (f"Recorded in {', '.join(str(w) for w in where)} (never overwritten; refits and test "
                        f"inference on {dev.type} are refused from now on)")
            else:
                note = "Not recorded: scored without --aggregator-lock, so this is not the pre-registered gate"
            raise click.ClickException(
                f"PARITY FAILED on {dev} (amp={use_amp}): |validation AUC({agg}) here - training device| = "
                f"{parity['abs_diff']:.4g} > {PARITY_TOL}: {after_parity_failure(dev.type)}. {note}")
        return

    # ---- test: predictions + per-second probabilities + provenance only, no metrics ----
    prov["key"] = str(Path(key).resolve())
    prov["key_sha256"] = sha256_file(key)
    prov["test_ids_sha256"] = hashlib.sha256(
        json.dumps(sorted(int(i) for i in split["test"])).encode()).hexdigest()
    declared = json.loads(Path(split_json).read_text()).get("ids_sha256")
    if isinstance(declared, dict):
        prov["split_declared_test_ids_sha256"] = declared.get("test")
    prov["parity"] = pp["parity"]
    pred = assemble_predictions(df.rename(columns={f"p_{agg}": "pred_prob"}), key_df, thr)
    if not _rows_aligned(pred, df):
        raise click.ClickException("assembled test predictions are not in the scored row order; the per-second "
                                   "array could not be row-aligned with test_predictions.parquet")
    target, tmp = out / "test_predictions.parquet", out / "test_predictions.tmp.parquet"
    sec_target, sec_tmp = out / "test_sec_probs.npy", out / "test_sec_probs.tmp.npy"
    pred.to_parquet(tmp, index=False)
    _save_npy(sec_tmp, sec30)
    prov.update({"aggregator": agg, "threshold": thr, "n_rows": int(len(pred)),
                 "n_subjects": int(pred["subject_id"].nunique()),
                 "test_predictions_sha256": sha256_file(tmp),
                 "test_sec_probs_sha256": sha256_file(sec_tmp), "test_sec_probs_shape": list(sec30.shape),
                 "test_sec_probs_dtype": str(sec30.dtype),
                 "test_sec_probs_note": "per-second probabilities, float32 (n_rows, 30), row-aligned with "
                                        "test_predictions.parquet (second s of epoch = column s)"})
    missing = [k for k in PREDICT_TEST_KEYS if k not in prov]
    if missing:
        raise click.ClickException(f"internal: predict_test.json would lack {missing}")
    if prior:  # archive the previous outputs together, only now that this run succeeded
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest, k = out / f"superseded_{stamp}", 1
        while dest.exists():  # evaluate_aim2_dl.archive_previous may have archived here in this UTC second
            dest, k = out / f"superseded_{stamp}_{k}", k + 1
        dest.mkdir()
        moved = []
        for name in TEST_OUTPUTS + EVALUATE_OUTPUTS:  # evaluate's metrics travel with their predictions
            if (out / name).exists():
                (out / name).rename(dest / name)
                moved.append(name)
        (dest / "superseded_reason.json").write_text(json.dumps(
            {"archived_utc": stamp, "rerun_reason": rerun_reason, "moved": moved, "prior": prior}, indent=2))
        prov["replaced_previous"] = str(dest)
    os.replace(tmp, target)
    os.replace(sec_tmp, sec_target)
    ptmp = out / "predict_test.json.tmp"
    ptmp.write_text(json.dumps(prov, indent=2))
    os.replace(ptmp, out / "predict_test.json")
    _log_attempt(out, {"event": "done", "test_predictions_sha256": prov["test_predictions_sha256"],
                       "test_sec_probs_sha256": prov["test_sec_probs_sha256"]})
    click.echo(f"wrote {target} ({len(pred):,} rows) and {sec_target}; no metrics computed here")


if __name__ == "__main__":
    main()
