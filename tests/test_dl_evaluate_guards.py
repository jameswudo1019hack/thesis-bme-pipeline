"""scripts/evaluate_aim2_dl.py --split test on SYNTHETIC run folders written by hand.

Every run folder follows the predict_test.json contract of predict_aim2_dl.py (spec B3.i) and
holds a training metrics.json; the matched cells hold metrics.json with the pinned model sha256
and the frozen prereg body; prereg.AIM2_DL_FREEZE points at a tmp freeze record. Checks:

* a valid M + P4 set is evaluated: claim-rule verdicts for the primary contrasts and the gap,
  descriptive ladder deltas, the short-hypopnoea endpoint, E1, mean +- SD of every metric, and
  the training facts of every run;
* an addendum appended to the note between predict and evaluate does not break evaluate;
* p-values only on the primary AUC-ROC contrasts (not AUC-PR, cross pairs, ladder, short hypopnoeas);
* a second evaluation moves the first summary and per-run metrics to superseded_<UTC>/;
* refusals (nothing is computed or written): whole-file-hash-only or wrong-body provenance,
  folder vs predict_test.json config / seed, two freeze commits, a run trained (resumed or
  finalised) at more than one commit, no passed code lineage, a threshold not refit at the
  freeze commit, seeds other than 42/43/44,
  an unpinned comparator, --contrast / --gap with --split test, PILOT ONLY training options,
  a recorded parity failure on the inference device, a subject without NSRR AHI, and the
  other provenance rules.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_aim2_dl  # noqa: E402
from thesis_pipeline import prereg  # noqa: E402
from thesis_pipeline.dl_eval import PINNED_REF_MODEL_SHA256, short_hypopnoea_mask  # noqa: E402
from thesis_pipeline.extended_metrics import write_extended_metrics  # noqa: E402

N_BOOT = 30
SUBJECTS = np.arange(400101, 400113, dtype=np.int32)
N_EP = 30
BODY = "# Aim 2 DL pre-registration (synthetic)\n\nPrimary contrasts: M vs LGBM-ECG, P4 vs LGBM-ECG+belt.\n\n"
FREEZE_COMMIT, TRAIN_COMMIT = "f" * 40, "c" * 40


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _pred(key: pd.DataFrame, rng: np.random.Generator, signal: float) -> pd.DataFrame:
    y = key["apnoea_label"].to_numpy()
    p = np.clip(signal * y + rng.random(len(y)) * (1 - signal), 0.0, 1.0)
    return pd.DataFrame({"subject_id": key["subject_id"].to_numpy(np.int32),
                         "epoch_idx": key["epoch_idx"].to_numpy(np.int32),
                         "apnoea_label": key["apnoea_label"].to_numpy(np.int8),
                         "pred_prob": p.astype(np.float64), "pred_label": (p > 0.5).astype(np.int64)})


class Env:
    def __init__(self, tmp: Path, monkeypatch):
        rng = np.random.default_rng(7)
        self.tmp = tmp
        sid = np.repeat(SUBJECTS, N_EP)
        y = (rng.random(len(sid)) < 0.35).astype(np.int8)
        y[::N_EP], y[1::N_EP] = 1, 0  # both classes in every subject
        self.key_df = pd.DataFrame({"subject_id": sid, "epoch_idx": np.tile(np.arange(N_EP, dtype=np.int32),
                                                                             len(SUBJECTS)), "apnoea_label": y})
        self.key = tmp / "key.parquet"
        self.key_df.to_parquet(self.key, index=False)
        kind = np.where(y == 1, np.where(rng.random(len(y)) < 0.7, "hypopnoea", "apnoea"), "")
        dur = np.where(y == 1, rng.uniform(10, 40, len(y)), np.nan)
        self.context = tmp / "test_epoch_context.parquet"
        pd.DataFrame({"subject_id": sid.astype(np.int64), "epoch_idx": self.key_df["epoch_idx"],
                      "sleep_stage": "N2", "event_duration_s": dur, "event_kind": kind,
                      "xml_label": y}).to_parquet(self.context, index=False)
        monkeypatch.setattr(evaluate_aim2_dl, "EPOCH_CONTEXT_SHA256", _sha(self.context))
        self.csv = tmp / "nsrr.csv"
        pd.DataFrame({"nsrrid": SUBJECTS.astype(int), "ahi_a0h3a": rng.uniform(0, 40, len(SUBJECTS))}).to_csv(
            self.csv, index=False)

        # pre-registration note and its freeze record
        self.note = tmp / "prereg.md"
        self.note.write_text(BODY + "## Addenda\n\n### 2026-10-05 recipe tier: A\n")
        self.body = hashlib.sha256(BODY.encode()).hexdigest()
        freeze = tmp / "aim2_dl_olsen_v1_freeze.json"
        freeze.write_text(json.dumps({"note": str(self.note), "body_sha256": self.body, "vault_commit": "v" * 40}))
        monkeypatch.setattr(prereg, "AIM2_DL_FREEZE", freeze)

        # matched LightGBM cells and one ladder reference
        self.refs = {}
        for name, signal in (("ecg_only", 0.20), ("ecg_belt", 0.30)):
            d = tmp / "cells" / name / "test"
            self._ref(d, _pred(self.key_df, rng, signal))
            (d / "metrics.json").write_text(json.dumps({
                "split": "test", "model_sha256": PINNED_REF_MODEL_SHA256[name],
                "prereg": {"prereg": str(self.note), "prereg_body_sha256": self.body}}))
            self.refs[name] = d
        self.ladder = tmp / "ladder" / "full"
        self._ref(self.ladder, _pred(self.key_df, rng, 0.40))

        # DL runs: M and P4 x seeds 42, 43, 44
        self.root = tmp / "runs"
        for cfg, signal in (("M", 0.25), ("P4", 0.35)):
            for seed in (42, 43, 44):
                self.write_run(cfg, seed, _pred(self.key_df, rng, signal + 0.01 * (seed - 42)))
        # one run that hit its pass cap
        tm = self.run_dir("P4", 44) / "metrics.json"
        tm.write_text(json.dumps({**json.loads(tm.read_text()), "stop_reason": "pass cap 25", "passes_run": 25}))

    def _ref(self, d: Path, df: pd.DataFrame) -> None:
        d.mkdir(parents=True)
        df.to_parquet(d / "test_predictions.parquet", index=False)
        write_extended_metrics(d, df, n_bootstrap_subj=N_BOOT)

    def run_dir(self, cfg: str, seed: int) -> Path:
        return self.root / cfg / f"seed{seed}"

    def write_run(self, cfg: str, seed: int, df: pd.DataFrame) -> None:
        d = self.run_dir(cfg, seed)
        d.mkdir(parents=True)
        df.to_parquet(d / "test_predictions.parquet", index=False)
        sec = np.repeat(df["pred_prob"].to_numpy(np.float32)[:, None], 30, axis=1)
        np.save(d / "test_sec_probs.npy", sec)
        ckpt_sha = hashlib.sha256(f"{cfg}{seed}".encode()).hexdigest()
        prov = {
            "split": "test", "config": cfg, "model_seed": seed, "freeze_tag": "aim2-dl-freeze-v1",
            "freeze_commit": FREEZE_COMMIT, "git_commit": FREEZE_COMMIT,
            "prereg": str(self.note), "prereg_sha256": _sha(self.note), "prereg_body_sha256": self.body,
            "prereg_freeze_sha256": "e" * 64, "ckpt": f"/drive/{cfg}/seed{seed}/best.pt", "ckpt_sha256": ckpt_sha,
            "train_git_commit": TRAIN_COMMIT, "train_git_commits": [TRAIN_COMMIT],
            "code_lineage": {"train_commit": TRAIN_COMMIT, "freeze_commit": FREEZE_COMMIT,
                             "train_is_ancestor_of_freeze": True, "model_files_changed": [], "passed": True},
            "train_prereg_body_sha256": self.body, "postproc_sha256": "d" * 64,
            "postproc_git_commit": FREEZE_COMMIT,
            "parity": {"aggregator": "mean", "train_device_auc": 0.80, "this_device_auc": 0.8004, "abs_diff": 4e-4,
                       "tol": 0.002, "passed": True, "device": "mps", "amp": True},
            "aggregator": "mean", "aggregator_locked": "mean", "aggregator_lock_sha256": "a" * 64,
            "threshold": 0.5, "device": "mps", "amp": True, "platform": "Darwin",
            "data_manifest_sha256": "b" * 64,
            "data_versions": {"frontend_version": "dl-frontend-1.1", "stage_b_version": "stage-b-1.0",
                              "edr_method": "psa"},
            "key": str(self.key), "key_sha256": _sha(self.key), "split_sha256": "8" * 64, "test_ids_sha256": "9" * 64,
            "test_predictions_sha256": _sha(d / "test_predictions.parquet"),
            "test_sec_probs_sha256": _sha(d / "test_sec_probs.npy"), "test_sec_probs_shape": list(sec.shape),
            "n_rows": len(df), "n_subjects": len(SUBJECTS), "frontend_detector_counts": {"neurokit": len(SUBJECTS)},
            "frontend_failed_subjects": [], "rerun_reason": None, "replaced_previous": None,
            "started": "2026-11-01T00:00:00Z", "predict_seconds": 1.0,
        }
        (d / "predict_test.json").write_text(json.dumps(prov))
        (d / "metrics.json").write_text(json.dumps({
            "config": cfg, "model_seed": seed, "best_ckpt_sha256": ckpt_sha, "git_commit": TRAIN_COMMIT,
            "git_commits": [TRAIN_COMMIT], "scaler_skipped_steps": 2,
            "recipe": {"name": "A", "batch": 128, "lr_patience": 3, "stop_patience": 6, "max_passes": 25,
                       "subsample": 1.0},
            "gpu_name": "NVIDIA A100-SXM4-40GB", "train_seconds_total": 3600.0 + seed, "val_seconds_total": 120.0,
            "passes_run": 12, "best_pass": 5, "stop_reason": "early stop: 6 passes without improvement",
            "amp": True, "nonfinite_losses": 0, "allow_unfrozen": False,
            "settings": {"config": cfg, "model_seed": seed, "recipe": "A", "amp": True, "device": "auto", "lr": 1e-3,
                         "weight_decay": 1e-4, "min_delta": 1e-4, "hidden": 128, "eval_batch": 512, "prefetch": 2,
                         "ckpt_every_steps": 0, "stop_after_passes": None, "max_passes": None,
                         "max_batches_per_pass": None, "aggregator": "mean", "allow_deferred": False,
                         "num_threads": None, "split_seed": 42, "allow_unfrozen": False}}))

    def prov(self, cfg: str, seed: int) -> dict:
        return json.loads((self.run_dir(cfg, seed) / "predict_test.json").read_text())

    def set_prov(self, cfg: str, seed: int, **kw) -> None:
        p = self.prov(cfg, seed)
        p.update(kw)
        for k in [k for k, v in kw.items() if v is _DROP]:
            del p[k]
        (self.run_dir(cfg, seed) / "predict_test.json").write_text(json.dumps(p))

    def base_args(self, refs: tuple[str, ...] = ("ecg_only", "ecg_belt")) -> list[str]:
        out = ["--runs-root", str(self.root), "--split", "test", "--key", str(self.key), "--csv", str(self.csv),
               "--subject-metadata", str(self.tmp / "none.parquet"), "--n-bootstrap", str(N_BOOT),
               "--epoch-context", str(self.context), "--prereg", str(self.note)]
        for r in refs:
            out += ["--ref", f"{r}={self.refs[r]}"]
        return out

    def invoke(self, *extra: str, refs: tuple[str, ...] = ("ecg_only", "ecg_belt"), args: list[str] | None = None):
        return CliRunner().invoke(evaluate_aim2_dl.main, (self.base_args(refs) if args is None else args)
                                  + list(extra))


_DROP = object()


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    return Env(tmp_path, monkeypatch)


def _nothing_written(env: Env) -> bool:
    return (not (env.root / "evaluation_summary_test.json").exists()
            and not any(env.root.glob("*/seed*/metrics_extended.json")))


# --------------------------------------------------------------------------- the valid set


def test_valid_set_writes_verdicts_and_every_endpoint(env):
    r = env.invoke("--ladder", f"full={env.ladder}")
    assert r.exit_code == 0, r.output
    s = json.loads((env.root / "evaluation_summary_test.json").read_text())

    assert s["prereg"] == prereg.check_prereg(env.note)
    assert s["prereg"]["prereg_body_sha256"] == env.body
    assert s["evaluated_configs"] == ["M", "P4"] and s["required_seeds"] == [42, 43, 44]
    assert s["freeze_commits"] == [FREEZE_COMMIT] and s["train_git_commit"] == TRAIN_COMMIT
    assert s["aggregator"] == "mean" and s["device_type"] == "mps" and s["recipe"]["name"] == "A"
    assert s["amp_by_config"] == {"M": True, "P4": True}
    assert "only for the three primary contrasts" in s["multiplicity"]
    assert s["refs"]["ecg_only"]["model_sha256"] == PINNED_REF_MODEL_SHA256["ecg_only"]

    for cfg, ref in (("M", "ecg_only"), ("P4", "ecg_belt")):
        c = s["configs"][cfg]
        assert c["seeds"] == [42, 43, 44] and list(c["contrasts"]) == [ref]
        blk = c["contrasts"][ref]
        assert len(blk["auc_per_seed"]) == len(blk["aupr_per_seed"]) == 3
        assert all(d["delta_point"] is not None for d in blk["auc_per_seed"] + blk["aupr_per_seed"])
        assert blk["auc_seed_averaged"]["delta_point"] is not None
        cr = blk["claim_rule_auc"]
        assert cr["verdict"] in {"A", "B", "C", "D", "E"} and cr["label"].startswith(cr["verdict"] + ":")
        assert cr["deltas"] == [d["delta_point"] for d in blk["auc_per_seed"]]
        assert cr["seed_averaged_ci"] == [blk["auc_seed_averaged"]["delta_ci_low"],
                                          blk["auc_seed_averaged"]["delta_ci_high"]]
        assert "claim_rule" not in {k for k in blk if k != "claim_rule_auc"}
        # p-values only on the primary AUC-ROC contrast; AUC-PR deltas are estimation only
        assert all("p_two_sided" in d and "p_text" in d for d in blk["auc_per_seed"] + [blk["auc_seed_averaged"]])
        assert not any(k.startswith("p_") for d in blk["aupr_per_seed"] + [blk["aupr_seed_averaged"]] for k in d)
        # mean +- SD of every numeric metric
        msd = c["metrics_mean_sd"]
        assert msd["auc_roc"]["mean"] == pytest.approx(c["auc_mean"]) and msd["auc_roc"]["sd"] == pytest.approx(
            c["auc_sd"])
        assert {"brier", "ece", "f1", "sensitivity", "specificity", "precision"} <= set(msd)
        assert "metric_notes.calibration" not in msd
        # per-run provenance and training facts
        for seed in ("42", "43", "44"):
            ps = c["per_seed"][seed]
            assert ps["provenance"]["prereg_body_sha256"] == env.body
            assert ps["training"]["gpu_name"] == "NVIDIA A100-SXM4-40GB" and ps["training"]["recipe"] == "A"
            assert ps["training"]["scaler_skipped_steps"] == 2
            assert ps["sec_probs"].startswith("test_sec_probs.npy saved")
    assert s["configs"]["P4"]["per_seed"]["44"]["training"]["cap_reached"] is True
    assert s["configs"]["M"]["training_cap_reached"] == {"42": False, "43": False, "44": False}

    gap = s["gaps"]["P4,M=ecg_belt,ecg_only"]
    assert gap["seeds"] == [42, 43, 44] and gap["pairing"] == "by model seed index"
    assert len(gap["per_seed"]) == 3 and len(gap["cross_pairs"]) == 9 and "P4_43-M_42" in gap["cross_pairs"]
    assert gap["claim_rule_auc"]["verdict"] in {"A", "B", "C", "D", "E"}
    assert all("p_two_sided" in d for d in gap["per_seed"] + [gap["seed_averaged"]])
    assert not any(k.startswith("p_") for d in gap["cross_pairs"].values() for k in d)  # reported only

    lad = s["ladder"]["full"]
    assert lad["predictions_sha256"] == _sha(env.ladder / "test_predictions.parquet")
    assert set(lad["configs"]) == {"M", "P4"}
    assert "claim_rule_auc" not in lad["configs"]["M"] and len(lad["configs"]["M"]["auc_per_seed"]) == 3
    lm = lad["configs"]["M"]
    assert not any(k.startswith("p_") for key in ("auc_per_seed", "aupr_per_seed") for d in lm[key] for k in d)
    assert not any(k.startswith("p_") for key in ("auc_seed_averaged", "aupr_seed_averaged") for k in lm[key])

    sh = s["short_hypopnoea"]
    mask, counts = short_hypopnoea_mask(pd.read_parquet(env.context), env.key_df)
    assert (sh["n_pos"], sh["n_neg"]) == (counts["n_pos"], counts["n_neg"]) and counts["n_pos"] > 0
    y = env.key_df["apnoea_label"].to_numpy()
    for seed in (42, 43, 44):
        p = pd.read_parquet(env.run_dir("M", seed) / "test_predictions.parquet")["pred_prob"].to_numpy()
        assert sh["configs"]["M"]["auc_per_seed"][str(seed)] == pytest.approx(roc_auc_score(y[mask], p[mask]))
    assert sh["configs"]["M"]["reference"] == "ecg_only" and len(sh["configs"]["M"]["delta_per_seed"]) == 3
    shm = sh["configs"]["M"]
    assert not any(k.startswith("p_") for d in shm["delta_per_seed"] + [shm["delta_seed_averaged"]] for k in d)
    assert set(sh["refs"]) == {"ecg_only", "ecg_belt"} and set(sh["ladder"]) == {"full"}
    e1 = s["expectations"]["E1"]
    assert e1["value"] == pytest.approx(sh["configs"]["M"]["auc_mean"]) and e1["bound"] == 0.9
    assert e1["met"] is (e1["value"] < 0.9)
    assert (env.run_dir("M", 42) / "metrics_extended.json").exists()


def test_addendum_appended_between_predict_and_evaluate_is_accepted(env):
    predicted_sha = env.prov("M", 42)["prereg_sha256"]
    with open(env.note, "a") as fh:
        fh.write("\n### 2026-11-02 test rerun note\nnothing rerun\n")
    r = env.invoke("--config", "M", refs=("ecg_only",))
    assert r.exit_code == 0, r.output
    s = json.loads((env.root / "evaluation_summary_test.json").read_text())
    assert s["prereg"]["prereg_sha256"] != predicted_sha and s["prereg"]["prereg_body_sha256"] == env.body
    assert s["evaluated_configs"] == ["M"] and s["gaps"] == "skipped: needs M and P4"
    assert s["configs"]["M"]["contrasts"]["ecg_only"]["claim_rule_auc"]["n_seeds"] == 3
    assert "E1" in s["expectations"]


def test_a_second_evaluation_keeps_the_first(env):
    """Seminar order: M first (--config M), the full set later. The M-only summary and M's per-run
    metrics are moved to superseded_<UTC>/ (byte-identical), never overwritten."""
    r = env.invoke("--config", "M", refs=("ecg_only",))
    assert r.exit_code == 0, r.output
    first_summary = (env.root / "evaluation_summary_test.json").read_bytes()
    m42 = env.run_dir("M", 42)
    first_run = {n: (m42 / n).read_bytes() for n in evaluate_aim2_dl.EVAL_RUN_OUTPUTS}
    assert json.loads(first_summary)["superseded_previous"] == {"summary": None, "runs": {}}
    r = env.invoke()
    assert r.exit_code == 0, r.output
    s = json.loads((env.root / "evaluation_summary_test.json").read_text())
    assert s["evaluated_configs"] == ["M", "P4"]
    arch = Path(s["superseded_previous"]["summary"])
    assert arch.parent == env.root and arch.name.startswith("superseded_")
    assert (arch / "evaluation_summary_test.json").read_bytes() == first_summary
    run_arch = Path(s["superseded_previous"]["runs"]["M"]["42"])
    assert run_arch.parent == m42 and {n: (run_arch / n).read_bytes() for n in first_run} == first_run
    why = json.loads((run_arch / "superseded_reason.json").read_text())
    assert why["predict_test_sha256"] == _sha(m42 / "predict_test.json") and sorted(why["moved"]) == sorted(first_run)
    assert "P4" not in s["superseded_previous"]["runs"]  # nothing of P4 existed before
    assert all((m42 / n).exists() for n in first_run)  # the new evaluation's outputs


def test_missing_sec_probs_is_recorded_not_refused(env):
    for seed in (42, 43, 44):
        env.set_prov("M", seed, test_sec_probs_sha256=_DROP, test_sec_probs_shape=_DROP)
        (env.run_dir("M", seed) / "test_sec_probs.npy").unlink()
    r = env.invoke("--config", "M", refs=("ecg_only",))
    assert r.exit_code == 0, r.output
    s = json.loads((env.root / "evaluation_summary_test.json").read_text())
    assert s["configs"]["M"]["per_seed"]["42"]["sec_probs"] == evaluate_aim2_dl.NO_SEC_PROBS


# --------------------------------------------------------------------------- refusals


def _edit_metrics(path: Path, **kw) -> None:
    path.write_text(json.dumps({**json.loads(path.read_text()), **kw}))


def _edit_settings(e: Env, cfg: str, seed: int, **kw) -> None:
    path = e.run_dir(cfg, seed) / "metrics.json"
    m = json.loads(path.read_text())
    _edit_metrics(path, settings={**m["settings"], **kw})


def _drop_nsrr_subject(e: Env) -> None:
    pd.read_csv(e.csv).iloc[1:].to_csv(e.csv, index=False)


REFUSALS = {
    # whole-file prereg hash only, or a different body
    "whole_file_hash_only": (lambda e: e.set_prov("M", 43, prereg_body_sha256=_DROP), "whole-file prereg hash alone"),
    "wrong_body": (lambda e: e.set_prov("M", 43, prereg_body_sha256="1" * 64), "not the frozen body"),
    "train_body": (lambda e: e.set_prov("P4", 42, train_prereg_body_sha256=None), "training did not start"),
    # folder vs predict_test.json
    "config_mismatch": (lambda e: e.set_prov("M", 42, config="P4"), "not this folder's M seed42"),
    "seed_mismatch": (lambda e: e.set_prov("M", 42, model_seed=43), "not this folder's M seed42"),
    "split_val": (lambda e: e.set_prov("M", 42, split="val"), "not 'test'"),
    # one batch
    "two_freeze_commits": (lambda e: e.set_prov("P4", 44, freeze_commit="0" * 40, postproc_git_commit="0" * 40,
                                                code_lineage={**e.prov("P4", 44)["code_lineage"],
                                                              "freeze_commit": "0" * 40}),
                           "runs disagree on freeze_commit"),
    "two_train_commits": (lambda e: (e.set_prov("M", 44, train_git_commit="0" * 40, train_git_commits=["0" * 40],
                                                code_lineage={**e.prov("M", 44)["code_lineage"],
                                                              "train_commit": "0" * 40}),
                                     _edit_metrics(e.run_dir("M", 44) / "metrics.json", git_commit="0" * 40,
                                                   git_commits=["0" * 40])),
                          "runs disagree on train_git_commit"),
    # one run trained (resumed) at more than one commit, seen in the checkpoint or in metrics.json
    "ckpt_commit_history": (lambda e: e.set_prov("P4", 43, train_git_commits=["0" * 40, TRAIN_COMMIT]),
                            "every invocation of a run"),
    "metrics_commit_history": (lambda e: _edit_metrics(e.run_dir("M", 43) / "metrics.json",
                                                       git_commits=[TRAIN_COMMIT, "0" * 40]),
                               "is not exactly C_train"),
    "metrics_finalised_elsewhere": (lambda e: _edit_metrics(e.run_dir("M", 43) / "metrics.json", git_commit="0" * 40),
                                    "is not exactly C_train"),
    "no_code_lineage": (lambda e: e.set_prov("M", 42, code_lineage=_DROP), "no passed code_lineage"),
    "lineage_other_commit": (lambda e: e.set_prov("M", 42, code_lineage={**e.prov("M", 42)["code_lineage"],
                                                                        "freeze_commit": "0" * 40}),
                             "no passed code_lineage"),
    "postproc_not_at_freeze": (lambda e: e.set_prov("P4", 42, postproc_git_commit="e" * 40),
                               "refit by code"),
    "two_aggregators": (lambda e: e.set_prov("M", 44, aggregator="max", aggregator_locked="max"),
                        "runs disagree on aggregator"),
    "two_devices": (lambda e: e.set_prov("P4", 43, device="cpu"), "runs disagree on device type"),
    "amp_within_config": (lambda e: e.set_prov("P4", 43, amp=False), "P4 runs disagree on amp"),
    "two_recipes": (lambda e: _edit_metrics(e.run_dir("M", 43) / "metrics.json",
                                            recipe={"name": "C", "batch": 256}), "runs disagree on recipe"),
    # files and gates
    "parquet_edited": (lambda e: pd.read_parquet(e.run_dir("M", 44) / "test_predictions.parquet").assign(
        pred_prob=0.5).to_parquet(e.run_dir("M", 44) / "test_predictions.parquet", index=False),
        "replaced or edited"),
    "sec_probs_edited": (lambda e: np.save(e.run_dir("M", 44) / "test_sec_probs.npy", np.zeros((2, 30), np.float32)),
                         "test_sec_probs.npy sha256"),
    "parity_failed": (lambda e: e.set_prov("P4", 42, parity={"passed": False}), "parity gate did not pass"),
    "lock_mismatch": (lambda e: e.set_prov("M", 42, aggregator="max"), "matching aggregator lock"),
    "no_training_metrics": (lambda e: (e.run_dir("M", 43) / "metrics.json").unlink(), "no training metrics.json"),
    "training_metrics_other_ckpt": (lambda e: _edit_metrics(e.run_dir("M", 43) / "metrics.json",
                                                            best_ckpt_sha256="0" * 64), "another checkpoint"),
    # training options: PILOT ONLY flags, the unfrozen flag, settings shared by every run
    "pass_cap_extended": (lambda e: _edit_settings(e, "P4", 43, max_passes=40), "PILOT ONLY options"),
    "passes_truncated": (lambda e: _edit_settings(e, "M", 42, max_batches_per_pass=10), "PILOT ONLY options"),
    "trained_unfrozen": (lambda e: _edit_metrics(e.run_dir("M", 44) / "metrics.json", allow_unfrozen=True),
                         "--allow-unfrozen"),
    "no_training_settings": (lambda e: _edit_metrics(e.run_dir("P4", 42) / "metrics.json", settings=None),
                             "records no settings"),
    "two_training_settings": (lambda e: _edit_settings(e, "M", 43, hidden=64), "runs disagree on training settings"),
    "training_amp_within_config": (lambda e: _edit_settings(e, "P4", 44, amp=False),
                                   "P4 runs disagree on training amp"),
    # a failed parity gate recorded for the inference device by predict_aim2_dl.py
    "parity_failed_on_device": (lambda e: (e.root / "PARITY_FAILED_mps.json").write_text(json.dumps(
        [{"config": "M", "model_seed": 43, "parity": {"passed": False, "abs_diff": 0.004}}])),
        "records a failed parity gate on mps"),
    # the NSRR AHI completeness check runs before any run is scored
    "nsrr_ahi_missing": (_drop_nsrr_subject, "11 of 12 subjects have ahi_a0h3a"),
    # seeds and configs
    "seeds_42_43": (lambda e: shutil.rmtree(e.run_dir("M", 44)), "needs exactly [42, 43, 44]"),
    "stray_seed_folder": (lambda e: e.run_dir("P4", 45).mkdir(), "needs exactly [42, 43, 44]"),
    "f6_config": (lambda e: shutil.copytree(e.run_dir("M", 42), e.root / "F6" / "seed42"),
                  "outside the pre-registration"),
    # comparators and context
    "unpinned_ref": (lambda e: _edit_metrics(e.refs["ecg_belt"] / "metrics.json", model_sha256="0" * 64),
                     "is not the pinned model"),
    "ref_other_prereg": (lambda e: _edit_metrics(e.refs["ecg_only"] / "metrics.json",
                                                 prereg={"path": "x", "sha256": "0" * 64}),
                         "not scored under the frozen pre-registration"),
    "context_sha": (lambda e: e.context.write_bytes(e.context.read_bytes() + b"\0"), "!= pinned"),
    "edited_note_body": (lambda e: e.note.write_text(e.note.read_text().replace("Primary", "Main")), "may not change"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_refusals_compute_nothing(env, case):
    tamper, msg = REFUSALS[case]
    tamper(env)
    r = env.invoke()
    assert r.exit_code != 0, r.output
    assert msg in r.output, r.output
    assert _nothing_written(env)


def test_load_ref_checks_the_pin_before_reading_predictions(tmp_path, monkeypatch):
    """A primary reference is refused on its metrics.json alone: no prediction is read and no AUC is
    computed for an unpinned model or one scored under another pre-registration."""
    d = tmp_path / "ecg_only" / "test"
    d.mkdir(parents=True)  # no predictions, no bootstrap arrays
    (d / "metrics.json").write_text(json.dumps({"model_sha256": "0" * 64, "prereg": {"prereg_body_sha256": "b"}}))
    monkeypatch.setattr(evaluate_aim2_dl.pd, "read_parquet", lambda *a, **k: pytest.fail("predictions read"))
    with pytest.raises(evaluate_aim2_dl.click.ClickException, match="is not the pinned model"):
        evaluate_aim2_dl.load_ref(d, "test", primary="ecg_only", freeze_body="b")
    (d / "metrics.json").write_text(json.dumps({"model_sha256": PINNED_REF_MODEL_SHA256["ecg_only"],
                                                "prereg": {"prereg_body_sha256": "other"}}))
    with pytest.raises(evaluate_aim2_dl.click.ClickException, match="not scored under the frozen"):
        evaluate_aim2_dl.load_ref(d, "test", primary="ecg_only", freeze_body="b")


def test_refuses_contrast_gap_and_missing_prereg(env):
    r = env.invoke("--contrast", "M=ecg_only")
    assert r.exit_code == 2 and "--contrast and --gap are not accepted with --split test" in r.output
    r = env.invoke("--gap", "P4,M=ecg_belt,ecg_only")
    assert r.exit_code == 2 and "not accepted" in r.output
    args = env.base_args()
    i = args.index("--prereg")
    r = env.invoke(args=args[:i] + args[i + 2:])
    assert r.exit_code == 2 and "--split test requires --prereg" in r.output
    assert _nothing_written(env)


def test_refs_must_be_exactly_the_matched_cells(env):
    r = env.invoke(refs=("ecg_only",))  # P4 evaluated, ecg_belt missing
    assert r.exit_code != 0 and "must be exactly the matched cells" in r.output
    r = env.invoke("--ref", f"physio={env.ladder}")
    assert r.exit_code != 0 and "must be exactly the matched cells" in r.output
    r = env.invoke("--config", "F6", refs=())
    assert r.exit_code != 0 and "no test predictions for config F6" in r.output
    assert _nothing_written(env)


def test_selecting_configs_ignores_others(env):
    """--config M evaluates M only; a P4 folder with a broken provenance is not even read."""
    env.set_prov("P4", 42, prereg_body_sha256=_DROP)
    r = env.invoke("--config", "M", refs=("ecg_only",))
    assert r.exit_code == 0, r.output
    assert not (env.run_dir("P4", 42) / "metrics_extended.json").exists()
