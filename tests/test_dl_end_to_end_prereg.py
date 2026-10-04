"""End to end on SYNTHETIC data: freeze -> train -> Mac predict (val parity, test) -> evaluate,
in the order of the pre-registration's Seminar exception (M first, P4 later).

Every step runs the real code; nothing touches SHHS data. Subjects 4000xx are the synthetic
subjects of tests/dl_synthetic.py (three of them in the test split, so the subject bootstrap
varies). The pipeline follows the pre-registered order of irreversible steps:

1. a tmp pre-registration note (body + '## Addenda') and its freeze record
   (``prereg.AIM2_DL_FREEZE`` patched); M and P4 x seeds 42 / 43 / 44 trained under it with a
   tiny recipe; seeds after M seed 42 use M seed 42's frozen aggregator, and the
   AGGREGATOR_LOCK.json is written from M/seed42/postproc.json the way notebook cell 9 does;
2. per M run: the training folder is 'downloaded' into predict_aim2_dl.DEFAULT_ROOT/M/seed<k>,
   validation is re-scored with --refit-postproc (parity gate), then --split test writes into
   that fixed folder (platform and freeze tag patched, as in tests/test_dl_guards.py), at the
   pre-registered eval batch 256 (the real PREREG_EVAL_BATCH);
3. the matched cell LGBM-ECG (synthetic predictions, metrics.json written the way
   fit_aim2_matched_cells.py writes it: model_sha256 + the check_prereg record) is built only
   once the cell script's ordering guard clears it (all M seeds predicted), while LGBM-ECG+belt
   is still refused;
4. a seminar addendum is appended, then evaluate_aim2_dl.py --split test --config M with
   LGBM-ECG only (the gap is skipped);
5. step 2 for every P4 run (after the M test result was seen, at the same freeze commit), then
   LGBM-ECG+belt, an addendum, and the full evaluate_aim2_dl.py --split test (pinned comparator
   shas and the epoch-context sha patched to the synthetic files), which archives the M-only
   summary and M's per-run metrics byte-identical.

The tests then read the summaries: claim-rule verdict fields for both contrasts and the gap,
mean +- SD of every metric, the compute / training facts, the short-hypopnoea endpoint and E1,
the seminar M-only evaluation and its archive, and the predict_test.json contract between
predict_aim2_dl.py and evaluate_aim2_dl.py.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_aim2_dl  # noqa: E402
import predict_aim2_dl  # noqa: E402
from dl_synthetic import SUBJECTS, build_packed_dataset  # noqa: E402
from test_dl_guards import lineage_stub  # noqa: E402
from thesis_pipeline import dl_train, prereg  # noqa: E402
from thesis_pipeline.dl_eval import PRIMARY_CONTRASTS, REQUIRED_SEEDS, load_nsrr_ahi  # noqa: E402
from thesis_pipeline.extended_metrics import write_extended_metrics  # noqa: E402

N_BOOT = 40
BODY = ("# Aim 2 DL pre-registration (synthetic end-to-end test)\n\n"
        "Primary contrasts: M vs LGBM-ECG, P4 vs LGBM-ECG+belt, and the gap.\n\n")
SEMINAR_ADDENDUM = ("\n### 2026-10-27 synthetic addendum: Seminar exception used\n"
                    "M and LGBM-ECG evaluated on test before P4 test inference.\n")
ADDENDUM = "\n### 2026-11-02 synthetic addendum written after test inference\nNo rule changed.\n"
FREEZE_COMMIT, TRAIN_COMMIT = "f" * 40, "c" * 40  # clean synthetic commits (the working tree may be dirty)
# two more synthetic test subjects (events are hypopnoeas, durations in s)
E2E_SUBJECTS = {
    **SUBJECTS,
    400007: ("test", 24, {"events": [(195.0, 15.0), (450.0, 30.0)]}),
    400008: ("test", 20, {"events": [(90.0, 12.0), (400.0, 25.0)]}),
}
CONFIGS = tuple(PRIMARY_CONTRASTS)  # ("M", "P4")
# every predict_test.json key evaluate_aim2_dl.py reads (test_provenance / check_across_runs)
EVALUATE_READS = ("split", "config", "model_seed", "prereg_body_sha256", "train_prereg_body_sha256",
                  "test_predictions_sha256", "test_sec_probs_sha256", "parity", "freeze_commit", "train_git_commit",
                  "aggregator", "aggregator_locked", "aggregator_lock_sha256", "ckpt_sha256", "device", "amp",
                  "train_git_commits", "code_lineage", "postproc_git_commit")
CLAIM_FIELDS = {"verdict", "label", "direction", "n_seeds", "deltas", "mean_delta", "sd_delta", "all_ci_above_0",
                "all_ci_below_0", "any_ci_excludes_0", "abs_mean_ge_min_effect", "abs_mean_gt_seed_sd",
                "seed_averaged_ci", "min_effect", "equivalence_margin", "rule"}
TRAINING_FIELDS = {"gpu_name", "train_seconds_total", "val_seconds_total", "passes_run", "best_pass", "stop_reason",
                   "cap_reached", "recipe", "amp", "nonfinite_losses", "scaler_skipped_steps"}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _ok(r) -> None:
    assert r.exit_code == 0, r.output + (repr(r.exception) if r.exception else "")


def _epoch_context(key: pd.DataFrame) -> pd.DataFrame:
    """test_epoch_context.parquet layout: the longest synthetic event overlapping each labelled epoch."""
    kind, dur = [], []
    for sid, e, y in key[["subject_id", "epoch_idx", "apnoea_label"]].itertuples(index=False):
        events = E2E_SUBJECTS[int(sid)][2].get("events", [])
        hits = [d for a, d in events if min(a + d, 30.0 * (e + 1)) - max(a, 30.0 * e) > 0]
        if y == 1 and hits:
            kind.append("hypopnoea")
            dur.append(float(max(hits)))
        else:
            kind.append("")
            dur.append(np.nan)
    return pd.DataFrame({"subject_id": key["subject_id"].to_numpy(np.int64), "epoch_idx": key["epoch_idx"],
                         "sleep_stage": "N2", "event_duration_s": dur, "event_kind": kind,
                         "xml_label": key["apnoea_label"].to_numpy(np.int8)})


def _lgbm_like(key: pd.DataFrame, signal: float, seed: int) -> pd.DataFrame:
    """A comparator's test_predictions.parquet (canonical schema) with a given signal strength."""
    rng = np.random.default_rng(seed)
    y = key["apnoea_label"].to_numpy()
    p = np.clip(signal * y + rng.random(len(y)) * (1.0 - signal), 0.0, 1.0)
    return pd.DataFrame({"subject_id": key["subject_id"].to_numpy(np.int32),
                         "epoch_idx": key["epoch_idx"].to_numpy(np.int32),
                         "apnoea_label": key["apnoea_label"].to_numpy(np.int8),
                         "pred_prob": p.astype(np.float64), "pred_label": (p > 0.5).astype(np.int64)})


def _train_all(root: Path, ds: dict) -> tuple[dict, str]:
    """M and P4 x 3 seeds under the freeze; M seed 42 chooses the aggregator, the rest use it."""
    drive, runs, agg = root / "drive", {}, "auto"
    for cfg in CONFIGS:
        for seed in REQUIRED_SEEDS:
            s = dl_train.TrainSettings(config=cfg, model_seed=seed, recipe="T", amp=False, device="cpu", hidden=8,
                                       eval_batch=16, prefetch=2, aggregator=agg)
            out, ck_dir = drive / cfg / f"seed{seed}", root / "ck" / cfg / f"seed{seed}"
            assert dl_train.Trainer(s, ds["out"] / "train", ds["out"] / "val", ds["split_json"], ck_dir, out,
                                    log=lambda *a: None).run() == "done"
            runs[(cfg, seed)] = {"ckpt": ck_dir / "best.pt", "drive": out}
            if (cfg, seed) == ("M", 42):
                agg = json.loads((out / "postproc.json").read_text())["aggregator"]
    return runs, agg


def _predict(args: list[str]):
    return CliRunner().invoke(predict_aim2_dl.main, args)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    from fit_aim2_matched_cells import missing_dl_test_runs  # LightGBM import only after torch is loaded

    root = tmp_path_factory.mktemp("e2e_prereg")
    ds = build_packed_dataset(root / "ds", subjects=E2E_SUBJECTS)
    note = root / "prereg.md"
    note.write_text(BODY + "## Addenda\n")
    body = prereg.body_sha256(note)
    freeze = root / "aim2_dl_olsen_v1_freeze.json"
    freeze.write_text(json.dumps({"note": str(note), "body_sha256": body, "vault_commit": "v" * 40,
                                  "frozen_utc": "2026-10-04T00:00:00Z"}))
    key = root / "key_test.parquet"
    ep = pd.read_parquet(ds["out"] / "test" / "epochs_test.parquet")
    key_df = ep[ep["sleep"]][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True).astype(
        {"subject_id": np.int32, "epoch_idx": np.int32, "apnoea_label": np.int8})
    key_df.to_parquet(key, index=False)
    csv = root / "nsrr.csv"
    ids = sorted(ds["split"]["test"] + ds["split"]["val"])
    pd.DataFrame({"nsrrid": ids, "ahi_a0h3a": np.linspace(3.0, 30.0, len(ids))}).to_csv(csv, index=False)
    dl_root = root / "models" / "aim2_dl_olsen_v1"  # predict_aim2_dl.DEFAULT_ROOT

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(prereg, "AIM2_DL_FREEZE", freeze)
        mp.setattr(dl_train, "git_commit", lambda: TRAIN_COMMIT)  # C_train
        mp.setitem(dl_train.RECIPES, "T", dl_train.Recipe("T", 4, 1, 2, 3, 1.0))
        runs, agg = _train_all(root, ds)
    m42_pp = runs[("M", 42)]["drive"] / "postproc.json"
    pp42 = json.loads(m42_pp.read_text())
    lock = root / "AGGREGATOR_LOCK.json"  # notebook cell 9
    lock.write_text(json.dumps({"aggregator": pp42["aggregator"], "rule": pp42["aggregator_rule"],
                                "from": str(m42_pp), "best_ckpt_sha256": pp42["best_ckpt_sha256"],
                                "postproc_sha256": _sha(m42_pp), "train_ref": "aim2-dl-train-v1",
                                "locked": "2026-10-20T00:00:00"}))

    def predict_patches(mp) -> None:
        mp.setattr(prereg, "AIM2_DL_FREEZE", freeze)
        mp.setattr(predict_aim2_dl, "DEFAULT_ROOT", dl_root)
        mp.setattr(predict_aim2_dl, "check_freeze", lambda tag: FREEZE_COMMIT)
        mp.setattr(predict_aim2_dl, "git_commit", lambda: FREEZE_COMMIT)  # the refit runs at the freeze tag
        mp.setattr(predict_aim2_dl, "check_code_lineage", lineage_stub)  # real one: test_dl_predict_guards.py
        mp.setattr(predict_aim2_dl, "_platform_system", lambda: "Darwin")

    common = ["--prereg", str(note), "--split-json", str(ds["split_json"]), "--device", "cpu", "--no-amp",
              "--eval-batch", "256", "--aggregator-lock", str(lock)]  # the pre-registered eval batch

    def predict_config(cfg: str) -> None:
        """Mac steps for every seed of ``cfg``: 'download', validation refit (parity gate), test inference."""
        with pytest.MonkeyPatch.context() as mp:
            predict_patches(mp)
            for seed in REQUIRED_SEEDS:
                run, out = runs[(cfg, seed)], dl_root / cfg / f"seed{seed}"  # 'download' from Drive
                out.mkdir(parents=True)
                for name in ("metrics.json", "postproc.json"):
                    shutil.copyfile(run["drive"] / name, out / name)
                _ok(_predict(["--ckpt", str(run["ckpt"]), "--postproc", str(out / "postproc.json"),
                              "--data-dir", str(ds["out"] / "val"), "--split", "val", "--out-dir", str(out),
                              "--refit-postproc", *common]))
                _ok(_predict(["--ckpt", str(run["ckpt"]), "--postproc", str(out / "postproc_cpu.json"),
                              "--data-dir", str(ds["out"] / "test"), "--split", "test", "--freeze-tag",
                              "aim2-dl-v1", "--key", str(key), *common]))

    def guard() -> dict:
        return {cell: missing_dl_test_runs(dl_root, cell) for cell in PRIMARY_CONTRASTS.values()}

    # matched LightGBM cells are scored on test only once the cell script's ordering guard clears them
    nsrr = load_nsrr_ahi(csv)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(prereg, "AIM2_DL_FREEZE", freeze)
        cell_prereg = prereg.check_prereg(note)  # what fit_aim2_matched_cells.py writes to metrics.json
    cells, pinned = {}, {}

    def build_cell(name: str) -> None:
        assert missing_dl_test_runs(dl_root, name) == [], name
        d = root / "models" / "aim2_matched_cells_v1" / name / "test"
        d.mkdir(parents=True)
        (d / "model_full.txt").write_text(f"synthetic LightGBM model for {name}\n")
        i = list(PRIMARY_CONTRASTS.values()).index(name)
        pred = _lgbm_like(key_df, 0.15 + 0.1 * i, seed=100 + i)
        pred.to_parquet(d / "test_predictions.parquet", index=False)
        write_extended_metrics(d, pred, subject_metadata=nsrr, n_bootstrap_subj=N_BOOT)
        (d / "metrics.json").write_text(json.dumps({"split": "test", "name": name, "prereg": cell_prereg,
                                                    "model_sha256": _sha(d / "model_full.txt")}))
        cells[name], pinned[name] = d, _sha(d / "model_full.txt")

    ladder = root / "models" / "recovery" / "full"  # one ladder rung
    ladder.mkdir(parents=True)
    lad = _lgbm_like(key_df, 0.6, seed=7)
    lad.to_parquet(ladder / "test_predictions.parquet", index=False)
    write_extended_metrics(ladder, lad, subject_metadata=nsrr, n_bootstrap_subj=N_BOOT)
    ctx = root / "test_epoch_context.parquet"
    _epoch_context(key_df).to_parquet(ctx, index=False)
    ev_args = ["--runs-root", str(dl_root), "--split", "test", "--prereg", str(note), "--key", str(key),
               "--csv", str(csv), "--subject-metadata", str(root / "none.parquet"), "--n-bootstrap", str(N_BOOT),
               "--epoch-context", str(ctx), "--ladder", f"full={ladder}"]

    def evaluate(extra: list[str], pins: dict):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(prereg, "AIM2_DL_FREEZE", freeze)
            mp.setattr(evaluate_aim2_dl, "PINNED_REF_MODEL_SHA256", pins)
            mp.setattr(evaluate_aim2_dl, "EPOCH_CONTEXT_SHA256", _sha(ctx))
            return CliRunner().invoke(evaluate_aim2_dl.main, ev_args + extra)

    # ---- Seminar exception: M and LGBM-ECG first
    guards = {"before": guard()}
    predicted_note_sha = {"M": _sha(note)}
    predict_config("M")
    guards["after_m"] = guard()
    build_cell("ecg_only")
    with open(note, "a") as fh:  # the seminar exception is recorded in an addendum
        fh.write(SEMINAR_ADDENDUM)
    seminar_note_sha = _sha(note)
    _ok(evaluate(["--config", "M", "--ref", f"ecg_only={cells['ecg_only']}"], dict(pinned)))
    m42 = dl_root / "M" / "seed42"
    seminar = {"summary": (dl_root / "evaluation_summary_test.json").read_bytes(), "note_sha": seminar_note_sha,
               "m42_outputs": {n: (m42 / n).read_bytes() for n in evaluate_aim2_dl.EVAL_RUN_OUTPUTS}}

    # ---- then P4 (same freeze tag), LGBM-ECG+belt and the full evaluation
    predicted_note_sha["P4"] = _sha(note)
    predict_config("P4")
    guards["after"] = guard()
    build_cell("ecg_belt")
    with open(note, "a") as fh:  # an addendum between test inference and evaluation
        fh.write(ADDENDUM)
    full = [arg for name, d in cells.items() for arg in ("--ref", f"{name}={d}")]
    _ok(evaluate(full, pinned))
    # a comparator that is not the pinned model is refused on the same, otherwise valid, set
    r_bad = evaluate(full, dict(pinned, ecg_belt="0" * 64))
    return {"root": root, "ds": ds, "note": note, "body": body, "freeze": freeze, "lock": lock, "agg": agg,
            "runs": runs, "dl_root": dl_root, "key_df": key_df, "cells": cells, "ctx": ctx,
            "predicted_note_sha": predicted_note_sha, "guards": guards, "seminar": seminar,
            "summary": json.loads((dl_root / "evaluation_summary_test.json").read_text()), "r_bad": r_bad}


# --------------------------------------------------------------------------- predict side and the contract


def test_predict_outputs_and_the_predict_test_contract(e2e):
    for (cfg, seed), run in e2e["runs"].items():
        out = e2e["dl_root"] / cfg / f"seed{seed}"
        prov = json.loads((out / "predict_test.json").read_text())
        assert not [k for k in predict_aim2_dl.PREDICT_TEST_KEYS if k not in prov]
        assert not [k for k in EVALUATE_READS if k not in prov]  # everything evaluate reads is written
        assert (prov["split"], prov["config"], prov["model_seed"]) == ("test", cfg, seed)
        assert prov["prereg_body_sha256"] == prov["train_prereg_body_sha256"] == e2e["body"]
        assert prov["prereg_sha256"] == e2e["predicted_note_sha"][cfg]  # P4: after the seminar addendum
        assert prov["ckpt_sha256"] == _sha(run["ckpt"])
        assert prov["ckpt_sha256"] == json.loads((out / "metrics.json").read_text())["best_ckpt_sha256"]
        assert prov["train_git_commit"] == json.loads((out / "metrics.json").read_text())["git_commit"]
        assert prov["train_git_commit"] == TRAIN_COMMIT and prov["train_git_commits"] == [TRAIN_COMMIT]
        assert json.loads((out / "metrics.json").read_text())["git_commits"] == [TRAIN_COMMIT]
        assert prov["code_lineage"]["passed"] is True and prov["postproc_git_commit"] == FREEZE_COMMIT
        assert prov["test_predictions_sha256"] == _sha(out / "test_predictions.parquet")
        assert prov["test_sec_probs_sha256"] == _sha(out / "test_sec_probs.npy")
        assert prov["aggregator"] == prov["aggregator_locked"] == e2e["agg"]
        assert prov["aggregator_lock_sha256"] == _sha(e2e["lock"]) and prov["freeze_commit"] == FREEZE_COMMIT
        pp = json.loads((out / "postproc_cpu.json").read_text())
        assert prov["parity"] == pp["parity"] and pp["parity"]["passed"] is True
        assert prov["postproc_sha256"] == _sha(out / "postproc_cpu.json")
        assert prov["device"] == "cpu" and prov["amp"] is False and prov["platform"] == "Darwin"
        pred = pd.read_parquet(out / "test_predictions.parquet")
        assert pred[["subject_id", "epoch_idx", "apnoea_label"]].equals(e2e["key_df"])
        assert np.load(out / "test_sec_probs.npy").shape == (len(pred), 30)
        assert prov["n_subjects"] == 3


def test_matched_cells_ordering_guard_follows_dl_test_inference(e2e):
    g = e2e["guards"]
    for cell, cfg in (("ecg_only", "M"), ("ecg_belt", "P4")):
        assert len(g["before"][cell]) == 3 and all(m.startswith(f"{cfg}/seed") for m in g["before"][cell])
        assert g["after"][cell] == []
    # seminar order: once every M seed is predicted, LGBM-ECG is cleared while LGBM-ECG+belt is still refused
    assert g["after_m"]["ecg_only"] == [] and g["after_m"]["ecg_belt"] == g["before"]["ecg_belt"]


def test_addendum_between_m_and_p4_test_inference_is_accepted(e2e):
    sha = e2e["predicted_note_sha"]
    assert sha["M"] != sha["P4"] == e2e["seminar"]["note_sha"]
    for cfg in CONFIGS:
        for seed in REQUIRED_SEEDS:
            prov = json.loads((e2e["dl_root"] / cfg / f"seed{seed}" / "predict_test.json").read_text())
            assert prov["prereg_body_sha256"] == e2e["body"] and prov["prereg_sha256"] == sha[cfg]


def test_seminar_exception_m_evaluated_first_then_archived(e2e):
    """evaluate --config M with LGBM-ECG only (no gap), before P4 test inference; the full evaluation
    later moves that summary and M's per-run metrics to superseded_<UTC>/ byte-identical and
    reproduces M's numbers from the same predictions."""
    sem, s = e2e["seminar"], e2e["summary"]
    s_m = json.loads(sem["summary"])
    assert s_m["evaluated_configs"] == ["M"] and s_m["gaps"] == "skipped: needs M and P4"
    assert list(s_m["configs"]) == ["M"] and set(s_m["refs"]) == {"ecg_only"}
    cr = s_m["configs"]["M"]["contrasts"]["ecg_only"]["claim_rule_auc"]
    assert cr["n_seeds"] == 3 and cr["verdict"] in {"A", "B", "C", "D", "E"}
    assert "E1" in s_m["expectations"] and set(s_m["short_hypopnoea"]["configs"]) == {"M"}
    assert s_m["superseded_previous"] == {"summary": None, "runs": {}}
    assert s_m["prereg"]["prereg_sha256"] == sem["note_sha"] and s_m["prereg"]["prereg_body_sha256"] == e2e["body"]
    arch = Path(s["superseded_previous"]["summary"])
    assert arch.parent == e2e["dl_root"] and (arch / "evaluation_summary_test.json").read_bytes() == sem["summary"]
    runs_arch = s["superseded_previous"]["runs"]
    assert set(runs_arch) == {"M"} and set(runs_arch["M"]) == {"42", "43", "44"}  # nothing of P4 existed before
    m42_arch = Path(runs_arch["M"]["42"])
    assert {n: (m42_arch / n).read_bytes() for n in sem["m42_outputs"]} == sem["m42_outputs"]
    assert s["configs"]["M"]["per_seed"] == s_m["configs"]["M"]["per_seed"]
    assert s["configs"]["M"]["contrasts"]["ecg_only"] == s_m["configs"]["M"]["contrasts"]["ecg_only"]


# --------------------------------------------------------------------------- evaluate summary


def test_addendum_between_predict_and_evaluate_is_accepted(e2e):
    s = e2e["summary"]
    assert e2e["note"].read_text().endswith(ADDENDUM)
    assert s["prereg"]["prereg_body_sha256"] == e2e["body"]
    assert s["prereg"]["prereg_sha256"] == _sha(e2e["note"]) and _sha(e2e["note"]) not in e2e[
        "predicted_note_sha"].values()
    assert s["prereg"]["prereg_freeze_sha256"] == _sha(e2e["freeze"])


def test_claim_rule_verdicts_for_both_contrasts_and_the_gap(e2e):
    s = e2e["summary"]
    assert s["evaluated_configs"] == ["M", "P4"] and s["required_seeds"] == [42, 43, 44]
    assert "only for the three primary contrasts" in s["multiplicity"]
    for cfg, ref in PRIMARY_CONTRASTS.items():
        c = s["configs"][cfg]
        assert c["seeds"] == [42, 43, 44] and list(c["contrasts"]) == [ref]
        blk = c["contrasts"][ref]
        cr = blk["claim_rule_auc"]
        assert set(cr) == CLAIM_FIELDS and cr["n_seeds"] == 3
        assert cr["verdict"] in {"A", "B", "C", "D", "E"} and cr["label"].startswith(cr["verdict"] + ":")
        # verdicts use the full-test-set point estimates: DL AUC of seed s minus the comparator AUC
        want = [c["per_seed"][str(k)]["auc"] - s["refs"][ref]["auc"] for k in (42, 43, 44)]
        assert cr["deltas"] == pytest.approx(want, abs=1e-12)
        assert cr["mean_delta"] == pytest.approx(np.mean(want), abs=1e-12)
        assert blk["auc_seed_averaged"]["delta_point"] == pytest.approx(np.mean(want), abs=1e-12)
        assert cr["seed_averaged_ci"] == [blk["auc_seed_averaged"]["delta_ci_low"],
                                          blk["auc_seed_averaged"]["delta_ci_high"]]
        assert len(blk["aupr_per_seed"]) == 3 and "claim_rule_aupr" not in blk  # AUC-PR: no verdict
        assert "p_two_sided" in blk["auc_seed_averaged"] and "p_two_sided" not in blk["aupr_seed_averaged"]
        assert s["refs"][ref]["model_sha256"] == _sha(e2e["cells"][ref] / "model_full.txt")
    gap = s["gaps"]["P4,M=ecg_belt,ecg_only"]
    cr = gap["claim_rule_auc"]
    assert set(cr) == CLAIM_FIELDS and cr["n_seeds"] == 3 and cr["verdict"] in {"A", "B", "C", "D", "E"}
    assert gap["pairing"] == "by model seed index" and len(gap["cross_pairs"]) == 9
    p4, m = s["configs"]["P4"]["per_seed"], s["configs"]["M"]["per_seed"]
    ref_gap = s["refs"]["ecg_belt"]["auc"] - s["refs"]["ecg_only"]["auc"]
    want = [p4[str(k)]["auc"] - m[str(k)]["auc"] - ref_gap for k in (42, 43, 44)]
    assert cr["deltas"] == pytest.approx(want, abs=1e-12)
    assert gap["cross_pairs"]["P4_43-M_42"]["delta_point"] == pytest.approx(p4["43"]["auc"] - m["42"]["auc"] - ref_gap,
                                                                            abs=1e-12)
    lad = s["ladder"]["full"]
    assert set(lad["configs"]) == {"M", "P4"} and "claim_rule_auc" not in lad["configs"]["M"]


def test_metrics_mean_sd_and_compute_fields(e2e):
    s = e2e["summary"]
    m_metrics = json.loads((e2e["dl_root"] / "M" / "seed42" / "metrics_extended.json").read_text())
    assert s["freeze_commits"] == [FREEZE_COMMIT] and s["device_type"] == "cpu"
    assert s["recipe"]["name"] == "T" and s["amp_by_config"] == {"M": False, "P4": False}
    assert s["aggregator"] == e2e["agg"] and s["aggregator_lock_sha256"] == _sha(e2e["lock"])
    trained = json.loads((e2e["runs"][("M", 42)]["drive"] / "metrics.json").read_text())
    assert s["train_git_commit"] == trained["git_commit"]
    for cfg in CONFIGS:
        c = s["configs"][cfg]
        msd = c["metrics_mean_sd"]
        assert {"auc_roc", "auc_pr", "brier", "ece", "f1", "sensitivity", "specificity", "precision"} <= set(msd)
        aucs = [c["per_seed"][str(k)]["auc"] for k in (42, 43, 44)]
        assert msd["auc_roc"]["mean"] == pytest.approx(np.mean(aucs))
        assert c["auc_mean"] == pytest.approx(np.mean(aucs))
        assert msd["auc_roc"]["sd"] == pytest.approx(np.std(aucs, ddof=1))
        assert set(msd["auc_roc"]["per_seed"]) == {"42", "43", "44"}
        if cfg == "M":
            assert msd["brier"]["per_seed"]["42"] == pytest.approx(m_metrics["brier"])
        assert set(c["training_cap_reached"]) == {"42", "43", "44"}
        for k in ("42", "43", "44"):
            ps = c["per_seed"][k]
            tr = ps["training"]
            assert set(tr) == TRAINING_FIELDS
            assert tr["recipe"] == "T" and tr["amp"] is False and tr["gpu_name"] is None  # trained on CPU
            assert tr["train_seconds_total"] >= 0.0 and tr["val_seconds_total"] >= 0.0  # per-pass, 0.01 s rounding
            assert 1 <= tr["best_pass"] + 1 <= tr["passes_run"] <= 3 and tr["nonfinite_losses"] == 0
            assert tr["scaler_skipped_steps"] == 0  # fp32 on CPU: no GradScaler
            assert tr["cap_reached"] == ("pass cap" in tr["stop_reason"]) == c["training_cap_reached"][k]
            assert ps["sec_probs"] == "test_sec_probs.npy saved; sha256 verified"
            assert ps["provenance"]["prereg_body_sha256"] == e2e["body"]
            assert (e2e["dl_root"] / cfg / f"seed{k}" / "metrics_extended.json").exists()


def test_short_hypopnoea_endpoint_and_e1(e2e):
    s = e2e["summary"]
    sh = s["short_hypopnoea"]
    y = e2e["key_df"]["apnoea_label"].to_numpy()
    assert sh["n_pos"] == 2 and sh["n_neg"] == int((y == 0).sum())  # 400007 (15 s) and 400008 (12 s)
    assert sh["context_sha256"] == _sha(e2e["ctx"])
    for cfg, ref in PRIMARY_CONTRASTS.items():
        c = sh["configs"][cfg]
        assert set(c["auc_per_seed"]) == {"42", "43", "44"} and c["reference"] == ref
        assert len(c["delta_per_seed"]) == 3 and c["delta_seed_averaged"]["delta_point"] is not None
    assert set(sh["refs"]) == set(PRIMARY_CONTRASTS.values()) and set(sh["ladder"]) == {"full"}
    e1 = s["expectations"]["E1"]
    assert e1["value"] == pytest.approx(sh["configs"]["M"]["auc_mean"]) and e1["met"] is (e1["value"] < 0.9)


def test_unpinned_comparator_refused_on_the_same_set(e2e):
    r = e2e["r_bad"]
    assert r.exit_code != 0 and "is not the pinned model" in r.output
