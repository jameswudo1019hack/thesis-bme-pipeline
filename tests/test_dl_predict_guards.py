"""predict_aim2_dl.py pre-registration guards (spec part B3) on SYNTHETIC data only.

Uses the fixtures of tests/test_dl_guards.py: M seeds 42 / 43 trained on the synthetic cache
under a tmp freeze record, the notebook-style AGGREGATOR_LOCK.json, and the ``frozen``
fixture (freeze record in force, DEFAULT_ROOT under tmp, HEAD == tag, Mac platform).

Covers: the prereg body check (addenda allowed, body edits refused); the fixed test output
folder; the Mac-only rule; pre-registered configs / seeds only; checkpoints that did not
start under the freeze; the training metrics.json; data-version checks (val and test); the
aggregator-lock rules; the parity gate (failure writes both files and a PARITY_FAILED record,
then exits non-zero; a failed or missing parity record blocks test; a failure cannot be
erased by a retry and blocks every run on that device type; the gate (any scoring under the
lock, refit or not) runs only at the pre-registered precision and eval batch, the refit records
its code commit and eval batch, and test refuses a refit not made at the freeze commit or at its
eval batch); a test rerun's archive folder never collides with evaluate's; PILOT ONLY training options; amp equality and the pre-registered test precision;
one clean training commit for every invocation of a run; the per-second output (row-aligned
with the parquet, sha recorded); the predict_test.json contract; front-end QC; and, on tmp git
repos, the real ``check_freeze`` (missing tag, HEAD != tag, tracked edit, untracked code) and
``check_code_lineage`` (freeze commit descends from C_train, model files byte-identical,
front-end changes recorded).
"""
from __future__ import annotations

import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import click
import pytest
import torch
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_dl_guards import (  # noqa: E402
    FREEZE_COMMIT,
    TRAIN_COMMIT,
    _args,
    _sha,
    apply_freeze_patches,
    build_env,
    make_test_run,
    refit_val,
    stage,
    test_args,
)

import predict_aim2_dl  # noqa: E402
from thesis_pipeline import prereg  # noqa: E402
from thesis_pipeline.dl_aggregate import AGGREGATORS, sec_to_epoch  # noqa: E402

AGGS = AGGREGATORS


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return build_env(tmp_path_factory.mktemp("predict_guards"))


@pytest.fixture()
def frozen(env, tmp_path, monkeypatch):
    return apply_freeze_patches(env, tmp_path / "runs", monkeypatch)


def _invoke(args):
    return CliRunner().invoke(predict_aim2_dl.main, args)


def _ckpt_variant(env, tmp_path: Path, name: str, edit) -> Path:
    """A copy of M seed 42's best.pt with ``edit(ck)`` applied (a different sha256)."""
    ck = torch.load(env["runs"][42]["ckpt"], map_location="cpu", weights_only=False)
    edit(ck)
    p = tmp_path / f"{name}.pt"
    torch.save(ck, p)
    return p


def _lock(env, tmp_path: Path, name: str, **kw) -> Path:
    lock = {**json.loads(env["lock"].read_text()), **kw}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({k: v for k, v in lock.items() if v is not None}))
    return p


def _refit_ok(env, runs: Path, seed: int = 42) -> Path:
    out = stage(env, runs, seed)
    r = refit_val(env, out, seed)
    assert r.exit_code == 0, r.output
    return out


# ---------------------------------------------------------------- pre-registration body


def test_prereg_body_checked_and_addenda_allowed(env, tmp_path, frozen):
    edited = tmp_path / "edited.md"
    edited.write_text(env["prereg"].read_text().replace("Primary contrasts", "Primary contrastz"))
    args = _args(env, "val", tmp_path / "a")
    r = _invoke([a if a != str(env["prereg"]) else str(edited) for a in args])
    assert r.exit_code != 0 and "may not change" in r.output
    no_addenda = tmp_path / "no_addenda.md"
    no_addenda.write_text("# a note without the addenda heading\n")
    r = _invoke([a if a != str(env["prereg"]) else str(no_addenda) for a in args])
    assert r.exit_code != 0 and "no '## Addenda' line" in r.output
    with_addendum = tmp_path / "with_addendum.md"
    with_addendum.write_text(env["prereg"].read_text() + "\n### 2026-10-20 aggregator lock: mean\n")
    r = _invoke([a if a != str(env["prereg"]) else str(with_addendum) for a in args])
    assert r.exit_code == 0, r.output
    prov = json.loads((tmp_path / "a" / "predict_val_cpu.json").read_text())
    assert prov["prereg_body_sha256"] == env["body_sha256"] and prov["prereg_sha256"] == _sha(with_addendum)
    assert prov["prereg_freeze_sha256"] == _sha(env["freeze"]) and prov["prereg_vault_commit"] == "v" * 40


def test_missing_freeze_record_is_refused(env, tmp_path, frozen, monkeypatch):
    monkeypatch.setattr(prereg, "AIM2_DL_FREEZE", tmp_path / "no_record.json")
    r = _invoke(_args(env, "val", tmp_path / "a"))
    assert r.exit_code != 0 and "no freeze record" in r.output


# ---------------------------------------------------------------- test-only refusals before any data


def test_wrong_out_dir_refused_for_test(env, tmp_path, frozen):
    out = _refit_ok(env, frozen)
    elsewhere = tmp_path / "elsewhere" / "M" / "seed42"
    shutil.copytree(out, elsewhere)
    r = _invoke(_args(env, "test", elsewhere, pp=elsewhere / "postproc_cpu.json", extra=test_args(env)))
    assert r.exit_code != 0 and "writes only into that run folder" in r.output
    assert not (elsewhere / "predict_test_attempts.jsonl").exists()
    # the default folder (no --out-dir) is accepted
    r = _invoke(_args(env, "test", None, pp=out / "postproc_cpu.json", extra=test_args(env)))
    assert r.exit_code == 0, r.output
    assert (out / "test_predictions.parquet").exists()


def test_non_darwin_refused_for_test(env, tmp_path, frozen, monkeypatch):
    out = _refit_ok(env, frozen)
    monkeypatch.setattr(predict_aim2_dl, "_platform_system", lambda: "Linux")
    r = _invoke(_args(env, "test", None, pp=out / "postproc_cpu.json", extra=test_args(env)))
    assert r.exit_code != 0 and "only on the Mac" in r.output
    assert not (out / "predict_test_attempts.jsonl").exists()


def test_unregistered_config_or_seed_refused_for_test(env, tmp_path, frozen):
    ck45 = _ckpt_variant(env, tmp_path, "seed45", lambda ck: ck["settings"].update(model_seed=45))
    r = _invoke(_args(env, "test", None, ckpt=ck45, extra=test_args(env)))
    assert r.exit_code != 0 and "pre-registered for test" in r.output


@pytest.mark.parametrize("name,edit,msg", [
    ("pilot", lambda ck: ck["state"].update(prereg_freeze_first=None), "did not start training under the frozen"),
    ("legacy", lambda ck: ck["state"].pop("prereg_freeze_first"), "did not start training under the frozen"),
    ("other_body", lambda ck: ck["state"].update(prereg_freeze_first={"body_sha256": "0" * 64}),
     "started training under pre-registration body"),
    ("unfrozen_flag", lambda ck: ck.update(allow_unfrozen=True), "--allow-unfrozen"),
    ("pass_cap_extended", lambda ck: ck["settings"].update(max_passes=40), "PILOT ONLY options"),
    ("passes_truncated", lambda ck: ck["settings"].update(max_batches_per_pass=10), "PILOT ONLY options"),
])
def test_checkpoint_not_started_under_the_freeze_refused(env, tmp_path, frozen, name, edit, msg):
    out = _refit_ok(env, frozen)
    ck = _ckpt_variant(env, tmp_path, name, edit)
    r = _invoke(_args(env, "test", None, ckpt=ck, pp=out / "postproc_cpu.json", extra=test_args(env)))
    assert r.exit_code != 0 and msg in r.output, r.output
    assert not (out / "predict_test_attempts.jsonl").exists()


def test_training_metrics_must_name_the_checkpoint(env, tmp_path, frozen):
    out = _refit_ok(env, frozen)
    (out / "metrics.json").unlink()
    run = lambda: _invoke(_args(env, "test", None, pp=out / "postproc_cpu.json", extra=test_args(env)))  # noqa: E731
    r = run()
    assert r.exit_code != 0 and "metrics.json missing" in r.output
    shutil.copyfile(env["runs"][43]["drive"] / "metrics.json", out / "metrics.json")  # another run's metrics
    r = run()
    assert r.exit_code != 0 and "is not the best checkpoint of this run" in r.output
    m = json.loads((env["runs"][42]["drive"] / "metrics.json").read_text())
    for edit, msg in (({"allow_unfrozen": True}, "--allow-unfrozen"),
                      ({"settings": {**m["settings"], "max_passes": 40}}, "PILOT ONLY options")):
        (out / "metrics.json").write_text(json.dumps({**m, **edit}))
        r = run()
        assert r.exit_code != 0 and msg in r.output, r.output
    assert not (out / "predict_test_attempts.jsonl").exists()


def test_data_versions_must_match_the_training_cache(env, tmp_path, frozen):
    for which in ("val", "test"):
        bad = tmp_path / f"{which}_other_frontend"
        shutil.copytree(env["ds"]["out"] / which, bad)
        man = json.loads((bad / "MANIFEST.json").read_text())
        (bad / "MANIFEST.json").write_text(json.dumps({**man, "frontend_version": "dl-frontend-9.9"}))
        if which == "val":
            args = _args(env, "val", tmp_path / "v")
        else:
            out = _refit_ok(env, frozen)
            args = _args(env, "test", None, pp=out / "postproc_cpu.json", extra=test_args(env))
        args[args.index("--data-dir") + 1] = str(bad)
        r = _invoke(args)
        assert r.exit_code != 0 and "different front-end" in r.output, (which, r.output)
    no_dv = _ckpt_variant(env, tmp_path, "no_dv", lambda ck: ck.pop("data_versions"))
    r = _invoke(_args(env, "val", tmp_path / "w", ckpt=no_dv, pp=_pp_for(env, tmp_path, no_dv)))
    assert r.exit_code != 0 and "records no data_versions" in r.output


def _pp_for(env, tmp_path: Path, ckpt: Path) -> Path:
    """M seed 42's training postproc re-pointed at another checkpoint file."""
    pp = json.loads(env["runs"][42]["pp"].read_text())
    p = tmp_path / f"postproc_{ckpt.stem}.json"
    p.write_text(json.dumps({**pp, "best_ckpt_sha256": _sha(ckpt)}))
    return p


# ---------------------------------------------------------------- aggregator lock


def test_lock_not_from_m_seed42_refused(env, tmp_path, frozen):
    for i, src in enumerate(("/drive/results/aim2_dl_olsen_v1/M/seed43/postproc.json",
                             "/drive/results/aim2_dl_olsen_v1/P4/seed42/postproc.json", "M seed 42 (by hand)")):
        lock = _lock(env, tmp_path, f"lock_{i}", **{"from": src})
        r = _invoke(_args(env, "val", tmp_path / f"v{i}", extra=["--aggregator-lock", str(lock)]))
        assert r.exit_code != 0 and "locked only from M seed 42" in r.output
    win = _lock(env, tmp_path, "lock_win", **{"from": "C:\\Thesis\\results\\M\\seed42\\postproc.json"})
    r = _invoke(_args(env, "val", tmp_path / "win", extra=["--aggregator-lock", str(win)]))
    assert r.exit_code == 0, r.output  # a Windows-style path to M/seed42 is fine


def test_lock_whose_aggregator_disagrees_with_its_own_aucs_refused(env, tmp_path, frozen):
    other = [a for a in AGGS if a != env["agg"]][0]
    aucs = {a: 0.60 for a in AGGS}
    aucs[other] = 0.90  # the rule now chooses ``other``; the lock still says env["agg"]
    rule = {**json.loads(env["lock"].read_text())["rule"], "aucs": aucs}
    lock = _lock(env, tmp_path, "lock_bad_rule", rule=rule)
    r = _invoke(_args(env, "val", tmp_path / "v", extra=["--aggregator-lock", str(lock)]))
    assert r.exit_code != 0 and f"chooses {other!r}" in r.output
    no_rule = _lock(env, tmp_path, "lock_no_rule", rule={"chosen": env["agg"]})
    r = _invoke(_args(env, "val", tmp_path / "w", extra=["--aggregator-lock", str(no_rule)]))
    assert r.exit_code != 0 and "no rule.aucs" in r.output


def test_m_seed42_with_a_different_ckpt_sha_refused(env, tmp_path, frozen):
    lock = _lock(env, tmp_path, "lock_old_m42", best_ckpt_sha256="0" * 64)
    r = _invoke(_args(env, "val", tmp_path / "v", extra=["--aggregator-lock", str(lock)]))
    assert r.exit_code != 0 and "the lock was made from another M seed 42 checkpoint" in r.output
    # another seed is not compared with the lock's checkpoint
    r = _invoke(_args(env, "val", tmp_path / "w", ckpt_seed=43, extra=["--aggregator-lock", str(lock)]))
    assert r.exit_code == 0, r.output
    # test needs the lock to record M seed 42's checkpoint at all
    out = _refit_ok(env, frozen)
    no_sha = _lock(env, tmp_path, "lock_no_sha", best_ckpt_sha256=None)
    r = _invoke(_args(env, "test", None, pp=out / "postproc_cpu.json", extra=[
        "--freeze-tag", "t", "--aggregator-lock", str(no_sha), "--key", str(env["key"])]))
    assert r.exit_code != 0 and "no best_ckpt_sha256" in r.output


# ---------------------------------------------------------------- parity gate


def _off_postproc(env, tmp_path: Path) -> Path:
    """M seed 42's training postproc whose "training-device" AUC is 0.01 away from this device's."""
    pp = json.loads(env["runs"][42]["pp"].read_text())
    rule = {**pp["aggregator_rule"], "aucs": {**pp["aggregator_rule"]["aucs"]}}
    rule["aucs"][env["agg"]] += 0.01  # 0.01 > PARITY_TOL
    off = tmp_path / "postproc_off.json"
    off.write_text(json.dumps({**pp, "aggregator_rule": rule}))
    return off


def test_parity_failure_raises_but_writes_both_files(env, tmp_path, frozen):
    agg = env["agg"]
    out = stage(env, frozen)
    r = refit_val(env, out, pp=_off_postproc(env, tmp_path))
    # on CPU (the pre-registered fp32 fallback) there is no further fallback to point at
    assert r.exit_code != 0 and "PARITY FAILED" in r.output and "no further fallback" in r.output, r.output
    assert "--device cpu --no-amp" not in r.output
    new_pp = json.loads((out / "postproc_cpu.json").read_text())
    prov = json.loads((out / "predict_val_cpu.json").read_text())
    for par in (new_pp["parity"], prov["parity"]):
        assert par["passed"] is False and par["abs_diff"] == pytest.approx(0.01)
        assert par["aggregator"] == agg and par["tol"] == predict_aim2_dl.PARITY_TOL
        assert par["device"] == "cpu" and par["amp"] is False
    # the failure is recorded in the run folder and, for a reportable checkpoint, device-wide
    for where in (out, frozen):
        recs = json.loads((where / "PARITY_FAILED_cpu.json").read_text())
        assert len(recs) == 1 and recs[0]["config"] == "M" and recs[0]["model_seed"] == 42
        assert recs[0]["parity"]["passed"] is False and recs[0]["refit_postproc"] is True
    # ... and test inference refuses that postproc
    r = _invoke(_args(env, "test", None, pp=out / "postproc_cpu.json", extra=test_args(env)))
    assert r.exit_code != 0 and "no passed parity gate" in r.output
    assert not (out / "predict_test_attempts.jsonl").exists()


def test_parity_failure_cannot_be_erased_and_blocks_every_run(env, tmp_path, frozen):
    out43 = _refit_ok(env, frozen, seed=43)  # seed 43 passed parity BEFORE seed 42 failed on this device
    out42 = stage(env, frozen)
    assert refit_val(env, out42, pp=_off_postproc(env, tmp_path)).exit_code != 0
    failed = (out42 / "postproc_cpu.json").read_bytes()
    # a retry of the refit on the same device type is refused and overwrites nothing: the failure record
    # survives (one at another eval batch is not the pre-registered gate, so it is refused before scoring)
    r = refit_val(env, out42, extra=["--eval-batch", "8"])
    assert r.exit_code != 0 and "eval batch" in r.output, r.output
    r = refit_val(env, out42)
    assert r.exit_code != 0 and "the parity gate already failed on cpu" in r.output, r.output
    assert (out42 / "postproc_cpu.json").read_bytes() == failed
    # even with the device-wide record gone, the run's own record still refuses the retry
    (frozen / "PARITY_FAILED_cpu.json").rename(tmp_path / "moved.json")
    r = refit_val(env, out42)
    assert r.exit_code != 0 and "already failed on cpu" in r.output
    (out42 / "PARITY_FAILED_cpu.json").unlink()
    r = refit_val(env, out42)
    assert r.exit_code != 0 and "records a failed parity gate" in r.output  # the failed postproc itself
    (tmp_path / "moved.json").rename(frozen / "PARITY_FAILED_cpu.json")
    # another run: its refit and its test inference on this device type are refused
    r = refit_val(env, out43, seed=43)
    assert r.exit_code != 0 and "already failed on cpu" in r.output
    r = _invoke(_args(env, "test", None, ckpt_seed=43, pp=out43 / "postproc_cpu.json", extra=test_args(env)))
    assert r.exit_code != 0 and "the parity gate already failed on cpu" in r.output, r.output
    assert not (out43 / "predict_test_attempts.jsonl").exists()
    # plain validation scoring (no refit) stays allowed for diagnostics
    r = _invoke(_args(env, "val", tmp_path / "diag", ckpt_seed=43))
    assert r.exit_code == 0, r.output


def test_parity_failure_without_the_lock_is_not_the_gate(env, tmp_path, frozen):
    """Scored without --aggregator-lock (diagnostics, no refit) a parity miss still exits non-zero, but
    it is not the pre-registered measurement, so it abandons no device."""
    r = _invoke(_args(env, "val", tmp_path / "diag", pp=_off_postproc(env, tmp_path)))
    assert r.exit_code != 0 and "PARITY FAILED" in r.output and "Not recorded" in r.output, r.output
    assert not list(tmp_path.glob("**/PARITY_FAILED_*.json")) and not list(frozen.glob("PARITY_FAILED_*.json"))
    _refit_ok(env, frozen)  # the gate itself is still open on this device


def test_after_parity_failure_messages_and_test_precision():
    mps = predict_aim2_dl.after_parity_failure("mps")
    assert "abandon mps for every run" in mps and "--device cpu --no-amp" in mps
    cpu = predict_aim2_dl.after_parity_failure("cpu")
    assert "no further fallback" in cpu and "--device cpu" not in cpu
    trained_fp16 = {"settings": {"amp": True}}
    trained_fp32 = {"settings": {"amp": False}}  # a config retrained under the NaN fallback
    predict_aim2_dl.check_test_precision(trained_fp16, "mps", True)
    predict_aim2_dl.check_test_precision(trained_fp32, "mps", False)
    predict_aim2_dl.check_test_precision(trained_fp16, "cpu", False)  # the CPU fp32 fallback
    with pytest.raises(click.ClickException, match="training precision"):
        predict_aim2_dl.check_test_precision(trained_fp16, "mps", False)
    with pytest.raises(click.ClickException, match="training precision"):
        predict_aim2_dl.check_test_precision(trained_fp32, "mps", True)
    with pytest.raises(click.ClickException, match="fp32"):
        predict_aim2_dl.check_test_precision(trained_fp16, "cpu", True)


def test_refit_needs_the_lock_and_test_needs_matching_amp(env, tmp_path, frozen):
    r = _invoke(_args(env, "val", tmp_path / "v", extra=["--refit-postproc"]))
    assert r.exit_code != 0 and "--refit-postproc requires --aggregator-lock" in r.output
    out = _refit_ok(env, frozen)
    pp = json.loads((out / "postproc_cpu.json").read_text())
    amp_pp = out / "postproc_cpu_amp.json"
    amp_pp.write_text(json.dumps({**pp, "amp": True}))
    r = _invoke(_args(env, "test", None, pp=amp_pp, extra=test_args(env)))
    assert r.exit_code != 0 and "refit with amp=True" in r.output


def test_parity_refit_runs_only_at_the_preregistered_precision(env, tmp_path, frozen, monkeypatch):
    """Scoring under the lock (the refit, or a locked scoring without it) IS the parity gate, measured
    at the test precision: at another precision it is refused before scoring, so it can neither pass
    the gate nor record a (permanent, device-wide) parity failure. Plain validation scoring without
    the lock (diagnostics) is not held to it."""
    seen = []

    def wrong_precision(ck, devtype, amp):
        seen.append((devtype, amp))
        raise click.ClickException("wrong inference precision (synthetic)")

    monkeypatch.setattr(predict_aim2_dl, "check_test_precision", wrong_precision)
    out = stage(env, frozen)
    r = refit_val(env, out, pp=_off_postproc(env, tmp_path))  # would fail the gate if it were scored
    assert r.exit_code != 0 and "wrong inference precision" in r.output, r.output
    assert seen == [("cpu", False)]
    # the same scoring under the lock without --refit-postproc is the gate too
    r = _invoke(_args(env, "val", out, pp=_off_postproc(env, tmp_path), extra=["--aggregator-lock", str(env["lock"])]))
    assert r.exit_code != 0 and "wrong inference precision" in r.output, r.output
    assert seen == [("cpu", False)] * 2
    assert not (out / "postproc_cpu.json").exists() and not (out / "predict_val_cpu.json").exists()
    assert not list(out.glob("PARITY_FAILED_*")) and not list(frozen.glob("PARITY_FAILED_*"))
    r = _invoke(_args(env, "val", tmp_path / "diag"))
    assert r.exit_code == 0, r.output
    assert seen == [("cpu", False)] * 2


def test_gate_and_test_run_only_at_the_preregistered_eval_batch(env, tmp_path, frozen, monkeypatch):
    """The pre-registration fixes eval batch 256 for test inference; the parity gate (any scoring under
    the lock) is measured at it too, and the refit records it. The other synthetic tests score at
    batch 16 (``apply_freeze_patches``); here the real value is in force."""
    assert {p.name: p.default for p in predict_aim2_dl.main.params}["eval_batch"] == 256  # the CLI default
    monkeypatch.setattr(predict_aim2_dl, "PREREG_EVAL_BATCH", 256)
    out = stage(env, frozen)
    for args in (_args(env, "val", out, pp=_off_postproc(env, tmp_path), extra=[  # would fail the gate if scored
                     "--refit-postproc", "--aggregator-lock", str(env["lock"])]),
                 _args(env, "val", out, pp=_off_postproc(env, tmp_path), extra=["--aggregator-lock", str(env["lock"])])):
        r = _invoke(args)  # batch 16
        assert r.exit_code != 0 and "eval batch 256" in r.output, r.output
    assert not (out / "postproc_cpu.json").exists() and not (out / "predict_val_cpu.json").exists()
    assert not list(out.glob("PARITY_FAILED_*")) and not list(frozen.glob("PARITY_FAILED_*"))
    assert _invoke(_args(env, "val", tmp_path / "diag")).exit_code == 0  # diagnostics without the lock: any batch
    at_256 = ["--eval-batch", "256"]
    r = refit_val(env, out, extra=at_256)
    assert r.exit_code == 0, r.output
    pp = json.loads((out / "postproc_cpu.json").read_text())
    assert pp["eval_batch"] == 256 and pp["parity"]["passed"] is True
    test = lambda extra=(), p=out / "postproc_cpu.json": _invoke(  # noqa: E731
        _args(env, "test", None, pp=p, extra=test_args(env, extra)))
    r = test()  # batch 16
    assert r.exit_code != 0 and "eval batch 256" in r.output, r.output
    other = out / "postproc_cpu_batch.json"
    other.write_text(json.dumps({**pp, "eval_batch": 128}))  # a refit recorded at another batch
    r = test(at_256, other)
    assert r.exit_code != 0 and "refit at eval batch 128" in r.output, r.output
    assert not (out / "predict_test_attempts.jsonl").exists()  # all refused before inference
    r = test(at_256)
    assert r.exit_code == 0, r.output
    assert json.loads((out / "predict_test.json").read_text())["eval_batch"] == 256


def test_test_rerun_archive_does_not_collide_with_an_evaluate_archive(env, frozen):
    """evaluate_aim2_dl.archive_previous may have made <run>/superseded_<UTC second> in the second a
    predict test rerun archives in; the rerun then uses superseded_<stamp>_<k> instead of failing
    after inference."""
    out, r = make_test_run(env, frozen)
    assert r.exit_code == 0, r.output
    first = {n: (out / n).read_bytes() for n in predict_aim2_dl.TEST_OUTPUTS}
    now = datetime.datetime.now(datetime.timezone.utc)
    taken = [out / f"superseded_{(now + datetime.timedelta(seconds=s)).strftime('%Y%m%dT%H%M%SZ')}"
             for s in range(120)]
    for d in taken:
        d.mkdir()
    r = _invoke(_args(env, "test", None, pp=out / "postproc_cpu.json",
                      extra=test_args(env, ["--rerun-reason", "bug W"])))
    assert r.exit_code == 0, r.output
    arch = [p for p in out.glob("superseded_*") if p not in taken]
    assert len(arch) == 1 and arch[0].name.endswith("_1") and arch[0].name[:-2] in {d.name for d in taken}
    assert {n: (arch[0] / n).read_bytes() for n in first} == first
    assert not any(any(d.iterdir()) for d in taken)  # the other archives are untouched
    assert json.loads((out / "predict_test.json").read_text())["replaced_previous"] == str(arch[0])


def test_test_refuses_a_threshold_not_refit_at_the_freeze_commit(env, frozen):
    out = _refit_ok(env, frozen)
    pp = json.loads((out / "postproc_cpu.json").read_text())
    assert pp["git_commit"] == FREEZE_COMMIT  # the refit records the code it ran at
    for name, commit in (("pre_freeze", "e" * 40), ("dirty", FREEZE_COMMIT + "-dirty"), ("none", None)):
        p = out / f"postproc_cpu_{name}.json"
        p.write_text(json.dumps({**pp, "git_commit": commit}))
        r = _invoke(_args(env, "test", None, pp=p, extra=test_args(env)))
        assert r.exit_code != 0 and "not the clean freeze commit" in r.output, r.output
    assert not (out / "predict_test_attempts.jsonl").exists()


@pytest.mark.parametrize("edit, msg", [
    (lambda m: {**m, "git_commits": ["0" * 40, TRAIN_COMMIT]}, "more than one commit"),  # resumed elsewhere
    (lambda m: {**m, "git_commit": "0" * 40}, "more than one commit"),  # finalised elsewhere
    (lambda m: {k: v for k, v in m.items() if k != "git_commits"}, "no training commit history"),
])
def test_test_refuses_a_run_trained_at_more_than_one_commit(env, frozen, edit, msg):
    out = _refit_ok(env, frozen)
    mpath = out / "metrics.json"
    mpath.write_text(json.dumps(edit(json.loads(mpath.read_text()))))
    r = _invoke(_args(env, "test", None, pp=out / "postproc_cpu.json", extra=test_args(env)))
    assert r.exit_code != 0 and msg in r.output, r.output
    assert not (out / "predict_test_attempts.jsonl").exists()


def test_train_commit_history_rules():
    a, b = "a" * 40, "b" * 40
    ck = {"git_commit": a, "state": {"git_commits": [a]}}
    assert predict_aim2_dl.train_commit_history(ck, {"git_commit": a, "git_commits": [a]}) == a
    # best.pt saved at A, the run then resumed and finalised at B: metrics.json shows both
    with pytest.raises(click.ClickException, match="more than one commit"):
        predict_aim2_dl.train_commit_history(ck, {"git_commit": b, "git_commits": [a, b]})
    for bad in (a + "-dirty", "unknown"):
        with pytest.raises(click.ClickException, match="not a clean commit"):
            predict_aim2_dl.train_commit_history({"git_commit": bad, "state": {"git_commits": [bad]}},
                                                 {"git_commit": bad, "git_commits": [bad]})
    with pytest.raises(click.ClickException, match="no training commit history"):
        predict_aim2_dl.train_commit_history({"git_commit": a, "state": {}}, {"git_commit": a, "git_commits": [a]})


# ---------------------------------------------------------------- outputs and the predict_test.json contract


def test_test_writes_row_aligned_sec_probs_and_the_contract(env, tmp_path, frozen):
    out, r = make_test_run(env, frozen)
    assert r.exit_code == 0, r.output
    pred = pd.read_parquet(out / "test_predictions.parquet")
    sec = np.load(out / "test_sec_probs.npy")
    assert sec.dtype == np.float32 and sec.shape == (len(pred), 30)
    prov = json.loads((out / "predict_test.json").read_text())
    agg = prov["aggregator"]
    np.testing.assert_allclose(sec_to_epoch(sec, agg), pred["pred_prob"].to_numpy(), rtol=0, atol=1e-7)
    assert prov["test_sec_probs_sha256"] == _sha(out / "test_sec_probs.npy")
    assert prov["test_sec_probs_shape"] == [len(pred), 30] and prov["test_sec_probs_dtype"] == "float32"
    missing = [k for k in predict_aim2_dl.PREDICT_TEST_KEYS if k not in prov]
    assert not missing, missing
    ck = torch.load(env["runs"][42]["ckpt"], map_location="cpu", weights_only=False)
    assert prov["split"] == "test" and prov["config"] == "M" and prov["model_seed"] == 42
    assert prov["freeze_commit"] == FREEZE_COMMIT and prov["freeze_tag"] == "t"
    assert prov["prereg_body_sha256"] == prov["train_prereg_body_sha256"] == env["body_sha256"]
    assert prov["train_git_commit"] == ck["git_commit"] and prov["ckpt_sha256"] == _sha(env["runs"][42]["ckpt"])
    assert prov["train_git_commits"] == [TRAIN_COMMIT] == ck["state"]["git_commits"] and ck["git_commit"] == TRAIN_COMMIT
    assert prov["code_lineage"]["passed"] is True and prov["code_lineage"]["train_commit"] == TRAIN_COMMIT
    assert prov["code_lineage"]["freeze_commit"] == FREEZE_COMMIT
    assert prov["postproc_git_commit"] == FREEZE_COMMIT == json.loads((out / "postproc_cpu.json").read_text())[
        "git_commit"]
    assert prov["parity"]["passed"] is True and prov["parity"] == json.loads(
        (out / "postproc_cpu.json").read_text())["parity"]
    assert prov["postproc_sha256"] == _sha(out / "postproc_cpu.json")
    assert prov["aggregator"] == prov["aggregator_locked"] == env["agg"]
    assert prov["data_versions"] == ck["data_versions"]["train"]
    assert prov["platform"] == "Darwin" and prov["device"] == "cpu" and prov["amp"] is False
    assert prov["n_rows"] == len(pred) and prov["n_subjects"] == pred["subject_id"].nunique()
    assert prov["rerun_reason"] is None and prov["replaced_previous"] is None
    qc = pd.read_json(env["ds"]["out"] / "test" / "qc_test.jsonl", lines=True)
    assert sum(prov["frontend_detector_counts"].values()) == len(qc)
    assert prov["frontend_failed_subjects"] == []


def test_frontend_qc_counts_and_failed_subjects(tmp_path):
    recs = [{"status": "done", "subject_id": 1, "ecg_ok": True, "ecg_detector": "neurokit"},
            {"status": "done", "subject_id": 2, "ecg_ok": True, "ecg_detector": "pantompkins"},
            {"status": "done", "subject_id": 3, "ecg_ok": False, "ecg_reason": "flat_or_short_ecg"},
            {"status": "done", "subject_id": 4, "ecg_ok": False, "ecg_reason": "too_few_peaks",
             "ecg_detector": "neurokit"},
            {"status": "error", "subject_id": 5, "error": "boom"},
            {"status": "done", "subject_id": 5, "ecg_ok": True, "ecg_detector": "neurokit"}]  # last record wins
    (tmp_path / "qc_test.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    q = predict_aim2_dl.frontend_qc(tmp_path, "test")
    assert q["frontend_detector_counts"] == {"neurokit": 3, "none": 1, "pantompkins": 1}
    assert q["frontend_failed_subjects"] == [3, 4]
    assert q["frontend_failed_reasons"] == {"3": "flat_or_short_ecg", "4": "too_few_peaks"}
    assert predict_aim2_dl.frontend_qc(tmp_path, "val")["frontend_detector_counts"] is None  # no ledger


# ---------------------------------------------------------------- check_freeze (the real function, tmp repo)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                           "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture()
def tagged_repo(tmp_path, monkeypatch):
    """A tiny git repo with code folders, committed and tagged 't'; predict_aim2_dl.CODE_ROOT points at it."""
    repo = tmp_path / "Code"
    for rel in ("thesis_pipeline/m.py", "scripts/s.py", "splits/split.json", "prereg/freeze.json", "README.md"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "freeze")
    _git(repo, "tag", "t")
    monkeypatch.setattr(predict_aim2_dl, "CODE_ROOT", repo)
    return repo


def test_check_freeze_accepts_a_clean_tagged_head(tagged_repo):
    head = _git(tagged_repo, "rev-parse", "HEAD")
    assert predict_aim2_dl.check_freeze("t") == head
    # untracked files outside the code folders, or not code, do not matter
    (tagged_repo / "models").mkdir()
    (tagged_repo / "models" / "out.json").write_text("{}")
    (tagged_repo / "scripts" / "notes.md").write_text("n")
    assert predict_aim2_dl.check_freeze("t") == head


def test_check_freeze_refuses_a_missing_tag(tagged_repo):
    with pytest.raises(click.ClickException, match="git tag 'nope' not found"):
        predict_aim2_dl.check_freeze("nope")


def test_check_freeze_refuses_head_not_at_the_tag(tagged_repo):
    (tagged_repo / "scripts" / "s.py").write_text("y\n")
    _git(tagged_repo, "commit", "-q", "-am", "after the freeze")
    with pytest.raises(click.ClickException, match="!= tag t"):
        predict_aim2_dl.check_freeze("t")


def test_check_freeze_refuses_a_tracked_edit(tagged_repo):
    (tagged_repo / "thesis_pipeline" / "m.py").write_text("edited\n")
    with pytest.raises(click.ClickException, match="tracked files modified"):
        predict_aim2_dl.check_freeze("t")


@pytest.mark.parametrize("rel", ["prereg/aim2_dl_olsen_v1_freeze.json", "scripts/new_script.py",
                                 "thesis_pipeline/sub/new_module.py", "splits/other.json"])
def test_check_freeze_refuses_untracked_code(tagged_repo, rel):
    (tagged_repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (tagged_repo / rel).write_text("{}")
    with pytest.raises(click.ClickException, match="untracked code not in the frozen tag"):
        predict_aim2_dl.check_freeze("t")


# ---------------------------------------------------------------- check_code_lineage (the real function, tmp repo)


@pytest.fixture()
def lineage_repo(tmp_path, monkeypatch):
    """A tmp repo holding the pinned model and front-end files, committed as C_train."""
    repo = tmp_path / "Code"
    for rel in (*predict_aim2_dl.MODEL_FILES, *predict_aim2_dl.FRONTEND_FILES, "scripts/predict_aim2_dl.py",
                "README.md"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(f"{rel} v1\n")
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "C_train")
    monkeypatch.setattr(predict_aim2_dl, "CODE_ROOT", repo)
    return repo, _git(repo, "rev-parse", "HEAD")


def _commit(repo: Path, rel: str, text: str) -> str:
    (repo / rel).write_text(text)
    _git(repo, "commit", "-q", "-am", f"edit {rel}")
    return _git(repo, "rev-parse", "HEAD")


def test_code_lineage_accepts_c_train_and_descendants_keeping_the_model_files(lineage_repo):
    repo, c = lineage_repo
    rec = predict_aim2_dl.check_code_lineage(c, c)
    assert rec["passed"] is True and rec["model_files_changed"] == [] and rec["frontend_changed_vs_train_commit"] == []
    assert rec["frontend_changed_vs_pack_commit"] is None  # the pack commit is not in this tmp repo
    f1 = _commit(repo, "scripts/predict_aim2_dl.py", "v2\n")
    assert predict_aim2_dl.check_code_lineage(c, f1)["passed"] is True
    # a front-end change is recorded, not refused (the test-cache contingency may need one)
    f2 = _commit(repo, "thesis_pipeline/dl_signals.py", "v2\n")
    rec = predict_aim2_dl.check_code_lineage(c, f2)
    assert rec["passed"] is True and rec["frontend_changed_vs_train_commit"] == ["thesis_pipeline/dl_signals.py"]


@pytest.mark.parametrize("rel", predict_aim2_dl.MODEL_FILES)
def test_code_lineage_refuses_a_model_file_change(lineage_repo, rel):
    repo, c = lineage_repo
    f = _commit(repo, rel, "changed after training\n")
    with pytest.raises(click.ClickException, match="byte-identical"):
        predict_aim2_dl.check_code_lineage(c, f)


def test_code_lineage_refuses_a_freeze_commit_not_descending_from_c_train(lineage_repo):
    repo, c = lineage_repo
    later = _commit(repo, "README.md", "v2\n")  # 'trained' at a later commit; the freeze tag is its parent
    with pytest.raises(click.ClickException, match="does not descend from the checkpoint's training commit"):
        predict_aim2_dl.check_code_lineage(later, c)
    with pytest.raises(click.ClickException, match="is not a commit of"):
        predict_aim2_dl.check_code_lineage("0" * 40, c)
