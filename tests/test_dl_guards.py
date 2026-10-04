"""Test-set guards of the Aim 2 DL arm on SYNTHETIC data only (subject 400006 is the
synthetic 'test' subject of tests/dl_synthetic.py, not an SHHS subject):

* predict_aim2_dl.py refuses a postproc from another checkpoint, device or aggregator;
* the once-only test-inference guard survives failed reruns and archives provenance;
* evaluate_aim2_dl.py --split test requires --prereg (frozen body) and refuses real predict
  outputs that are not a full pre-registered set (the accepting path is
  tests/test_dl_end_to_end_prereg.py);
* bench_bigru.py --procs N parses the child's result line (the Colab gate on CUDA).

``build_env`` (also used by tests/test_dl_predict_guards.py) trains M seeds 42 and 43 on the
synthetic cache under a tmp pre-registration freeze record, writes an AGGREGATOR_LOCK.json
the way notebook cell 9 does, and the canonical key. The ``frozen`` fixture points
prereg.AIM2_DL_FREEZE at that record, predict_aim2_dl.DEFAULT_ROOT at a tmp runs folder,
pretends the freeze tag equals HEAD and that the platform is the Mac. ``make_test_run``
walks a run through the Mac steps: download (copy metrics.json + postproc.json into
DEFAULT_ROOT/M/seed<k>), validation refit with the parity gate, then test inference.
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

import bench_bigru  # noqa: E402
import evaluate_aim2_dl  # noqa: E402
import predict_aim2_dl  # noqa: E402
from dl_synthetic import build_packed_dataset  # noqa: E402
from thesis_pipeline import dl_train, prereg  # noqa: E402

BODY = "# Aim 2 DL pre-registration (synthetic test)\n\nPrimary contrasts: M vs LGBM-ECG, P4 vs LGBM-ECG+belt.\n\n"
# the synthetic runs are trained at a clean C_train and predicted at a clean freeze commit (the real
# working tree may be dirty, so dl_train / predict_aim2_dl.git_commit are patched)
TRAIN_COMMIT, FREEZE_COMMIT = "c" * 40, "f" * 40


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lineage_stub(train_commit: str, freeze_commit: str) -> dict:
    """check_code_lineage for the synthetic commits (the real function is tested on a tmp git repo)."""
    return {"train_commit": train_commit, "freeze_commit": freeze_commit, "train_is_ancestor_of_freeze": True,
            "model_files_changed": [], "passed": True, "note": "synthetic test stub"}


def build_env(root: Path) -> dict:
    """Synthetic cache + frozen note + M seeds 42 / 43 trained under the freeze + lock + key."""
    ds = build_packed_dataset(root / "ds")
    note = root / "prereg.md"
    note.write_text(BODY + "## Addenda\n")
    freeze = root / "aim2_dl_olsen_v1_freeze.json"
    freeze.write_text(json.dumps({"note": str(note), "body_sha256": prereg.body_sha256(note),
                                  "vault_commit": "v" * 40, "frozen_utc": "2026-10-04T00:00:00Z"}))
    drive = root / "drive"  # RESULTS/<config>/seed<k> on Drive (training outputs)
    runs = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(prereg, "AIM2_DL_FREEZE", freeze)
        mp.setattr(dl_train, "git_commit", lambda: TRAIN_COMMIT)
        mp.setitem(dl_train.RECIPES, "T", dl_train.Recipe("T", 4, 1, 2, 3, 1.0))
        agg = "auto"
        for seed in (42, 43):  # seed 43 is trained with the frozen choice of seed 42, as in the notebook
            s = dl_train.TrainSettings(config="M", model_seed=seed, recipe="T", amp=False, device="cpu", hidden=8,
                                       eval_batch=16, prefetch=2, aggregator=agg)
            out = drive / "M" / f"seed{seed}"
            dl_train.Trainer(s, ds["out"] / "train", ds["out"] / "val", ds["split_json"], root / f"ck{seed}",
                             out, log=lambda *a: None).run()
            runs[seed] = {"ckpt": root / f"ck{seed}" / "best.pt", "pp": out / "postproc.json", "drive": out}
            if seed == 42:
                agg = json.loads((out / "postproc.json").read_text())["aggregator"]
    pp42 = json.loads(runs[42]["pp"].read_text())
    lock = root / "AGGREGATOR_LOCK.json"
    lock.write_text(json.dumps({"aggregator": pp42["aggregator"], "rule": pp42["aggregator_rule"],
                                "from": str(runs[42]["pp"]), "best_ckpt_sha256": pp42["best_ckpt_sha256"],
                                "postproc_sha256": _sha(runs[42]["pp"]), "train_ref": "aim2-dl-train-v1",
                                "locked": "2026-10-04T00:00:00"}))
    ep = pd.read_parquet(ds["out"] / "test" / "epochs_test.parquet")
    key = root / "key_test.parquet"
    ep[ep["sleep"]][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True).astype(
        {"subject_id": np.int32, "epoch_idx": np.int32, "apnoea_label": np.int8}).to_parquet(key, index=False)
    csv = root / "nsrr.csv"
    pd.DataFrame({"nsrrid": ds["split"]["test"] + ds["split"]["val"], "ahi_a0h3a": [12.0, 4.0, 22.0]}).to_csv(
        csv, index=False)
    return {"root": root, "ds": ds, "runs": runs, "prereg": note, "freeze": freeze, "lock": lock,
            "agg": pp42["aggregator"], "key": key, "csv": csv, "body_sha256": prereg.body_sha256(note)}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return build_env(tmp_path_factory.mktemp("guards"))


def _args(env, which, out, ckpt_seed=42, pp=None, extra=(), ckpt=None):
    """predict_aim2_dl.py arguments; ``out=None`` leaves --out-dir to its default."""
    ds = env["ds"]
    args = ["--ckpt", str(ckpt or env["runs"][ckpt_seed]["ckpt"]),
            "--postproc", str(pp or env["runs"][ckpt_seed]["pp"]),
            "--data-dir", str(ds["out"] / which), "--split", which, "--prereg", str(env["prereg"]),
            "--split-json", str(ds["split_json"]), "--device", "cpu", "--no-amp", "--eval-batch", "16"]
    if out is not None:
        args += ["--out-dir", str(out)]
    return args + list(extra)


def test_args(env, extra=()):
    """Test-split extras: freeze tag, lock and the synthetic key."""
    return ["--freeze-tag", "t", "--aggregator-lock", str(env["lock"]), "--key", str(env["key"])] + list(extra)


test_args.__test__ = False  # a helper, not a test


def apply_freeze_patches(env, runs: Path, monkeypatch) -> Path:
    """Freeze record in force, DEFAULT_ROOT = ``runs``, HEAD == tag on a clean tree (the code commit
    predict records is the freeze commit), the code lineage check passes, the Mac platform, and the
    pre-registered eval batch is the synthetic tests' 16."""
    monkeypatch.setattr(prereg, "AIM2_DL_FREEZE", env["freeze"])
    monkeypatch.setattr(predict_aim2_dl, "DEFAULT_ROOT", runs)
    monkeypatch.setattr(predict_aim2_dl, "check_freeze", lambda tag: FREEZE_COMMIT)
    monkeypatch.setattr(predict_aim2_dl, "git_commit", lambda: FREEZE_COMMIT)
    monkeypatch.setattr(predict_aim2_dl, "check_code_lineage", lineage_stub)
    monkeypatch.setattr(predict_aim2_dl, "_platform_system", lambda: "Darwin")
    # the synthetic runs are scored at eval batch 16 (the real 256: test_gate_and_test_run_only_at_the_
    # preregistered_eval_batch and tests/test_dl_end_to_end_prereg.py)
    monkeypatch.setattr(predict_aim2_dl, "PREREG_EVAL_BATCH", 16)
    return runs


@pytest.fixture()
def frozen(env, tmp_path, monkeypatch):
    return apply_freeze_patches(env, tmp_path / "runs", monkeypatch)


def stage(env, runs: Path, seed: int = 42) -> Path:
    """'Download' a run's training outputs into DEFAULT_ROOT/M/seed<k> (metrics.json + postproc.json)."""
    out = runs / "M" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "postproc.json"):
        shutil.copyfile(env["runs"][seed]["drive"] / name, out / name)
    return out


def refit_val(env, out: Path, seed: int = 42, pp=None, extra=()):
    """--split val --refit-postproc with the lock (parity gate) into ``out``."""
    return CliRunner().invoke(predict_aim2_dl.main, _args(env, "val", out, ckpt_seed=seed, pp=pp, extra=[
        "--refit-postproc", "--aggregator-lock", str(env["lock"]), *extra]))


def make_test_run(env, runs: Path, seed: int = 42, extra=()):
    """Stage, refit validation on this device, then run test inference into the default folder."""
    out = stage(env, runs, seed)
    r = refit_val(env, out, seed)
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", None, ckpt_seed=seed,
                                                       pp=out / "postproc_cpu.json", extra=test_args(env, extra)))
    return out, r


# ---------------------------------------------------------------- postproc guards


def test_postproc_from_another_checkpoint_or_aggregator_is_refused(env, tmp_path, frozen):
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "val", tmp_path / "a", pp=env["runs"][43]["pp"]))
    assert r.exit_code != 0 and "belongs to another checkpoint" in r.output
    other = [a for a in ("mean", "max", "k10", "c10") if a != env["agg"]][0]
    pp = json.loads(env["runs"][42]["pp"].read_text())
    other_pp = tmp_path / "postproc_other_agg.json"
    other_pp.write_text(json.dumps({**pp, "aggregator": other}))
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "val", tmp_path / "b", pp=other_pp,
                                                       extra=["--aggregator-lock", str(env["lock"])]))
    assert r.exit_code != 0 and "!= locked" in r.output
    # test mode: the lock is required, and the postproc must be fitted on this device type
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", None, extra=["--freeze-tag", "t"]))
    assert r.exit_code != 0 and "--aggregator-lock" in r.output
    out = stage(env, frozen)
    cuda_pp = tmp_path / "postproc_cuda.json"
    cuda_pp.write_text(json.dumps({**pp, "device": "cuda:0"}))
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", None, pp=cuda_pp, extra=test_args(env)))
    assert r.exit_code != 0 and "fitted on device 'cuda:0'" in r.output
    # the training postproc (fitted on this device type, CPU) has no parity record: refused too
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", None, extra=test_args(env)))
    assert r.exit_code != 0 and "no passed parity gate" in r.output
    assert not (out / "predict_test_attempts.jsonl").exists()  # refused before any inference
    # a matched val refit records the lock and the parity gate, and passes
    r = refit_val(env, tmp_path / "e")
    assert r.exit_code == 0, r.output
    new_pp = json.loads((tmp_path / "e" / "postproc_cpu.json").read_text())
    assert new_pp["aggregator_lock_sha256"] == _sha(env["lock"]) and new_pp["best_ckpt_sha256"] == _sha(
        env["runs"][42]["ckpt"])
    assert new_pp["parity"]["passed"] is True and new_pp["parity"]["tol"] == predict_aim2_dl.PARITY_TOL


# ---------------------------------------------------------------- once-only test inference


def test_test_inference_once_only_survives_failed_reruns(env, tmp_path, frozen, monkeypatch):
    out = stage(env, frozen)
    assert refit_val(env, out).exit_code == 0
    pp = out / "postproc_cpu.json"
    run = lambda extra=(): CliRunner().invoke(predict_aim2_dl.main, _args(  # noqa: E731
        env, "test", None, pp=pp, extra=test_args(env, extra)))
    r = run(["--rerun-reason", "x"])
    assert r.exit_code != 0 and "holds no earlier test inference" in r.output
    r = run()
    assert r.exit_code == 0, r.output
    names = ("test_predictions.parquet", "test_sec_probs.npy", "predict_test.json")
    first = {n: (out / n).read_bytes() for n in names}
    prov = json.loads(first["predict_test.json"])
    assert prov["test_predictions_sha256"] == _sha(out / "test_predictions.parquet")
    assert prov["test_sec_probs_sha256"] == _sha(out / "test_sec_probs.npy")
    assert prov["aggregator"] == prov["aggregator_locked"] == env["agg"]
    assert prov["aggregator_lock_sha256"] == _sha(env["lock"]) and len(prov["data_manifest_sha256"]) == 64
    r = run()
    assert r.exit_code != 0 and "already ran" in r.output

    # failed rerun BEFORE inference (bad key path): nothing moved, no inference logged
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", None, pp=pp, extra=[
        "--freeze-tag", "t", "--aggregator-lock", str(env["lock"]), "--key", str(tmp_path / "missing.parquet"),
        "--rerun-reason", "bug X"]))
    assert r.exit_code != 0
    assert {n: (out / n).read_bytes() for n in first} == first
    # failed rerun AFTER inference: still nothing moved, but the attempt is logged
    monkeypatch.setattr(predict_aim2_dl, "assemble_predictions",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("label mismatch")))
    r = run(["--rerun-reason", "bug Y"])
    assert r.exit_code != 0 and {n: (out / n).read_bytes() for n in first} == first
    monkeypatch.undo()
    apply_freeze_patches(env, frozen, monkeypatch)
    log = [json.loads(ln) for ln in (out / "predict_test_attempts.jsonl").read_text().splitlines()]
    assert [e["event"] for e in log] == ["inference_start", "done", "inference_start"]
    assert log[-1]["rerun_reason"] == "bug Y"
    r = run()  # the guard still holds after the failed reruns
    assert r.exit_code != 0 and "already ran" in r.output

    # what evaluate_aim2_dl.py computed from the first predictions travels with them into the archive
    ev_out = {n: f"evaluate output {n}".encode() for n in predict_aim2_dl.EVALUATE_OUTPUTS}
    for n, b in ev_out.items():
        (out / n).write_bytes(b)
    r = run(["--rerun-reason", "bug Z"])
    assert r.exit_code == 0, r.output
    arch = sorted(out.glob("superseded_*"))
    assert len(arch) == 1
    assert {n: (arch[0] / n).read_bytes() for n in first} == first  # parquet, npy AND provenance kept
    assert {n: (arch[0] / n).read_bytes() for n in ev_out} == ev_out and not any((out / n).exists() for n in ev_out)
    reason = json.loads((arch[0] / "superseded_reason.json").read_text())
    assert reason["rerun_reason"] == "bug Z" and sorted(reason["moved"]) == sorted(names + tuple(ev_out))
    prov2 = json.loads((out / "predict_test.json").read_text())
    assert prov2["rerun_reason"] == "bug Z" and prov2["replaced_previous"] == str(arch[0])
    assert prov2["test_sec_probs_sha256"] == _sha(out / "test_sec_probs.npy")
    assert not (out / "test_predictions.tmp.parquet").exists() and not (out / "test_sec_probs.tmp.npy").exists()


# ---------------------------------------------------------------- evaluate --split test provenance


def test_evaluate_test_requires_prereg_and_provenance(env, tmp_path, frozen):
    """Refusals on REAL predict_aim2_dl.py outputs; the accepting path (M and P4 x 3 seeds) is
    tests/test_dl_end_to_end_prereg.py, the hand-written refusal grid tests/test_dl_evaluate_guards.py."""
    runs_root = frozen
    out, r = make_test_run(env, runs_root)
    assert r.exit_code == 0, r.output
    base = ["--runs-root", str(runs_root), "--split", "test", "--key", str(env["key"]), "--csv", str(env["csv"]),
            "--subject-metadata", str(tmp_path / "none.parquet"), "--n-bootstrap", "20"]
    ev = lambda extra=(): CliRunner().invoke(evaluate_aim2_dl.main, base + list(extra))  # noqa: E731
    r = ev()
    assert r.exit_code != 0 and "--prereg" in r.output
    other = tmp_path / "other_prereg.md"
    other.write_text("# a different pre-registration\n")
    r = ev(["--prereg", str(other)])
    assert r.exit_code != 0 and "no '## Addenda' line" in r.output
    edited = tmp_path / "edited_prereg.md"
    edited.write_text(env["prereg"].read_text().replace("Primary contrasts", "Main contrasts"))
    r = ev(["--prereg", str(edited)])
    assert r.exit_code != 0 and "may not change" in r.output

    # one finished seed is not a pre-registered set
    r = ev(["--prereg", str(env["prereg"])])
    assert r.exit_code != 0 and "needs exactly [42, 43, 44]" in r.output
    # seed43 / seed44 folders holding seed 42's outputs: folder vs predict_test.json is refused
    for k in (43, 44):
        d = runs_root / "M" / f"seed{k}"
        d.mkdir(parents=True, exist_ok=True)
        for name in ("test_predictions.parquet", "test_sec_probs.npy", "predict_test.json", "metrics.json"):
            shutil.copyfile(out / name, d / name)
    r = ev(["--prereg", str(env["prereg"])])
    assert r.exit_code != 0 and "not this folder's M seed43" in r.output
    # predictions edited after predict_test.json was written
    df = pd.read_parquet(out / "test_predictions.parquet")
    df.assign(pred_prob=1.0 - df["pred_prob"]).to_parquet(out / "test_predictions.parquet", index=False)
    r = ev(["--prereg", str(env["prereg"])])
    assert r.exit_code != 0 and "replaced or edited" in r.output
    assert not (runs_root / "evaluation_summary_test.json").exists()  # nothing computed on a refusal
    assert not any(runs_root.glob("*/seed*/metrics_extended.json"))


# ---------------------------------------------------------------- bench gate (--procs on CUDA)

_RESULT = {"device": "cuda", "gpu": "NVIDIA A100-SXM4-40GB", "batch": 128, "c_in": 2, "amp": True,
           "train_windows_per_sec": 600.0, "infer_windows_per_sec": 2000.0}


def test_bench_parent_parses_the_child_result_line(tmp_path, monkeypatch):
    gate = json.dumps({"gate_recipe": "B", "single_128": 600.0, "multi_total": None, "gpu": _RESULT["gpu"]})
    assert bench_bigru.parse_child_result(json.dumps(_RESULT) + "\n" + gate) == _RESULT
    with pytest.raises(RuntimeError, match="no result line"):
        bench_bigru.parse_child_result(gate)
    seen = []

    class FakeProc:
        def __init__(self, cmd, **kw):
            seen.append(cmd)

        def communicate(self):  # what a pre-fix child printed: result line, then a gate line
            return json.dumps(_RESULT) + "\n" + gate + "\n", None

    monkeypatch.setattr(bench_bigru.subprocess, "Popen", FakeProc)
    out = tmp_path / "gate_multi.json"
    r = CliRunner().invoke(bench_bigru.main, ["--device", "cuda", "--amp", "--cin", "2", "--batch", "128",
                                              "--procs", "3", "--json-out", str(out)], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    assert len(seen) == 3 and all("--child" in c for c in seen)
    res = json.loads(out.read_text())[0]
    assert res["total_train_windows_per_sec"] == 1800.0 and res["procs"] == 3
    assert json.loads(r.output.strip().splitlines()[-1])["gate_recipe"] == "A"  # 3 procs >= 1,500 in total


def test_bench_child_prints_only_its_result(monkeypatch):
    monkeypatch.setattr(bench_bigru, "run_one", lambda *a, **k: dict(_RESULT))
    r = CliRunner().invoke(bench_bigru.main, ["--device", "cuda", "--batch", "128", "--procs", "1", "--child"],
                           catch_exceptions=False)
    lines = r.output.strip().splitlines()
    assert r.exit_code == 0 and len(lines) == 1 and json.loads(lines[0]) == _RESULT
    r = CliRunner().invoke(bench_bigru.main, ["--device", "cuda", "--batch", "128"], catch_exceptions=False)
    assert json.loads(r.output.strip().splitlines()[-1])["gate_recipe"] == "B"  # a top-level run still gates
