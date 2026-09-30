"""Test-set guards of the Aim 2 DL arm on SYNTHETIC data only (subject 400006 is the
synthetic 'test' subject of tests/dl_synthetic.py, not an SHHS subject):

* predict_aim2_dl.py refuses a postproc from another checkpoint, device or aggregator;
* the once-only test-inference guard survives failed reruns and archives provenance;
* evaluate_aim2_dl.py --split test requires --prereg and predict_test.json provenance;
* bench_bigru.py --procs N parses the child's result line (the Colab gate on CUDA).
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
from thesis_pipeline import dl_train  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("guards")
    ds = build_packed_dataset(root / "ds")
    dl_train.RECIPES["T"] = dl_train.Recipe("T", 4, 1, 2, 3, 1.0)
    runs = {}
    for seed in (42, 43):
        s = dl_train.TrainSettings(config="M", model_seed=seed, recipe="T", amp=False, device="cpu", hidden=8,
                                   eval_batch=16, prefetch=2)
        dl_train.Trainer(s, ds["out"] / "train", ds["out"] / "val", ds["split_json"], root / f"ck{seed}",
                         root / f"out{seed}", log=lambda *a: None).run()
        runs[seed] = {"ckpt": root / f"ck{seed}" / "best.pt", "pp": root / f"out{seed}" / "postproc.json"}
    del dl_train.RECIPES["T"]
    prereg = root / "prereg.md"
    prereg.write_text("# pre-registration (synthetic test)\n")
    agg = json.loads(runs[42]["pp"].read_text())["aggregator"]
    lock = root / "AGGREGATOR_LOCK.json"
    lock.write_text(json.dumps({"aggregator": agg, "from": "M seed 42 (synthetic)"}))
    ep = pd.read_parquet(ds["out"] / "test" / "epochs_test.parquet")
    key = root / "key_test.parquet"
    ep[ep["sleep"]][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True).astype(
        {"subject_id": np.int32, "epoch_idx": np.int32, "apnoea_label": np.int8}).to_parquet(key, index=False)
    csv = root / "nsrr.csv"
    pd.DataFrame({"nsrrid": ds["split"]["test"] + ds["split"]["val"], "ahi_a0h3a": [12.0, 4.0, 22.0]}).to_csv(
        csv, index=False)
    return {"root": root, "ds": ds, "runs": runs, "prereg": prereg, "lock": lock, "agg": agg, "key": key, "csv": csv}


def _args(env, which, out, ckpt_seed=42, pp=None, extra=()):
    ds = env["ds"]
    return (["--ckpt", str(env["runs"][ckpt_seed]["ckpt"]), "--postproc", str(pp or env["runs"][ckpt_seed]["pp"]),
             "--data-dir", str(ds["out"] / which), "--split", which, "--prereg", str(env["prereg"]),
             "--out-dir", str(out), "--split-json", str(ds["split_json"]), "--device", "cpu", "--no-amp",
             "--eval-batch", "16"] + list(extra))


@pytest.fixture()
def frozen(monkeypatch):
    monkeypatch.setattr(predict_aim2_dl, "check_freeze", lambda tag: "f" * 40)


# ---------------------------------------------------------------- postproc guards


def test_postproc_from_another_checkpoint_or_aggregator_is_refused(env, tmp_path, frozen):
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "val", tmp_path / "a", pp=env["runs"][43]["pp"]))
    assert r.exit_code != 0 and "belongs to another checkpoint" in r.output
    other = [a for a in ("mean", "max", "k10", "c10") if a != env["agg"]][0]
    bad_lock = tmp_path / "lock_other.json"
    bad_lock.write_text(json.dumps({"aggregator": other}))
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "val", tmp_path / "b",
                                                       extra=["--aggregator-lock", str(bad_lock)]))
    assert r.exit_code != 0 and "!= locked" in r.output
    # test mode: the lock is required, and the postproc must be fitted on this device type
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", tmp_path / "c", extra=["--freeze-tag", "t"]))
    assert r.exit_code != 0 and "--aggregator-lock" in r.output
    pp = json.loads(env["runs"][42]["pp"].read_text())
    cuda_pp = tmp_path / "postproc_cuda.json"
    cuda_pp.write_text(json.dumps({**pp, "device": "cuda:0"}))
    r = CliRunner().invoke(predict_aim2_dl.main, _args(
        env, "test", tmp_path / "d", pp=cuda_pp,
        extra=["--freeze-tag", "t", "--aggregator-lock", str(env["lock"]), "--key", str(env["key"])]))
    assert r.exit_code != 0 and "fitted on device 'cuda:0'" in r.output
    assert not (tmp_path / "d" / "predict_test_attempts.jsonl").exists()  # refused before any inference
    # a matched val refit records the lock and passes
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "val", tmp_path / "e", extra=[
        "--refit-postproc", "--aggregator-lock", str(env["lock"])]), catch_exceptions=False)
    assert r.exit_code == 0, r.output
    new_pp = json.loads((tmp_path / "e" / "postproc_cpu.json").read_text())
    assert new_pp["aggregator_lock_sha256"] == _sha(env["lock"]) and new_pp["best_ckpt_sha256"] == _sha(
        env["runs"][42]["ckpt"])


# ---------------------------------------------------------------- once-only test inference


def test_test_inference_once_only_survives_failed_reruns(env, tmp_path, frozen, monkeypatch):
    out = tmp_path / "M" / "seed42"
    test_args = ["--freeze-tag", "t", "--aggregator-lock", str(env["lock"]), "--key", str(env["key"])]
    run = lambda extra=(): CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", out,  # noqa: E731
                                                                          extra=test_args + list(extra)))
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", out, extra=test_args + ["--rerun-reason", "x"]))
    assert r.exit_code != 0 and "holds no earlier test inference" in r.output
    r = run()
    assert r.exit_code == 0, r.output
    first = {n: (out / n).read_bytes() for n in ("test_predictions.parquet", "predict_test.json")}
    prov = json.loads(first["predict_test.json"])
    assert prov["test_predictions_sha256"] == _sha(out / "test_predictions.parquet")
    assert prov["aggregator"] == prov["aggregator_locked"] == env["agg"]
    assert prov["aggregator_lock_sha256"] == _sha(env["lock"]) and len(prov["data_manifest_sha256"]) == 64
    r = run()
    assert r.exit_code != 0 and "already ran" in r.output

    # failed rerun BEFORE inference (bad key path): nothing moved, no inference logged
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", out, extra=[
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
    monkeypatch.setattr(predict_aim2_dl, "check_freeze", lambda tag: "f" * 40)
    log = [json.loads(ln) for ln in (out / "predict_test_attempts.jsonl").read_text().splitlines()]
    assert [e["event"] for e in log] == ["inference_start", "done", "inference_start"]
    assert log[-1]["rerun_reason"] == "bug Y"
    r = run()  # the guard still holds after the failed reruns
    assert r.exit_code != 0 and "already ran" in r.output

    r = run(["--rerun-reason", "bug Z"])
    assert r.exit_code == 0, r.output
    arch = sorted(out.glob("superseded_*"))
    assert len(arch) == 1
    assert {n: (arch[0] / n).read_bytes() for n in first} == first  # parquet AND its provenance kept
    reason = json.loads((arch[0] / "superseded_reason.json").read_text())
    assert reason["rerun_reason"] == "bug Z"
    prov2 = json.loads((out / "predict_test.json").read_text())
    assert prov2["rerun_reason"] == "bug Z" and prov2["replaced_previous"] == str(arch[0])
    assert not (out / "test_predictions.tmp.parquet").exists()


# ---------------------------------------------------------------- evaluate --split test provenance


def test_evaluate_test_requires_prereg_and_provenance(env, tmp_path, frozen):
    runs_root = tmp_path / "runs"
    out = runs_root / "M" / "seed42"
    r = CliRunner().invoke(predict_aim2_dl.main, _args(env, "test", out, extra=[
        "--freeze-tag", "t", "--aggregator-lock", str(env["lock"]), "--key", str(env["key"])]))
    assert r.exit_code == 0, r.output
    base = ["--runs-root", str(runs_root), "--split", "test", "--key", str(env["key"]), "--csv", str(env["csv"]),
            "--subject-metadata", str(tmp_path / "none.parquet"), "--n-bootstrap", "20"]
    ev = lambda extra=(): CliRunner().invoke(evaluate_aim2_dl.main, base + list(extra))  # noqa: E731
    r = ev()
    assert r.exit_code != 0 and "--prereg" in r.output
    other = tmp_path / "other_prereg.md"
    other.write_text("# a different pre-registration\n")
    r = ev(["--prereg", str(other)])
    assert r.exit_code != 0 and "pre-registration" in r.output

    s43 = runs_root / "M" / "seed43"
    s43.mkdir(parents=True)
    shutil.copyfile(out / "test_predictions.parquet", s43 / "test_predictions.parquet")  # no predict_test.json
    r = ev(["--prereg", str(env["prereg"])])
    assert r.exit_code != 0 and "no predict_test.json" in r.output
    shutil.copyfile(out / "predict_test.json", s43 / "predict_test.json")
    df = pd.read_parquet(s43 / "test_predictions.parquet")
    df.assign(pred_prob=1.0 - df["pred_prob"]).to_parquet(s43 / "test_predictions.parquet", index=False)
    r = ev(["--prereg", str(env["prereg"])])
    assert r.exit_code != 0 and "replaced or edited" in r.output
    shutil.copyfile(out / "test_predictions.parquet", s43 / "test_predictions.parquet")
    p43 = json.loads((s43 / "predict_test.json").read_text())
    other_agg = [a for a in ("mean", "max", "k10", "c10") if a != env["agg"]][0]
    (s43 / "predict_test.json").write_text(json.dumps({**p43, "aggregator": other_agg, "aggregator_locked": other_agg,
                                                       "aggregator_lock_sha256": "0" * 64}))
    r = ev(["--prereg", str(env["prereg"])])
    assert r.exit_code != 0 and "runs disagree on aggregator" in r.output
    assert not (runs_root / "evaluation_summary_test.json").exists()  # nothing computed on a refusal

    shutil.rmtree(s43)
    r = ev(["--prereg", str(env["prereg"])])
    assert r.exit_code == 0, r.output
    summ = json.loads((runs_root / "evaluation_summary_test.json").read_text())
    assert summ["prereg_sha256"] == _sha(env["prereg"]) and summ["freeze_commits"] == ["f" * 40]
    pv = summ["configs"]["M"]["per_seed"]["42"]["provenance"]
    assert pv["test_predictions_sha256"] == _sha(out / "test_predictions.parquet")
    assert pv["aggregator_lock_sha256"] == _sha(env["lock"])


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
