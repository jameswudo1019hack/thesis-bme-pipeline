"""Test-protection and layout guards of scripts/fit_aim2_matched_cells.py, on synthetic data.

A 100-subject synthetic features tar (the real 217-column physio_only list, random
values, hr_mean carrying the label) runs the real CLI end to end:

* train-only mode writes one split per folder (<cell>/val/, with validation
  bootstrap arrays and metrics_extended.json) and never opens a test member;
* full mode refuses to run without a saved train-only run, aborts before any test
  row is read when the refit's model bytes or validation predictions differ from
  it, and refuses a second test scoring unless --rerun-reason is given (the old
  outputs are archived, not overwritten);
* every run record carries code_sha256 and the input-match caveats;
* scripts/evaluate_aim2_dl.py load_ref can only pair a folder with its own split.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis_pipeline.splits import split_record, write_split  # noqa: E402

CANONICAL_LIST = (
    ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only" / "feature_list.json"
)
SCRIPT = ROOT / "scripts" / "fit_aim2_matched_cells.py"
EVALUATOR = ROOT / "scripts" / "evaluate_aim2_dl.py"
N_SUBJ, N_EPOCHS, N_WAKE = 100, 60, 10
VERSION = "2026-05-01-phase1batch-v1"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fam = _load(SCRIPT, "fit_aim2_matched_cells_under_test")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _add(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


class Env:
    def __init__(self, root: Path):
        if not CANONICAL_LIST.exists():
            pytest.skip(f"canonical feature list not present: {CANONICAL_LIST}")
        physio = json.loads(CANONICAL_LIST.read_text())
        rng = np.random.default_rng(0)
        self.ids = np.arange(200001, 200001 + N_SUBJ, dtype=np.int64)
        self.tar = root / "features.tar"
        labels = {}
        with tarfile.open(self.tar, "w") as tf:
            for sid in self.ids:
                y = (rng.random(N_EPOCHS) < 0.3).astype(np.int8)
                y[N_WAKE], y[N_WAKE + 1] = 1, 0  # both classes among every subject's sleep epochs
                X = rng.normal(size=(N_EPOCHS, len(physio)))
                X[:, physio.index("hr_mean")] += 1.5 * y
                X[rng.random(N_EPOCHS) < 0.05, physio.index("hr_mean")] = np.nan
                df = pd.DataFrame(X, columns=physio)
                df.insert(0, "subject_id", int(sid))
                df.insert(1, "epoch_idx", np.arange(N_EPOCHS, dtype=np.int32))
                df.insert(2, "sleep_stage", ["W"] * N_WAKE + ["N2"] * (N_EPOCHS - N_WAKE - 5) + ["REM"] * 5)
                df.insert(3, "apnoea_label", y)
                df["features_version"] = VERSION
                _add(tf, f"features/shhs1-{sid}.parquet", _parquet_bytes(df))
                labels[int(sid)] = y[N_WAKE:]
            sm = pd.DataFrame({"subject_id": [f"shhs1-{s}" for s in self.ids],
                               "ahi_a0h3a": rng.uniform(0, 40, N_SUBJ)})
            _add(tf, "features/subject_metadata.parquet", _parquet_bytes(sm))

        tr, va, te = fam.split_subjects(self.ids, 42)
        self.split = {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}
        self.test_ids = {int(s) for s in te}
        self.split_file = root / "split.json"
        write_split(self.split_file, split_record(self.split, "synthetic", N_SUBJ))

        ytr = np.concatenate([labels[int(s)] for s in sorted(tr)])
        pos = float(ytr.sum())
        spw = (float(len(ytr)) - pos) / max(pos, 1.0)
        self.canon = root / "canon" / "physio_only"
        self.canon.mkdir(parents=True)
        (self.canon / "feature_list.json").write_text(json.dumps(physio))
        (self.canon / "metrics.json").write_text(json.dumps({"scale_pos_weight": spw}))
        pd.DataFrame({"subject_id": np.repeat(np.sort(te).astype(np.int32), 3)}).to_parquet(
            self.canon / "test_predictions.parquet", index=False)
        self.prereg = root / "prereg.md"
        self.prereg.write_text("synthetic pre-registration\n")

    def invoke(self, out: Path, *extra: str):
        args = ["--features-tar", str(self.tar), "--split-file", str(self.split_file),
                "--canonical-dir", str(self.canon), "--out-root", str(out), *extra]
        return CliRunner().invoke(fam.main, args)


@pytest.fixture(scope="module")
def env(tmp_path_factory) -> Env:
    return Env(tmp_path_factory.mktemp("synthetic_matched_cells"))


@pytest.fixture(scope="module")
def base_ecg_only(env, tmp_path_factory) -> Path:
    """A saved ecg_only --train-only run, copied into each test's own out-root."""
    out = tmp_path_factory.mktemp("base")
    r = env.invoke(out, "--cell", "ecg_only", "--train-only")
    assert r.exit_code == 0, r.output
    return out / "ecg_only" / "val"


@pytest.fixture
def spies(monkeypatch):
    """Record every tar member read and every load_rows subject list."""
    reads: list[int] = []
    loads: list[set[int]] = []
    orig_data, orig_load = fam.TarSource.data, fam.load_rows

    def data(self, sid):
        reads.append(int(sid))
        return orig_data(self, sid)

    def load_rows(src, index, subjects, cols):
        loads.append({int(s) for s in subjects})
        return orig_load(src, index, subjects, cols)

    monkeypatch.setattr(fam.TarSource, "data", data)
    monkeypatch.setattr(fam, "load_rows", load_rows)
    return reads, loads


def _with_saved_run(base_val: Path, out: Path) -> Path:
    val = out / "ecg_only" / "val"
    shutil.copytree(base_val, val)
    return val


# --------------------------------------------------------------------------- train-only

def test_train_only_writes_val_folder_and_never_opens_test_members(env, tmp_path, spies):
    reads, loads = spies
    r = env.invoke(tmp_path, "--cell", "ecg_belt", "--train-only")
    assert r.exit_code == 0, r.output

    assert reads and not set(reads) & env.test_ids
    assert set(reads) <= {int(s) for s in env.split["train"]} | {int(s) for s in env.split["val"]}
    assert all(not s & env.test_ids for s in loads)

    cell = tmp_path / "ecg_belt"
    assert [p.name for p in cell.iterdir()] == ["val"]
    val = cell / "val"
    assert {p.name for p in val.iterdir()} == {
        "val_predictions.parquet", "bootstrap_aucs_subject.npy", "bootstrap_auprs_subject.npy",
        "metrics_extended.json", "metrics_trainonly.json", "model.txt", "feature_list.json",
        "qc_hr_mean_nan_trainval.parquet"}

    m = json.loads((val / "metrics_trainonly.json").read_text())
    ext = json.loads((val / "metrics_extended.json").read_text())
    assert m["split"] == "val" and m["mode"] == "train_only"
    assert ext["auc_roc"] == m["inner_val"]["val_auc_roc"]
    assert ext["n_test_subjects"] == ext["n_subjects_with_nsrr_ahi"] == len(env.split["val"])
    assert np.load(val / "bootstrap_aucs_subject.npy").shape == (1000,)
    assert m["model_sha256"] == _sha(val / "model.txt")
    assert "no test parquet" in m["test_set"]

    # code provenance and caveats
    assert m["code_sha256"]["scripts/fit_aim2_matched_cells.py"] == _sha(SCRIPT)
    assert set(m["code_sha256"]) == set(fam.PROTOCOL_FILES)
    assert "belt_label_proximity" in m["input_match_caveats"]
    assert "absolute_scale" in m["input_match_caveats"]

    qc = pd.read_parquet(val / "qc_hr_mean_nan_trainval.parquet")
    assert set(qc["partition"]) == {"train", "val"}
    assert len(qc) == len(env.split["train"]) + len(env.split["val"])
    assert 0 < m["hr_mean_nan_qc"]["train"]["epoch_nan_frac"] < 0.2


# --------------------------------------------------------------------------- full-mode guards

def test_full_mode_refuses_without_saved_trainonly_run(env, tmp_path, spies):
    reads, loads = spies
    r = env.invoke(tmp_path, "--cell", "ecg_only", "--prereg", str(env.prereg))
    assert r.exit_code != 0
    assert "no saved --train-only run" in r.output
    assert reads == [] and loads == []  # refused before anything was read
    assert not (tmp_path / "ecg_only").exists()


def test_full_mode_scores_once_after_matching_trainonly_run(env, base_ecg_only, tmp_path, spies):
    reads, loads = spies
    val = _with_saved_run(base_ecg_only, tmp_path)
    r = env.invoke(tmp_path, "--cell", "ecg_only", "--prereg", str(env.prereg))
    assert r.exit_code == 0, r.output

    test = tmp_path / "ecg_only" / "test"
    assert {p.name for p in test.iterdir()} == {
        "test_predictions.parquet", "bootstrap_aucs_subject.npy", "bootstrap_auprs_subject.npy",
        "metrics_extended.json", "metrics.json", "model_full.txt", "feature_list.json"}
    assert not (tmp_path / "ecg_only" / fam.PARTIAL_MODEL).exists()
    assert sum(1 for s in loads if s & env.test_ids) == 1
    m = json.loads((test / "metrics.json").read_text())
    saved = json.loads((val / "metrics_trainonly.json").read_text())
    assert m["split"] == "test"
    assert m["trainonly_guard"]["passed"] is True
    assert m["trainonly_guard"]["val_pred_max_abs_diff"] <= fam.PROB_TOL
    assert m["model_sha256"] == _sha(test / "model_full.txt") == saved["model_sha256"]
    assert m["prereg"]["sha256"] == _sha(env.prereg)
    assert m["rerun"] is None
    assert m["code_sha256"]["thesis_pipeline/matched_cells.py"] == _sha(ROOT / "thesis_pipeline" / "matched_cells.py")

    # the evaluator can only pair a folder with its own split
    try:
        ev = _load(EVALUATOR, "evaluate_aim2_dl_under_test")
    except Exception as e:  # pragma: no cover - evaluator belongs to another track
        pytest.skip(f"evaluator not importable: {e}")
    ref_val, ref_test = ev.load_ref(val, "val"), ev.load_ref(test, "test")
    assert ref_val["auc"] == saved["inner_val"]["val_auc_roc"]
    assert ref_test["auc"] == m["test_auc_roc"]
    assert len(ref_val["key"]) == saved["inner_val"]["n_val_epochs"]
    assert np.array_equal(ref_test["boot_auc"], np.load(test / "bootstrap_aucs_subject.npy"), equal_nan=True)
    with pytest.raises(Exception):
        ev.load_ref(val, "test")
    with pytest.raises(Exception):
        ev.load_ref(test, "val")


@pytest.mark.parametrize("tamper", ["model_sha256", "val_predictions"])
def test_full_mode_aborts_before_test_rows_when_refit_differs(env, base_ecg_only, tmp_path, spies, tamper):
    reads, loads = spies
    val = _with_saved_run(base_ecg_only, tmp_path)
    if tamper == "model_sha256":
        m = json.loads((val / "metrics_trainonly.json").read_text())
        m["model_sha256"] = "0" * 64
        (val / "metrics_trainonly.json").write_text(json.dumps(m))
    else:
        p = pd.read_parquet(val / "val_predictions.parquet")
        p["pred_prob"] = np.clip(p["pred_prob"] + 1e-3, 0, 1)
        p.to_parquet(val / "val_predictions.parquet", index=False)
    r = env.invoke(tmp_path, "--cell", "ecg_only", "--prereg", str(env.prereg))
    assert r.exit_code == 1
    assert "not scoring test" in r.output
    assert loads and all(not s & env.test_ids for s in loads)  # no test row was read
    assert not (tmp_path / "ecg_only" / "test").exists()
    assert not (tmp_path / "ecg_only" / fam.PARTIAL_MODEL).exists()


def test_second_scoring_needs_rerun_reason_and_keeps_old_outputs(env, base_ecg_only, tmp_path, spies):
    reads, loads = spies
    _with_saved_run(base_ecg_only, tmp_path)
    assert env.invoke(tmp_path, "--cell", "ecg_only", "--prereg", str(env.prereg)).exit_code == 0
    test = tmp_path / "ecg_only" / "test"
    first = {p.name: _sha(p) for p in test.iterdir() if p.is_file()}

    reads.clear()
    loads.clear()
    r = env.invoke(tmp_path, "--cell", "ecg_only", "--prereg", str(env.prereg))
    assert r.exit_code == 2 and "--rerun-reason" in r.output
    assert reads == [] and loads == []
    assert {p.name: _sha(p) for p in test.iterdir() if p.is_file()} == first

    r = env.invoke(tmp_path, "--cell", "ecg_only", "--prereg", str(env.prereg),
                   "--rerun-reason", "synthetic bug fix")
    assert r.exit_code == 0, r.output
    archives = [p for p in test.iterdir() if p.name.startswith("superseded_")]
    assert len(archives) == 1
    arch = archives[0]
    assert {p.name: _sha(p) for p in arch.iterdir() if p.name != "superseded_reason.json"} == first
    reason = json.loads((arch / "superseded_reason.json").read_text())
    assert reason["rerun_reason"] == "synthetic bug fix"
    m = json.loads((test / "metrics.json").read_text())
    assert m["rerun"]["reason"] == "synthetic bug fix"
    assert m["rerun"]["superseded_dir"] == str(arch)


def test_rerun_reason_is_refused_without_earlier_outputs_or_in_train_only(env, base_ecg_only, tmp_path, spies):
    reads, loads = spies
    _with_saved_run(base_ecg_only, tmp_path)
    r = env.invoke(tmp_path, "--cell", "ecg_only", "--prereg", str(env.prereg), "--rerun-reason", "x")
    assert r.exit_code == 2 and "no earlier test outputs" in r.output
    r = env.invoke(tmp_path, "--cell", "ecg_only", "--train-only", "--rerun-reason", "x")
    assert r.exit_code == 2
    assert reads == [] and loads == []
