"""thesis_pipeline.splits: reproduction of the Aim 2 split and the frozen split file.

The equivalence test shows why splitting the sorted id list reproduces the
Aim 2 fits, which split the sleep-filtered epoch frame. The real-data tests
use only local, small files (subject_metadata.parquet, the canonical
test_predictions key file, splits/aim2_seed42.json) and skip if absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis_pipeline import splits  # noqa: E402

METADATA = ROOT / "features" / "subject_metadata.parquet"
KEY = ROOT / "models" / "recovery_2026-05-03" / "aim2_v6_past_only" / "test_predictions.parquet"
FROZEN = ROOT / "splits" / "aim2_seed42.json"


def _fit_script_split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The subject sets exactly as fit_aim2_v85_taxonomy_ablation.py builds them (rows, not ids)."""
    groups = df["subject_id"].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tv_idx, test_idx = next(outer.split(np.empty(len(df)), df["apnoea_label"].to_numpy(), groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    tr_rel, va_rel = next(inner.split(np.empty(len(tv_idx)), None, groups[tv_idx]))
    return {
        "train": np.unique(groups[tv_idx[tr_rel]]),
        "val": np.unique(groups[tv_idx[va_rel]]),
        "test": np.unique(groups[test_idx]),
    }


@pytest.mark.parametrize("n_subjects,seed", [(97, 0), (500, 1), (1234, 2)])
def test_id_split_equals_epoch_level_split(n_subjects, seed):
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(np.arange(200001, 206000), n_subjects, replace=False))
    counts = rng.integers(1, 40, n_subjects)
    df = pd.DataFrame({
        "subject_id": np.repeat(ids, counts),
        "apnoea_label": rng.integers(0, 2, counts.sum()),
    })
    # epoch rows arrive in sorted-file order in the fit scripts; shuffle too to show order is irrelevant
    for frame in (df, df.sample(frac=1.0, random_state=seed).reset_index(drop=True)):
        want = _fit_script_split(frame)
        got = splits.aim2_split(rng.permutation(ids))
        for k in splits.SPLIT_NAMES:
            assert np.array_equal(got[k], want[k]), k


def test_split_is_partition_and_sorted():
    ids = np.arange(1000, 1400)
    s = splits.aim2_split(ids)
    allids = np.concatenate([s[k] for k in splits.SPLIT_NAMES])
    assert np.array_equal(np.sort(allids), ids)
    for k in splits.SPLIT_NAMES:
        assert np.all(np.diff(s[k]) > 0)
    assert len(s["test"]) == 80 and len(s["val"]) == 16  # ceil(0.2*400), ceil(0.05*320)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        splits.aim2_split([1, 2, 2, 3])


def test_write_load_roundtrip_and_freeze(tmp_path):
    s = splits.aim2_split(np.arange(10, 110))
    rec = splits.split_record(s, ids_source="synthetic", n_ids=100)
    p = tmp_path / "s.json"
    sha, changed = splits.write_split(p, rec)
    assert changed and (tmp_path / "s.json.sha256").read_text().startswith(sha)
    sha2, changed2 = splits.write_split(p, rec)
    assert sha2 == sha and not changed2
    loaded = splits.load_split(p)
    assert loaded["sha256"] == sha
    for k in splits.SPLIT_NAMES:
        assert np.array_equal(loaded[k], s[k]) and loaded[k].dtype == np.int64
    # a different split is refused without force
    other = splits.split_record(splits.aim2_split(np.arange(10, 111)), ids_source="synthetic", n_ids=101)
    with pytest.raises(FileExistsError):
        splits.write_split(p, other)
    splits.write_split(p, other, force=True)
    assert len(splits.load_split(p)["test"]) == 21


def test_load_detects_tampering(tmp_path):
    s = splits.aim2_split(np.arange(10, 110))
    p = tmp_path / "s.json"
    splits.write_split(p, splits.split_record(s, ids_source="synthetic", n_ids=100))
    rec = json.loads(p.read_text())
    rec["test"][0], rec["train"][0] = rec["train"][0], rec["test"][0]
    p.write_text(json.dumps(rec, indent=1) + "\n")
    with pytest.raises(ValueError, match="sha256"):
        splits.load_split(p)
    (tmp_path / "s.json.sha256").unlink()
    with pytest.raises(FileNotFoundError):
        splits.load_split(p)


def test_overlap_detected(tmp_path):
    bad = {"train": np.array([1, 2, 3]), "val": np.array([4]), "test": np.array([3, 5])}
    with pytest.raises(AssertionError, match="share"):
        splits._assert_disjoint(bad)


# --------------------------------------------------------------------------- real Aim 2 artefacts

needs_real = pytest.mark.skipif(not (METADATA.exists() and KEY.exists()), reason="Aim 2 artefacts not local")


@needs_real
def test_reproduces_canonical_aim2_split():
    meta = splits.read_subject_metadata(METADATA)
    assert len(meta) == splits.AIM2_EXPECTED_N_IDS
    s = splits.aim2_split(meta["sid"].to_numpy())
    out = splits.assert_canonical(s, KEY, meta)
    assert out["sizes"] == {"train": 4402, "val": 232, "test": 1159}
    assert out["sleep_epochs"] == {"train": 3_177_094, "val": 166_030, "test": 828_795}
    assert out["train_neg_pos"] == pytest.approx(2.5138042967345924, abs=1e-12)
    key_ids = np.unique(pd.read_parquet(KEY, columns=["subject_id"])["subject_id"])
    assert np.array_equal(s["test"], key_ids)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert np.intersect1d(s[a], s[b]).size == 0


@needs_real
def test_frozen_file_matches_reproduction():
    if not FROZEN.exists():
        pytest.skip("splits/aim2_seed42.json not written yet (run scripts/make_aim2_split.py)")
    loaded = splits.load_split(FROZEN)  # verifies the .sha256
    meta = splits.read_subject_metadata(METADATA)
    s = splits.aim2_split(meta["sid"].to_numpy())
    for k in splits.SPLIT_NAMES:
        assert np.array_equal(loaded[k], s[k]), k
    splits.assert_canonical(loaded, KEY, meta)


@needs_real
def test_wrong_split_fails_canonical_check():
    meta = splits.read_subject_metadata(METADATA)
    ids = meta["sid"].to_numpy()
    s = splits.aim2_split(ids[1:])  # one subject dropped -> a different shuffle
    with pytest.raises(AssertionError):
        splits.assert_canonical(s, KEY, meta)


def test_make_split_check_tar_refuses_placeholder(tmp_path, monkeypatch):
    """--check-tar must not iterate an iCloud-evicted tar (that would re-download ~17 GB)."""
    import tarfile

    from click.testing import CliRunner

    sys.path.insert(0, str(ROOT / "scripts"))
    import make_aim2_split as m

    tar = tmp_path / "features.tar"
    with tarfile.open(tar, "w"):
        pass
    dummy = tmp_path / "dummy.parquet"
    dummy.write_bytes(b"")

    def boom(*a, **k):
        raise AssertionError("tar opened")

    monkeypatch.setattr(m.shhs, "is_local", lambda p: False)  # as if UF_DATALESS were set
    monkeypatch.setattr(m.tarfile, "open", boom)
    monkeypatch.setattr(m.splits, "read_subject_metadata", boom)
    args = ["--metadata", str(dummy), "--key", str(dummy), "--out", str(tmp_path / "s.json"), "--check-tar", str(tar)]
    r = CliRunner().invoke(m.main, args)
    assert r.exit_code == 1 and "placeholder" in r.output and not (tmp_path / "s.json").exists()
