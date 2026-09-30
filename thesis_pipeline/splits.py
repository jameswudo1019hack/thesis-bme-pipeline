"""The frozen Aim 2 patient-level split, shared by the LightGBM and DL arms.

The Aim 2 LightGBM scripts never stored their split; they rebuilt it on every
run from the sleep-filtered epoch frame:

    outer = GroupShuffleSplit(n_splits=1, test_size=0.2,  random_state=42)   # fit_aim2_v85_taxonomy_ablation.py:378
    inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)   # ... :237, on the train+val rows

``GroupShuffleSplit`` shuffles ``np.unique(groups)`` (the sorted unique
subject ids) and ignores rows, so the subject sets depend only on the sorted
id list. ``aim2_split`` therefore reproduces them from the 5,793 ids alone,
and ``make_aim2_split.py`` freezes the result as ``splits/aim2_seed42.json``
(+ ``.sha256``). DL code loads that file and never re-derives the split from
the DL cohort: dropping one subject would reshuffle everything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SPLIT_NAMES = ("train", "val", "test")

AIM2_EXPECTED_N_IDS = 5793
AIM2_EXPECTED_SUBJECTS = {"train": 4402, "val": 232, "test": 1159}
AIM2_EXPECTED_SLEEP_EPOCHS = {"train": 3_177_094, "val": 166_030, "test": 828_795}
# Stored scale_pos_weight of every Aim 2 sleep-only LightGBM fit
# (e.g. models/recovery_2026-05-03/aim2_v85_taxonomy/physio_only/metrics.json).
AIM2_TRAIN_NEG_POS = 2.5138042967345924


def aim2_split(
    ids,
    outer_seed: int = 42,
    outer_test_size: float = 0.2,
    inner_seed: int = 42,
    inner_test_size: float = 0.05,
) -> dict[str, np.ndarray]:
    """Reproduce the Aim 2 train / val / test subject sets from the cohort's subject ids.

    Returns sorted int64 arrays keyed ``train``, ``val``, ``test``.
    """
    arr = np.asarray(ids, dtype=np.int64)
    uniq = np.unique(arr)
    if uniq.size != arr.size:
        raise ValueError(f"{arr.size - uniq.size} duplicated subject id(s)")
    outer = GroupShuffleSplit(n_splits=1, test_size=outer_test_size, random_state=outer_seed)
    tv_rel, te_rel = next(outer.split(uniq, groups=uniq))
    tv = np.sort(uniq[tv_rel])
    inner = GroupShuffleSplit(n_splits=1, test_size=inner_test_size, random_state=inner_seed)
    tr_rel, va_rel = next(inner.split(tv, groups=tv))
    return {
        "train": np.sort(tv[tr_rel]),
        "val": np.sort(tv[va_rel]),
        "test": np.sort(uniq[te_rel]),
    }


def ids_sha256(ids) -> str:
    """Content hash of an id list (sorted, comma-joined decimal)."""
    s = ",".join(str(int(i)) for i in np.sort(np.asarray(ids, dtype=np.int64)))
    return hashlib.sha256(s.encode()).hexdigest()


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_subject_metadata(path: Path | str) -> pd.DataFrame:
    """``features/subject_metadata.parquet`` with an int64 ``sid`` column and sid index.

    The parquet stores ``subject_id`` as ``"shhs1-200001"`` strings.
    """
    m = pd.read_parquet(path)
    sid = m["subject_id"].astype(str).str.replace(r"^shhs\d-", "", regex=True).astype(np.int64)
    m = m.assign(sid=sid.to_numpy())
    if not m["sid"].is_unique:
        raise ValueError(f"{path}: duplicated subject ids")
    return m.set_index("sid", drop=False).sort_index()


def sleep_epoch_counts(split: dict, metadata: pd.DataFrame) -> dict[str, int]:
    """Sleep epochs per split from ``tst_min * 2`` (tst_min counts N1-REM epochs x 0.5 min)."""
    out = {}
    for name in SPLIT_NAMES:
        v = metadata.loc[np.asarray(split[name]), "tst_min"].to_numpy() * 2
        if not np.allclose(v, np.round(v)):
            raise ValueError("tst_min * 2 is not integral; cannot derive sleep-epoch counts")
        out[name] = int(np.round(v).sum())
    return out


def train_neg_pos(split: dict, metadata: pd.DataFrame) -> float:
    """(sleep epochs - apnoea sleep epochs) / apnoea sleep epochs over the train subjects."""
    m = metadata.loc[np.asarray(split["train"])]
    pos = float(m["n_apnoea_epochs"].sum())
    n = float((m["tst_min"] * 2).sum())
    return (n - pos) / pos


def split_record(split: dict, ids_source: str, n_ids: int, extra: dict | None = None) -> dict:
    """JSON-able, deterministic description of a split (no timestamps, so the file hash is stable)."""
    import sklearn

    rec = {
        "name": "aim2_seed42",
        "description": (
            "Aim 2 patient-level split reproduced from the sorted SHHS-1 subject ids. "
            "Identical to the subject sets of every Aim 2 sleep-only LightGBM fit."
        ),
        "method": {
            "outer": "GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) over sorted unique ids",
            "inner": "GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42) over sorted train+val ids",
            "reference": "scripts/fit_aim2_v85_taxonomy_ablation.py:237 (inner), :378 (outer)",
            "sklearn_version": sklearn.__version__,
        },
        "ids_source": ids_source,
        "n_ids": int(n_ids),
        "counts": {k: int(len(split[k])) for k in SPLIT_NAMES},
        "ids_sha256": {k: ids_sha256(split[k]) for k in SPLIT_NAMES},
    }
    if extra:
        rec.update(extra)
    for k in SPLIT_NAMES:
        rec[k] = [int(i) for i in split[k]]
    return rec


def write_split(path: Path | str, record: dict, force: bool = False) -> tuple[str, bool]:
    """Write the split JSON and ``<path>.sha256``. Returns (sha256, changed).

    Refuses to overwrite a different existing split unless ``force``.
    """
    path = Path(path)
    text = json.dumps(record, indent=1) + "\n"
    new_sha = hashlib.sha256(text.encode()).hexdigest()
    if path.exists():
        old_sha = file_sha256(path)
        if old_sha == new_sha:
            _write_sha(path, new_sha)
            return new_sha, False
        if not force:
            raise FileExistsError(
                f"{path} exists with different content (sha256 {old_sha[:12]} vs {new_sha[:12]}); "
                "the split is frozen. Use force only with a logged reason."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    _write_sha(path, new_sha)
    return new_sha, True


def _write_sha(path: Path, sha: str) -> None:
    sp = path.with_name(path.name + ".sha256")
    sp.write_text(f"{sha}  {path.name}\n")


def load_split(path: Path | str, verify_sha: bool = True) -> dict:
    """Load a frozen split. Returns the record with sorted int64 arrays for train/val/test.

    Verifies the ``.sha256`` sidecar (if present, or required when
    ``verify_sha``) and that the three sets are disjoint.
    """
    path = Path(path)
    sha = file_sha256(path)
    sp = path.with_name(path.name + ".sha256")
    if verify_sha and not sp.exists():
        raise FileNotFoundError(f"missing {sp}")
    if sp.exists():
        want = sp.read_text().split()[0]
        if want != sha:
            raise ValueError(f"{path}: sha256 {sha} != recorded {want}")
    rec = json.loads(path.read_text())
    for k in SPLIT_NAMES:
        rec[k] = np.asarray(rec[k], dtype=np.int64)
        if rec[k].size and np.any(np.diff(rec[k]) <= 0):
            raise ValueError(f"{path}: {k} ids not strictly increasing")
    _assert_disjoint(rec)
    rec["sha256"] = sha
    return rec


def _assert_disjoint(split: dict) -> None:
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = np.intersect1d(split[a], split[b])
        if both.size:
            raise AssertionError(f"{a} and {b} share {both.size} subject(s), e.g. {both[:5].tolist()}")


def subject_to_split(split: dict) -> dict[int, str]:
    return {int(s): name for name in SPLIT_NAMES for s in split[name]}


def assert_canonical(
    split: dict,
    key_parquet: Path | str,
    metadata: pd.DataFrame | None = None,
) -> dict:
    """Assert ``split`` is the Aim 2 split. Returns the numbers it checked.

    * sizes 4,402 / 232 / 1,159, pairwise disjoint;
    * test ids == subject set of the canonical ``test_predictions.parquet`` (``key_parquet``);
    * with ``metadata``: sleep epochs 3,177,094 / 166,030 / 828,795 and the
      train neg/pos ratio equal to the stored LightGBM ``scale_pos_weight``.
    """
    _assert_disjoint(split)
    sizes = {k: int(len(split[k])) for k in SPLIT_NAMES}
    if sizes != AIM2_EXPECTED_SUBJECTS:
        raise AssertionError(f"split sizes {sizes} != {AIM2_EXPECTED_SUBJECTS}")
    key_ids = np.unique(pd.read_parquet(key_parquet, columns=["subject_id"])["subject_id"].to_numpy())
    if not np.array_equal(np.sort(split["test"]), key_ids.astype(np.int64)):
        raise AssertionError(
            f"test ids differ from {key_parquet}: "
            f"{np.setdiff1d(split['test'], key_ids).size} only in split, "
            f"{np.setdiff1d(key_ids, split['test']).size} only in key"
        )
    out: dict = {"sizes": sizes, "test_ids_match_key": True}
    if metadata is not None:
        se = sleep_epoch_counts(split, metadata)
        if se != AIM2_EXPECTED_SLEEP_EPOCHS:
            raise AssertionError(f"sleep epochs {se} != {AIM2_EXPECTED_SLEEP_EPOCHS}")
        ratio = train_neg_pos(split, metadata)
        if not np.isclose(ratio, AIM2_TRAIN_NEG_POS, rtol=0, atol=1e-12):
            raise AssertionError(f"train neg/pos {ratio!r} != {AIM2_TRAIN_NEG_POS!r}")
        out.update(sleep_epochs=se, train_neg_pos=ratio)
    return out
