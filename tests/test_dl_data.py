"""Tests for stage-B packing, window geometry, split guards and evaluation helpers
(design section 8, tests 3, 5, 6, 7, 8 plus pairing / claim-rule checks)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl_synthetic import SPIKE_EPOCHS, SUBJECTS, build_packed_dataset, write_split_json  # noqa: E402
from thesis_pipeline.dl_data import (  # noqa: E402
    CTX_S,
    PackedSplit,
    TestDataRefused,
    assert_ids_in_split,
    centre_epochs,
    eval_windows,
    iter_batches,
    load_split,
    pass_order,
    sleep_epoch_key,
    train_window_starts,
)
from thesis_pipeline.dl_eval import (  # noqa: E402
    PRED_DTYPES,
    assemble_predictions,
    check_schema,
    claim_rule,
    gap_per_seed,
    metadata_with_ahi,
    paired_delta,
    seed_averaged_delta,
    threshold_f1max,
)
from thesis_pipeline.dl_stage_b import (  # noqa: E402
    CHANNELS,
    derive_subject,
    load_stage_a,
    verify_manifest,
)
from thesis_pipeline.extended_metrics import _subject_bootstrap_aucs  # noqa: E402

ALL = CHANNELS


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return build_packed_dataset(tmp_path_factory.mktemp("dlpack"))


@pytest.fixture(scope="module")
def train(ds):
    return PackedSplit(ds["out"] / "train", ALL)


@pytest.fixture(scope="module")
def val(ds):
    return PackedSplit(ds["out"] / "val", ALL)


# ---------------------------------------------------------------- packing


def test_manifest_hashes_chunks_and_counts(ds, train):
    man = train.manifest
    assert man["split"] == "train" and man["channels"] == list(CHANNELS) and man["fs"] == 4
    assert man["pad_s"] == 60 and len(man["chunks"]) == 3  # forced small chunks
    assert man["split_sha256"] == load_split(ds["split_json"])["sha256"]
    assert verify_manifest(ds["out"] / "train") == []
    n_ep = sum(v[1] for v in SUBJECTS.values() if v[0] == "train")
    assert man["n_epochs"] == n_ep == len(train.epochs)
    assert train.index["ok_RR"].all() and train.index["ok_THOR"].all()


def test_pack_roundtrip_equals_direct_derivation(ds, train):
    sid = 400002
    res = derive_subject(load_stage_a(ds["raw"] / f"shhs1-{sid}.npz"))
    r = train.index.index[train.index["subject_id"] == sid][0]
    k, o1, n_ep = (int(train.index.loc[r, c]) for c in ("chunk", "off1", "n_epochs"))
    for ci, c in enumerate(CHANNELS):
        got = np.asarray(train.sig[c][k][o1 * 4:(o1 + n_ep * 30) * 4])
        assert np.array_equal(got, res["signals"][ci]), c
        pre = np.asarray(train.sig[c][k][(o1 - 60) * 4:o1 * 4])
        assert not pre.any()  # 60 s of zero padding before the night
    assert np.array_equal(np.asarray(train.target[k][o1:o1 + n_ep * 30]), res["sec_target"])


def test_pack_resume_skips_done_subjects(ds, tmp_path):
    from thesis_pipeline.dl_stage_b import pack_split

    ids = ds["split"]["val"]
    out = tmp_path / "m"
    pack_split("val", ids, ds["raw"], out, split_json=ds["split_json"], workers=1, log=lambda *a: None)
    first = json.loads((out / "val" / "MANIFEST.json").read_text())["sha256"]
    msgs = []
    pack_split("val", ids, ds["raw"], out, split_json=ds["split_json"], workers=1, log=msgs.append)
    assert "2 already packed, 0 to derive" in msgs[0]
    again = json.loads((out / "val" / "MANIFEST.json").read_text())["sha256"]
    sig = {k: v for k, v in first.items() if k.startswith(("signals_", "sec_"))}
    assert sig == {k: again[k] for k in sig}


def test_pack_with_worker_pool_matches_serial(ds, tmp_path):
    from thesis_pipeline.dl_stage_b import pack_split

    ids = ds["split"]["val"]
    pack_split("val", ids, ds["raw"], tmp_path / "par", split_json=ds["split_json"], workers=2,
               max_chunk_bytes=1000 * 8, log=lambda *a: None)
    a = json.loads((tmp_path / "par" / "val" / "MANIFEST.json").read_text())["sha256"]
    b = json.loads((ds["out"] / "val" / "MANIFEST.json").read_text())["sha256"]
    keys = [k for k in b if k.startswith(("signals_", "sec_"))]
    assert {k: a[k] for k in keys} == {k: b[k] for k in keys}


# ---------------------------------------------------------------- window geometry


def test_spike_at_epoch_i_lands_in_centre_of_window_i_div_6(train):
    """Design test 3."""
    win = eval_windows(train)
    row = int(np.flatnonzero(train.subject_ids == 400001)[0])
    thor = CHANNELS.index("THOR")
    for i in SPIKE_EPOCHS:
        w = np.flatnonzero((win.row == row) & (win.j == i // 6))
        assert len(w) == 1
        bd = train.gather(win, w)
        pos = (CTX_S + 30 * (i % 6) + 15) * 4
        centre = slice(CTX_S * 4, (CTX_S + 180) * 4)
        assert int(np.argmax(bd["x"][0, centre, thor])) + CTX_S * 4 == pos
        # per-second target: exactly one positive second in the window centre
        y_c = bd["y"][0, CTX_S:CTX_S + 180]
        assert (np.flatnonzero(y_c) + CTX_S).tolist() == [CTX_S + 30 * (i % 6) + 15]


def _block_bounds(ps, rows):
    off1 = ps.index["off1"].to_numpy()[rows]
    n_pad = ps.index["n_epochs_pad"].to_numpy()[rows]
    return off1 - 60, off1 + n_pad * 30 + 60


def test_no_window_crosses_a_subject_boundary(train, val):
    """Design test 5, for evaluation windows and phase-shifted training windows."""
    for ps in (train, val):
        wins = [eval_windows(ps)] + [train_window_starts(ps, s, p) for s in (42, 43, 44) for p in range(6)]
        for win in wins:
            lo, hi = _block_bounds(ps, win.row)
            assert (win.s0 >= lo).all() and (win.s0 + 300 <= hi).all()
            assert (win.chunk == ps.index["chunk"].to_numpy()[win.row]).all()


def test_train_windows_phase_sleep_determinism_and_subsample(train):
    a = train_window_starts(train, 42, 3)
    b = train_window_starts(train, 42, 3)
    c = train_window_starts(train, 42, 4)
    assert np.array_equal(a.s0, b.s0) and not np.array_equal(a.s0, c.s0)
    assert ((a.phase >= 0) & (a.phase < 180)).all()
    c0 = a.s0 + CTX_S - train.index["off1"].to_numpy()[a.row]
    assert (train.sleep_seconds(a.row, c0, c0 + 180) > 0).all()
    half = train_window_starts(train, 42, 3, subsample=0.5)
    assert len(half) == int(np.floor(0.5 * len(a)))
    assert np.array_equal(pass_order(len(a), 42, 3), pass_order(len(a), 42, 3))


def test_every_sleep_epoch_scored_exactly_once(val):
    """Design test 6: the fixed grid scores each sleep epoch once; skipped windows hold no sleep."""
    win = eval_windows(val)
    sec = np.repeat(np.arange(len(win), dtype=np.float32)[:, None], 300, axis=1)
    df = centre_epochs(val, win, sec)
    assert not df.duplicated(["subject_id", "epoch_idx"]).any()
    sl = df[df["sleep"]]
    key = sleep_epoch_key(val)
    assert np.array_equal(sl["subject_id"].to_numpy(), key["subject_id"].to_numpy())
    assert np.array_equal(sl["epoch_idx"].to_numpy(), key["epoch_idx"].to_numpy())
    assert np.array_equal(sl["apnoea_label"].to_numpy(), key["apnoea_label"].to_numpy())
    # window index recovered from the centre seconds matches j = epoch_idx // 6
    w_of = df.attrs["sec"][:, 0].astype(int)
    assert np.array_equal(win.j[w_of], df["epoch_idx"].to_numpy() // 6)


def test_gather_masks_padding_by_index_and_sleep(train):
    win = eval_windows(train)
    row = int(np.flatnonzero(train.subject_ids == 400003)[0])
    w0 = np.flatnonzero((win.row == row) & (win.j == 0))
    bd = train.gather(win, w0)
    assert not bd["valid"][0, :240].any() and bd["valid"][0, 240:].all()  # 60 s pre-pad
    assert not bd["x"][0, :240].any()
    # stages: 3 wake epochs first -> loss mask off for padding + wake, on for N2
    assert not bd["mask"][0, :60 + 90].any() and bd["mask"][0, 60 + 90:].all()
    last = np.flatnonzero((win.row == row) & (win.j == win.j[win.row == row].max()))
    bd = train.gather(win, last)
    n_ep = int(train.index.loc[row, "n_epochs"])
    tail_valid = bd["valid"][0].sum() // 4
    assert tail_valid == (n_ep * 30 - (180 * int(win.j[last][0]) - 60))


def test_iter_batches_prefetch_equals_serial(train):
    win = train_window_starts(train, 42, 0)
    order = pass_order(len(win), 42, 0)
    a = [(b, s, d["x"].copy()) for b, s, d in iter_batches(train, win, 7, order, prefetch=0)]
    b = [(b, s, d["x"].copy()) for b, s, d in iter_batches(train, win, 7, order, start_batch=2, prefetch=3)]
    assert [x[0] for x in b] == [x[0] for x in a][2:]
    for (_, s1, x1), (_, s2, x2) in zip(a[2:], b):
        assert np.array_equal(s1, s2) and np.array_equal(x1, x2)


# ---------------------------------------------------------------- test protection


def test_packed_test_split_is_refused(ds):
    """Design test 7 (data side): a test folder cannot be opened without allow_test."""
    with pytest.raises(TestDataRefused):
        PackedSplit(ds["out"] / "test", ("RR", "EDR"))
    ps = PackedSplit(ds["out"] / "test", ("RR", "EDR"), allow_test=True)
    assert ps.split == "test"


def test_split_guards(tmp_path):
    sp = load_split(write_split_json(tmp_path / "s.json", [1, 2, 3], [4], [5, 6]))
    assert_ids_in_split([1, 3], sp, "train")
    with pytest.raises(TestDataRefused):
        assert_ids_in_split([1, 5], sp, "train")
    with pytest.raises(ValueError):
        assert_ids_in_split([4], sp, "train")
    with pytest.raises(ValueError, match="both"):
        load_split(write_split_json(tmp_path / "bad.json", [1, 2], [2], [3]))


# ---------------------------------------------------------------- evaluation helpers


def _synthetic_key(n_subj=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        n = rng.integers(20, 40)
        ep = np.sort(rng.choice(60, size=n, replace=False))
        rows.append(pd.DataFrame({"subject_id": 500000 + s, "epoch_idx": ep,
                                  "apnoea_label": (rng.random(n) < 0.3).astype(np.int8)}))
    k = pd.concat(rows, ignore_index=True)
    return k.astype({"subject_id": np.int32, "epoch_idx": np.int32, "apnoea_label": np.int8})


def test_assemble_predictions_schema_and_equals():
    """Design test 8."""
    key = _synthetic_key()
    rng = np.random.default_rng(1)
    scores = key.sample(frac=1.0, random_state=2).assign(pred_prob=rng.random(len(key)))
    out = assemble_predictions(scores, key, 0.4)
    assert list(out.columns) == list(PRED_DTYPES)
    assert out[["subject_id", "epoch_idx", "apnoea_label"]].equals(key.reset_index(drop=True))
    assert {c: str(out[c].dtype) for c in out} == {"subject_id": "int32", "epoch_idx": "int32",
                                                  "apnoea_label": "int8", "pred_prob": "float64",
                                                  "pred_label": "int64"}
    assert ((out["pred_prob"] > 0.4) == (out["pred_label"] == 1)).all()
    with pytest.raises(ValueError, match="no finite prediction"):
        assemble_predictions(scores.iloc[1:], key, 0.4)
    bad = scores.copy()
    bad.loc[bad.index[0], "apnoea_label"] = 1 - bad.iloc[0]["apnoea_label"]
    with pytest.raises(ValueError, match="differs"):
        assemble_predictions(bad, key, 0.4)


def test_assemble_predictions_refuses_score_rows_outside_the_key():
    """A cache sleep epoch the canonical key lacks is an error, not silently dropped."""
    key = _synthetic_key()
    scores = key.assign(pred_prob=0.3)
    in_subject = pd.DataFrame({"subject_id": [500000], "epoch_idx": [999], "apnoea_label": [0], "pred_prob": [0.5]})
    new_subject = pd.DataFrame({"subject_id": [999999], "epoch_idx": [0], "apnoea_label": [0], "pred_prob": [0.5]})
    for extra in (in_subject, new_subject):
        with pytest.raises(ValueError, match="1 score rows are not in the key"):
            assemble_predictions(pd.concat([scores, extra], ignore_index=True), key, 0.4)


def test_pred_dtypes_match_canonical_key_file():
    p = ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only" / "test_predictions.parquet"
    if not p.exists():
        pytest.skip("canonical key file not present")
    import pyarrow.parquet as pq

    schema = pq.read_schema(p)  # schema only: no rows, no metrics
    got = {f.name: f.type for f in schema}
    assert str(got["subject_id"]) == "int32" and str(got["epoch_idx"]) == "int32"
    assert str(got["apnoea_label"]) == "int8" and str(got["pred_prob"]) == "double"
    assert str(got["pred_label"]) == "int64"


def test_check_schema_rejects_unsorted():
    key = _synthetic_key()
    df = assemble_predictions(key.assign(pred_prob=0.3), key, 0.5)
    with pytest.raises(ValueError, match="sorted"):
        check_schema(df.iloc[::-1].reset_index(drop=True))


def test_threshold_rule_is_strict_greater_than():
    grid = np.linspace(0.05, 0.95, 91)
    y = np.array([1, 1, 0])
    p = np.array([0.5, 0.5, grid[10]])  # negative scored exactly at a grid value
    thr, f1 = threshold_f1max(y, p)
    assert thr == grid[10] and f1 == 1.0  # with >= the first perfect threshold would be grid[11]


def _load_recovery_bootstrap():
    spec = importlib.util.spec_from_file_location("rpb", ROOT / "scripts" / "recovery_paired_bootstrap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_paired_delta_equals_subject_paired_bootstrap():
    key = _synthetic_key(n_subj=30, seed=3)
    rng = np.random.default_rng(4)
    y = key["apnoea_label"].to_numpy()
    pa = np.clip(0.3 * y + rng.random(len(y)) * 0.8, 0, 1)
    pb = np.clip(0.2 * y + rng.random(len(y)) * 0.8, 0, 1)
    da = key.assign(pred_prob=pa, pred_label=0)
    db = key.assign(pred_prob=pb, pred_label=0)
    a_auc, _ = _subject_bootstrap_aucs(da, n_resamples=200, seed=42)
    b_auc, _ = _subject_bootstrap_aucs(db, n_resamples=200, seed=42)
    mine = paired_delta(a_auc, b_auc)
    rpb = _load_recovery_bootstrap()
    ref = rpb.subject_paired_bootstrap(pd.DataFrame({"subject_id": key["subject_id"], "epoch_idx": key["epoch_idx"],
                                                     "y": y, "p_a": pa, "p_b": pb}), n_resamples=200, seed=42)
    assert np.allclose(a_auc - b_auc, ref["deltas_all"], atol=1e-12, equal_nan=True)
    assert mine["p_two_sided"] == ref["p_two_sided"]
    assert mine["delta_ci_low"] == pytest.approx(ref["delta_ci_low"])


def test_claim_rule_seed_average_and_gap():
    """Pre-registered verdicts (A-E, none) on paired_delta records; full cases in test_dl_eval_rules.py."""
    base = np.linspace(0.80, 0.82, 100)
    hi = [base + 0.03 + 0.001 * k for k in range(3)]
    per_seed = [paired_delta(h, base, float(h.mean()), float(base.mean())) for h in hi]
    sa = seed_averaged_delta(hi, base, [float(h.mean()) for h in hi], float(base.mean()))
    cr = claim_rule(per_seed, sa)
    assert cr["verdict"] == "A" and cr["direction"] == "DL > reference" and cr["abs_mean_gt_seed_sd"]
    small = [paired_delta(base + 0.005, base, 0.815, 0.81)] * 3
    assert claim_rule(small)["verdict"] == "B"  # every CI above 0 but |delta| < 0.01
    # all CIs above 0 and |mean| >= 0.01, but the mean lies within one seed SD -> B, not A
    spread = [paired_delta(base + d, base, float(base.mean()) + d, float(base.mean())) for d in (0.001, 0.002, 0.040)]
    cr = claim_rule(spread)
    assert cr["all_ci_above_0"] and cr["abs_mean_ge_min_effect"] and not cr["abs_mean_gt_seed_sd"]
    assert cr["mean_delta"] == pytest.approx(0.014333, abs=1e-6) and cr["sd_delta"] == pytest.approx(0.022234, abs=1e-6)
    assert cr["verdict"] == "B" and cr["direction"] is None
    neg = [paired_delta(base - d, base, float(base.mean()) - d, float(base.mean())) for d in (0.001, 0.002, 0.040)]
    assert claim_rule(neg)["verdict"] == "B" and claim_rule(neg)["all_ci_below_0"]
    assert claim_rule(per_seed[:2])["verdict"] is None
    assert sa["delta_boot_mean"] == pytest.approx(0.031) and sa["p_text"] == "p < 0.001"
    assert sa["delta_point"] == pytest.approx(np.mean([r["delta_point"] for r in per_seed]), abs=1e-12)
    runs =lambda arrs: {s: {"auc": float(a.mean()), "boot_auc": a} for s, a in zip((42, 43, 44), arrs)}  # noqa: E731
    g = gap_per_seed(runs(hi), runs([base] * 3), {"auc": float((base + 0.01).mean()), "boot_auc": base + 0.01},
                     {"auc": float(base.mean()), "boot_auc": base})
    assert g["seed_averaged"]["delta_boot_mean"] == pytest.approx(0.021)
    assert g["seed_averaged"]["delta_point"] == pytest.approx(0.021)
    assert g["claim_rule_auc"]["verdict"] == "A"


def test_metadata_with_ahi_handles_string_ids():
    sm = pd.DataFrame({"subject_id": ["shhs1-200001", "shhs1-200002"], "tst_min": [300.0, 320.0]})
    nsrr = pd.DataFrame({"subject_id": [200001, 200002, 200003], "ahi_a0h3a": [5.0, 12.0, 1.0]})
    m = metadata_with_ahi(sm, nsrr)
    assert m["subject_id"].tolist() == [200001, 200002] and m["ahi_a0h3a"].tolist() == [5.0, 12.0]
