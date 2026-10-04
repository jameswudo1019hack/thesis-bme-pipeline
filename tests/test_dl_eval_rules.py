"""Pre-registered evaluation rules of thesis_pipeline/dl_eval.py, on synthetic numbers.

* claim_rule: the examiner's synthetic cases for verdicts A-E and the fewer-than-3-seeds case;
  point estimates (never the bootstrap mean) drive the verdict;
* gap_per_seed: index pairing, seed-averaged point = mean of per-seed points, all 9 cross pairs;
* subset_bootstrap_auc: the same resamples as extended_metrics._subject_bootstrap_aucs;
* short_hypopnoea_mask: definition and alignment checks;
* integration (skipped when the files are absent): the short-hypopnoea ladder values of the
  pre-registration reproduce from the recovery test predictions (no new test scoring).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thesis_pipeline import dl_eval  # noqa: E402
from thesis_pipeline.dl_eval import (  # noqa: E402
    EQUIV_MARGIN,
    MIN_EFFECT,
    claim_rule,
    gap_per_seed,
    paired_delta,
    short_hypopnoea_mask,
    subset_bootstrap_auc,
)
from thesis_pipeline.extended_metrics import _subject_bootstrap_aucs  # noqa: E402


def _pd(point: float, lo: float, hi: float, boot_mean: float | None = None) -> dict:
    """A paired_delta-shaped record with a given point estimate and CI."""
    return {"delta_point": point, "delta_ci_low": lo, "delta_ci_high": hi,
            "delta_boot_mean": point if boot_mean is None else boot_mean}


def _above(*points: float) -> list[dict]:
    return [_pd(d, d / 2, d * 2) for d in points]  # every CI strictly above 0


def _sa(per_seed: list[dict], lo: float, hi: float) -> dict:
    return _pd(float(np.mean([r["delta_point"] for r in per_seed])), lo, hi)


# --------------------------------------------------------------------------- claim rule


def test_constants():
    assert MIN_EFFECT == EQUIV_MARGIN == 0.01
    assert dl_eval.PRIMARY_CONTRASTS == {"M": "ecg_only", "P4": "ecg_belt"}
    assert dl_eval.PRIMARY_GAP == ("P4", "M", "ecg_belt", "ecg_only")
    assert dl_eval.REQUIRED_SEEDS == (42, 43, 44)
    assert dl_eval.SHORT_HYP_MAX_S == 20.0 and dl_eval.E1_BOUND == 0.900
    assert all(len(v) == 64 for v in dl_eval.PINNED_REF_MODEL_SHA256.values())


@pytest.mark.parametrize("points", [(0.010, 0.060, 0.012), (0.001, 0.002, 0.040), (0.004, 0.005, 0.006)])
def test_consistent_but_small_or_within_spread_is_B(points):
    ps = _above(*points)
    cr = claim_rule(ps, _sa(ps, 0.001, 0.05))
    assert cr["verdict"] == "B" and cr["direction"] is None
    assert cr["label"] == "B: detectable but below the pre-declared 0.01 AUC / within seed spread"  # the note's wording
    assert cr["all_ci_above_0"] and not cr["all_ci_below_0"] and cr["any_ci_excludes_0"]
    assert cr["deltas"] == list(points)


def test_B_case_numbers_match_the_examiner():
    cr = claim_rule(_above(0.010, 0.060, 0.012))
    assert cr["mean_delta"] == pytest.approx(0.027333, abs=1e-6)
    assert cr["sd_delta"] == pytest.approx(0.028307, abs=1e-6)  # mean 0.027 < SD 0.028
    assert cr["abs_mean_ge_min_effect"] and not cr["abs_mean_gt_seed_sd"]
    cr = claim_rule(_above(0.001, 0.002, 0.040))
    assert cr["mean_delta"] == pytest.approx(0.014333, abs=1e-6) and cr["sd_delta"] == pytest.approx(0.022234, abs=1e-6)
    cr = claim_rule(_above(0.004, 0.005, 0.006))
    assert not cr["abs_mean_ge_min_effect"] and cr["abs_mean_gt_seed_sd"]


def test_mixed_seeds_are_C():
    ps = [_pd(0.0295, 0.020, 0.039), _pd(0.030, 0.020, 0.040), _pd(-0.002, -0.021, 0.017)]
    cr = claim_rule(ps, _sa(ps, -0.001, 0.035))
    assert cr["verdict"] == "C" and cr["label"] == "C: seed-dependent"
    assert cr["any_ci_excludes_0"] and not cr["all_ci_above_0"]


def test_opposite_sign_seeds_are_C():
    ps = [_pd(0.03, 0.02, 0.04), _pd(0.03, 0.02, 0.04), _pd(-0.03, -0.04, -0.02)]
    assert claim_rule(ps)["verdict"] == "C"


def test_clear_effect_is_A_in_both_directions():
    ps = _above(0.030, 0.031, 0.032)
    cr = claim_rule(ps, _sa(ps, 0.02, 0.04))
    assert cr["verdict"] == "A" and cr["label"] == "A: DL > reference" and cr["direction"] == "DL > reference"
    assert cr["abs_mean_ge_min_effect"] and cr["abs_mean_gt_seed_sd"]
    assert cr["seed_averaged_ci"] == (0.02, 0.04)
    neg = [_pd(-d, -2 * d, -d / 2) for d in (0.030, 0.031, 0.032)]
    cr = claim_rule(neg, _sa(neg, -0.04, -0.02))
    assert cr["verdict"] == "A" and cr["direction"] == "DL < reference" and cr["all_ci_below_0"]


def test_straddling_seeds_are_D_or_E_by_the_seed_averaged_ci():
    ps = [_pd(0.001, -0.006, 0.008), _pd(0.002, -0.005, 0.009), _pd(0.0, -0.007, 0.007)]
    cr = claim_rule(ps, _sa(ps, -0.004, 0.006))
    assert cr["verdict"] == "D" and cr["label"] == "D: equivalent within ±0.01"
    assert not cr["any_ci_excludes_0"]
    assert claim_rule(ps, _sa(ps, -0.02, 0.015))["verdict"] == "E"
    assert claim_rule(ps, _sa(ps, -0.02, 0.015))["label"] == "E: inconclusive"
    assert claim_rule(ps)["verdict"] == "E"  # no seed-averaged CI: D cannot apply
    assert claim_rule(ps, _sa(ps, -0.01, 0.005))["verdict"] == "E"  # strictly inside the margin


def test_fewer_than_three_seeds_gives_no_verdict():
    cr = claim_rule(_above(0.03, 0.031))
    assert cr["verdict"] is None and cr["direction"] is None
    assert cr["label"] == "no verdict: fewer than 3 seeds (\U0001f6a7)"
    assert cr["n_seeds"] == 2


def test_point_estimates_not_bootstrap_means_drive_the_verdict():
    ps = [_pd(d, d / 2, d * 2, boot_mean=0.001) for d in (0.030, 0.031, 0.032)]
    assert claim_rule(ps)["verdict"] == "A"
    with pytest.raises(ValueError, match="delta_point"):
        claim_rule([{**p, "delta_point": None} for p in ps])
    with pytest.raises(ValueError, match="seed-averaged"):
        claim_rule(ps, _pd(0.5, 0.02, 0.04))


def test_return_fields():
    cr = claim_rule(_above(0.03, 0.031, 0.032))
    assert set(cr) == {"verdict", "label", "direction", "n_seeds", "deltas", "mean_delta", "sd_delta",
                       "all_ci_above_0", "all_ci_below_0", "any_ci_excludes_0", "abs_mean_ge_min_effect",
                       "abs_mean_gt_seed_sd", "seed_averaged_ci", "min_effect", "equivalence_margin", "rule"}
    assert cr["seed_averaged_ci"] is None and cr["min_effect"] == 0.01 and cr["equivalence_margin"] == 0.01
    assert "design section" not in dl_eval.__doc__ and "pre-registration" in dl_eval.__doc__


# --------------------------------------------------------------------------- gap contrast


def _runs(base: np.ndarray, gains: dict[int, float]) -> dict[int, dict]:
    return {s: {"auc": float((base + g).mean()), "boot_auc": base + g} for s, g in gains.items()}


def test_gap_per_seed_pairs_by_index_and_reports_cross_pairs():
    rng = np.random.default_rng(1)
    base = 0.80 + 0.01 * rng.random(200)
    hi = _runs(base, {42: 0.06, 43: 0.061, 44: 0.062})
    lo = _runs(base, {42: 0.02, 43: 0.0205, 44: 0.021})
    ref_hi = {"auc": float((base + 0.01).mean()), "boot_auc": base + 0.01}
    ref_lo = {"auc": float(base.mean()), "boot_auc": base}
    g = gap_per_seed(hi, lo, ref_hi, ref_lo)
    assert g["seeds"] == [42, 43, 44] and g["pairing"] == "by model seed index"
    want = [0.06 - 0.02 - 0.01, 0.061 - 0.0205 - 0.01, 0.062 - 0.021 - 0.01]
    assert [r["delta_point"] for r in g["per_seed"]] == pytest.approx(want, abs=1e-12)
    assert g["seed_averaged"]["delta_point"] == pytest.approx(np.mean(want), abs=1e-12)
    assert len(g["cross_pairs"]) == 9 and "P4_42-M_44" in g["cross_pairs"]
    assert g["cross_pairs"]["P4_42-M_44"]["delta_point"] == pytest.approx(0.06 - 0.021 - 0.01, abs=1e-12)
    # cross pairs are reported only: estimation, no p-value; the index-paired estimates keep theirs
    assert not any(k.startswith("p_") for d in g["cross_pairs"].values() for k in d)
    assert all("p_two_sided" in d for d in g["per_seed"] + [g["seed_averaged"]])
    assert g["claim_rule_auc"]["verdict"] == "A" and g["claim_rule_auc"]["direction"] == "DL > reference"
    # identical to paired_delta on the per-resample gap
    direct = paired_delta(hi[43]["boot_auc"] - lo[43]["boot_auc"], ref_hi["boot_auc"] - ref_lo["boot_auc"],
                          hi[43]["auc"] - lo[43]["auc"], ref_hi["auc"] - ref_lo["auc"])
    assert g["per_seed"][1] == direct


def test_gap_per_seed_refuses_different_seed_sets():
    base = np.linspace(0.8, 0.82, 50)
    ref = {"auc": 0.81, "boot_auc": base}
    with pytest.raises(ValueError, match="identical seed sets"):
        gap_per_seed(_runs(base, {42: 0.0, 43: 0.0}), _runs(base, {42: 0.0, 44: 0.0}), ref, ref)


# --------------------------------------------------------------------------- subset bootstrap


def _frame(n_subj: int = 25, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_subj):
        n = int(rng.integers(5, 30))
        y = (rng.random(n) < (0.05 if k % 7 == 0 else 0.35)).astype(np.int8)
        rows.append(pd.DataFrame({"subject_id": np.int32(300000 + k), "epoch_idx": np.arange(n, dtype=np.int32),
                                  "apnoea_label": y, "pred_prob": np.clip(0.4 * y + rng.random(n) * 0.7, 0, 1)}))
    return pd.concat(rows, ignore_index=True)


def test_subset_bootstrap_with_all_rows_reproduces_extended_metrics_exactly():
    df = _frame()
    want, _ = _subject_bootstrap_aucs(df, n_resamples=200, seed=42)
    got = subset_bootstrap_auc(df["subject_id"].to_numpy(), df["apnoea_label"].to_numpy(),
                               df["pred_prob"].to_numpy(), np.ones(len(df), bool), n_resamples=200, seed=42)
    assert np.array_equal(got, want, equal_nan=True)
    # a frame whose rows are not grouped by subject draws the same resamples too
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    want, _ = _subject_bootstrap_aucs(shuffled, n_resamples=50, seed=42)
    got = subset_bootstrap_auc(shuffled["subject_id"].to_numpy(), shuffled["apnoea_label"].to_numpy(),
                               shuffled["pred_prob"].to_numpy(), np.ones(len(df), bool), n_resamples=50, seed=42)
    assert np.array_equal(got, want, equal_nan=True)


def test_subset_bootstrap_uses_the_full_frame_resamples():
    df = _frame()
    sid, y, p = df["subject_id"].to_numpy(), df["apnoea_label"].to_numpy(), df["pred_prob"].to_numpy()
    mask = (df["epoch_idx"].to_numpy() % 3 != 0) | (y == 1)
    got = subset_bootstrap_auc(sid, y, p, mask, n_resamples=30, seed=42)
    # brute force: the same draws as the full frame, restricted to the mask
    rng = np.random.default_rng(42)
    subjects = np.unique(sid)
    want = np.full(30, np.nan)
    for i in range(30):
        drawn = rng.choice(subjects.astype(np.int64), size=len(subjects), replace=True)
        idx = np.concatenate([np.where((sid == s) & mask)[0] for s in drawn])
        if len(np.unique(y[idx])) == 2:
            want[i] = roc_auc_score(y[idx], p[idx])
    assert np.array_equal(got, want, equal_nan=True)
    # a resample whose subset is single-class is NaN, and later resamples are unaffected
    only_neg = y == 0
    nan_run = subset_bootstrap_auc(sid, y, p, only_neg, n_resamples=10, seed=42)
    assert np.isnan(nan_run).all()


# --------------------------------------------------------------------------- short-hypopnoea mask


def _key_and_context():
    key = pd.DataFrame({"subject_id": np.repeat(np.int32([1, 2]), 6), "epoch_idx": np.tile(np.arange(6, dtype=np.int32), 2),
                        "apnoea_label": np.int8([0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1])})
    ctx = pd.DataFrame({
        "subject_id": key["subject_id"].astype(np.int64), "epoch_idx": key["epoch_idx"],
        "event_kind": ["", "hypopnoea", "hypopnoea", "apnoea", "hypopnoea", "", "", "", "hypopnoea", "apnoea", "",
                       "hypopnoea"],
        "event_duration_s": [np.nan, 12.0, 20.0, 15.0, 19.9, np.nan, np.nan, np.nan, 35.0, 8.0, np.nan, 10.0],
        "xml_label": key["apnoea_label"].to_numpy(np.int8),
    })
    return key, ctx


def test_short_hypopnoea_mask_definition():
    key, ctx = _key_and_context()
    mask, counts = short_hypopnoea_mask(ctx, key)
    # positives: hypopnoea < 20 s with label 1 (rows 1, 4, 11); negatives: every label-0 row
    assert counts == {"n_pos": 3, "n_neg": 5}
    assert np.flatnonzero(mask).tolist() == [0, 1, 4, 5, 6, 7, 10, 11]


def test_short_hypopnoea_mask_refuses_misaligned_context():
    key, ctx = _key_and_context()
    with pytest.raises(ValueError, match="epoch_idx"):
        short_hypopnoea_mask(ctx.assign(epoch_idx=ctx["epoch_idx"][::-1].to_numpy()), key)
    with pytest.raises(ValueError, match="xml_label"):
        short_hypopnoea_mask(ctx.assign(xml_label=np.int8(0)), key)
    with pytest.raises(ValueError, match="rows"):
        short_hypopnoea_mask(ctx.iloc[:-1], key)


# --------------------------------------------------------------------------- integration (real files, read-only)

RECOVERY = ROOT / "models" / "recovery_2026-05-03"
CONTEXT = ROOT / "models" / "aim2_analysis_v1" / "test_epoch_context.parquet"
LADDER = {
    "physio_only": (RECOVERY / "aim2_v85_taxonomy" / "physio_only", 0.8313),
    "exp2": (RECOVERY / "aim2_phase1_batch" / "physio_only" / "exp2", 0.8374),
    "full": (RECOVERY / "aim2_v85_taxonomy" / "full", 0.9696),
}


@pytest.mark.skipif(not CONTEXT.exists() or not all((d / "test_predictions.parquet").exists()
                                                    for d, _ in LADDER.values()),
                    reason="recovery predictions or epoch context not present")
def test_short_hypopnoea_ladder_reproduces_from_recovery_predictions():
    from thesis_pipeline.dl_stage_b import sha256_file

    assert sha256_file(CONTEXT) == dl_eval.EPOCH_CONTEXT_SHA256
    ctx = pd.read_parquet(CONTEXT)
    key = pd.read_parquet(LADDER["physio_only"][0] / "test_predictions.parquet",
                          columns=["subject_id", "epoch_idx", "apnoea_label"])
    mask, counts = short_hypopnoea_mask(ctx, key)
    assert counts == {"n_pos": 97271, "n_neg": 587056}
    y = key["apnoea_label"].to_numpy()
    for name, (d, want) in LADDER.items():
        pred = pd.read_parquet(d / "test_predictions.parquet")
        assert pred[["subject_id", "epoch_idx", "apnoea_label"]].equals(key), name
        p = pred["pred_prob"].to_numpy()
        assert round(float(roc_auc_score(y[mask], p[mask])), 4) == want, name


def test_paired_delta_without_p_is_estimation_only():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.80, 0.01, 200), rng.normal(0.78, 0.01, 200)
    with_p = dl_eval.paired_delta(a, b, 0.80, 0.78)
    est = dl_eval.paired_delta(a, b, 0.80, 0.78, with_p=False)
    assert {"p_two_sided", "p_left", "p_right", "p_text"} <= set(with_p)
    assert not any(k.startswith("p_") for k in est)
    assert {k: v for k, v in with_p.items() if not k.startswith("p_")} == est  # same estimate and CI
    sa = dl_eval.seed_averaged_delta([a, a + 0.001], b, [0.80, 0.801], 0.78, with_p=False)
    assert not any(k.startswith("p_") for k in sa) and sa["delta_point"] == pytest.approx(0.0205)
