"""Evaluation helpers for the Aim 2 DL arm.

The pre-registration note (``Thesis Vault/Experiments/2026-09-30 - Aim 2 DL Olsen BiGRU
pre-registration.md``, frozen body checked by ``thesis_pipeline.prereg``) governs every rule
here; where code and note disagree, the code is fixed.

* ``threshold_f1max``: the LightGBM rule (``fit_aim2_v85_taxonomy_ablation.py``):
  F1-max over ``np.linspace(0.05, 0.95, 91)`` with strict ``>`` and first-argmax ties.
* ``assemble_predictions``: epoch scores -> the ``test_predictions.parquet`` schema
  (subject_id int32, epoch_idx int32, apnoea_label int8, pred_prob float64,
  pred_label int64), in the key file's row order, with coverage and label checks.
* ``paired_delta``: subtract saved subject-bootstrap arrays. ``write_extended_metrics``
  resamples ``np.unique(subject_id)`` with ``default_rng(42)``, so two models scored
  on the same subjects share resamples and the arrays can be subtracted directly.
  p is the percentile two-sided p of ``recovery_paired_bootstrap.subject_paired_bootstrap``;
  it is reported only for the primary AUC-ROC contrasts (``with_p=False`` elsewhere: the
  pre-registration gives estimation-only endpoints no p-value).
* ``claim_rule``: the five pre-registered verdicts (A direction, B consistent but small or
  within seed spread, C seed-dependent, D equivalent within +-0.01, E inconclusive; no
  verdict with fewer than 3 seeds), on the full-test-set point estimates of the per-seed
  deltas, their per-seed CIs and the CI of the seed-averaged delta.
* ``gap_per_seed``: contrast 3, (P4 - M) - (LGBM-ECG+belt - LGBM-ECG), per index-paired
  seed, seed-averaged, and all cross pairs (reported only).
* ``short_hypopnoea_mask`` / ``subset_bootstrap_auc``: the key secondary endpoint (AUC on
  short-hypopnoea positives vs all negatives) on the same bootstrap resamples as the
  full-set metrics.

The primary contrasts, required seeds, pinned comparator models and the epoch-context file
are fixed constants below; ``scripts/evaluate_aim2_dl.py`` refuses anything else on test.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

KEY_COLS = ["subject_id", "epoch_idx", "apnoea_label"]
PRED_DTYPES = {
    "subject_id": np.int32,
    "epoch_idx": np.int32,
    "apnoea_label": np.int8,
    "pred_prob": np.float64,
    "pred_label": np.int64,
}
THRESHOLDS = np.linspace(0.05, 0.95, 91)
MIN_EFFECT = 0.01
EQUIV_MARGIN = MIN_EFFECT  # verdict D: seed-averaged CI strictly inside (-0.01, +0.01)

# Fixed by the pre-registration: the evaluated configs and their matched LightGBM cells.
PRIMARY_CONTRASTS = {"M": "ecg_only", "P4": "ecg_belt"}
PRIMARY_GAP = ("P4", "M", "ecg_belt", "ecg_only")  # (hi, lo, ref_hi, ref_lo)
REQUIRED_SEEDS = (42, 43, 44)
PINNED_REF_MODEL_SHA256 = {
    "ecg_only": "69b4cc840ac6dd5091b4c9c92d2d1e4844b35d1aba64e48debf54a7709e0fa3f",
    "ecg_belt": "741f4740a084c1c3e9e6966a77f7dd63dd206ec4bf5b903926f8979d13a73c2f",
}
# models/aim2_analysis_v1/test_epoch_context.parquet (per-epoch longest labelling event)
EPOCH_CONTEXT_SHA256 = "73ddabae7104131e2ad4363c4ba52fbe06a9279d17dd579cd762e470bc0641df"
SHORT_HYP_MAX_S = 20.0
PARITY_TOL = 0.002  # |validation AUC on the test-inference device - training-device AUC| (parity gate)
# Record of a failed parity gate on one device type (format with the device type). Written by
# predict_aim2_dl.py into the run folder and, for a reportable checkpoint, into the runs root;
# never overwritten or removed by code, so a failure cannot be erased by a passing retry.
PARITY_FAILED_FILE = "PARITY_FAILED_{}.json"
# train_aim2_dl.py options marked PILOT ONLY: a reportable run trained with either is refused.
PILOT_ONLY_SETTINGS = ("max_passes", "max_batches_per_pass")
# TrainSettings.run_fields() minus config, model_seed and amp: identical across every reported run.
SHARED_TRAIN_SETTINGS = ("recipe", "lr", "weight_decay", "min_delta", "hidden", "max_passes",
                         "max_batches_per_pass", "split_seed")
E1_BOUND = 0.900  # expectation E1: M seed-averaged short-hypopnoea AUC < 0.900


def threshold_f1max(y: np.ndarray, p: np.ndarray, grid: np.ndarray = THRESHOLDS) -> tuple[float, float]:
    """(threshold, F1) maximising F1 of ``p > t`` over ``grid`` (first max wins)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    f1s = [f1_score(y, (p > t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), float(f1s[i])


def assemble_predictions(
    scores: pd.DataFrame,
    key: pd.DataFrame,
    threshold: float,
    prob_col: str = "pred_prob",
) -> pd.DataFrame:
    """Join epoch scores onto the key rows and return the canonical schema.

    ``scores`` needs subject_id, epoch_idx, ``prob_col`` and apnoea_label (the label
    derived from the DL cache). Every key row must get exactly one finite score, every
    score row must match a key row (so a sleep-mask disagreement in either direction
    raises), and the cache label must equal the key label on every row.
    """
    need = {"subject_id", "epoch_idx", prob_col, "apnoea_label"}
    if need - set(scores.columns):
        raise ValueError(f"scores missing {sorted(need - set(scores.columns))}")
    k = key[KEY_COLS].reset_index(drop=True)
    s = scores[["subject_id", "epoch_idx", prob_col, "apnoea_label"]].rename(
        columns={prob_col: "_p", "apnoea_label": "_y_cache"}
    )
    if s.duplicated(["subject_id", "epoch_idx"]).any():
        raise ValueError("duplicate (subject_id, epoch_idx) in scores")
    s = s.astype({"subject_id": np.int64, "epoch_idx": np.int64})
    m = k.astype({"subject_id": np.int64, "epoch_idx": np.int64}).merge(
        s, on=["subject_id", "epoch_idx"], how="left", validate="one_to_one"
    )
    if m["_p"].isna().any():
        n = int(m["_p"].isna().sum())
        raise ValueError(f"{n} key rows have no finite prediction")
    n_extra = len(s) - int(m["_y_cache"].notna().sum())
    if n_extra:
        extra = s.merge(k[["subject_id", "epoch_idx"]].astype(np.int64), on=["subject_id", "epoch_idx"],
                        how="left", indicator=True)
        extra = extra[extra["_merge"] == "left_only"]
        raise ValueError(f"{n_extra} score rows are not in the key (e.g. "
                         f"{extra[['subject_id', 'epoch_idx']].head(3).to_numpy().tolist()}); "
                         "the DL sleep mask disagrees with the canonical key")
    if not np.array_equal(m["_y_cache"].to_numpy(np.int64), m["apnoea_label"].to_numpy(np.int64)):
        n = int((m["_y_cache"].to_numpy() != m["apnoea_label"].to_numpy()).sum())
        raise ValueError(f"cache apnoea_label differs from key on {n} rows")
    p = m["_p"].to_numpy(np.float64)
    out = pd.DataFrame({
        "subject_id": k["subject_id"].to_numpy(np.int32),
        "epoch_idx": k["epoch_idx"].to_numpy(np.int32),
        "apnoea_label": k["apnoea_label"].to_numpy(np.int8),
        "pred_prob": p,
        "pred_label": (p > float(threshold)).astype(np.int64),
    })
    check_schema(out, key)
    return out


def check_schema(df: pd.DataFrame, key: pd.DataFrame | None = None) -> None:
    """Assert dtypes, order, no NaN and (optionally) key-column equality with ``key``."""
    if list(df.columns) != list(PRED_DTYPES):
        raise ValueError(f"columns {list(df.columns)} != {list(PRED_DTYPES)}")
    for c, dt in PRED_DTYPES.items():
        if df[c].dtype != np.dtype(dt):
            raise ValueError(f"{c} dtype {df[c].dtype} != {np.dtype(dt)}")
    if df.isna().any().any():
        raise ValueError("NaN in predictions")
    sid, eid = df["subject_id"].to_numpy(np.int64), df["epoch_idx"].to_numpy(np.int64)
    ordered = np.all((np.diff(sid) > 0) | ((np.diff(sid) == 0) & (np.diff(eid) > 0)))
    if not ordered:
        raise ValueError("rows not sorted by (subject_id, epoch_idx) or duplicated")
    if key is not None:
        k = key[KEY_COLS].reset_index(drop=True)
        k = k.astype({"subject_id": np.int32, "epoch_idx": np.int32, "apnoea_label": np.int8})
        if not df[KEY_COLS].reset_index(drop=True).equals(k):
            raise ValueError("key columns do not DataFrame.equals the key file")


def _p_two(valid: np.ndarray) -> tuple[float, float, float]:
    p_left = float(np.mean(valid <= 0.0))
    p_right = float(np.mean(valid >= 0.0))
    return 2.0 * min(p_left, p_right), p_left, p_right


def paired_delta(
    a_boot: np.ndarray,
    b_boot: np.ndarray,
    a_point: float | None = None,
    b_point: float | None = None,
    with_p: bool = True,
) -> dict:
    """Delta = a - b on shared bootstrap resamples (NaN resamples dropped).

    ``with_p=False`` (estimation-only endpoints: the pre-registration's Multiplicity section
    gives them a point estimate and a 95 % CI but no p-value) leaves out the p fields.
    """
    a = np.asarray(a_boot, dtype=np.float64)
    b = np.asarray(b_boot, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"bootstrap arrays differ in shape: {a.shape} vs {b.shape}")
    d = a - b
    valid = d[np.isfinite(d)]
    if len(valid) == 0:
        raise ValueError("no finite paired resamples")
    lo, hi = float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))
    out = {
        "delta_point": (float(a_point) - float(b_point)) if a_point is not None and b_point is not None else None,
        "delta_boot_mean": float(valid.mean()),
        "delta_ci_low": lo,
        "delta_ci_high": hi,
    }
    if with_p:
        p2, pl, pr = _p_two(valid)
        out |= {
            "p_two_sided": p2,
            "p_left": pl,
            "p_right": pr,
            "p_text": "p < 0.001" if (valid > 0).all() or (valid < 0).all() else f"p = {p2:.3f}",
        }
    out |= {"n_resamples_valid": int(len(valid)), "n_resamples_total": int(len(d))}
    return out


def seed_averaged_delta(
    seed_boots: Sequence[np.ndarray], ref_boot: np.ndarray, seed_points: Sequence[float] | None = None,
    ref_point: float | None = None, with_p: bool = True,
) -> dict:
    """Per resample: mean over seeds of the DL AUC minus the reference AUC."""
    stack = np.vstack([np.asarray(b, dtype=np.float64) for b in seed_boots])
    mean_boot = stack.mean(axis=0)
    a_point = float(np.mean(seed_points)) if seed_points is not None else None
    return paired_delta(mean_boot, ref_boot, a_point, ref_point, with_p=with_p)


def gap_per_seed(
    hi_runs: Mapping[int, Mapping],
    lo_runs: Mapping[int, Mapping],
    ref_hi: Mapping,
    ref_lo: Mapping,
    hi_name: str = "P4",
    lo_name: str = "M",
) -> dict:
    """Contrast 3: (AUC hi - AUC lo) - (AUC ref_hi - AUC ref_lo), seeds paired by index.

    ``hi_runs`` / ``lo_runs``: {seed: {"auc": float, "boot_auc": array}}; refs: {"auc", "boot_auc"}.
    The seed sets must be identical. Per seed s: paired_delta of the per-resample gap. Seed-averaged:
    mean over seeds of the DL AUCs before differencing (its point equals the mean of the per-seed
    points). Cross pairs (hi seed i, lo seed j), all of them, are reported only (no verdict, no
    p-value: estimation only).
    """
    if set(hi_runs) != set(lo_runs):
        raise ValueError(f"gap needs identical seed sets: {hi_name} {sorted(hi_runs)} vs {lo_name} {sorted(lo_runs)}")
    seeds = sorted(int(s) for s in hi_runs)
    if not seeds:
        raise ValueError("gap needs at least one seed")
    rb = np.asarray(ref_hi["boot_auc"], dtype=np.float64) - np.asarray(ref_lo["boot_auc"], dtype=np.float64)
    rp = float(ref_hi["auc"]) - float(ref_lo["auc"])

    def boot(r: Mapping) -> np.ndarray:
        return np.asarray(r["boot_auc"], dtype=np.float64)

    per_seed = [paired_delta(boot(hi_runs[s]) - boot(lo_runs[s]), rb,
                             float(hi_runs[s]["auc"]) - float(lo_runs[s]["auc"]), rp) for s in seeds]
    hi_mean = np.vstack([boot(hi_runs[s]) for s in seeds]).mean(axis=0)
    lo_mean = np.vstack([boot(lo_runs[s]) for s in seeds]).mean(axis=0)
    hi_pt = float(np.mean([float(hi_runs[s]["auc"]) for s in seeds]))
    lo_pt = float(np.mean([float(lo_runs[s]["auc"]) for s in seeds]))
    seed_averaged = paired_delta(hi_mean - lo_mean, rb, hi_pt - lo_pt, rp)
    cross = {
        f"{hi_name}_{i}-{lo_name}_{j}": paired_delta(boot(hi_runs[i]) - boot(lo_runs[j]), rb,
                                                     float(hi_runs[i]["auc"]) - float(lo_runs[j]["auc"]), rp,
                                                     with_p=False)
        for i in seeds for j in seeds
    }
    return {
        "seeds": seeds,
        "pairing": "by model seed index",
        "per_seed": per_seed,
        "seed_averaged": seed_averaged,
        "cross_pairs": cross,
        "claim_rule_auc": claim_rule(per_seed, seed_averaged),
    }


CLAIM_RULE_TEXT = (
    "Exactly one verdict, the first that applies: no verdict with fewer than 3 seeds; "
    "A (direction) if every per-seed 95% CI excludes 0 with the same sign, |mean delta| >= 0.01 and "
    "|mean delta| > the SD (ddof 1) of the per-seed point deltas; B (detectable but below the "
    "pre-declared 0.01 AUC / within seed spread) if every per-seed CI excludes 0 with the same sign but A fails; "
    "C (seed-dependent) if at least one per-seed CI excludes 0; D (equivalent within +-0.01) if the "
    "seed-averaged 95% CI lies strictly inside (-0.01, +0.01); otherwise E (inconclusive)."
)


def claim_rule(
    per_seed: Sequence[Mapping],
    seed_averaged: Mapping | None = None,
    min_effect: float = MIN_EFFECT,
    n_required: int = 3,
) -> dict:
    """Pre-registered verdict (A-E, or none) for one primary contrast on AUC-ROC.

    ``per_seed``: one ``paired_delta`` per model seed, each with a non-None ``delta_point`` (the
    full-test-set point estimate; the bootstrap mean is never used). ``seed_averaged``: the
    ``paired_delta`` of the seed-averaged contrast; its CI decides verdict D, and its point (if
    any) must equal the mean of the per-seed points.
    """
    for r in per_seed:
        if r.get("delta_point") is None:
            raise ValueError("claim_rule needs delta_point (the full-test-set point estimate) for every seed")
    deltas = [float(r["delta_point"]) for r in per_seed]
    lows = np.array([float(r["delta_ci_low"]) for r in per_seed])
    highs = np.array([float(r["delta_ci_high"]) for r in per_seed])
    n = len(per_seed)
    mean_d = float(np.mean(deltas)) if n else float("nan")
    sd_d = float(np.std(deltas, ddof=1)) if n > 1 else float("nan")
    if seed_averaged is not None and seed_averaged.get("delta_point") is not None and n:
        if not abs(float(seed_averaged["delta_point"]) - mean_d) < 1e-9:
            raise ValueError(f"seed-averaged delta_point {seed_averaged['delta_point']!r} != mean of the "
                             f"per-seed points {mean_d!r}")
    all_pos = bool(n and (lows > 0).all())
    all_neg = bool(n and (highs < 0).all())
    any_excl = bool(((lows > 0) | (highs < 0)).any())
    big = bool(np.isfinite(mean_d) and abs(mean_d) >= min_effect)
    gt_sd = bool(np.isfinite(sd_d) and abs(mean_d) > sd_d)
    margin = EQUIV_MARGIN
    sa_ci = None
    if seed_averaged is not None:
        sa_ci = (float(seed_averaged["delta_ci_low"]), float(seed_averaged["delta_ci_high"]))

    direction = None
    if n < n_required:
        verdict, label = None, f"no verdict: fewer than {n_required} seeds (\U0001f6a7)"
    elif (all_pos or all_neg) and big and gt_sd:
        direction = "DL > reference" if all_pos else "DL < reference"
        verdict, label = "A", f"A: {direction}"
    elif all_pos or all_neg:
        # the wording the pre-registration prescribes for verdict B
        verdict, label = "B", f"B: detectable but below the pre-declared {min_effect:g} AUC / within seed spread"
    elif any_excl:
        verdict, label = "C", "C: seed-dependent"
    elif sa_ci is not None and -margin < sa_ci[0] and sa_ci[1] < margin:
        verdict, label = "D", f"D: equivalent within \u00b1{margin:g}"
    else:
        verdict, label = "E", "E: inconclusive"
    return {
        "verdict": verdict,
        "label": label,
        "direction": direction,
        "n_seeds": n,
        "deltas": deltas,
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "all_ci_above_0": all_pos,
        "all_ci_below_0": all_neg,
        "any_ci_excludes_0": any_excl,
        "abs_mean_ge_min_effect": big,
        "abs_mean_gt_seed_sd": gt_sd,
        "seed_averaged_ci": sa_ci,
        "min_effect": min_effect,
        "equivalence_margin": margin,
        "rule": CLAIM_RULE_TEXT,
    }


# --------------------------------------------------------------------------- key secondary endpoint

SHORT_HYP_DEFINITION = (
    "AUC-ROC of pred_prob with positives = test sleep epochs with apnoea_label 1 whose longest labelling "
    "event in models/aim2_analysis_v1/test_epoch_context.parquet is a hypopnoea (event_kind 'hypopnoea') "
    f"with event_duration_s < {SHORT_HYP_MAX_S:g}, and negatives = all apnoea_label 0 test sleep epochs; "
    "CIs from the same subject-bootstrap resamples as the full-set metrics, restricted to these rows."
)


def short_hypopnoea_mask(context: pd.DataFrame, key: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Rows of ``key`` in the short-hypopnoea endpoint, and {n_pos, n_neg}.

    ``context`` (test_epoch_context.parquet) must be row-aligned with ``key`` (same subject_id
    and epoch_idx arrays) and its xml_label must equal the key's apnoea_label on every row.
    """
    if len(context) != len(key):
        raise ValueError(f"epoch context has {len(context)} rows, key {len(key)}")
    for c in ("subject_id", "epoch_idx"):
        if not np.array_equal(context[c].to_numpy(np.int64), key[c].to_numpy(np.int64)):
            raise ValueError(f"epoch context {c} differs from the key (row order or rows)")
    y = key["apnoea_label"].to_numpy(np.int64)
    if not np.array_equal(context["xml_label"].to_numpy(np.int64), y):
        n = int((context["xml_label"].to_numpy(np.int64) != y).sum())
        raise ValueError(f"epoch context xml_label differs from the key apnoea_label on {n} rows")
    dur = context["event_duration_s"].to_numpy(np.float64)
    with np.errstate(invalid="ignore"):
        short = dur < SHORT_HYP_MAX_S  # NaN (no event) -> False
    pos = (y == 1) & (context["event_kind"].astype(str).to_numpy() == "hypopnoea") & short
    neg = y == 0
    return pos | neg, {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}


def subset_bootstrap_auc(
    subject_id: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    mask: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Subject-bootstrap AUCs on the ``mask`` rows, with the full frame's resamples.

    Draws exactly what ``extended_metrics._subject_bootstrap_aucs`` draws on the full frame
    (default_rng(seed) over np.unique(subject_id) of ALL rows, one ``rng.choice`` per resample,
    also for resamples that turn out undefined), then scores only the drawn subjects' rows that
    are inside ``mask``. NaN where the resampled subset has a single class.
    """
    sids = np.asarray(subject_id)
    y = np.asarray(y).astype(np.int8)
    p = np.asarray(p)
    mask = np.asarray(mask, dtype=bool)
    if not (len(sids) == len(y) == len(p) == len(mask)):
        raise ValueError("subject_id, y, p and mask differ in length")
    rng = np.random.default_rng(seed)
    subjects = np.unique(sids)
    n_subj = len(subjects)
    order = np.argsort(sids, kind="stable")  # ascending row index within each subject, like np.where
    bounds = np.searchsorted(sids[order], subjects, side="left")
    ends = np.append(bounds[1:], len(order))
    by_subject = {}
    for s, a, b in zip(subjects, bounds, ends):
        rows = order[a:b]
        by_subject[int(s)] = rows[mask[rows]]
    subjects_int = np.asarray(list(by_subject.keys()), dtype=np.int64)
    aucs = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        sampled = rng.choice(subjects_int, size=n_subj, replace=True)
        idx = np.concatenate([by_subject[int(s)] for s in sampled])
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        aucs[i] = roc_auc_score(yi, p[idx])
    return aucs


def load_nsrr_ahi(csv_path: str | Path) -> pd.DataFrame:
    """(subject_id int64, ahi_a0h3a) from the NSRR SHHS-1 CSV (read-only)."""
    df = pd.read_csv(csv_path, usecols=["nsrrid", "ahi_a0h3a"])
    return pd.DataFrame({
        "subject_id": df["nsrrid"].astype(np.int64),
        "ahi_a0h3a": pd.to_numeric(df["ahi_a0h3a"], errors="coerce"),
    })


def metadata_with_ahi(subject_metadata: pd.DataFrame | None, nsrr_ahi: pd.DataFrame) -> pd.DataFrame:
    """subject_metadata (subject_id may be 'shhs1-<id>') merged with ahi_a0h3a."""
    if subject_metadata is None:
        return nsrr_ahi.copy()
    sm = subject_metadata.copy()
    sid = sm["subject_id"]
    if sid.dtype == object:
        sid = pd.to_numeric(sid.astype(str).str.extract(r"(\d+)$", expand=False), errors="coerce")
    sm["subject_id"] = sid.astype(np.int64)
    sm = sm.drop(columns=[c for c in ("ahi_a0h3a",) if c in sm.columns])
    return sm.merge(nsrr_ahi, on="subject_id", how="left")
