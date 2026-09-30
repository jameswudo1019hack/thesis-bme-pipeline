"""Evaluation helpers for the Aim 2 DL arm (design sections 4 and 7).

* ``threshold_f1max``: the LightGBM rule (``fit_aim2_v85_taxonomy_ablation.py``):
  F1-max over ``np.linspace(0.05, 0.95, 91)`` with strict ``>`` and first-argmax ties.
* ``assemble_predictions``: epoch scores -> the ``test_predictions.parquet`` schema
  (subject_id int32, epoch_idx int32, apnoea_label int8, pred_prob float64,
  pred_label int64), in the key file's row order, with coverage and label checks.
* ``paired_delta``: subtract saved subject-bootstrap arrays. ``write_extended_metrics``
  resamples ``np.unique(subject_id)`` with ``default_rng(42)``, so two models scored
  on the same subjects share resamples and the arrays can be subtracted directly.
  p is the percentile two-sided p of ``recovery_paired_bootstrap.subject_paired_bootstrap``.
* ``claim_rule``: a direction is claimed only if every per-seed CI excludes 0 with the
  same sign, |mean delta| >= 0.01 and |mean delta| > the seed SD of the per-seed deltas
  (design section 7: "Anything smaller than 0.01, or within the seed SD, is reported as
  no difference"). The reference is fixed across seeds, so the SD of the per-seed
  deltas equals the SD of the DL per-seed AUCs; both readings give the same rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

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
) -> dict:
    """Delta = a - b on shared bootstrap resamples (NaN resamples dropped)."""
    a = np.asarray(a_boot, dtype=np.float64)
    b = np.asarray(b_boot, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"bootstrap arrays differ in shape: {a.shape} vs {b.shape}")
    d = a - b
    valid = d[np.isfinite(d)]
    if len(valid) == 0:
        raise ValueError("no finite paired resamples")
    p2, pl, pr = _p_two(valid)
    lo, hi = float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))
    out = {
        "delta_point": (float(a_point) - float(b_point)) if a_point is not None and b_point is not None else None,
        "delta_boot_mean": float(valid.mean()),
        "delta_ci_low": lo,
        "delta_ci_high": hi,
        "p_two_sided": p2,
        "p_left": pl,
        "p_right": pr,
        "p_text": "p < 0.001" if (valid > 0).all() or (valid < 0).all() else f"p = {p2:.3f}",
        "n_resamples_valid": int(len(valid)),
        "n_resamples_total": int(len(d)),
    }
    return out


def seed_averaged_delta(
    seed_boots: Sequence[np.ndarray], ref_boot: np.ndarray, seed_points: Sequence[float] | None = None,
    ref_point: float | None = None,
) -> dict:
    """Per resample: mean over seeds of the DL AUC minus the reference AUC."""
    stack = np.vstack([np.asarray(b, dtype=np.float64) for b in seed_boots])
    mean_boot = stack.mean(axis=0)
    a_point = float(np.mean(seed_points)) if seed_points is not None else None
    return paired_delta(mean_boot, ref_boot, a_point, ref_point)


def gap_contrast(
    dl_hi: Sequence[np.ndarray], dl_lo: Sequence[np.ndarray], ml_hi: np.ndarray, ml_lo: np.ndarray
) -> dict:
    """(mean_seeds DL_hi - mean_seeds DL_lo) - (ML_hi - ML_lo) on shared resamples.

    With DL_hi = P4, DL_lo = M, ML_hi = ECG+belt LightGBM, ML_lo = ECG-only LightGBM
    this compares the belt gain across model classes.
    """
    a = np.vstack(dl_hi).mean(axis=0) - np.vstack(dl_lo).mean(axis=0)
    b = np.asarray(ml_hi, dtype=np.float64) - np.asarray(ml_lo, dtype=np.float64)
    return paired_delta(a, b)


def claim_rule(per_seed: Sequence[Mapping], min_effect: float = MIN_EFFECT) -> dict:
    """Pre-declared claim rule over per-seed ``paired_delta`` results."""
    lows = np.array([r["delta_ci_low"] for r in per_seed])
    highs = np.array([r["delta_ci_high"] for r in per_seed])
    pts = np.array([
        r["delta_point"] if r.get("delta_point") is not None else r["delta_boot_mean"] for r in per_seed
    ])
    all_pos = bool((lows > 0).all())
    all_neg = bool((highs < 0).all())
    mean_d = float(pts.mean())
    sd_d = float(pts.std(ddof=1)) if len(pts) > 1 else float("nan")
    big = abs(mean_d) >= min_effect
    within_sd = bool(np.isfinite(sd_d) and abs(mean_d) <= sd_d)
    if len(per_seed) < 3:
        verdict = "insufficient seeds (< 3): no claim"
    elif all_pos and big and not within_sd:
        verdict = "DL > reference"
    elif all_neg and big and not within_sd:
        verdict = "DL < reference"
    else:
        verdict = "no difference claimed"
    return {
        "verdict": verdict,
        "n_seeds": len(per_seed),
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "all_ci_above_0": all_pos,
        "all_ci_below_0": all_neg,
        "abs_mean_ge_min_effect": big,
        "abs_mean_within_seed_sd": within_sd,
        "min_effect": min_effect,
        "rule": "direction only if every per-seed CI excludes 0 with the same sign, |mean delta| >= "
                f"{min_effect} and |mean delta| > seed SD (ddof 1); otherwise no difference",
    }


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
