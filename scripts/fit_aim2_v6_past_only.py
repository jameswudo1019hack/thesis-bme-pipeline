"""Aim 2 v6 past-only context — does the v6 lift survive without future leakage?

Pre-registration: see [[Experiments/2026-04-30 - v6 past-only context
(pre-registration)]] in the vault. Hypothesis table is locked in git BEFORE
this script runs. Three pre-committed outcomes:

  Survival (≥ 0.840):  past-only matches v6 future-inclusive (0.8455).
                        v6 lift is real-time deployable.
  Mixed    (0.800-0.839): both past + future contexts contribute.
  Collapse (≤ 0.799):  v6 lift was substantially future-leakage; the
                        deployable AUC is much closer to v7 (0.7794).

Methodology
-----------
Starts from the audit-v6 schema (`2026-04-26-audit-v6`, 203 cols). DROPS:
  - _lead1 (explicit future)
  - _roll5_mean / _roll5_std / _roll11_mean / _roll11_std  (centred rolling
    looks both directions)

KEEPS:
  - 29 base features
  - 29 _lag1 columns (true past)

RECOMPUTES at fit time, per subject (right-aligned rolling, min_periods=1):
  - _past_roll5_mean / _past_roll5_std
  - _past_roll11_mean / _past_roll11_std

Final feature count: 29 + 29 + 116 = 174 (vs v6's 203). Slightly fewer
because _lead1 has no past-only equivalent.

Same FIXED LightGBM params, same seed=42 outer split, same sleep-only filter,
same subject-level bootstrap as v8.5-tax / Phase 1 batch. Direct paired-test
against `aim2_cv_v6` (future-inclusive baseline) and `aim2_cv_v7` (no-context
baseline) is the headline comparison.

Usage:
    cd Code && python3 scripts/fit_aim2_v6_past_only.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import click
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.epochs import sleep_mask  # noqa: E402
from thesis_pipeline.extended_metrics import write_extended_metrics  # noqa: E402

FEATURES_DIR = CODE_ROOT / "features"
OUT_DIR = CODE_ROOT / "models" / "aim2_v6_past_only"

NON_FEATURE = {
    "subject_id", "cohort", "epoch_idx", "epoch_start_sec",
    "sleep_stage", "apnoea_label", "features_version",
}

# Suffixes that mark a column as a contextual derivative of a base feature.
# Future-leaking ones get DROPPED; _lag1 is past-only and stays.
FUTURE_LEAKING_SUFFIXES = ("_lead1", "_roll5_mean", "_roll5_std", "_roll11_mean", "_roll11_std")
PAST_KEPT_SUFFIXES = ("_lag1",)
ALL_DERIVATIVE_SUFFIXES = FUTURE_LEAKING_SUFFIXES + PAST_KEPT_SUFFIXES

PAST_ROLL_WINDOWS = (5, 11)

# Pre-reg specifies audit-v6 feature subset. The local cohort is now on
# phase1batch-v1 (a superset of audit-v6). At fit time we drop the Phase 1
# NEW base columns + their contextual derivatives to recover the audit-v6
# feature set. Documented in the pre-reg operational-deviation appendix.
PHASE1_NEW_BASE_COLS = (
    # Exp 1
    "spo2_sampen", "hrv_sampen",
    # Exp 2
    "spo2_psd_apnea_band", "spo2_psd_total", "spo2_psd_apnea_ratio",
    # Exp 3
    "cpc_amp_cv", "cpc_amp_iqr",
    # Exp 4
    "ecg_band_low_power", "ecg_band_mid_low_power", "ecg_band_mid_power",
    "ecg_band_mid_high_power", "ecg_band_high_power", "ecg_band_total_power",
    "ecg_band_low_rel", "ecg_band_mid_low_rel", "ecg_band_mid_rel",
    "ecg_band_mid_high_rel", "ecg_band_high_rel",
)

# Sprint 1 (2026-04-26) added these 10 base cols on top of the original
# 29-base aim2_cv_v6 schema: 4 freq-HRV + 1 hypoxic-burden + 5 position.
# The --match-v6-schema flag drops these too, recovering the EXACT
# 29-base × 7-suffix = 203-col schema aim2_cv_v6 was trained on. This is the
# proper paired comparator for the original "v6 contextual lift" question.
SPRINT1_NEW_BASE_COLS = (
    # T3 frequency-HRV
    "hrv_lf_power", "hrv_hf_power", "hrv_lf_hf_ratio", "hrv_total_power_freq",
    # T4 hypoxic burden
    "hypoxic_burden_epoch",
    # T5 position
    "position_right_frac", "position_left_frac", "position_supine_frac",
    "position_prone_frac", "position_upright_frac",
)

ALL_SUFFIXES_INCLUDING_BASE = ("",) + ALL_DERIVATIVE_SUFFIXES


def _expand_drop_list(base_cols: tuple[str, ...]) -> set[str]:
    """Expand base col names to {base, base_lag1, base_lead1, base_roll5_mean, ...}."""
    return {base + sfx for base in base_cols for sfx in ALL_SUFFIXES_INCLUDING_BASE}


def phase1_columns_to_drop(all_cols: list[str]) -> list[str]:
    targets = _expand_drop_list(PHASE1_NEW_BASE_COLS)
    return sorted(c for c in all_cols if c in targets)


def sprint1_columns_to_drop(all_cols: list[str]) -> list[str]:
    targets = _expand_drop_list(SPRINT1_NEW_BASE_COLS)
    return sorted(c for c in all_cols if c in targets)

FIXED_PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 48,
    "min_child_samples": 100,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.0,
    "reg_lambda": 0.1,
    "objective": "binary",
    "metric": "auc",
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
FIXED_N_ESTIMATORS = 800
FIXED_EARLY_STOPPING = 50


def is_future_leaking(col: str) -> bool:
    return any(col.endswith(s) for s in FUTURE_LEAKING_SUFFIXES)


def is_derivative(col: str) -> bool:
    return any(col.endswith(s) for s in ALL_DERIVATIVE_SUFFIXES)


def find_base_features(all_cols: list[str], non_feature: set[str]) -> list[str]:
    """Base features = non-meta, non-derivative columns."""
    return sorted(c for c in all_cols if c not in non_feature and not is_derivative(c))


def _compute_keep_columns(probe_path: Path, match_v6_schema: bool, include_future_context: bool) -> list[str]:
    """Probe one parquet to determine which columns to load.

    Always drops: Phase 1 cols (recovers audit-v6).
    Optionally drops Sprint 1 cols when match_v6_schema=True, recovering the
    exact 203-col aim2_cv_v6 schema.
    Optionally KEEPS _lead1 + centred _roll* when include_future_context=True
    (matched future-inclusive baseline run; no past-only recomputation needed).
    """
    head = pd.read_parquet(probe_path)
    cols = list(head.columns)
    keep_meta = {"subject_id", "epoch_idx", "apnoea_label", "features_version", "sleep_stage"}
    p1_drop = set(phase1_columns_to_drop(cols))
    s1_drop = set(sprint1_columns_to_drop(cols)) if match_v6_schema else set()

    keep: list[str] = []
    for c in cols:
        if c in keep_meta:
            keep.append(c)
            continue
        if c in p1_drop or c in s1_drop:
            continue
        if not include_future_context and any(c.endswith(s) for s in FUTURE_LEAKING_SUFFIXES):
            continue
        keep.append(c)
    return keep


def load_cohort(features_dir: Path, version: str, match_v6_schema: bool,
                include_future_context: bool) -> pd.DataFrame:
    """Load needed columns per parquet, filter to features_version. NO sleep filter here.

    Sleep filter is applied LATER, after past-only rolling is computed on the
    full PSG sequence (so rolling windows respect real-time epoch order rather
    than jumping over wake periods).
    """
    files = [
        f for f in sorted(features_dir.glob("shhs1-*.parquet"))
        if not f.name.startswith("._")
    ]
    if not files:
        raise FileNotFoundError(f"No per-subject parquet files in {features_dir}")

    keep_cols = _compute_keep_columns(files[0], match_v6_schema=match_v6_schema,
                                      include_future_context=include_future_context)
    total_cols = len(pd.read_parquet(files[0]).columns)
    print(f"  loading {len(files)} parquet files "
          f"(reading {len(keep_cols)}/{total_cols} cols/file; "
          f"match_v6_schema={match_v6_schema}, future_context={include_future_context})...")

    KEEP_META = {"subject_id", "epoch_idx", "apnoea_label", "features_version", "sleep_stage"}

    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=keep_cols)
        for c in df.columns:
            if c in KEEP_META:
                continue
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        if "subject_id" in df.columns:
            df["subject_id"] = df["subject_id"].astype(np.int32)
        if "apnoea_label" in df.columns:
            df["apnoea_label"] = df["apnoea_label"].astype(np.int8)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True, copy=False)
    n_before = df["subject_id"].nunique()
    df = df[df["features_version"] == version].reset_index(drop=True)
    n_after = df["subject_id"].nunique()
    print(f"  features_version filter: {version!r} → {n_after}/{n_before} subjects ({len(df):,} pre-filter epochs)")
    return df


def apply_sleep_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep SLEEP epochs (sleep_mask returns True for N1/N2/N3/REM)."""
    n_before = len(df)
    mask = sleep_mask(df.sleep_stage.values)
    df = df[mask].reset_index(drop=True)
    assert df["sleep_stage"].isin(["N1", "N2", "N3", "REM"]).all(), (
        f"sleep filter failed: found stages {sorted(df['sleep_stage'].unique())} after filter"
    )
    print(f"  sleep-only filter: kept {len(df):,}/{n_before:,} epochs ({100*len(df)/n_before:.1f}%)")
    return df


def add_past_only_rolling(df: pd.DataFrame, base_cols: list[str], windows: tuple[int, ...]) -> pd.DataFrame:
    """Add right-aligned past-only rolling mean/std per subject for each base column.

    Sorts by (subject_id, epoch_idx) before rolling so the rolling windows
    respect within-subject epoch order. Returns the input df with new columns
    appended; original df is not mutated.
    """
    print(f"  computing past-only rolling features for {len(base_cols)} base cols × {len(windows)} windows × 2 stats...")
    t0 = time.time()
    df = df.sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True)
    g = df.groupby("subject_id", sort=False)

    new_cols: dict[str, np.ndarray] = {}
    for w in windows:
        for col in base_cols:
            r = g[col].rolling(window=w, min_periods=1)
            mean_series = r.mean().reset_index(level=0, drop=True)
            std_series = r.std().reset_index(level=0, drop=True)
            new_cols[f"{col}_past_roll{w}_mean"] = mean_series.astype(np.float32).values
            new_cols[f"{col}_past_roll{w}_std"] = std_series.astype(np.float32).values

    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    print(f"  rolling done in {time.time()-t0:.0f}s; added {len(new_cols)} columns")
    return out


def subject_bootstrap(
    df_test: pd.DataFrame, probs: np.ndarray, n_resamples: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    sids = df_test["subject_id"].values
    y = df_test["apnoea_label"].values.astype(np.int8)

    subjects = np.unique(sids)
    n_subj = len(subjects)
    by_subject = {int(s): np.where(sids == s)[0] for s in subjects}
    subjects_int = np.asarray(list(by_subject.keys()), dtype=np.int64)

    aucs = np.full(n_resamples, np.nan)
    auprs = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        sampled = rng.choice(subjects_int, size=n_subj, replace=True)
        idx = np.concatenate([by_subject[int(s)] for s in sampled])
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            continue
        pi = probs[idx]
        aucs[i] = roc_auc_score(yi, pi)
        auprs[i] = average_precision_score(yi, pi)

    valid_auc = aucs[np.isfinite(aucs)]
    valid_aupr = auprs[np.isfinite(auprs)]
    return aucs, auprs, {
        "auc_ci_low": float(np.percentile(valid_auc, 2.5)),
        "auc_ci_high": float(np.percentile(valid_auc, 97.5)),
        "aupr_ci_low": float(np.percentile(valid_aupr, 2.5)),
        "aupr_ci_high": float(np.percentile(valid_aupr, 97.5)),
        "n_bootstrap_valid": int(len(valid_auc)),
    }


def _resolve_out_dir(match_v6_schema: bool, include_future_context: bool) -> Path:
    suffix_schema = "_strict" if match_v6_schema else ""
    if include_future_context:
        return CODE_ROOT / "models" / f"aim2_v6_future_inclusive{suffix_schema}"
    return CODE_ROOT / "models" / f"aim2_v6_past_only{suffix_schema}"


@click.command()
@click.option("--features-version", default="2026-05-01-phase1batch-v1", show_default=True,
              help="Local cohort is on phase1batch-v1 (superset of audit-v6). Phase 1 cols "
                   "are always dropped at fit time to recover the audit-v6 feature set.")
@click.option("--seed", type=int, default=42, show_default=True,
              help="MUST be 42 to keep test split identical to v6 / v7 / v8.5-tax for paired tests")
@click.option("--match-v6-schema", is_flag=True,
              help="Also drop Sprint 1 cols (freq-HRV, hypoxic_burden, position) to recover "
                   "the exact 29-base × 7-suffix = 203-col aim2_cv_v6 schema.")
@click.option("--include-future-context", is_flag=True,
              help="KEEP _lead1 and centred _roll* columns; SKIP past-only rolling. "
                   "Use to fit the matched future-inclusive baseline on the same "
                   "5,793-subject + seed=42 cohort, enabling valid paired bootstrap.")
def main(features_version: str, seed: int, match_v6_schema: bool, include_future_context: bool) -> None:
    out_dir = _resolve_out_dir(match_v6_schema, include_future_context)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_tag = "strict v6 schema" if match_v6_schema else "audit-v6 schema"
    ctx_tag = "future-inclusive" if include_future_context else "past-only"
    print(f"\n=== Aim 2 — v6 {ctx_tag} context ({schema_tag}) ===")
    print(f"  Features dir: {FEATURES_DIR}")
    print(f"  Output dir:   {out_dir}")
    print(f"  Features version: {features_version}")
    print(f"  Seed: {seed}")
    print(f"  Match v6 schema: {match_v6_schema}")
    print(f"  Include future context: {include_future_context}")
    print()

    df = load_cohort(FEATURES_DIR, version=features_version,
                     match_v6_schema=match_v6_schema,
                     include_future_context=include_future_context)
    feature_cols_all = [c for c in df.columns if c not in NON_FEATURE]
    base_cols = find_base_features(list(df.columns), NON_FEATURE)
    n_lag1 = sum(1 for c in feature_cols_all if c.endswith("_lag1"))
    n_lead1 = sum(1 for c in feature_cols_all if c.endswith("_lead1"))
    n_roll = sum(1 for c in feature_cols_all if any(c.endswith(s) for s in ("_roll5_mean", "_roll5_std", "_roll11_mean", "_roll11_std")))
    print(f"  Loaded schema: {len(base_cols)} base + {n_lag1} _lag1 + {n_lead1} _lead1 + {n_roll} _roll* = {len(feature_cols_all)} cols")
    assert len(base_cols) > 0, "No base features identified — check schema"
    assert n_lag1 == len(base_cols), f"_lag1 count {n_lag1} != base count {len(base_cols)}"
    if include_future_context:
        assert n_lead1 == len(base_cols), f"_lead1 count {n_lead1} != base count {len(base_cols)}"
        assert n_roll == 4 * len(base_cols), f"_roll count {n_roll} != 4 × base count {4 * len(base_cols)}"

    # Past-only mode: recompute right-aligned rolling on FULL PSG sequence
    # (BEFORE sleep filter — matches v6's extraction-time methodology where
    # rolling spans wake/sleep transitions naturally).
    if not include_future_context:
        df = add_past_only_rolling(df, base_cols, windows=PAST_ROLL_WINDOWS)

    # Sleep filter applied AFTER rolling so windows respect real-time epoch order
    df = apply_sleep_filter(df)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE]
    print(f"  Final feature count ({ctx_tag}): {len(feature_cols)}\n")

    # Outer split — IDENTICAL to all v6 / v7 / v8.5-tax / Phase 1 work
    X_dummy = np.empty(len(df))
    y = df["apnoea_label"].values
    groups_arr = df["subject_id"].values
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, test_idx = next(outer.split(X_dummy, y, groups_arr))
    print(f"  Outer split (seed={seed}):")
    print(f"    TV pool: {len(tv_idx):,} epochs / {len(np.unique(groups_arr[tv_idx]))} subjects")
    print(f"    Test:    {len(test_idx):,} epochs / {len(np.unique(groups_arr[test_idx]))} subjects")

    # Inner train/val carve from TV pool
    X = df[feature_cols].values
    inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    tr_inner_rel, va_inner_rel = next(inner.split(X[tv_idx], y[tv_idx], groups_arr[tv_idx]))
    tr = tv_idx[tr_inner_rel]
    va = tv_idx[va_inner_rel]

    pos = float(np.sum(y[tr])); neg = float(len(tr) - pos)
    spw = neg / max(pos, 1.0)
    print(f"  inner train: {len(tr):,} epochs / {len(np.unique(groups_arr[tr]))} subj  ({y[tr].mean()*100:.1f}% positive)")
    print(f"  inner val:   {len(va):,} epochs / {len(np.unique(groups_arr[va]))} subj")
    print(f"  scale_pos_weight: {spw:.3f}\n")

    # Fit
    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=FIXED_N_ESTIMATORS,
        scale_pos_weight=spw,
        **FIXED_PARAMS,
    )
    model.fit(
        X[tr], y[tr],
        eval_set=[(X[va], y[va])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(FIXED_EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
    )
    fit_time = time.time() - t0
    best_iter = int(model.best_iteration_ or FIXED_N_ESTIMATORS)
    print(f"  fit in {fit_time:.0f}s  (best_iter={best_iter})")

    # Test eval
    probs = model.predict_proba(X[test_idx])[:, 1]
    test_auc = float(roc_auc_score(y[test_idx], probs))
    test_aupr = float(average_precision_score(y[test_idx], probs))

    # Tuned-threshold F1 from inner val
    va_probs = model.predict_proba(X[va])[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    va_f1s = [f1_score(y[va], (va_probs > t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(va_f1s))])
    preds = (probs > best_thresh).astype(int)
    test_f1 = float(f1_score(y[test_idx], preds, zero_division=0))
    test_p = float(precision_score(y[test_idx], preds, zero_division=0))
    test_r = float(recall_score(y[test_idx], preds, zero_division=0))

    print(f"\n  Test AUC: {test_auc:.4f}   AUC-PR: {test_aupr:.4f}")
    print(f"  F1@{best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    # Subject-level bootstrap
    df_test = df.iloc[test_idx][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True)
    print(f"  computing subject-level bootstrap CI (1000 resamples)...")
    t0 = time.time()
    aucs, auprs, ci = subject_bootstrap(df_test, probs, n_resamples=1000, seed=42)
    print(f"  bootstrap in {time.time()-t0:.0f}s")
    print(f"  AUC subject CI:    [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
    print(f"  AUC-PR subject CI: [{ci['aupr_ci_low']:.4f}, {ci['aupr_ci_high']:.4f}]")

    # Pre-registered outcome interpretation (only meaningful for past-only runs)
    if not include_future_context:
        print(f"\n  Pre-registered outcome interpretation:")
        if test_auc >= 0.840:
            print(f"    SURVIVAL — past-only AUC {test_auc:.4f} ≥ 0.840: v6 lift is real-time deployable")
        elif test_auc >= 0.800:
            print(f"    MIXED — past-only AUC {test_auc:.4f} in [0.800, 0.840): both past + future contribute")
        else:
            print(f"    COLLAPSE — past-only AUC {test_auc:.4f} < 0.800: v6 lift was substantially future-leakage")
    else:
        print(f"\n  (future-inclusive run — pre-reg verdict applies to past-only only)")

    # Save artifacts
    test_pred_df = pd.DataFrame({
        "subject_id": df.iloc[test_idx]["subject_id"].values,
        "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
        "apnoea_label": y[test_idx],
        "pred_prob": probs,
        "pred_label": preds,
    })
    test_pred_df.to_parquet(out_dir / "test_predictions.parquet", index=False)
    np.save(out_dir / "bootstrap_aucs_subject.npy", aucs)
    np.save(out_dir / "bootstrap_auprs_subject.npy", auprs)
    (out_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2))

    try:
        sm_path = FEATURES_DIR / "subject_metadata.parquet"
        sm = pd.read_parquet(sm_path) if sm_path.exists() else None
        write_extended_metrics(out_dir, test_pred_df, subject_metadata=sm)
    except Exception as _e:
        print(f"  ! extended metrics failed: {_e}")

    metrics = {
        "name": out_dir.name,
        "model": "lightgbm",
        "filter_used": "sleep-only",
        "features_version": features_version,
        "match_v6_schema": match_v6_schema,
        "include_future_context": include_future_context,
        "n_features": len(feature_cols),
        "n_test_subjects": int(len(np.unique(groups_arr[test_idx]))),
        "n_test_epochs": int(len(test_idx)),
        "best_iter": best_iter,
        "fixed_params": FIXED_PARAMS,
        "scale_pos_weight": spw,
        "best_threshold": best_thresh,
        "test_auc_roc": test_auc,
        "test_auc_pr": test_aupr,
        "test_auc_ci_low": ci["auc_ci_low"],
        "test_auc_ci_high": ci["auc_ci_high"],
        "test_aupr_ci_low": ci["aupr_ci_low"],
        "test_aupr_ci_high": ci["aupr_ci_high"],
        "test_f1_tuned": test_f1,
        "test_precision_tuned": test_p,
        "test_recall_tuned": test_r,
        "fit_seconds": fit_time,
        "preregistration_note": "Vault: Experiments/2026-04-30 - v6 past-only context (pre-registration).md",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n  saved → {out_dir}")


if __name__ == "__main__":
    main()
