"""Aim 2 Phase 1 — Experiment 5: sleep-stage-as-feature (vs sleep-only filter).

Hypothesis
----------
Including sleep_stage as an *input feature* (with no wake-epoch filter) gives
the model strictly more information than dropping wake epochs entirely,
because:

  (1) The model can learn stage-specific apnoea patterns (e.g. REM-related
      events, N1 transitions) rather than treating all sleep epochs uniformly.
  (2) The wake epochs themselves contain useful baseline information (resting
      HRV, baseline SpO2) the filtered approach throws away.
  (3) For the physiological-only feature subset, this should narrow the gap
      to the rule-recapitulation regime by leveraging stage information.

Expected outcome
----------------
- physio_only + stage_feat (all-stages):  0.70-0.78  (was 0.6737 sleep-only)
- aasm_only + stage_feat (all-stages):    ~0.95     (rule features already dominate)
- full + stage_feat (all-stages):         ~0.96     (small marginal gain)

If physio_only goes 0.67 → 0.75+, the hypothesis is confirmed: stage info is
genuinely useful, and filtering wake throws away signal. If it stays ~0.67,
the wake-vs-sleep boost is purely about removing trivial negatives, not
about using stage info.

Protocol
--------
Same as 8.5-tax, with two changes:
  - NO wake_mask filter (all-stages)
  - sleep_stage included as a categorical feature (LightGBM native)

Three configs run in sequence (all on identical seed=42 outer split):
  full + stage:        273 + 1 features
  physio_only + stage: 217 + 1 features
  aasm_only + stage:   56 + 1 features

For each, output: metrics.json, test_predictions.parquet,
bootstrap_aucs_subject.npy, bootstrap_auprs_subject.npy.

Usage (Colab high-RAM):
  %cd /content/Code
  !python scripts/fit_aim2_phase1_exp5_sleep_stage.py
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

from thesis_pipeline.feature_groups import feature_subset, split_features  # noqa: E402

FEATURES_DIR = CODE_ROOT / "features"
OUT_ROOT = CODE_ROOT / "models" / "aim2_phase1_exp5_sleep_stage"
NON_FEATURE = {
    "subject_id", "cohort", "epoch_idx", "epoch_start_sec",
    "apnoea_label", "features_version",
    # NOTE: sleep_stage is INTENTIONALLY OMITTED from this set so it is
    # included as a feature rather than treated as metadata.
}

# Same fixed HP as 8.5-tax for direct comparability
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


def load_cohort_all_stages(features_dir: Path, version: str = "2026-04-26-audit-v6") -> pd.DataFrame:
    """Load all per-subject parquets, filter to features_version. NO sleep filter.

    AppleDouble shadows are explicitly excluded from the glob.
    """
    files = [
        f for f in sorted(features_dir.glob("*.parquet"))
        if f.name != "subject_metadata.parquet" and not f.name.startswith("._")
    ]
    if not files:
        raise FileNotFoundError(f"No per-subject parquet files in {features_dir}")
    print(f"  loading {len(files)} parquet files...")

    KEEP_META = {"subject_id", "epoch_idx", "apnoea_label", "features_version", "sleep_stage"}
    DROP_META = {"cohort", "epoch_start_sec"}

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df = df.drop(columns=[c for c in DROP_META if c in df.columns], errors="ignore")
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
    print(f"  features_version filter: {version!r} → {n_after}/{n_before} subjects")
    print(f"  ALL-STAGES retained: {len(df):,} epochs (no wake_mask filter applied)")
    return df


def encode_sleep_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure sleep_stage is integer-coded for LightGBM categorical handling.

    Logs the value distribution so we can see the encoding.
    """
    print("\n  Sleep_stage diagnostics:")
    print(f"    dtype: {df.sleep_stage.dtype}")
    vc = df.sleep_stage.value_counts(dropna=False).sort_index()
    print(f"    value counts:\n{vc.to_string()}")

    # Coerce to integer (handles strings → ints, NaN → -1)
    if df.sleep_stage.dtype.kind not in ("i", "u"):
        # If string, try to parse; otherwise treat NaN as -1 wake-equivalent
        try:
            df["sleep_stage"] = df["sleep_stage"].fillna(-1).astype(np.int8)
        except (ValueError, TypeError):
            df["sleep_stage"] = pd.factorize(df["sleep_stage"])[0].astype(np.int8)
            print(f"    (factorized non-numeric stages)")
    else:
        df["sleep_stage"] = df["sleep_stage"].fillna(-1).astype(np.int8)

    print(f"    encoded dtype: {df.sleep_stage.dtype}, unique values: {sorted(df.sleep_stage.unique().tolist())}")
    return df


def subject_bootstrap(df_test, probs, n_resamples=1000, seed=42):
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


def fit_one_config(name: str, df: pd.DataFrame, feature_cols: list[str],
                   tv_idx: np.ndarray, test_idx: np.ndarray,
                   out_dir: Path, categorical_features: list[str]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*78}\n=== Config: {name}  ({len(feature_cols)} features incl. sleep_stage)\n{'='*78}")

    if len(feature_cols) == 0:
        return {"name": name, "n_features": 0, "skipped": True}

    # Identify positional indices of categorical features
    cat_indices = [feature_cols.index(c) for c in categorical_features if c in feature_cols]

    X = df[feature_cols].values
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    inner = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    tr_inner_rel, va_inner_rel = next(inner.split(X[tv_idx], y[tv_idx], groups[tv_idx]))
    tr = tv_idx[tr_inner_rel]
    va = tv_idx[va_inner_rel]

    pos = float(np.sum(y[tr])); neg = float(len(tr) - pos)
    spw = neg / max(pos, 1.0)
    print(f"  inner train: {len(tr):,} ({y[tr].mean()*100:.1f}% pos)  val: {len(va):,}  test: {len(test_idx):,}")
    print(f"  scale_pos_weight: {spw:.3f}")
    print(f"  categorical feature indices: {cat_indices} ({categorical_features})")

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
        categorical_feature=cat_indices if cat_indices else "auto",
        callbacks=[lgb.early_stopping(FIXED_EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)],
    )
    fit_time = time.time() - t0
    best_iter = int(model.best_iteration_ or FIXED_N_ESTIMATORS)
    print(f"  fit in {fit_time:.0f}s  (best_iter={best_iter})")

    probs = model.predict_proba(X[test_idx])[:, 1]
    test_auc = float(roc_auc_score(y[test_idx], probs))
    test_aupr = float(average_precision_score(y[test_idx], probs))

    va_probs = model.predict_proba(X[va])[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    va_f1s = [f1_score(y[va], (va_probs > t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(va_f1s))])
    preds = (probs > best_thresh).astype(int)
    test_f1 = float(f1_score(y[test_idx], preds, zero_division=0))
    test_p = float(precision_score(y[test_idx], preds, zero_division=0))
    test_r = float(recall_score(y[test_idx], preds, zero_division=0))

    print(f"  Test AUC: {test_auc:.4f}   AUC-PR: {test_aupr:.4f}")
    print(f"  F1@{best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    df_test = df.iloc[test_idx][["subject_id", "epoch_idx", "apnoea_label"]].reset_index(drop=True)
    print(f"  computing subject-level bootstrap CI (1000 resamples)...")
    t0 = time.time()
    aucs, auprs, ci = subject_bootstrap(df_test, probs, n_resamples=1000, seed=42)
    print(f"  bootstrap in {time.time()-t0:.0f}s")
    print(f"  AUC subject CI:    [{ci['auc_ci_low']:.4f}, {ci['auc_ci_high']:.4f}]")
    print(f"  AUC-PR subject CI: [{ci['aupr_ci_low']:.4f}, {ci['aupr_ci_high']:.4f}]")

    pd.DataFrame({
        "subject_id": df.iloc[test_idx]["subject_id"].values,
        "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
        "apnoea_label": y[test_idx],
        "pred_prob": probs,
        "pred_label": preds,
    }).to_parquet(out_dir / "test_predictions.parquet", index=False)
    np.save(out_dir / "bootstrap_aucs_subject.npy", aucs)
    np.save(out_dir / "bootstrap_auprs_subject.npy", auprs)
    (out_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2))

    metrics = {
        "name": name,
        "experiment": "phase1_exp5_sleep_stage_feature",
        "n_features": len(feature_cols),
        "categorical_features": categorical_features,
        "all_stages": True,
        "sleep_stage_as_feature": True,
        "n_test_subjects": int(len(np.unique(groups[test_idx]))),
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
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  saved → {out_dir}")
    return metrics


@click.command()
@click.option("--features-version", default="2026-04-26-audit-v6", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def main(features_version: str, seed: int) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Aim 2 Phase 1 — Exp 5: sleep-stage-as-feature ===\n")

    df = load_cohort_all_stages(FEATURES_DIR, version=features_version)
    df = encode_sleep_stage(df)

    # All non-meta columns are candidate features. sleep_stage is now included.
    feature_cols_all = [c for c in df.columns if c not in NON_FEATURE]
    # Exclude sleep_stage from the taxonomy classification (it's neither AASM nor physio)
    feature_cols_for_split = [c for c in feature_cols_all if c != "sleep_stage"]
    groups = split_features(feature_cols_for_split)

    print(f"\n  Total feature candidates (incl sleep_stage): {len(feature_cols_all)}")
    print(f"  Feature taxonomy (sleep_stage NOT in either group):")
    print(f"    AASM-rule: {len(groups['aasm'])}")
    print(f"    Physiological: {len(groups['physio'])}")
    print(f"    Unknown→physio: {len(groups['unknown'])}")

    # Outer split — IDENTICAL seed=42 to all v6/v8/v8.5/8.5-tax/Lit-2019
    X_dummy = np.empty(len(df))
    y = df["apnoea_label"].values
    groups_arr = df["subject_id"].values
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, test_idx = next(outer.split(X_dummy, y, groups_arr))
    print(f"\n  Outer split (seed={seed}, ALL-STAGES):")
    print(f"    TV pool: {len(tv_idx):,} epochs / {len(np.unique(groups_arr[tv_idx]))} subj")
    print(f"    Test:    {len(test_idx):,} epochs / {len(np.unique(groups_arr[test_idx]))} subj")

    # Build the three configs: full / physio_only / aasm_only — each + sleep_stage feature
    configs = {
        "full_plus_stage":         feature_subset(feature_cols_for_split, "full") + ["sleep_stage"],
        "physio_only_plus_stage":  feature_subset(feature_cols_for_split, "physio_only") + ["sleep_stage"],
        "aasm_only_plus_stage":    feature_subset(feature_cols_for_split, "aasm_only") + ["sleep_stage"],
    }

    summary = {"experiment": "phase1_exp5_sleep_stage_feature", "configs": {}}
    for name, fcols in configs.items():
        out_dir = OUT_ROOT / name
        result = fit_one_config(name, df, fcols, tv_idx, test_idx, out_dir,
                                categorical_features=["sleep_stage"])
        summary["configs"][name] = result

    print(f"\n\n{'='*78}\n=== EXP 5 COMPACT COMPARISON (vs 8.5-tax sleep-only baselines)\n{'='*78}\n")
    print(f"{'Config':30s}  {'Features':>9s}  {'Test AUC':>9s}  {'Subject 95% CI':>20s}  {'vs 8.5-tax':>10s}")
    BASELINES = {
        "full_plus_stage":        ("8.5-tax full",        0.9591),
        "physio_only_plus_stage": ("8.5-tax physio_only", 0.6737),
        "aasm_only_plus_stage":   ("8.5-tax aasm_only",   0.9488),
    }
    for name, r in summary["configs"].items():
        if r.get("skipped"):
            continue
        ref_name, ref_auc = BASELINES.get(name, (None, None))
        delta = r["test_auc_roc"] - ref_auc if ref_auc else 0.0
        ci = f"[{r['test_auc_ci_low']:.4f}, {r['test_auc_ci_high']:.4f}]"
        print(f"{name:30s}  {r['n_features']:>9d}  {r['test_auc_roc']:>9.4f}  {ci:>20s}  {delta:+10.4f}  (vs {ref_name})")

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {OUT_ROOT / 'summary.json'}")
    print(f"\n=== HYPOTHESIS CHECK ===")
    print(f"Hypothesis: physio_only + stage_feat (all-stages) > 0.6737 (8.5-tax physio_only sleep-only)")
    physio_result = summary["configs"].get("physio_only_plus_stage", {}).get("test_auc_roc", 0.0)
    print(f"Result: physio_only + stage_feat = {physio_result:.4f}")
    delta_physio = physio_result - 0.6737
    if delta_physio > 0.01:
        print(f"  → CONFIRMED. Stage-as-feature adds {delta_physio:+.4f} AUC over wake-filter (physio-only track).")
    elif delta_physio < -0.01:
        print(f"  → DISCONFIRMED. Stage-as-feature loses {delta_physio:+.4f} vs wake-filter; sleep-only filter is better.")
    else:
        print(f"  → INCONCLUSIVE. Stage-as-feature ≈ wake-filter (Δ = {delta_physio:+.4f}). Both extract similar info.")


if __name__ == "__main__":
    main()
