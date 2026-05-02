"""Aim 2 baseline v4 — LightGBM with GroupKFold CV + Optuna HPO + bootstrap CIs.

Experiment log: Vault/Experiments/2026-04-24 — LightGBM GroupKFold CV + Optuna HPO.md

Protocol:
    1. Hold out 20% of subjects (patient-level) as the final test set.
    2. 5-fold GroupKFold CV on the remaining 80% — Optuna objective = mean val AUC.
    3. Refit best params on full train+val pool, evaluate on held-out test.
    4. 1000-sample bootstrap 95% CI on test AUC.

Output directory: Code/models/aim2_cv_v4/
    best_params.json
    per_fold_metrics.json
    optuna_study.db             (sqlite study for post-hoc plotting)
    bootstrap_aucs.npy
    metrics.json                (headline metrics)
    test_predictions.parquet

Usage:
    python scripts/fit_aim2_cv.py --trials 30 --timeout 1800
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import click
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thesis_pipeline.epochs import sleep_mask

NON_FEATURE_COLS = {
    "subject_id",
    "cohort",
    "epoch_idx",
    "epoch_start_sec",
    "sleep_stage",
    "apnoea_label",
    "features_version",
}

CODE_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = CODE_ROOT / "features"
OUT_DIR = CODE_ROOT / "models" / "aim2_cv_v4"


def load_cohort(features_dir: Path) -> pd.DataFrame:
    """Memory-efficient cohort load: float32 features, drop unused columns.

    Per-subject reads everything (we need features_version for filtering),
    but downcasts feature columns to float32 and drops string metadata
    that isn't needed downstream. ~2× memory reduction vs pd.concat default.
    """
    # Restrict to per-subject epoch parquets; exclude `subject_metadata.parquet`
    # (which has one row per subject and string subject_id, would crash int32 cast).
    files = [f for f in sorted(features_dir.glob("*.parquet"))
             if f.name != "subject_metadata.parquet"]
    if not files:
        raise FileNotFoundError(f"No per-subject parquet files in {features_dir}")

    KEEP_META = {"subject_id", "epoch_idx", "apnoea_label", "features_version", "sleep_stage"}
    DROP_META = {"cohort", "epoch_start_sec"}

    frames: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        # Drop columns we don't need
        df = df.drop(columns=[c for c in DROP_META if c in df.columns], errors="ignore")
        # Downcast feature floats to float32
        for c in df.columns:
            if c in KEEP_META:
                continue
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        # subject_id can be int32 (range 200000-205804)
        if "subject_id" in df.columns and df["subject_id"].dtype != np.int32:
            df["subject_id"] = df["subject_id"].astype(np.int32)
        if "apnoea_label" in df.columns and df["apnoea_label"].dtype != np.int8:
            df["apnoea_label"] = df["apnoea_label"].astype(np.int8)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, copy=False)


def _fold_scale_pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y))
    neg = float(len(y) - pos)
    return neg / max(pos, 1.0)


def _fit_and_score(params: dict, X_tr, y_tr, X_va, y_va) -> tuple[float, int, np.ndarray]:
    """Fit LightGBM with early stopping on val; return (val AUC, best iteration, val probs).

    Val probs returned so the caller can collect out-of-fold (OOF) predictions
    across CV folds — used by F10 (threshold selection on unbiased OOF preds
    instead of the training data the model has already seen).
    """
    full_params = dict(params)
    full_params["scale_pos_weight"] = _fold_scale_pos_weight(y_tr)
    full_params["verbose"] = -1
    full_params["n_jobs"] = -1
    full_params["random_state"] = 42

    model = lgb.LGBMClassifier(**full_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    probs = model.predict_proba(X_va)[:, 1]
    return (
        float(roc_auc_score(y_va, probs)),
        int(model.best_iteration_ or full_params.get("n_estimators", 1000)),
        probs,
    )


def make_objective(X, y, groups, tv_idx: np.ndarray, k: int = 5):
    """Returns an Optuna objective: mean val AUC across k GroupKFolds."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Search space tightened 2026-04-27 for v8.5 audit-v6 + sleep-only run.
            # Original lr floor 0.01 + num_leaves up to 127 caused single trials to
            # take 40+ min on 4.2M rows × 280 cols. Bumping lr floor to 0.03 and
            # capping num_leaves at 64 keeps each trial ~2-3 min with no expected
            # AUC loss (v6 best landed at lr=0.113, num_leaves=49, well within range).
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 64),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "n_estimators": 2000,
        }
        gkf = GroupKFold(n_splits=k)
        fold_aucs = []
        for tr_rel, va_rel in gkf.split(X[tv_idx], y[tv_idx], groups[tv_idx]):
            tr = tv_idx[tr_rel]
            va = tv_idx[va_rel]
            auc, _, _ = _fit_and_score(params, X[tr], y[tr], X[va], y[va])
            fold_aucs.append(auc)
            # Optuna pruning: report intermediate and allow early stopping of trials
            trial.report(float(np.mean(fold_aucs)), step=len(fold_aucs))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_aucs))

    return objective


def bootstrap_auc_ci(
    y_true: np.ndarray, probs: np.ndarray, n: int = 1000, seed: int = 42
) -> tuple[np.ndarray, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    aucs = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        # Skip degenerate resamples (single-class)
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], probs[idx])
    valid = aucs[np.isfinite(aucs)]
    ci = float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))
    return aucs, ci


@click.command()
@click.option("--trials", type=int, default=30, show_default=True)
@click.option("--timeout", type=int, default=1800, show_default=True, help="Optuna wall-time cap (seconds)")
@click.option("--k", type=int, default=5, show_default=True, help="GroupKFold splits")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--features-version", default=None, help="Restrict cohort to a single features_version tag")
@click.option("--out-name", default=None, help="Subdirectory name under models/ (default aim2_cv_v4)")
@click.option("--sleep-only/--all-stages", default=True,
              help="Filter to sleep epochs only (N1/N2/N3/REM). Default true.")
def main(trials: int, timeout: int, k: int, seed: int, features_version: str | None, out_name: str | None, sleep_only: bool) -> None:
    out_dir = OUT_DIR if out_name is None else CODE_ROOT / "models" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    df = load_cohort(FEATURES_DIR)
    if features_version and "features_version" in df.columns:
        n_before = df["subject_id"].nunique()
        df = df[df["features_version"] == features_version].reset_index(drop=True)
        n_after = df["subject_id"].nunique()
        print(f"  features_version filter: {features_version!r} → {n_after} of {n_before} subjects")
    if sleep_only:
        n_before = len(df)
        mask = sleep_mask(df.sleep_stage.values)
        df = df.loc[mask].reset_index(drop=True)
        assert df["sleep_stage"].isin(["N1", "N2", "N3", "REM"]).all(), (
            f"sleep filter failed: found stages {sorted(df['sleep_stage'].unique())} after filter"
        )
        click.echo(f"Sleep-only filter: kept {len(df)}/{n_before} epochs ({100*len(df)/n_before:.1f}%)")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"  {len(df):,} epochs, {df['subject_id'].nunique()} subjects, {len(feature_cols)} features")

    X = df[feature_cols].values
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    # Outer 80/20 patient-level split
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, test_idx = next(outer.split(X, y, groups))

    print(
        f"  Train+val pool: {len(tv_idx):,} epochs / {len(np.unique(groups[tv_idx]))} subjects  "
        f"({y[tv_idx].mean()*100:.1f}% positive)"
    )
    print(
        f"  Held-out test: {len(test_idx):,} epochs / {len(np.unique(groups[test_idx]))} subjects  "
        f"({y[test_idx].mean()*100:.1f}% positive)"
    )

    # Optuna study (TPE) with sqlite backend so you can browse later
    storage_url = f"sqlite:///{out_dir / 'optuna_study.db'}"
    sampler = TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        study_name=f"aim2_cv_{seed}",
        load_if_exists=True,
    )

    print(f"\n▶ Running Optuna: up to {trials} trials, {timeout}s timeout, {k}-fold GroupKFold")
    objective = make_objective(X, y, groups, tv_idx, k=k)
    study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params
    best_mean_auc = study.best_value
    print(f"\n  best mean-fold val AUC: {best_mean_auc:.4f}")
    print(f"  best params: {json.dumps(best, indent=2)}")

    # Per-fold breakdown at the best hyperparameters (for reporting). Also
    # collect out-of-fold (OOF) predictions on the TV pool so the threshold
    # can be tuned on validation data the model didn't train on (F10 fix —
    # was tuned on tv predictions from a model fit on the entire TV pool,
    # introducing optimistic bias on F1 / precision / recall / best_threshold).
    print("\n▶ Recomputing per-fold AUCs at best params + collecting OOF preds...")
    final_params = dict(best)
    final_params["n_estimators"] = 2000
    gkf = GroupKFold(n_splits=k)
    per_fold = []
    n_tv = len(tv_idx)
    oof_probs = np.full(n_tv, np.nan, dtype=float)
    for i, (tr_rel, va_rel) in enumerate(gkf.split(X[tv_idx], y[tv_idx], groups[tv_idx])):
        tr = tv_idx[tr_rel]
        va = tv_idx[va_rel]
        auc, best_iter, va_probs = _fit_and_score(final_params, X[tr], y[tr], X[va], y[va])
        oof_probs[va_rel] = va_probs
        per_fold.append({"fold": i + 1, "auc": auc, "best_iter": best_iter,
                          "n_train_subjects": int(len(np.unique(groups[tr]))),
                          "n_val_subjects": int(len(np.unique(groups[va])))})
        print(f"   fold {i+1}: AUC {auc:.4f}  (iter {best_iter})")
    mean_auc = float(np.mean([f["auc"] for f in per_fold]))
    std_auc = float(np.std([f["auc"] for f in per_fold], ddof=1))
    mean_best_iter = int(np.round(np.mean([f["best_iter"] for f in per_fold])))
    print(f"\n  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}  (5-fold, patient-level)")

    # Final refit on full TV pool, then evaluate on held-out test
    # Use mean-best-iter as a sensible fixed n_estimators for the final model
    final_params_refit = dict(best)
    final_params_refit["n_estimators"] = max(100, int(1.1 * mean_best_iter))  # small safety margin
    final_params_refit["scale_pos_weight"] = _fold_scale_pos_weight(y[tv_idx])
    final_params_refit["verbose"] = -1
    final_params_refit["n_jobs"] = -1
    final_params_refit["random_state"] = seed

    print(f"\n▶ Refitting on full train+val pool (n_estimators={final_params_refit['n_estimators']})...")
    model = lgb.LGBMClassifier(**final_params_refit)
    model.fit(X[tv_idx], y[tv_idx])
    probs = model.predict_proba(X[test_idx])[:, 1]
    test_auc = float(roc_auc_score(y[test_idx], probs))
    test_ap = float(average_precision_score(y[test_idx], probs))
    # F10 fix: pick threshold on OOF predictions (unbiased — each prediction
    # came from a fold where that subject was held out). The previous code
    # picked threshold on `model.predict_proba(X[tv_idx])` which is the
    # training data the final model was just fit on — overfit threshold,
    # optimistically biased F1 / precision / recall.
    print("\n▶ Tuning threshold on OOF CV predictions (F10 fix)...")
    thresholds = np.linspace(0.05, 0.95, 91)
    oof_f1s = [
        f1_score(y[tv_idx], (oof_probs > t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_thresh = float(thresholds[int(np.argmax(oof_f1s))])
    preds = (probs > best_thresh).astype(int)
    test_f1 = float(f1_score(y[test_idx], preds, zero_division=0))
    test_p = float(precision_score(y[test_idx], preds, zero_division=0))
    test_r = float(recall_score(y[test_idx], preds, zero_division=0))

    print("\n▶ Bootstrap 95% CI on test AUC (1000 resamples)...")
    bootstrap_aucs, (ci_lo, ci_hi) = bootstrap_auc_ci(y[test_idx], probs, n=1000, seed=seed)

    print(f"\n=== Held-out test set ===")
    print(f"  AUC-ROC : {test_auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  AUC-PR  : {test_ap:.4f}")
    print(f"  F1 @ {best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    # Persist
    (out_dir /"best_params.json").write_text(json.dumps(best, indent=2))
    (out_dir /"per_fold_metrics.json").write_text(json.dumps(per_fold, indent=2))
    np.save(out_dir /"bootstrap_aucs.npy", bootstrap_aucs)
    pd.DataFrame({
        "subject_id": df.iloc[test_idx]["subject_id"].values,
        "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
        "apnoea_label": y[test_idx],
        "pred_prob": probs,
        "pred_label": preds,
    }).to_parquet(out_dir /"test_predictions.parquet", index=False)
    model.booster_.save_model(str(out_dir /"model.txt"))

    metrics = {
        "filter_used": "sleep-only" if sleep_only else "all-stages",
        "cv_mean_auc": mean_auc,
        "cv_std_auc": std_auc,
        "cv_best_value_optuna": best_mean_auc,
        "test_auc_roc": test_auc,
        "test_auc_pr": test_ap,
        "test_auc_ci_low": ci_lo,
        "test_auc_ci_high": ci_hi,
        "test_f1_tuned": test_f1,
        "test_precision_tuned": test_p,
        "test_recall_tuned": test_r,
        "best_threshold": best_thresh,
        "n_trials_completed": len(study.trials),
        "n_trials_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        "feature_cols": feature_cols,
        "best_params": best,
        "mean_best_iter": mean_best_iter,
        "n_cohort_subjects": int(df["subject_id"].nunique()),
        "n_test_subjects": int(len(np.unique(groups[test_idx]))),
    }
    (out_dir /"metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
