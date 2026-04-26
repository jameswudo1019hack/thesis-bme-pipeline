"""Aim 2 baseline v8.5 — RF + LogReg on GPU via NVIDIA RAPIDS cuML.

Experiment log: Vault/Experiments/2026-04-26 - v8 model ablation (Plan A).md

Purpose: complete the Plan A model survey by adding the two sklearn-style
baselines (Random Forest, Logistic Regression) that crashed the local 16 GB
laptop earlier today. cuML's GPU implementations make these tractable.

Same protocol as fit_aim2_v8_ablation.py and fit_aim2_cv.py:
    1. Hold out 20% of subjects (patient-level) — IDENTICAL split to v6 (seed=42).
    2. 5-fold GroupKFold CV on the remaining 80% — Optuna objective = mean val AUC.
    3. Refit best params on full train+val pool, evaluate on held-out test.
    4. 1000-sample bootstrap 95% CI on test AUC.

Models supported:
    rf       — cuml.ensemble.RandomForestClassifier
    logreg   — cuml.linear_model.LogisticRegression (median-imputed + scaled)

Output directory: Code/models/aim2_v8_<model>/  (matches the GPU GBM run layout)

Usage (on Colab with RAPIDS / cuML installed):
    python scripts/fit_aim2_v8_cuml.py --model rf      --trials 15 --timeout 1800 \
        --features-dir /content/v8_work/features --models-dir /content/v8_work/models
    python scripts/fit_aim2_v8_cuml.py --model logreg  --trials 30 --timeout 1200 \
        --features-dir /content/v8_work/features --models-dir /content/v8_work/models
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import click
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


# ============================================================================
# Cohort load — identical to fit_aim2_v8_ablation.py
# ============================================================================


def load_cohort(features_dir: Path) -> pd.DataFrame:
    files = sorted(features_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {features_dir}")

    KEEP_META = {"subject_id", "epoch_idx", "apnoea_label", "features_version"}
    DROP_META = {"cohort", "epoch_start_sec", "sleep_stage"}

    frames: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        df = df.drop(columns=[c for c in DROP_META if c in df.columns], errors="ignore")
        for c in df.columns:
            if c in KEEP_META:
                continue
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        if "subject_id" in df.columns and df["subject_id"].dtype != np.int32:
            df["subject_id"] = df["subject_id"].astype(np.int32)
        if "apnoea_label" in df.columns and df["apnoea_label"].dtype != np.int8:
            df["apnoea_label"] = df["apnoea_label"].astype(np.int8)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, copy=False)


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """sklearn-style class_weight='balanced' as per-row sample weights.

    Used by LogReg (cuML LogisticRegression accepts sample_weight).
    """
    n = len(y)
    n_pos = float(np.sum(y))
    n_neg = float(n - n_pos)
    w_pos = n / (2.0 * max(n_pos, 1.0))
    w_neg = n / (2.0 * max(n_neg, 1.0))
    return np.where(y == 1, w_pos, w_neg).astype(np.float32)


def _balanced_undersample_idx(y: np.ndarray, ratio: float = 1.0, seed: int = 42) -> np.ndarray:
    """Return indices that balance positive:negative to the given ratio (default 1:1).

    Used by cuML RF — its fit() doesn't accept sample_weight or class_weight,
    so we manually undersample the majority class to handle imbalance.
    All positive examples are kept; majority is randomly downsampled.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg_target = int(n_pos / ratio)
    if n_neg_target >= len(neg_idx):
        return np.arange(len(y))
    neg_sampled = rng.choice(neg_idx, n_neg_target, replace=False)
    return np.sort(np.concatenate([pos_idx, neg_sampled]))


def bootstrap_auc_ci(
    y_true: np.ndarray, probs: np.ndarray, n: int = 1000, seed: int = 42
) -> tuple[np.ndarray, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    aucs = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], probs[idx])
    valid = aucs[np.isfinite(aucs)]
    ci = float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))
    return aucs, ci


# ============================================================================
# Per-model handlers (cuML)
# ============================================================================


class CuMLRFHandler:
    """cuML Random Forest — GPU-accelerated drop-in for sklearn's RandomForest."""
    name = "rf"

    def suggest(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 24),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

    def _impute_train(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Median impute, returning (X_imputed, medians) so val/test can use the same."""
        medians = np.nanmedian(X, axis=0)
        X_out = np.where(np.isnan(X), medians, X).astype(np.float32)
        return X_out, medians

    def _impute_apply(self, X: np.ndarray, medians: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(X), medians, X).astype(np.float32)

    def _build(self, params):
        from cuml.ensemble import RandomForestClassifier
        # cuML RF doesn't support max_features=fractions in older versions;
        # convert to int if needed
        max_f = params["max_features"]
        clean = {k: v for k, v in params.items() if k != "max_features"}
        return RandomForestClassifier(
            **clean,
            max_features=max_f,
            n_streams=1,            # deterministic
            random_state=42,
            n_bins=128,             # GPU histogram bin count
        )

    def fit_fold(self, params, X_tr, y_tr, X_va, y_va):
        # cuML RF doesn't accept sample_weight or class_weight in v24.10 —
        # balance via majority-class undersampling. Keeps all positives, randomly
        # downsamples negatives to 1:1 ratio. Standard imbalance-handling technique;
        # AUC is the headline metric so threshold-tuning happens later.
        sel = _balanced_undersample_idx(y_tr, ratio=1.0, seed=42)
        Xtr_bal = X_tr[sel]
        ytr_bal = y_tr[sel]
        Xtr, med = self._impute_train(Xtr_bal)
        Xva = self._impute_apply(X_va, med)
        model = self._build(params)
        model.fit(Xtr, ytr_bal.astype(np.int32))
        probs = np.asarray(model.predict_proba(Xva))[:, 1]
        return float(roc_auc_score(y_va, probs)), int(params["n_estimators"])

    def fit_full(self, params, X_tv, y_tv, n_est_unused):
        sel = _balanced_undersample_idx(y_tv, ratio=1.0, seed=42)
        Xtv_bal = X_tv[sel]
        ytv_bal = y_tv[sel]
        Xtv, med = self._impute_train(Xtv_bal)
        model = self._build(params)
        model.fit(Xtv, ytv_bal.astype(np.int32))
        model._train_medians = med
        return model

    def predict_proba(self, model, X):
        X_imp = self._impute_apply(X, model._train_medians)
        return np.asarray(model.predict_proba(X_imp))[:, 1]

    def save(self, model, path_no_ext: Path):
        # cuML RF pickling can be flaky across versions — save best_params
        # and let the user retrain if they need the model later.
        # (The test_predictions.parquet is the load-bearing artefact.)
        try:
            import joblib
            joblib.dump(model, path_no_ext.with_suffix(".joblib"))
        except Exception as e:
            print(f"  ! could not pickle cuML RF model: {e}; predictions saved separately")


class CuMLLogRegHandler:
    """cuML Logistic Regression — GPU-accelerated, supports L1/L2, class_weight via sample_weight."""
    name = "logreg"

    def suggest(self, trial: optuna.Trial) -> dict:
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
        }

    def _impute_and_scale_train(self, X: np.ndarray):
        medians = np.nanmedian(X, axis=0)
        X_imp = np.where(np.isnan(X), medians, X).astype(np.float32)
        means = X_imp.mean(axis=0)
        stds = X_imp.std(axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)  # avoid div-by-zero on constant cols
        X_scaled = (X_imp - means) / stds
        return X_scaled, medians, means, stds

    def _impute_and_scale_apply(self, X: np.ndarray, medians, means, stds) -> np.ndarray:
        X_imp = np.where(np.isnan(X), medians, X).astype(np.float32)
        return (X_imp - means) / stds

    def _build(self, params):
        from cuml.linear_model import LogisticRegression
        return LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            max_iter=500,
            tol=1e-4,
            fit_intercept=True,
        )

    def fit_fold(self, params, X_tr, y_tr, X_va, y_va):
        Xtr, med, mu, sd = self._impute_and_scale_train(X_tr)
        Xva = self._impute_and_scale_apply(X_va, med, mu, sd)
        sw = _balanced_sample_weight(y_tr)
        model = self._build(params)
        model.fit(Xtr, y_tr.astype(np.int32), sample_weight=sw)
        probs = np.asarray(model.predict_proba(Xva))[:, 1]
        return float(roc_auc_score(y_va, probs)), 0

    def fit_full(self, params, X_tv, y_tv, n_est_unused):
        Xtv, med, mu, sd = self._impute_and_scale_train(X_tv)
        sw = _balanced_sample_weight(y_tv)
        model = self._build(params)
        model.fit(Xtv, y_tv.astype(np.int32), sample_weight=sw)
        model._train_stats = (med, mu, sd)
        return model

    def predict_proba(self, model, X):
        med, mu, sd = model._train_stats
        X_proc = self._impute_and_scale_apply(X, med, mu, sd)
        return np.asarray(model.predict_proba(X_proc))[:, 1]

    def save(self, model, path_no_ext: Path):
        try:
            import joblib
            joblib.dump(model, path_no_ext.with_suffix(".joblib"))
        except Exception as e:
            print(f"  ! could not pickle cuML LogReg model: {e}; predictions saved separately")


HANDLERS = {
    "rf": CuMLRFHandler,
    "logreg": CuMLLogRegHandler,
}


# ============================================================================
# Optuna driver
# ============================================================================


def make_objective(handler, X, y, groups, tv_idx: np.ndarray, k: int):
    def objective(trial: optuna.Trial) -> float:
        params = handler.suggest(trial)
        gkf = GroupKFold(n_splits=k)
        fold_aucs = []
        for tr_rel, va_rel in gkf.split(X[tv_idx], y[tv_idx], groups[tv_idx]):
            tr = tv_idx[tr_rel]
            va = tv_idx[va_rel]
            auc, _ = handler.fit_fold(params, X[tr], y[tr], X[va], y[va])
            fold_aucs.append(auc)
            trial.report(float(np.mean(fold_aucs)), step=len(fold_aucs))
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_aucs))
    return objective


# ============================================================================
# CLI
# ============================================================================


@click.command()
@click.option("--model", type=click.Choice(list(HANDLERS.keys())), required=True)
@click.option("--trials", type=int, default=20, show_default=True)
@click.option("--timeout", type=int, default=1800, show_default=True, help="Optuna wall-time cap (seconds)")
@click.option("--k", type=int, default=5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True,
              help="MUST be 42 to keep test split identical to v6 for DeLong")
@click.option("--features-version", default="2026-04-25-contextual-v1",
              show_default=True, help="Restrict cohort to a single features_version tag")
@click.option("--out-name", default=None, help="Subdirectory under models/ (default aim2_v8_<model>)")
@click.option("--features-dir", default=None, type=click.Path(),
              help="Override features directory (default: Code/features/).")
@click.option("--models-dir", default=None, type=click.Path(),
              help="Override models output directory (default: Code/models/).")
def main(model: str, trials: int, timeout: int, k: int, seed: int,
         features_version: str | None, out_name: str | None,
         features_dir: str | None, models_dir: str | None) -> None:
    handler = HANDLERS[model]()

    features_path = Path(features_dir) if features_dir else FEATURES_DIR
    models_root = Path(models_dir) if models_dir else (CODE_ROOT / "models")
    out_dir = models_root / (out_name or f"aim2_v8_{model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== v8 cuML ablation: {model}  (device=cuda) ===")
    print(f"  Features dir: {features_path}")
    print(f"  Output dir:   {out_dir}")

    print("Loading features...")
    df = load_cohort(features_path)
    if features_version and "features_version" in df.columns:
        n_before = df["subject_id"].nunique()
        df = df[df["features_version"] == features_version].reset_index(drop=True)
        n_after = df["subject_id"].nunique()
        print(f"  features_version filter: {features_version!r} → {n_after} of {n_before} subjects")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"  {len(df):,} epochs, {df['subject_id'].nunique()} subjects, {len(feature_cols)} features")

    X = df[feature_cols].values
    y = df["apnoea_label"].values.astype(np.int8)
    groups = df["subject_id"].values

    # IDENTICAL outer split to v6 / v8 (seed=42)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tv_idx, test_idx = next(outer.split(X, y, groups))

    print(
        f"  Train+val pool: {len(tv_idx):,} epochs / {len(np.unique(groups[tv_idx]))} subjects  "
        f"({y[tv_idx].mean()*100:.1f}% positive)"
    )
    print(
        f"  Held-out test:  {len(test_idx):,} epochs / {len(np.unique(groups[test_idx]))} subjects  "
        f"({y[test_idx].mean()*100:.1f}% positive)"
    )

    storage_url = f"sqlite:///{out_dir / 'optuna_study.db'}"
    sampler = TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        study_name=f"aim2_v8_{model}_cuml_{seed}",
        load_if_exists=True,
    )

    print(f"\n▶ Optuna: up to {trials} trials, {timeout}s timeout, {k}-fold GroupKFold")
    objective = make_objective(handler, X, y, groups, tv_idx, k=k)
    study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=False)

    best = study.best_params
    best_mean_auc = study.best_value
    print(f"\n  best mean-fold val AUC: {best_mean_auc:.4f}")
    print(f"  best params: {json.dumps(best, indent=2)}")

    # Per-fold breakdown at best params
    print("\n▶ Recomputing per-fold AUCs at best params...")
    gkf = GroupKFold(n_splits=k)
    per_fold = []
    for i, (tr_rel, va_rel) in enumerate(gkf.split(X[tv_idx], y[tv_idx], groups[tv_idx])):
        tr = tv_idx[tr_rel]
        va = tv_idx[va_rel]
        auc, best_iter = handler.fit_fold(best, X[tr], y[tr], X[va], y[va])
        per_fold.append({
            "fold": i + 1, "auc": auc, "best_iter": best_iter,
            "n_train_subjects": int(len(np.unique(groups[tr]))),
            "n_val_subjects": int(len(np.unique(groups[va]))),
        })
        print(f"   fold {i+1}: AUC {auc:.4f}")
    mean_auc = float(np.mean([f["auc"] for f in per_fold]))
    std_auc = float(np.std([f["auc"] for f in per_fold], ddof=1))
    print(f"\n  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Final refit on full TV pool
    print(f"\n▶ Refitting on full train+val pool...")
    final_model = handler.fit_full(best, X[tv_idx], y[tv_idx], 0)
    probs = handler.predict_proba(final_model, X[test_idx])
    test_auc = float(roc_auc_score(y[test_idx], probs))
    test_ap = float(average_precision_score(y[test_idx], probs))

    # Tuned-threshold F1 from TV pool
    tv_probs = handler.predict_proba(final_model, X[tv_idx])
    thresholds = np.linspace(0.05, 0.95, 91)
    tv_f1s = [f1_score(y[tv_idx], (tv_probs > t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(tv_f1s))])
    preds = (probs > best_thresh).astype(int)
    test_f1 = float(f1_score(y[test_idx], preds, zero_division=0))
    test_p = float(precision_score(y[test_idx], preds, zero_division=0))
    test_r = float(recall_score(y[test_idx], preds, zero_division=0))

    print("\n▶ Bootstrap 95% CI on test AUC (1000 resamples)...")
    bootstrap_aucs, (ci_lo, ci_hi) = bootstrap_auc_ci(y[test_idx], probs, n=1000, seed=seed)

    print(f"\n=== Held-out test set ({model}, cuML) ===")
    print(f"  AUC-ROC : {test_auc:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  AUC-PR  : {test_ap:.4f}")
    print(f"  F1 @ {best_thresh:.2f}: {test_f1:.4f}  (P {test_p:.3f}, R {test_r:.3f})")

    # Persist
    (out_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    (out_dir / "per_fold_metrics.json").write_text(json.dumps(per_fold, indent=2))
    np.save(out_dir / "bootstrap_aucs.npy", bootstrap_aucs)
    pd.DataFrame({
        "subject_id": df.iloc[test_idx]["subject_id"].values,
        "epoch_idx": df.iloc[test_idx]["epoch_idx"].values,
        "apnoea_label": y[test_idx],
        "pred_prob": probs,
        "pred_label": preds,
    }).to_parquet(out_dir / "test_predictions.parquet", index=False)
    handler.save(final_model, out_dir / "model")

    metrics = {
        "model": model,
        "framework": "cuml",
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
        "n_cohort_subjects": int(df["subject_id"].nunique()),
        "n_test_subjects": int(len(np.unique(groups[test_idx]))),
        "features_version": features_version,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
