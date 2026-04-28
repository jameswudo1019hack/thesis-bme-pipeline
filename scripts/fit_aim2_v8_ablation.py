"""Aim 2 baseline v8 — model ablation on v6 contextual features.

Experiment log: Vault/Experiments/2026-04-26 - v8 model ablation.md

Same protocol as fit_aim2_cv.py (v6):
    1. Hold out 20% of subjects (patient-level) — IDENTICAL split to v6 (seed=42).
    2. 5-fold GroupKFold CV on the remaining 80% — Optuna objective = mean val AUC.
    3. Refit best params on full train+val pool, evaluate on held-out test.
    4. 1000-sample bootstrap 95% CI on test AUC.

Models supported (Plan A):
    xgboost   — XGBoost (hist, scale_pos_weight)         [supports --device cuda]
    catboost  — CatBoost (auto_class_weights="Balanced") [supports --device cuda]
    rf        — RandomForestClassifier (class_weight="balanced")  [CPU only]
    logreg    — LogisticRegression (StandardScaler + L2/L1)       [CPU only]

GPU support (Colab Pro+):
    --device cuda  → XGBoost uses GPU (`device='cuda'`, ~5-10× faster on >1M rows)
                  → CatBoost uses GPU (`task_type='GPU'`, ~4-5× faster)
                  → RF and LogReg fall back to CPU (sklearn has no GPU path)

LightGBM is *not* run here — that line is the v6 result (Code/models/aim2_cv_v6/),
which already used this exact protocol with seed=42. Re-running it would just burn
another 2 hr for the same number.

Output directory: Code/models/aim2_v8_<model>/
    best_params.json
    per_fold_metrics.json
    optuna_study.db
    bootstrap_aucs.npy
    metrics.json
    test_predictions.parquet
    model.<ext>          (joblib for sklearn, json/cbm for boosters)

Usage:
    python scripts/fit_aim2_v8_ablation.py --model xgboost   --trials 30 --timeout 5400
    python scripts/fit_aim2_v8_ablation.py --model catboost  --trials 20 --timeout 7200
    python scripts/fit_aim2_v8_ablation.py --model rf        --trials 15 --timeout 5400
    python scripts/fit_aim2_v8_ablation.py --model logreg    --trials 30 --timeout 1200
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import click
import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
# Cohort load (identical to fit_aim2_cv.py — keep these in sync)
# ============================================================================


def load_cohort(features_dir: Path) -> pd.DataFrame:
    # Restrict to per-subject epoch parquets; exclude `subject_metadata.parquet`
    # (one row per subject; would corrupt the epoch-level training frame).
    files = [f for f in sorted(features_dir.glob("*.parquet"))
             if f.name != "subject_metadata.parquet"]
    if not files:
        raise FileNotFoundError(f"No per-subject parquet files in {features_dir}")

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


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y))
    neg = float(len(y) - pos)
    return neg / max(pos, 1.0)


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
# Per-model handlers
#
# Each handler exposes:
#   suggest(trial)                            -> params dict
#   fit_fold(params, X_tr, y_tr, X_va, y_va)  -> (val_auc, refit_n_estimators)
#   fit_full(params, X_tv, y_tv, n_est)       -> fitted model
#   predict_proba(model, X)                   -> 1d array of P(y=1)
#   save(model, path_no_ext)                  -> writes model.<ext>
# ============================================================================


# ----- XGBoost --------------------------------------------------------------

class XGBHandler:
    name = "xgboost"

    def __init__(self, device: str = "cpu"):
        self.device = device  # "cpu" or "cuda"

    def suggest(self, trial: optuna.Trial) -> dict:
        # max_depth capped at 7 (was 10) — depth=10 means 1024 leaves and 8× the
        # per-tree compute vs depth=7, with no AUC benefit on this scale of tabular
        # data. min_child_weight floor at 5 (was 1) forces effectively shallower
        # trees, further reducing per-trial compute. Together these make each
        # 5-fold trial ~25 min instead of ~3 hr, with no expected AUC loss.
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        }

    def _common(self, params: dict, y_for_spw: np.ndarray) -> dict:
        full = dict(params)
        full["scale_pos_weight"] = _scale_pos_weight(y_for_spw)
        full["tree_method"] = "hist"
        full["device"] = self.device  # "cpu" or "cuda"
        full["objective"] = "binary:logistic"
        full["eval_metric"] = "auc"
        full["random_state"] = 42
        full["n_jobs"] = -1
        full["verbosity"] = 0
        return full

    def fit_fold(self, params, X_tr, y_tr, X_va, y_va):
        import xgboost as xgb
        full = self._common(params, y_tr)
        model = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **full)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        probs = model.predict_proba(X_va)[:, 1]
        best_iter = int(model.best_iteration or 1000)
        return float(roc_auc_score(y_va, probs)), best_iter

    def fit_full(self, params, X_tv, y_tv, n_est):
        import xgboost as xgb
        full = self._common(params, y_tv)
        model = xgb.XGBClassifier(n_estimators=n_est, **full)
        model.fit(X_tv, y_tv, verbose=False)
        return model

    def predict_proba(self, model, X):
        return model.predict_proba(X)[:, 1]

    def save(self, model, path_no_ext: Path):
        model.save_model(str(path_no_ext.with_suffix(".json")))


# ----- CatBoost -------------------------------------------------------------

class CatBoostHandler:
    name = "catboost"

    def __init__(self, device: str = "cpu"):
        # CatBoost uses task_type="CPU"/"GPU"; map from our --device flag.
        self.task_type = "GPU" if device == "cuda" else "CPU"

    def suggest(self, trial: optuna.Trial) -> dict:
        # depth capped at 7 (was 10) — CatBoost uses symmetric trees, so depth=10 is
        # 1024 leaves and ~8× per-tree compute vs depth=7 with no AUC gain on this
        # data. learning_rate floor at 0.03 (was 0.01) — combined with the depth cap,
        # this keeps each 5-fold trial near 25 min instead of multi-hour.
        # Note: bagging_temperature is incompatible with GPU (uses Bayesian
        # bootstrap by default there), so we drop it when on GPU.
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 254),
        }
        if self.task_type == "CPU":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
        return params

    def _common(self, params: dict) -> dict:
        full = dict(params)
        full["loss_function"] = "Logloss"
        full["eval_metric"] = "AUC"
        full["auto_class_weights"] = "Balanced"
        full["random_seed"] = 42
        full["task_type"] = self.task_type
        if self.task_type == "GPU":
            full["devices"] = "0"
        else:
            full["thread_count"] = -1
        full["verbose"] = False
        full["allow_writing_files"] = False
        return full

    def fit_fold(self, params, X_tr, y_tr, X_va, y_va):
        from catboost import CatBoostClassifier
        full = self._common(params)
        model = CatBoostClassifier(iterations=2000, early_stopping_rounds=50, **full)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        probs = model.predict_proba(X_va)[:, 1]
        best_iter = int(model.get_best_iteration() or 1000)
        return float(roc_auc_score(y_va, probs)), best_iter

    def fit_full(self, params, X_tv, y_tv, n_est):
        from catboost import CatBoostClassifier
        full = self._common(params)
        model = CatBoostClassifier(iterations=n_est, **full)
        model.fit(X_tv, y_tv, verbose=False)
        return model

    def predict_proba(self, model, X):
        return model.predict_proba(X)[:, 1]

    def save(self, model, path_no_ext: Path):
        model.save_model(str(path_no_ext.with_suffix(".cbm")))


# ----- Random Forest --------------------------------------------------------

class RFHandler:
    name = "rf"

    def suggest(self, trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }

    def _build(self, params):
        # Median-impute NaNs (e.g. desat_depth, lag/lead at subject boundaries).
        # add_indicator=False to keep memory bounded — the missingness signal is mostly
        # captured by the median value being a flat baseline that RF can split on.
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("clf", RandomForestClassifier(
                **params,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ])

    def fit_fold(self, params, X_tr, y_tr, X_va, y_va):
        model = self._build(params)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_va)[:, 1]
        # RF has no early stopping; "n_estimators_for_refit" is just whatever was tuned
        return float(roc_auc_score(y_va, probs)), int(params["n_estimators"])

    def fit_full(self, params, X_tv, y_tv, n_est):
        # n_est is ignored — RF uses its tuned n_estimators directly
        model = self._build(params)
        model.fit(X_tv, y_tv)
        return model

    def predict_proba(self, model, X):
        return model.predict_proba(X)[:, 1]

    def save(self, model, path_no_ext: Path):
        joblib.dump(model, path_no_ext.with_suffix(".joblib"))


# ----- Logistic Regression --------------------------------------------------

class LogRegHandler:
    name = "logreg"

    def suggest(self, trial: optuna.Trial) -> dict:
        # SGDClassifier with log_loss = streaming logistic regression.
        # alpha = inverse of LogisticRegression's C (regularisation strength).
        return {
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    def _build(self, params):
        # Median-impute (no indicator — keeps memory bounded), then standardise.
        # SGDClassifier with log_loss is the streaming logistic regression — designed
        # for n > 1M, low memory, fast (multi-threaded via OpenMP). Same loss function
        # as LogisticRegression, so coefficients are interpretable the same way.
        # alpha is the regularisation strength (inverse of LogisticRegression's C);
        # l1_ratio mixes L1 and L2 when penalty="elasticnet".
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(
                loss="log_loss",
                alpha=params["alpha"],
                penalty=params["penalty"],
                l1_ratio=params["l1_ratio"],
                class_weight="balanced",
                max_iter=50,
                tol=1e-4,
                early_stopping=False,
                n_jobs=-1,
                random_state=42,
            )),
        ])

    def fit_fold(self, params, X_tr, y_tr, X_va, y_va):
        model = self._build(params)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_va)[:, 1]
        return float(roc_auc_score(y_va, probs)), 0  # n_est not meaningful

    def fit_full(self, params, X_tv, y_tv, n_est):
        model = self._build(params)
        model.fit(X_tv, y_tv)
        return model

    def predict_proba(self, model, X):
        return model.predict_proba(X)[:, 1]

    def save(self, model, path_no_ext: Path):
        joblib.dump(model, path_no_ext.with_suffix(".joblib"))


HANDLERS = {
    "xgboost": XGBHandler,
    "catboost": CatBoostHandler,
    "rf": RFHandler,
    "logreg": LogRegHandler,
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
@click.option("--trials", type=int, default=30, show_default=True)
@click.option("--timeout", type=int, default=5400, show_default=True, help="Optuna wall-time cap (seconds)")
@click.option("--k", type=int, default=5, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True,
              help="MUST be 42 to keep test split identical to v6 for DeLong")
@click.option("--features-version", default="2026-04-25-contextual-v1",
              show_default=True, help="Restrict cohort to a single features_version tag")
@click.option("--out-name", default=None, help="Subdirectory under models/ (default aim2_v8_<model>)")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", show_default=True,
              help="Compute device for XGBoost / CatBoost. RF and LogReg ignore this (sklearn has no GPU path).")
@click.option("--features-dir", default=None, type=click.Path(),
              help="Override features directory (default: Code/features/). Useful for Colab where features live elsewhere.")
@click.option("--models-dir", default=None, type=click.Path(),
              help="Override models output directory (default: Code/models/).")
def main(model: str, trials: int, timeout: int, k: int, seed: int,
         features_version: str | None, out_name: str | None,
         device: str, features_dir: str | None, models_dir: str | None) -> None:
    # Try to instantiate handler with device kwarg; falls back for handlers that don't take it.
    try:
        handler = HANDLERS[model](device=device)
    except TypeError:
        handler = HANDLERS[model]()
        if device == "cuda":
            print(f"  ! warning: {model} handler doesn't support GPU; falling back to CPU.")

    features_path = Path(features_dir) if features_dir else FEATURES_DIR
    models_root = Path(models_dir) if models_dir else (CODE_ROOT / "models")
    out_dir = models_root / (out_name or f"aim2_v8_{model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== v8 ablation: {model}  (device={device}) ===")
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

    # IDENTICAL outer split to v6 (seed=42) — gives the same held-out test set.
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
        study_name=f"aim2_v8_{model}_{seed}",
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
        print(f"   fold {i+1}: AUC {auc:.4f}  (best_iter {best_iter})")
    mean_auc = float(np.mean([f["auc"] for f in per_fold]))
    std_auc = float(np.std([f["auc"] for f in per_fold], ddof=1))
    mean_best_iter = int(np.round(np.mean([f["best_iter"] for f in per_fold])))
    print(f"\n  CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # Final refit on full TV pool
    n_est_refit = max(100, int(1.1 * mean_best_iter)) if mean_best_iter > 0 else 0
    print(f"\n▶ Refitting on full train+val pool (n_estimators={n_est_refit if n_est_refit else 'model-default'})...")
    final_model = handler.fit_full(best, X[tv_idx], y[tv_idx], n_est_refit)
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

    print(f"\n=== Held-out test set ({model}) ===")
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
        "features_version": features_version,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
