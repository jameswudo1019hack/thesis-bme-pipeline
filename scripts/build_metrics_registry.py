"""Build a machine-readable metrics registry from Code/models/<*>/metrics*.json.

Walks every subdir of Code/models/, reads metrics_extended.json (preferred) and
metrics.json (for fields the extended file doesn't carry), and emits one row per
run_id to Thesis Vault/Aims/_metrics_registry.csv.

Idempotent — re-run any time to refresh after new experiments land. Sorts by
run_id so diffs in git stay stable.

Usage:
    cd Code && python3 scripts/build_metrics_registry.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = CODE_ROOT / "models"
VAULT_DIR = CODE_ROOT.parent / "Thesis Vault"
OUT_PATH = VAULT_DIR / "Aims" / "_metrics_registry.csv"

COLUMNS = [
    "run_id", "date", "model", "filter_used",
    "n_features",
    "cv_auc_mean", "cv_auc_std",
    "test_auc", "test_auc_ci_low", "test_auc_ci_high",
    "test_auprc", "test_auprc_ci_low", "test_auprc_ci_high",
    "f1", "sens", "spec", "precision", "balanced_acc",
    "brier", "ece",
    "ahi_mae", "ahi_corr",
    "sev_acc", "sev_kappa",
    "n_test_epochs", "n_test_subjects",
    "threshold", "n_optuna_trials",
    "ci_basis",
    "source_path",
]


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _infer_filter_used(model_dir: Path) -> str | None:
    """For old runs without explicit filter_used field, infer from run_id naming.

    Returns None if uninferable; the registry row will have an empty cell which
    makes the legacy-vs-recovered split visible at a glance.
    """
    name = str(model_dir).lower()
    if "phase1_exp5_sleep_stage" in name:
        return "all-stages"
    if any(tag in name for tag in ("aim2_cv_v4", "aim2_cv_v5", "aim2_cv_v6", "aim2_cv_v7")):
        return "sleep-only"
    if any(tag in name for tag in ("v85_taxonomy", "lit_2019", "phase1_batch", "phase1_exp6_tabnet", "v6_past_only", "v6_future_inclusive")):
        # Pre-recovery these were wake-only-bug; post-recovery they're sleep-only.
        # Without a metrics.json field, we can't tell — return None and let the
        # downstream user mark manually.
        return None
    if name.endswith(("v8_xgboost", "v8_catboost", "v8_logreg", "v8_rf", "v8_ensemble")):
        # Original v8 series had no filter at all
        return "all-stages-bug"
    return None


def _row_for(model_dir: Path) -> dict | None:
    ext = _load_json(model_dir / "metrics_extended.json")
    basic = _load_json(model_dir / "metrics.json")
    if ext is None and basic is None:
        return None

    src = (model_dir / "metrics_extended.json") if ext else (model_dir / "metrics.json")
    data: dict = {}
    if basic:
        data.update(basic)
    if ext:
        data.update(ext)  # extended supersedes any duplicate keys

    feat = basic.get("feature_cols") if basic else None
    n_features = len(feat) if isinstance(feat, list) else None

    return {
        "run_id": str(model_dir.relative_to(MODELS_DIR)),
        "date": datetime.fromtimestamp(src.stat().st_mtime).strftime("%Y-%m-%d"),
        "model": data.get("model"),
        "filter_used": data.get("filter_used") or _infer_filter_used(model_dir),
        "n_features": n_features,
        "cv_auc_mean": data.get("cv_mean_auc"),
        "cv_auc_std": data.get("cv_std_auc"),
        "test_auc": data.get("auc_roc") or data.get("test_auc_roc"),
        "test_auc_ci_low": data.get("auc_roc_ci_low") or data.get("test_auc_ci_low"),
        "test_auc_ci_high": data.get("auc_roc_ci_high") or data.get("test_auc_ci_high"),
        "test_auprc": data.get("auc_pr") or data.get("test_auc_pr"),
        "test_auprc_ci_low": data.get("auc_pr_ci_low"),
        "test_auprc_ci_high": data.get("auc_pr_ci_high"),
        "f1": data.get("f1") or data.get("test_f1_tuned"),
        "sens": data.get("sensitivity") or data.get("test_recall_tuned"),
        "spec": data.get("specificity"),
        "precision": data.get("precision") or data.get("test_precision_tuned"),
        "balanced_acc": data.get("balanced_accuracy"),
        "brier": data.get("brier"),
        "ece": data.get("ece"),
        "ahi_mae": data.get("ahi_mae"),
        "ahi_corr": data.get("ahi_corr"),
        "sev_acc": data.get("severity_accuracy"),
        "sev_kappa": data.get("severity_weighted_kappa"),
        "n_test_epochs": data.get("n_test_epochs") or data.get("n_test"),
        "n_test_subjects": data.get("n_test_subjects"),
        "threshold": data.get("best_threshold"),
        "n_optuna_trials": data.get("n_trials_completed"),
        # CIs from metrics_extended.json are subject-level bootstrap;
        # CIs from metrics.json (legacy) are epoch-level. Tag explicitly.
        "ci_basis": "subject" if ext else "epoch",
        "source_path": str(src.relative_to(CODE_ROOT.parent)),
    }


def main() -> None:
    if not MODELS_DIR.exists():
        raise SystemExit(f"models dir not found: {MODELS_DIR}")

    rows: list[dict] = []
    skipped: list[str] = []
    for model_dir in sorted(MODELS_DIR.rglob("*")):
        if not model_dir.is_dir():
            continue
        if not (model_dir / "metrics.json").exists() and not (model_dir / "metrics_extended.json").exists():
            continue
        row = _row_for(model_dir)
        if row is None:
            skipped.append(str(model_dir.relative_to(MODELS_DIR)))
            continue
        rows.append(row)

    if not rows:
        raise SystemExit("no metrics found under Code/models/")

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("run_id").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, float_format="%.6f")

    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    if skipped:
        print(f"Skipped (unparseable): {len(skipped)}")
        for s in skipped:
            print(f"  - {s}")
    print("\n=== Headline (run_id, model, test_auc, sev_kappa, ci_basis) ===")
    print(df[["run_id", "model", "test_auc", "sev_kappa", "ci_basis"]].to_string(index=False))


if __name__ == "__main__":
    main()
