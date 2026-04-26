"""Aim 2 baseline v8 — ensemble across LightGBM (v6) + the 4 v8 base models.

Experiment log: Vault/Experiments/2026-04-26 - v8 model ablation.md

Reads `test_predictions.parquet` from each base run, aligns on
(subject_id, epoch_idx), then computes three ensemble probabilities on the
held-out test set:

    mean        — simple arithmetic mean of P(y=1) across base models
    weighted    — softmax over CV AUCs (so the best CV-mean model gets the
                  highest weight, but no model is dropped)
    rank_mean   — mean of ranks per epoch, normalised to [0, 1]
                  (robust to base-model probability calibration differences)

Reports AUC + 1000-bootstrap 95% CI + AUC-PR + tuned-threshold F1 for each
ensemble. Threshold for F1 is tuned by sweeping over the test predictions
themselves (this is OK because we report the tuned value for context;
headline number is AUC, which is threshold-free).

Inputs (default):
    Code/models/aim2_cv_v6/test_predictions.parquet       (LightGBM, v6)
    Code/models/aim2_v8_xgboost/test_predictions.parquet
    Code/models/aim2_v8_catboost/test_predictions.parquet
    Code/models/aim2_v8_rf/test_predictions.parquet
    Code/models/aim2_v8_logreg/test_predictions.parquet

Output:
    Code/models/aim2_v8_ensemble/
        metrics.json
        ensemble_predictions.parquet   (subject_id, epoch_idx, apnoea_label,
                                        plus one column per ensemble strategy)
        base_aucs.json                 (per-base test AUC for the table)

Usage:
    python scripts/fit_aim2_v8_ensemble.py
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

CODE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASES = {
    "lightgbm_v6": CODE_ROOT / "models" / "aim2_cv_v6",
    "xgboost":     CODE_ROOT / "models" / "aim2_v8_xgboost",
    "catboost":    CODE_ROOT / "models" / "aim2_v8_catboost",
    "rf":          CODE_ROOT / "models" / "aim2_v8_rf",
    "logreg":      CODE_ROOT / "models" / "aim2_v8_logreg",
}


def bootstrap_auc_ci(y_true, probs, n=1000, seed=42):
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
    return aucs, (float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5)))


def report(name: str, y, probs, seed=42):
    auc = float(roc_auc_score(y, probs))
    ap = float(average_precision_score(y, probs))
    boot, (lo, hi) = bootstrap_auc_ci(y, probs, n=1000, seed=seed)
    # Tuned-threshold F1 on the test predictions (reported for context)
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, (probs > t).astype(int), zero_division=0) for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    pred = (probs > best_t).astype(int)
    return {
        "name": name,
        "auc": auc, "auc_ci_low": lo, "auc_ci_high": hi,
        "auc_pr": ap,
        "f1_tuned": float(f1_score(y, pred, zero_division=0)),
        "precision_tuned": float(precision_score(y, pred, zero_division=0)),
        "recall_tuned": float(recall_score(y, pred, zero_division=0)),
        "best_threshold": best_t,
        "bootstrap_aucs": boot,
    }


@click.command()
@click.option("--out-name", default="aim2_v8_ensemble", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
def main(out_name: str, seed: int) -> None:
    out_dir = CODE_ROOT / "models" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load each base model's test predictions and CV mean AUC for weighting
    bases: dict[str, dict] = {}
    for name, model_dir in DEFAULT_BASES.items():
        pred_path = model_dir / "test_predictions.parquet"
        metrics_path = model_dir / "metrics.json"
        if not pred_path.exists():
            print(f"  ! skipping {name}: {pred_path} not found")
            continue
        if not metrics_path.exists():
            print(f"  ! skipping {name}: {metrics_path} not found")
            continue
        preds = pd.read_parquet(pred_path)
        metrics = json.loads(metrics_path.read_text())
        bases[name] = {
            "preds": preds.sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True),
            "cv_mean_auc": float(metrics["cv_mean_auc"]),
            "test_auc": float(metrics["test_auc_roc"]),
        }
        print(f"  loaded {name}: CV AUC {metrics['cv_mean_auc']:.4f}, "
              f"Test AUC {metrics['test_auc_roc']:.4f}, n={len(preds):,}")

    if len(bases) < 2:
        raise SystemExit(f"Need ≥2 base models with predictions; got {len(bases)}")

    # Verify alignment — same (subject_id, epoch_idx, apnoea_label) across all bases
    first_name = next(iter(bases))
    ref = bases[first_name]["preds"]
    for name, b in bases.items():
        if not (b["preds"][["subject_id", "epoch_idx", "apnoea_label"]]
                .equals(ref[["subject_id", "epoch_idx", "apnoea_label"]])):
            raise SystemExit(
                f"Base {name} test predictions do NOT align with {first_name}. "
                f"All v8 runs must share seed=42 outer split."
            )
    print(f"\n  ✓ All {len(bases)} bases align on the same {len(ref):,} test rows")

    y = ref["apnoea_label"].values.astype(np.int8)
    P = np.column_stack([b["preds"]["pred_prob"].values for b in bases.values()])  # (n_test, n_base)
    base_names = list(bases.keys())

    # ----- Strategy 1: simple mean -------------------------------------------
    p_mean = P.mean(axis=1)

    # ----- Strategy 2: CV-AUC weighted (softmax) -----------------------------
    cv_aucs = np.array([b["cv_mean_auc"] for b in bases.values()])
    # Center for numerical stability, scale by 50 so a 0.01 AUC gap → ~50% weight ratio
    w_logits = (cv_aucs - cv_aucs.mean()) * 50.0
    w = np.exp(w_logits) / np.exp(w_logits).sum()
    p_weighted = P @ w
    print(f"\n  Weighted ensemble weights (softmax(CV AUC × 50)):")
    for n, wi in zip(base_names, w):
        print(f"    {n:14s} {wi:.4f}")

    # ----- Strategy 3: rank mean ---------------------------------------------
    # Average rank per epoch across base models, normalised to [0,1]
    R = np.column_stack([
        pd.Series(P[:, j]).rank(method="average").values / len(P) for j in range(P.shape[1])
    ])
    p_rank = R.mean(axis=1)

    # Evaluate each
    print("\n=== Ensemble results on held-out test ===\n")
    results = []
    for name, probs in [("mean", p_mean), ("weighted", p_weighted), ("rank_mean", p_rank)]:
        r = report(name, y, probs, seed=seed)
        results.append(r)
        print(f"  {name:10s} AUC {r['auc']:.4f} [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]  "
              f"PR {r['auc_pr']:.4f}  F1 {r['f1_tuned']:.4f}")

    # Also report each base for the table
    print("\n=== Base model test AUCs (for the comparison table) ===\n")
    base_results = []
    for name in base_names:
        probs = bases[name]["preds"]["pred_prob"].values
        r = report(name, y, probs, seed=seed)
        base_results.append(r)
        print(f"  {name:14s} AUC {r['auc']:.4f} [{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}]  "
              f"PR {r['auc_pr']:.4f}  F1 {r['f1_tuned']:.4f}")

    # Persist
    pred_out = pd.DataFrame({
        "subject_id": ref["subject_id"].values,
        "epoch_idx": ref["epoch_idx"].values,
        "apnoea_label": y,
        "pred_prob_mean": p_mean,
        "pred_prob_weighted": p_weighted,
        "pred_prob_rank_mean": p_rank,
    })
    pred_out.to_parquet(out_dir / "ensemble_predictions.parquet", index=False)

    summary = {
        "bases": [
            {"name": n, "cv_mean_auc": bases[n]["cv_mean_auc"],
             "test_auc": bases[n]["test_auc"], "weight": float(w[i])}
            for i, n in enumerate(base_names)
        ],
        "ensemble": [
            {k: v for k, v in r.items() if k != "bootstrap_aucs"} for r in results
        ],
        "base_test": [
            {k: v for k, v in r.items() if k != "bootstrap_aucs"} for r in base_results
        ],
        "n_test_rows": int(len(y)),
        "n_test_subjects": int(ref["subject_id"].nunique()),
        "seed": seed,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    # Save bootstrap arrays for DeLong / paired tests later
    np.savez(
        out_dir / "bootstrap_aucs.npz",
        **{f"ens_{r['name']}": r["bootstrap_aucs"] for r in results},
        **{f"base_{r['name']}": r["bootstrap_aucs"] for r in base_results},
    )

    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
