"""Extended evaluation metrics for Aim 2 — single source of truth.

Quick API for fit scripts (one-liner at the end of each script)::

    from thesis_pipeline.extended_metrics import write_extended_metrics
    write_extended_metrics(out_dir, test_predictions, subject_metadata=sm)

This writes:
  out_dir/metrics_extended.json
  out_dir/bootstrap_aucs_subject.npy
  out_dir/bootstrap_auprs_subject.npy


Per the methodology, every Aim 2 fit reports the following metric set:

  Threshold-free       AUC-ROC, AUC-PR, both with subject-level bootstrap 95% CI
  Threshold-tuned      F1, sensitivity, specificity, precision, balanced accuracy
  Calibration          Brier score, ECE (Expected Calibration Error)
  Clinical             AHI MAE, AHI correlation (vs NSRR ahi_a0h3a),
                       severity-tier accuracy, severity-tier weighted κ
  Subject counts       n_test_epochs, n_test_subjects, n_subjects_with_nsrr_ahi

If a metric cannot be computed (missing subject_metadata, single-class fold, etc.)
it is set to ``None`` and a one-line reason is appended to ``metric_notes``.
**Never silently omit a metric** — downstream tables count on the consistent schema.

This helper is called by:
  - All ``fit_aim2_*`` scripts at the end of training (compute on test predictions)
  - ``scripts/recompute_extended_metrics.py`` (retroactive recompute on saved predictions)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

EPOCH_SECONDS = 30  # 30-s epochs (AASM standard) — used for AHI proxy calc

# AASM apnoea severity tiers (events per hour). Right-edges of bins.
# < 5: none, 5–15: mild, 15–30: moderate, ≥ 30: severe
SEVERITY_BINS = (5.0, 15.0, 30.0)
SEVERITY_LABELS = ("none", "mild", "moderate", "severe")


def severity_tier(ahi: np.ndarray) -> np.ndarray:
    """Bin per-subject AHI values into integer severity tiers (0=none, 3=severe)."""
    return np.digitize(np.asarray(ahi, dtype=float), SEVERITY_BINS).astype(np.int8)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error.

    Splits [0, 1] into ``n_bins`` equally-spaced bins, computes per-bin
    |accuracy − mean-confidence|, weighted by bin size.

    Lower is better (0 = perfectly calibrated). Common thresholds in literature:
    ECE < 0.05 = well-calibrated; > 0.10 = poorly calibrated.

    NaN-safe: drops NaN rows.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[keep]
    y_prob = y_prob[keep]
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize returns 1..n_bins for [0,1); clip to n_bins-1
    bin_ids = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for k in range(n_bins):
        in_bin = bin_ids == k
        if not in_bin.any():
            continue
        bin_size = in_bin.sum()
        bin_acc = float(y_true[in_bin].mean())
        bin_conf = float(y_prob[in_bin].mean())
        ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return float(ece)


def per_subject_ahi_proxy(test_predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-epoch predictions into per-subject AHI proxies.

    AHI proxy = (positive_epoch_count) / (n_epochs × 30 s / 3600)
              = positive_epoch_count × 120 / n_epochs   (in events / hour)

    NOTE on the proxy semantics: this counts every epoch overlapping ≥10 s of an
    AASM-scored apnoea/hypopnoea event as "1 event". A long event spanning
    multiple epochs gets counted multiple times. Therefore this proxy
    systematically overestimates clinical AHI by a factor of ~2–3×. It's still
    useful as a *within-method* comparison metric (subjects' relative ranking
    is preserved) and for severity-tier classification (after appropriate
    thresholds are applied separately to predicted vs ground-truth AHI).

    Returns DataFrame indexed by subject_id with columns:
        n_epochs, n_apnoea_gt, n_apnoea_pred, ahi_proxy_gt, ahi_proxy_pred
    """
    grp = test_predictions.groupby("subject_id").agg(
        n_epochs=("apnoea_label", "count"),
        n_apnoea_gt=("apnoea_label", "sum"),
        n_apnoea_pred=("pred_label", "sum"),
    )
    tst_hours = grp["n_epochs"] * EPOCH_SECONDS / 3600.0
    grp["ahi_proxy_gt"] = grp["n_apnoea_gt"] / tst_hours
    grp["ahi_proxy_pred"] = grp["n_apnoea_pred"] / tst_hours
    return grp


def _subject_bootstrap_aucs(
    test_predictions: pd.DataFrame,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Subject-level bootstrap of AUC-ROC and AUC-PR. Returns (aucs, auprs)."""
    rng = np.random.default_rng(seed)
    sids = test_predictions["subject_id"].values
    y = test_predictions["apnoea_label"].values.astype(np.int8)
    p = test_predictions["pred_prob"].values

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
        pi = p[idx]
        aucs[i] = roc_auc_score(yi, pi)
        auprs[i] = average_precision_score(yi, pi)
    return aucs, auprs


def compute_extended_metrics(
    test_predictions: pd.DataFrame,
    subject_metadata: Optional[pd.DataFrame] = None,
    n_bootstrap_subj: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute the full Aim 2 extended metric set.

    Parameters
    ----------
    test_predictions : pd.DataFrame
        Must contain columns: ``subject_id``, ``epoch_idx``, ``apnoea_label``,
        ``pred_prob``, ``pred_label``.
    subject_metadata : pd.DataFrame, optional
        For clinical metrics. If provided, must contain ``subject_id`` and
        ``ahi_a0h3a`` (NSRR's clinical AHI). If None, clinical metrics are
        set to None with a note.
    n_bootstrap_subj : int
        Subject-level bootstrap iterations (default 1000).
    seed : int
        RNG seed.

    Returns
    -------
    dict with the full extended metric set + ``metric_notes`` for any metric
    that couldn't be computed. Suitable for ``json.dump → metrics.json``.

    Also returns ``bootstrap_aucs_subject`` and ``bootstrap_auprs_subject`` as
    lists for downstream paired-bootstrap tests. (Caller may choose to write
    these to .npy and exclude from the JSON.)
    """
    required = {"subject_id", "epoch_idx", "apnoea_label", "pred_prob", "pred_label"}
    missing = required - set(test_predictions.columns)
    if missing:
        raise ValueError(f"test_predictions missing columns: {missing}")

    y = test_predictions["apnoea_label"].values.astype(int)
    p = test_predictions["pred_prob"].values.astype(float)
    pred = test_predictions["pred_label"].values.astype(int)

    metric_notes: dict[str, str] = {}

    # ─── threshold-free ──────────────────────────────────────────────────
    auc_roc = float(roc_auc_score(y, p))
    auc_pr = float(average_precision_score(y, p))

    aucs, auprs = _subject_bootstrap_aucs(
        test_predictions, n_resamples=n_bootstrap_subj, seed=seed
    )
    valid_auc = aucs[np.isfinite(aucs)]
    valid_aupr = auprs[np.isfinite(auprs)]
    auc_ci_low = float(np.percentile(valid_auc, 2.5))
    auc_ci_high = float(np.percentile(valid_auc, 97.5))
    aupr_ci_low = float(np.percentile(valid_aupr, 2.5))
    aupr_ci_high = float(np.percentile(valid_aupr, 97.5))

    # ─── threshold-tuned ─────────────────────────────────────────────────
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    prec = float(precision_score(y, pred, zero_division=0))
    f1 = float(f1_score(y, pred, zero_division=0))
    if not (np.isnan(sens) or np.isnan(spec)):
        bal_acc = (sens + spec) / 2.0
    else:
        bal_acc = float("nan")
        metric_notes["balanced_accuracy"] = "Sens or spec NaN (degenerate label)."

    # ─── calibration ─────────────────────────────────────────────────────
    brier = float(brier_score_loss(y, p))
    ece = expected_calibration_error(y, p, n_bins=10)
    metric_notes["calibration"] = (
        "Brier and ECE computed on raw model output (no post-hoc calibration). "
        "LightGBM with scale_pos_weight produces miscalibrated probabilities by "
        "default; values typically 0.05–0.15 ECE without Platt/isotonic."
    )

    # ─── clinical (per-subject AHI vs NSRR) ──────────────────────────────
    ahi_mae: Optional[float] = None
    ahi_corr: Optional[float] = None
    severity_acc: Optional[float] = None
    severity_kappa: Optional[float] = None
    n_with_nsrr: Optional[int] = None

    if subject_metadata is not None and "ahi_a0h3a" in subject_metadata.columns:
        per_subj = per_subject_ahi_proxy(test_predictions)
        sm = subject_metadata.copy()

        # Subject ID normalisation: predictions use int (e.g. 200001) but
        # subject_metadata may store strings like 'shhs1-200001'. Strip the
        # prefix and coerce to int so the join succeeds.
        if "subject_id" in sm.columns:
            sid_series = sm["subject_id"]
            if sid_series.dtype == object:
                # Try string→int (after stripping any non-digit prefix)
                sid_int = sid_series.astype(str).str.extract(r"(\d+)$", expand=False)
                sm = sm.assign(subject_id_int=pd.to_numeric(sid_int, errors="coerce"))
                sm = sm.dropna(subset=["subject_id_int"])
                sm["subject_id_int"] = sm["subject_id_int"].astype(int)
                sm = sm.set_index("subject_id_int")
            else:
                sm = sm.set_index("subject_id")
        per_subj["ahi_a0h3a"] = per_subj.index.map(sm["ahi_a0h3a"])
        per_subj = per_subj.dropna(subset=["ahi_a0h3a"])
        n_with_nsrr = int(len(per_subj))

        if n_with_nsrr >= 10:
            diff = per_subj["ahi_proxy_pred"] - per_subj["ahi_a0h3a"]
            ahi_mae = float(np.abs(diff).mean())
            ahi_corr = float(per_subj["ahi_proxy_pred"].corr(per_subj["ahi_a0h3a"]))

            gt_severity = severity_tier(per_subj["ahi_a0h3a"].values)
            pred_severity = severity_tier(per_subj["ahi_proxy_pred"].values)
            severity_acc = float((gt_severity == pred_severity).mean())
            severity_kappa = float(
                cohen_kappa_score(gt_severity, pred_severity, weights="quadratic")
            )
            metric_notes["ahi_mae"] = (
                "Predicted AHI is an epoch-overlap proxy (counts every epoch "
                "overlapping an AASM event as one event). Systematically "
                "overestimates clinical AHI by ~2–3× because long events span "
                "multiple epochs. Use as within-method comparison metric. "
                "Severity-tier weighted κ is the more clinically meaningful number."
            )
        else:
            metric_notes["clinical"] = (
                f"Only {n_with_nsrr} subjects in test set have ahi_a0h3a; "
                "skipping AHI MAE / severity κ (need ≥10)."
            )
    else:
        metric_notes["clinical"] = (
            "subject_metadata.parquet not provided or missing ahi_a0h3a column; "
            "clinical metrics not computed."
        )

    out_dict = {
        # threshold-free
        "auc_roc": auc_roc,
        "auc_roc_ci_low": auc_ci_low,
        "auc_roc_ci_high": auc_ci_high,
        "auc_pr": auc_pr,
        "auc_pr_ci_low": aupr_ci_low,
        "auc_pr_ci_high": aupr_ci_high,
        # threshold-tuned
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "balanced_accuracy": bal_acc,
        # calibration
        "brier": brier,
        "ece": ece,
        # clinical
        "ahi_mae": ahi_mae,
        "ahi_corr": ahi_corr,
        "severity_accuracy": severity_acc,
        "severity_weighted_kappa": severity_kappa,
        # counts
        "n_test_epochs": int(len(test_predictions)),
        "n_test_subjects": int(test_predictions["subject_id"].nunique()),
        "n_subjects_with_nsrr_ahi": n_with_nsrr,
        # bootstrap arrays — caller decides whether to keep in JSON or save .npy
        "_bootstrap_aucs_subject": aucs.tolist(),
        "_bootstrap_auprs_subject": auprs.tolist(),
        # provenance
        "metric_notes": metric_notes,
        "extended_metrics_version": "1.0",
    }
    return out_dict


def write_extended_metrics(
    out_dir: str | Path,
    test_predictions: pd.DataFrame,
    subject_metadata: Optional[pd.DataFrame] = None,
    n_bootstrap_subj: int = 1000,
    seed: int = 42,
) -> dict:
    """Convenience wrapper: compute + persist extended metrics in standard layout.

    Writes (next to the existing metrics.json):
      ``{out_dir}/metrics_extended.json``
      ``{out_dir}/bootstrap_aucs_subject.npy``
      ``{out_dir}/bootstrap_auprs_subject.npy``

    Returns the metrics dict (without bootstrap arrays — those are persisted
    only as .npy to keep JSON small).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_extended_metrics(
        test_predictions=test_predictions,
        subject_metadata=subject_metadata,
        n_bootstrap_subj=n_bootstrap_subj,
        seed=seed,
    )
    aucs = np.asarray(metrics.pop("_bootstrap_aucs_subject"))
    auprs = np.asarray(metrics.pop("_bootstrap_auprs_subject"))
    np.save(out_dir / "bootstrap_aucs_subject.npy", aucs)
    np.save(out_dir / "bootstrap_auprs_subject.npy", auprs)
    (out_dir / "metrics_extended.json").write_text(json.dumps(metrics, indent=2))
    return metrics
