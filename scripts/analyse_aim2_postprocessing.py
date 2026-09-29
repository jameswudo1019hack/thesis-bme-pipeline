"""Aim 2 analysis trio on the canonical sleep-only recovery runs (T1-T3).

No model is refitted and no threshold is re-chosen: every quantity is computed
from the saved ``test_predictions.parquet`` of the 2026-05-03 recovery snapshot,
using each run's own validation-tuned ``pred_label``.

T1  Per-subject event index (AHI-type) agreement
    Two estimators per subject, both per hour of sleep:
      * epoch count  - positive epochs / h (the existing "epoch-overlap proxy")
      * run collapse - maximal runs of positive epochs / h; runs break at any
                       gap in epoch_idx, so events either side of a removed
                       wake bout never merge
    each computed from the model's pred_label AND from the true labels (the
    latter isolates the estimator's own error from model error). Compared
    against four references:
      * xml_ahi   - scored respiratory events in the NSRR XML whose midpoint
                    falls in a sleep epoch / hours of sleep. Same event
                    definition as the epoch labels.
      * rdi0p     - NSRR all-apnoea + all-hypopnoea index (no desat criterion).
      * ahi_a0h3a - NSRR AHI, hypopnoeas need >=3 % desat or arousal.
      * ahi_a0h4  - NSRR AHI, hypopnoeas need >=4 % desat (classic SHHS papers).
    The labels count every scored hypopnoea, so the model estimates an
    rdi0p-like quantity; the gap to ahi_a0h3a is a definition mismatch, not
    model error. Reported separately so the two can be decomposed.

T2  Calibration
    ECE (equal-width and equal-mass), Brier, Brier skill, calibration-in-the-
    large, logistic calibration slope/intercept, reliability curves, with
    subject-level bootstrap CIs and paired deltas between runs. Reported for
    the raw output AND a prior-corrected output that removes LightGBM's
    scale_pos_weight analytically (odds / spw, spw taken from the TRAINING
    rows in metrics.json). Nothing is fitted on the test set.

T3  Error anatomy
    Sensitivity / specificity (at the run's saved threshold) and AUC per cell
    for six cuts: sleep stage, subject severity tier (NSRR ahi_a0h3a), scored
    event duration, scored event type, epoch desaturation depth, and supine vs
    non-supine posture. Paired subject-bootstrap delta-AUC (physio_only - full) per cell.

Inputs (read-only): models/recovery_2026-05-03/*/test_predictions.parquet,
features/shhs1-*.parquet (test subjects only), the NSRR XMLs on the OneDrive
share, and Dataset/shhs/csv/shhs1-dataset-0.15.0.csv.

Outputs: models/aim2_analysis_v1/ (context cache, JSON + CSV tables, figures).

Usage:
  python scripts/analyse_aim2_postprocessing.py                 # all three
  python scripts/analyse_aim2_postprocessing.py --tasks events
  python scripts/analyse_aim2_postprocessing.py --n-boot 200     # quicker
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.epochs import APNOEA_OVERLAP_THRESHOLD, EPOCH_SECONDS  # noqa: E402
from thesis_pipeline.io import read_nsrr_xml  # noqa: E402

RECOVERY = CODE_ROOT / "models" / "recovery_2026-05-03"
FEATURES_DIR = CODE_ROOT / "features"
OUT_ROOT = CODE_ROOT / "models" / "aim2_analysis_v1"
NSRR_CSV = CODE_ROOT.parent / "Dataset" / "shhs" / "csv" / "shhs1-dataset-0.15.0.csv"
ONEDRIVE = Path.home() / (
    "Library/CloudStorage/OneDrive-SharedLibraries-TheUniversityofSydney(Staff)/Philip de Chazal - SHHS"
)

# Canonical tiers quoted in the thesis (sleep-only, seed 42, 1,159 test subjects).
RUNS = {
    "full": "aim2_v85_taxonomy/full",
    "aasm_only": "aim2_v85_taxonomy/aasm_only",
    "physio_only": "aim2_v85_taxonomy/physio_only",
    "physio_psd": "aim2_phase1_batch/physio_only/exp2",
    "past_only": "aim2_v6_past_only",
    "lit_2019": "aim2_lit_2019",
}
ANATOMY_RUNS = ("full", "aasm_only", "physio_only", "past_only")
POSITION_COLS = {
    "supine": "position_supine_frac", "left": "position_left_frac", "right": "position_right_frac",
    "prone": "position_prone_frac", "upright": "position_upright_frac",
}
SEVERITY_EDGES = (5.0, 15.0, 30.0)
SEVERITY_LABELS = ("none", "mild", "moderate", "severe")
NSRR_REFS = ("rdi0p", "ahi_a0h3a", "ahi_a0h4")


# ============================================================================ loading

def load_predictions(runs: dict[str, str]) -> dict[str, pd.DataFrame]:
    preds = {}
    for name, rel in runs.items():
        df = pd.read_parquet(RECOVERY / rel / "test_predictions.parquet")
        df = df.sort_values(["subject_id", "epoch_idx"]).reset_index(drop=True)
        preds[name] = df
    ref = preds[next(iter(preds))]
    for name, df in preds.items():
        same = (len(df) == len(ref)
                and np.array_equal(df["subject_id"].values, ref["subject_id"].values)
                and np.array_equal(df["epoch_idx"].values, ref["epoch_idx"].values)
                and np.array_equal(df["apnoea_label"].values, ref["apnoea_label"].values))
        assert same, f"run {name} does not share the reference test rows; paired analysis invalid"
    print(f"  {len(preds)} runs share {len(ref):,} test epochs / {ref['subject_id'].nunique()} subjects")
    return preds


def _subject_context(sid: int, epoch_idx: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Per-epoch context for one test subject + its XML event summary."""
    cols = ["epoch_idx", "sleep_stage", "desat_depth", *POSITION_COLS.values()]
    feat = pq.read_table(FEATURES_DIR / f"shhs1-{sid}.parquet", columns=cols).to_pandas()
    feat = feat.set_index("epoch_idx").reindex(epoch_idx)

    pos = feat[list(POSITION_COLS.values())].to_numpy(dtype=float)
    has_pos = np.isfinite(pos).any(axis=1)
    dominant = np.full(len(epoch_idx), "unknown", dtype=object)
    if has_pos.any():
        names = np.array(list(POSITION_COLS))
        dominant[has_pos] = names[np.nanargmax(np.where(np.isfinite(pos[has_pos]), pos[has_pos], -1), axis=1)]
    dominant = np.where(np.isin(dominant, ["left", "right"]), "lateral", dominant)

    _, events = read_nsrr_xml(ONEDRIVE / f"shhs1-{sid}-nsrr.xml")
    sleep_set = set(int(e) for e in epoch_idx)
    ev_dur = np.full(len(epoch_idx), np.nan)
    ev_kind = np.full(len(epoch_idx), "", dtype=object)
    xml_label = np.zeros(len(epoch_idx), dtype=np.int8)
    pos_of = {int(e): i for i, e in enumerate(epoch_idx)}
    n_sleep_events = 0
    for ev in events:
        mid_epoch = int((ev.start_sec + ev.duration_sec / 2) // EPOCH_SECONDS)
        if mid_epoch in sleep_set:
            n_sleep_events += 1
        end = ev.start_sec + ev.duration_sec
        for i in range(int(ev.start_sec // EPOCH_SECONDS), int(end // EPOCH_SECONDS) + 1):
            overlap = min(end, (i + 1) * EPOCH_SECONDS) - max(ev.start_sec, i * EPOCH_SECONDS)
            if overlap >= APNOEA_OVERLAP_THRESHOLD and i in pos_of:
                k = pos_of[i]
                xml_label[k] = 1
                if not (ev_dur[k] >= ev.duration_sec):  # keep the longest labelling event
                    ev_dur[k] = ev.duration_sec
                    ev_kind[k] = "hypopnoea" if ev.kind.lower().startswith("hypopn") else "apnoea"

    ctx = pd.DataFrame({
        "subject_id": sid, "epoch_idx": epoch_idx,
        "sleep_stage": feat["sleep_stage"].astype(str).values,
        "desat_depth": feat["desat_depth"].to_numpy(dtype=float),
        "position": dominant,
        "event_duration_s": ev_dur, "event_kind": ev_kind, "xml_label": xml_label,
    })
    return ctx, {"subject_id": sid, "n_sleep_events_xml": n_sleep_events, "n_events_xml_total": len(events)}


def build_context(ref: pd.DataFrame, out_dir: Path, rebuild: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx_path, subj_path = out_dir / "test_epoch_context.parquet", out_dir / "test_subject_context.parquet"
    if ctx_path.exists() and subj_path.exists() and not rebuild:
        print("  using cached context")
        return pd.read_parquet(ctx_path), pd.read_parquet(subj_path)

    groups = {int(s): g["epoch_idx"].to_numpy() for s, g in ref.groupby("subject_id")}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda kv: _subject_context(*kv), groups.items()))
    ctx = pd.concat([r[0] for r in results], ignore_index=True)
    subj = pd.DataFrame([r[1] for r in results]).set_index("subject_id")
    print(f"  built context for {len(subj)} subjects in {time.time()-t0:.0f}s")

    nsrr = pd.read_csv(NSRR_CSV, usecols=["nsrrid", *NSRR_REFS]).set_index("nsrrid")
    subj = subj.join(nsrr, how="left")
    ctx.to_parquet(ctx_path, index=False)
    subj.to_parquet(subj_path)
    return ctx, subj


# ============================================================================ bootstrap helpers

def bootstrap_weights(n_subj: int, n_boot: int, seed: int = 42) -> np.ndarray:
    """(n_boot, n_subj) multinomial resample counts; row b = how often each subject is drawn."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_subj, n_subj), minlength=n_subj) for _ in range(n_boot)])


def ci(a: np.ndarray) -> list[float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a.size else [np.nan, np.nan]


def paired_p(delta_boot: np.ndarray) -> float:
    d = delta_boot[np.isfinite(delta_boot)]
    if not d.size:
        return float("nan")
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


# ============================================================================ T1 events

def count_runs(epoch_idx: np.ndarray, flag: np.ndarray) -> int:
    """Number of maximal runs of flagged epochs with consecutive epoch_idx."""
    idx = epoch_idx[flag.astype(bool)]
    return 0 if idx.size == 0 else 1 + int(np.sum(np.diff(idx) > 1))


def icc_a1(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(A,1): two-way random effects, absolute agreement, single measure."""
    data = np.column_stack([x, y])
    n, k = data.shape
    grand = data.mean()
    msr = k * ((data.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((data.mean(0) - grand) ** 2).sum() / (k - 1)
    sse = ((data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k / n * (msc - mse)))


def severity(ahi: np.ndarray) -> np.ndarray:
    return np.digitize(ahi, SEVERITY_EDGES)


def quad_kappa(a: np.ndarray, b: np.ndarray, k: int = 4) -> float:
    cm = np.zeros((k, k))
    np.add.at(cm, (a, b), 1)
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    exp = np.outer(cm.sum(1), cm.sum(0)) / cm.sum()
    return float(1 - (w * cm).sum() / (w * exp).sum())


def agreement(est: np.ndarray, ref: np.ndarray) -> dict:
    d = est - ref
    return {
        "mae": float(np.abs(d).mean()), "bias": float(d.mean()),
        "loa": [float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
        "pearson_r": float(stats.pearsonr(est, ref)[0]),
        "spearman_rho": float(stats.spearmanr(est, ref)[0]),
        "icc_a1": icc_a1(est, ref),
        "severity_accuracy": float((severity(est) == severity(ref)).mean()),
        "severity_quadratic_kappa": quad_kappa(severity(est), severity(ref)),
    }


def plot_t1(per: pd.DataFrame, out_dir: Path, run: str = "full") -> None:
    """Three Bland-Altman panels for one run: the old comparison, the like-for-like one, run collapse."""
    p = per[per["run"] == run]
    panels = [
        ("epoch_proxy_pred", "ahi_a0h3a", "(a) Epoch count vs NSRR ahi_a0h3a\n(old metric: different event definition)"),
        ("epoch_proxy_pred", "xml_ahi", "(b) Epoch count vs scored events\n(same event definition as the labels)"),
        ("run_pred", "xml_ahi", "(c) Run collapse vs scored events\n(merges back-to-back events)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (est, ref, title) in zip(axes, panels):
        q = p[[est, ref]].dropna()
        m = (q[est] + q[ref]) / 2
        d = q[est] - q[ref]
        ax.scatter(m, d, s=5, alpha=0.35)
        ax.axhline(0, color="grey", lw=0.8)
        ax.axhline(d.mean(), color="k", label=f"bias {d.mean():+.1f}")
        for sgn in (-1, 1):
            ax.axhline(d.mean() + sgn * 1.96 * d.std(), ls="--", color="grey")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("mean of estimate and reference (events/h)")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("estimate - reference (events/h)")
    fig.suptitle(f"{run} model: per-subject event index agreement, {p.shape[0]:,} test subjects", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"t1_bland_altman_{run}.png", dpi=150)
    plt.close(fig)


def run_events(preds: dict, subj: pd.DataFrame, out_dir: Path, n_boot: int) -> dict:
    print("\n=== T1 event-run post-processing ===")
    rows = []
    for name, df in preds.items():
        for sid, g in df.groupby("subject_id", sort=True):
            e = g["epoch_idx"].to_numpy()
            tst_h = len(g) * EPOCH_SECONDS / 3600.0
            rows.append({
                "run": name, "subject_id": int(sid), "tst_h": tst_h,
                "epoch_proxy_pred": g["pred_label"].sum() / tst_h,
                "run_pred": count_runs(e, g["pred_label"].to_numpy()) / tst_h,
                "epoch_proxy_true": g["apnoea_label"].sum() / tst_h,
                "run_true": count_runs(e, g["apnoea_label"].to_numpy()) / tst_h,
            })
    per = pd.DataFrame(rows).set_index("subject_id").join(subj, how="left")
    per["xml_ahi"] = per["n_sleep_events_xml"] / per["tst_h"]
    per.reset_index().to_parquet(out_dir / "t1_per_subject_indices.parquet", index=False)

    refs = ("xml_ahi", *NSRR_REFS)
    estimators = ("run_pred", "epoch_proxy_pred", "run_true", "epoch_proxy_true")
    subjects = np.sort(per.index.unique())
    W = bootstrap_weights(len(subjects), n_boot)
    out: dict = {"n_subjects": int(len(subjects)), "n_boot": n_boot, "runs": {}}

    for name in preds:
        p = per[per["run"] == name].loc[subjects]
        res = {}
        for est in estimators:
            for ref in refs:
                ok = p[[est, ref]].notna().all(1).to_numpy()
                x, y = p[est].to_numpy()[ok], p[ref].to_numpy()[ok]
                a = agreement(x, y)
                idx = np.flatnonzero(ok)
                boot_mae, boot_icc, boot_kap = [], [], []
                for w in W:
                    sel = np.repeat(np.arange(len(subjects)), w)
                    sel = sel[np.isin(sel, idx)]
                    xs, ys = p[est].to_numpy()[sel], p[ref].to_numpy()[sel]
                    boot_mae.append(np.abs(xs - ys).mean())
                    boot_icc.append(icc_a1(xs, ys))
                    boot_kap.append(quad_kappa(severity(xs), severity(ys)))
                a.update({"n": int(ok.sum()), "mae_ci": ci(boot_mae), "icc_ci": ci(boot_icc),
                          "kappa_ci": ci(boot_kap)})
                res[f"{est}__vs__{ref}"] = a
        out["runs"][name] = res
        r = res["epoch_proxy_pred__vs__xml_ahi"]
        q = res["epoch_proxy_pred__vs__ahi_a0h3a"]
        c = res["run_pred__vs__xml_ahi"]
        print(f"  {name:12s} epoch-count vs XML: MAE {r['mae']:.1f} bias {r['bias']:+.1f} ICC {r['icc_a1']:.3f} "
              f"κ {r['severity_quadratic_kappa']:.3f} | vs ahi_a0h3a: MAE {q['mae']:.1f} bias {q['bias']:+.1f} "
              f"| run-collapse vs XML: bias {c['bias']:+.1f}")

    # Reference-definition gap, independent of any model.
    s = per[per["run"] == next(iter(preds))].loc[subjects]
    out["reference_gap"] = {
        "median": {r: float(s[r].median()) for r in refs},
        "xml_vs_rdi0p": agreement(*s[["xml_ahi", "rdi0p"]].dropna().to_numpy().T),
        "xml_vs_ahi_a0h3a": agreement(*s[["xml_ahi", "ahi_a0h3a"]].dropna().to_numpy().T),
        "run_true_vs_xml": agreement(*s[["run_true", "xml_ahi"]].dropna().to_numpy().T),
    }
    (out_dir / "t1_events.json").write_text(json.dumps(out, indent=2))

    plot_t1(per, out_dir)
    return out


# ============================================================================ T2 calibration

def _cal_stats(y, p, sid_codes, n_subj, edges):
    """Per-subject sufficient statistics for binned calibration + Brier."""
    b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    nb = len(edges) - 1
    key = sid_codes * nb + b
    cnt = np.bincount(key, minlength=n_subj * nb).reshape(n_subj, nb)
    sy = np.bincount(key, weights=y, minlength=n_subj * nb).reshape(n_subj, nb)
    sp = np.bincount(key, weights=p, minlength=n_subj * nb).reshape(n_subj, nb)
    se = np.bincount(sid_codes, weights=(p - y) ** 2, minlength=n_subj)
    return cnt, sy, sp, se


def _ece_brier(W, cnt, sy, sp, se):
    C, Y, P = W @ cnt, W @ sy, W @ sp
    n = C.sum(1)
    ece = (np.abs(Y - P)).sum(1) / n
    brier = (W @ se) / n
    prev = Y.sum(1) / n
    skill = 1 - brier / (prev * (1 - prev))
    citl = P.sum(1) / n - prev
    return ece, brier, skill, citl


def prior_correct(p: np.ndarray, spw: float) -> np.ndarray:
    """Undo LightGBM scale_pos_weight: divide the odds by the training-set weight.

    scale_pos_weight = n_neg / n_pos on the TRAINING rows, stored in each run's
    metrics.json, so this uses no test information.
    """
    p = np.clip(p, 1e-9, 1 - 1e-9)
    odds = p / (1 - p) / spw
    return odds / (1 + odds)


def run_calibration(preds: dict, out_dir: Path, n_boot: int) -> dict:
    print("\n=== T2 calibration ===")
    ref = preds[next(iter(preds))]
    subjects, sid_codes = np.unique(ref["subject_id"].to_numpy(), return_inverse=True)
    n_subj = len(subjects)
    W = np.vstack([np.ones(n_subj, dtype=np.int64), bootstrap_weights(n_subj, n_boot)])  # row 0 = point estimate
    y = ref["apnoea_label"].to_numpy(dtype=float)
    width_edges = np.linspace(0, 1, 11)

    def evaluate(p: np.ndarray) -> tuple[dict, dict, list]:
        mass_edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        mass_edges[0], mass_edges[-1] = 0.0, 1.0 + 1e-12
        e_w, b_w, s_w, c_w = _ece_brier(W, *_cal_stats(y, p, sid_codes, n_subj, width_edges))
        e_m, _, _, _ = _ece_brier(W, *_cal_stats(y, p, sid_codes, n_subj, mass_edges))
        lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        fit = LogisticRegression(penalty=None, max_iter=1000).fit(lp.reshape(-1, 1), y.astype(int))
        res = {
            "ece_width10": float(e_w[0]), "ece_width10_ci": ci(e_w[1:]),
            "ece_mass10": float(e_m[0]), "ece_mass10_ci": ci(e_m[1:]),
            "brier": float(b_w[0]), "brier_ci": ci(b_w[1:]),
            "brier_skill": float(s_w[0]), "brier_skill_ci": ci(s_w[1:]),
            "calibration_in_the_large": float(c_w[0]), "citl_ci": ci(c_w[1:]),
            "calibration_slope": float(fit.coef_[0, 0]), "calibration_intercept": float(fit.intercept_[0]),
            "prevalence": float(y.mean()), "mean_predicted": float(p.mean()),
        }
        b = np.clip(np.digitize(p, width_edges[1:-1]), 0, 9)
        curve = [{"bin": k, "n": int((b == k).sum()), "mean_pred": float(p[b == k].mean()),
                  "observed": float(y[b == k].mean())} for k in range(10) if (b == k).any()]
        return res, {"ece": e_w[1:], "brier": b_w[1:]}, curve

    out: dict = {"n_boot": n_boot, "variants": {"raw": {}, "prior_corrected": {}}, "paired_deltas": {}}
    boots: dict = {"raw": {}, "prior_corrected": {}}
    curves = []
    for name, df in preds.items():
        spw = json.loads((RECOVERY / RUNS[name] / "metrics.json").read_text())["scale_pos_weight"]
        p_raw = df["pred_prob"].to_numpy(dtype=float)
        for variant, p in (("raw", p_raw), ("prior_corrected", prior_correct(p_raw, spw))):
            res, bt, curve = evaluate(p)
            res["scale_pos_weight_train"] = float(spw)
            out["variants"][variant][name] = res
            boots[variant][name] = bt
            curves += [{"run": name, "variant": variant, **c} for c in curve]
        r, c = out["variants"]["raw"][name], out["variants"]["prior_corrected"][name]
        print(f"  {name:12s} raw ECE {r['ece_width10']:.4f} (int {r['calibration_intercept']:+.2f}, -log spw {-np.log(spw):+.2f})"
              f" -> prior-corrected ECE {c['ece_width10']:.4f} {[round(v, 4) for v in c['ece_width10_ci']]}"
              f"  Brier {r['brier']:.4f} -> {c['brier']:.4f}")

    pairs = [("physio_only", "full"), ("aasm_only", "full"), ("physio_only", "aasm_only"),
             ("past_only", "full"), ("physio_psd", "physio_only")]
    for variant in boots:
        for a, b in pairs:
            if a in boots[variant] and b in boots[variant]:
                for metric, key in (("ece", "ece_width10"), ("brier", "brier")):
                    d = boots[variant][a][metric] - boots[variant][b][metric]
                    pt = out["variants"][variant][a][key] - out["variants"][variant][b][key]
                    out["paired_deltas"][f"{variant}:{metric}:{a}-{b}"] = {"delta": float(pt), "ci": ci(d), "p": paired_p(d)}

    pd.DataFrame(curves).to_csv(out_dir / "t2_reliability_bins.csv", index=False)
    (out_dir / "t2_calibration.json").write_text(json.dumps(out, indent=2))

    cv = pd.DataFrame(curves)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for col, variant in enumerate(("raw", "prior_corrected")):
        ax, ax2 = axes[0, col], axes[1, col]
        ax.plot([0, 1], [0, 1], ls=":", color="grey", label="perfect")
        for name in ("full", "aasm_only", "physio_only", "past_only"):
            c = cv[(cv.run == name) & (cv.variant == variant)]
            ece = out["variants"][variant][name]["ece_width10"]
            ax.plot(c.mean_pred, c.observed, marker="o", ms=3, label=f"{name} (ECE {ece:.3f})")
            spw = out["variants"][variant][name]["scale_pos_weight_train"]
            p = preds[name]["pred_prob"].to_numpy()
            ax2.hist(p if variant == "raw" else prior_correct(p, spw), bins=50, histtype="step")
        ax.set_title("Raw model output" if variant == "raw" else "Prior-corrected (training class weight removed)", fontsize=10)
        ax.set_ylabel("observed apnoea fraction")
        ax.legend(fontsize=7)
        ax2.set_xlabel("predicted probability")
        ax2.set_yscale("log")
        ax2.set_ylabel("epochs")
    fig.tight_layout()
    fig.savefig(out_dir / "t2_reliability.png", dpi=150)
    plt.close(fig)
    return out


# ============================================================================ T3 error anatomy

def assign_cells(ctx: pd.DataFrame, subj: pd.DataFrame) -> dict[str, pd.Series]:
    sev = pd.Series(np.array(SEVERITY_LABELS)[severity(subj["ahi_a0h3a"].to_numpy())], index=subj.index)
    dur = pd.cut(ctx["event_duration_s"], [0, 20, 40, np.inf], right=False,
                 labels=["short <20 s", "medium 20-40 s", "long >=40 s"]).astype(object)
    # desat_depth is only defined for detected desaturations (>=3 % below a 100-s rolling-max
    # baseline for >=10 s) and in practice starts at ~4.3 %, so bins start there.
    desat = pd.cut(ctx["desat_depth"], [-np.inf, 6, 9, np.inf], right=False,
                   labels=["<6 %", "6-9 %", ">=9 %"]).astype(object).where(ctx["desat_depth"].notna(), "no desat")
    # Only supine vs non-supine is used: position_supine_frac agrees with NSRR supinep
    # (r = 1.00), but the right/left/prone code mapping could not be verified independently.
    posture = np.where(ctx["position"] == "supine", "supine",
                       np.where(ctx["position"] == "unknown", "unknown", "non-supine"))
    return {
        "sleep_stage": ctx["sleep_stage"],
        "severity_tier": ctx["subject_id"].map(sev),
        "event_duration": dur.where(ctx["event_duration_s"].notna(), None),
        "event_type": ctx["event_kind"].replace("", None),
        "desat_depth": desat,
        "position": pd.Series(posture, index=ctx.index),
    }


def run_anatomy(preds: dict, ctx: pd.DataFrame, subj: pd.DataFrame, out_dir: Path, n_boot: int) -> dict:
    print("\n=== T3 error anatomy ===")
    ref = preds[next(iter(preds))]
    assert np.array_equal(ctx["epoch_idx"].to_numpy(), ref["epoch_idx"].to_numpy())
    agree = float((ctx["xml_label"].to_numpy() == ref["apnoea_label"].to_numpy()).mean())
    print(f"  XML-rederived labels agree with stored apnoea_label on {100*agree:.3f}% of epochs")

    cells = assign_cells(ctx, subj)
    y = ref["apnoea_label"].to_numpy()
    subjects, sid_codes = np.unique(ref["subject_id"].to_numpy(), return_inverse=True)
    rows_of = [np.flatnonzero(sid_codes == i) for i in range(len(subjects))]
    rng = np.random.default_rng(42)
    boot_sets = [np.concatenate([rows_of[i] for i in rng.integers(0, len(subjects), len(subjects))])
                 for _ in range(n_boot)]
    runs = [r for r in ANATOMY_RUNS if r in preds]

    table = []
    for cut, lab in cells.items():
        lab = np.asarray(lab, dtype=object)
        for cell in [c for c in pd.unique(lab) if c is not None and c == c]:
            m = lab == cell
            n, npos = int(m.sum()), int(y[m].sum())
            row = {"cut": cut, "cell": str(cell), "n_epochs": n, "n_pos": npos,
                   "n_subjects": int(np.unique(sid_codes[m]).size)}
            aucs = {}
            for r in runs:
                pl, pp = preds[r]["pred_label"].to_numpy()[m], preds[r]["pred_prob"].to_numpy()[m]
                row[f"{r}_sens"] = float(pl[y[m] == 1].mean()) if npos else np.nan
                row[f"{r}_spec"] = float(1 - pl[y[m] == 0].mean()) if n - npos else np.nan
                both = 20 <= npos <= n - 20
                aucs[r] = float(roc_auc_score(y[m], pp)) if both else np.nan
                row[f"{r}_auc"] = aucs[r]
            if np.isfinite(aucs.get("physio_only", np.nan)) and np.isfinite(aucs.get("full", np.nan)):
                d = []
                pf, pp_ = preds["full"]["pred_prob"].to_numpy(), preds["physio_only"]["pred_prob"].to_numpy()
                for s in boot_sets:
                    s = s[m[s]]
                    ys = y[s]
                    if 0 < ys.sum() < len(ys):
                        d.append(roc_auc_score(ys, pp_[s]) - roc_auc_score(ys, pf[s]))
                d = np.asarray(d)
                row["delta_auc_physio_minus_full"] = aucs["physio_only"] - aucs["full"]
                row["delta_auc_ci"] = ci(d)
                row["delta_auc_p"] = paired_p(d)
            table.append(row)
            print(f"  {cut:15s} {str(cell):16s} n={n:>7,} pos={npos:>6,}  "
                  + "  ".join(f"{r}:{row[f'{r}_auc']:.3f}" for r in runs if np.isfinite(row[f"{r}_auc"])))

    tab = pd.DataFrame(table)
    tab.to_csv(out_dir / "t3_error_anatomy.csv", index=False)
    (out_dir / "t3_error_anatomy.json").write_text(json.dumps(
        {"n_boot": n_boot, "label_agreement_xml_vs_stored": agree, "runs": runs,
         "threshold_note": "sens/spec use each run's saved validation-tuned pred_label",
         "cells": table}, indent=2, default=float))

    t = tab.dropna(subset=["full_auc", "physio_only_auc"])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(t) + 1.2))
    yy = np.arange(len(t))
    ax.scatter(t["full_auc"], yy, label="full", marker="o")
    ax.scatter(t["physio_only_auc"], yy, label="physio_only", marker="s")
    ax.set_yticks(yy, [f"{c}: {v}" for c, v in zip(t["cut"], t["cell"])], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("AUC within cell")
    ax.legend(fontsize=8)
    ax.set_title("Error anatomy: where physiology-only and full models differ", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "t3_anatomy_auc.png", dpi=150)
    plt.close(fig)
    return {"n_cells": len(table)}


# ============================================================================ CLI

@click.command()
@click.option("--tasks", default="events,calibration,anatomy", show_default=True,
              help="Comma-separated: events, calibration, anatomy, t1fig (redraw T1 figures from saved indices).")
@click.option("--n-boot", type=int, default=1000, show_default=True)
@click.option("--out-root", type=click.Path(path_type=Path), default=OUT_ROOT, show_default=True)
@click.option("--rebuild-context", is_flag=True, help="Re-read parquets and XMLs even if the cache exists.")
@click.option("--subjects-file", type=click.Path(exists=True, path_type=Path), default=None,
              help="Testing only: restrict to subject ids listed one per line. Never for reported numbers.")
def main(tasks, n_boot, out_root, rebuild_context, subjects_file) -> None:
    tasks = {t.strip() for t in tasks.split(",")}
    out_root.mkdir(parents=True, exist_ok=True)
    print("=== Aim 2 analysis trio (recovery_2026-05-03, sleep-only) ===")
    preds = load_predictions(RUNS)
    if subjects_file is not None:
        keep = {int(x) for x in subjects_file.read_text().split()}
        preds = {k: v[v["subject_id"].isin(keep)].reset_index(drop=True) for k, v in preds.items()}
        print(f"  TEST SUBSET: {preds['full']['subject_id'].nunique()} subjects (not for reporting)")
    summary_path = out_root / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({"source_snapshot": str(RECOVERY.relative_to(CODE_ROOT)), "runs": RUNS, "n_boot": n_boot})
    ctx = subj = None
    if tasks & {"events", "anatomy"}:
        ctx, subj = build_context(preds[next(iter(preds))], out_root, rebuild_context)
    if "t1fig" in tasks:
        per = pd.read_parquet(out_root / "t1_per_subject_indices.parquet")
        for r in ("full", "physio_only"):
            plot_t1(per, out_root, r)
    if "events" in tasks:
        summary["t1_events"] = run_events(preds, subj, out_root, n_boot)["reference_gap"]["median"]
    if "calibration" in tasks:
        cal = run_calibration(preds, out_root, n_boot)["variants"]
        summary["t2_calibration_ece"] = {v: {k: r["ece_width10"] for k, r in runs.items()} for v, runs in cal.items()}
    if "anatomy" in tasks:
        summary["t3_anatomy"] = run_anatomy(preds, ctx, subj, out_root, min(n_boot, 200))
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nsaved -> {out_root}")


if __name__ == "__main__":
    main()
