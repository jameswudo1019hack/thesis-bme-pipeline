"""Evaluate Aim 2 Olsen BiGRU runs: provenance, extended metrics and the pre-registered contrasts.

The pre-registration note governs (its body is checked against the committed freeze record by
``thesis_pipeline.prereg.check_prereg``; dated addenda below '## Addenda' never break this).

``--split test`` (once: after DL test inference of every run and test scoring of the matched
LightGBM cells). Everything below is checked before any metric is computed or any file written:

  * ``--prereg`` is required; the full check_prereg record goes into the summary.
  * Runs ``<runs-root>/<config>/seed<k>/`` for the configs in ``--config`` (default: every
    discovered config). Every evaluated config must be M or P4 with seed folders exactly
    42, 43, 44 (F6 and stray seeds are refused).
  * Per run, predict_test.json (written by ``predict_aim2_dl.py --split test``): split 'test';
    config / model_seed equal to the folder's; prereg_body_sha256 and train_prereg_body_sha256
    equal to the frozen body (a whole-file prereg hash alone is refused);
    test_predictions_sha256 == sha256(test_predictions.parquet); test_sec_probs_sha256 ==
    sha256(test_sec_probs.npy) when recorded (absent: event-level F1 is dropped, and that is
    recorded); parity.passed is True; aggregator == aggregator_locked; train_git_commits ==
    [train_git_commit] (every invocation, resumes included, at C_train); a passed code_lineage
    (freeze commit descends from C_train, model files byte-identical); postproc_git_commit ==
    freeze_commit (the parity gate and threshold were refit by the frozen code).
  * Per run, the training metrics.json (copied from Drive): best_ckpt_sha256 == the predicted
    checkpoint's sha256; git_commit == C_train and git_commits == [C_train]; its settings show
    neither --allow-unfrozen nor a PILOT ONLY option (--max-passes, --max-batches-per-pass: the
    pass cap is never extended); GPU, timings, passes, stop reason, cap flag, recipe, AMP,
    non-finite and GradScaler-skipped step counts are copied into the summary.
  * Across runs: one aggregator and lock, one freeze_commit, one train_git_commit (C_train),
    one device type, one recipe, one set of training settings (TrainSettings.run_fields()
    minus config, model seed and amp); inference AMP and training AMP each identical within a
    config; no PARITY_FAILED_<devtype>.json for that device type in the runs root (written by
    predict_aim2_dl.py when the parity gate fails).
  * Comparators: the matched LightGBM cells of PRIMARY_CONTRASTS (``--ref NAME=DIR``, names
    exactly the ones the evaluated configs need; default models/aim2_matched_cells_v1/<name>/test),
    each with metrics.json model_sha256 equal to the pinned model and prereg_body_sha256 equal to
    the frozen body. ``--contrast`` / ``--gap`` are refused: contrasts are fixed.
  * ``--epoch-context`` (default models/aim2_analysis_v1/test_epoch_context.parquet) must have
    the pinned sha256 and be row-aligned with the key.
  * Per run: schema check and key equality with the key file (default the canonical
    physio_only test predictions); every subject has an NSRR ahi_a0h3a.

  Then (test) any earlier evaluation_summary_test.json and each run folder's earlier
  metrics_extended.json / bootstrap arrays are moved into superseded_<UTC>/ (never
  overwritten; recorded in summary["superseded_previous"]); per run ``write_extended_metrics``
  with the NSRR AHI. Per config: mean +- SD (ddof 1) of every numeric metric; per primary
  contrast the per-seed and seed-averaged paired deltas (AUC-ROC with p-values, AUC-PR without)
  and the claim rule on AUC-ROC; the gap contrast (``dl_eval.gap_per_seed``) when M and P4 are
  both evaluated; the ladder (``--ladder NAME=DIR``, descriptive deltas, no verdict, no
  p-value); the short-hypopnoea endpoint (estimation only, no p-value) and expectation E1; the
  multiplicity statement.

``--split val`` runs the same metrics on validation predictions (written to <run>/val_eval/),
with free ``--ref`` / ``--contrast`` / ``--gap`` pairings and no provenance or pin checks.

Outputs stay under ``--runs-root`` (default Code/models/aim2_dl_olsen_v1):
evaluation_summary_<split>.json, plus each run's metrics_extended.json and bootstrap arrays.
"""
from __future__ import annotations

import datetime
import json
import re
import sys
import time
from pathlib import Path
from typing import Mapping

import click
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.dl_eval import (  # noqa: E402
    E1_BOUND,
    EPOCH_CONTEXT_SHA256,
    KEY_COLS,
    PARITY_FAILED_FILE,
    PILOT_ONLY_SETTINGS,
    PINNED_REF_MODEL_SHA256,
    PRED_DTYPES,
    PRIMARY_CONTRASTS,
    PRIMARY_GAP,
    REQUIRED_SEEDS,
    SHARED_TRAIN_SETTINGS,
    SHORT_HYP_DEFINITION,
    check_schema,
    claim_rule,
    gap_per_seed,
    load_nsrr_ahi,
    metadata_with_ahi,
    paired_delta,
    seed_averaged_delta,
    short_hypopnoea_mask,
    subset_bootstrap_auc,
)
from thesis_pipeline.dl_stage_b import sha256_file  # noqa: E402
from thesis_pipeline.extended_metrics import write_extended_metrics  # noqa: E402
from thesis_pipeline.prereg import PreregMismatch, check_prereg  # noqa: E402

DEFAULT_ROOT = CODE_ROOT / "models" / "aim2_dl_olsen_v1"
DEFAULT_KEY = CODE_ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only" / \
    "test_predictions.parquet"
DEFAULT_CSV = CODE_ROOT.parent / "Dataset" / "shhs" / "csv" / "shhs1-dataset-0.15.0.csv"
DEFAULT_SM = CODE_ROOT / "features" / "subject_metadata.parquet"
DEFAULT_REF_ROOT = CODE_ROOT / "models" / "aim2_matched_cells_v1"
DEFAULT_EPOCH_CONTEXT = CODE_ROOT / "models" / "aim2_analysis_v1" / "test_epoch_context.parquet"

NO_SEC_PROBS = "per-second output not saved: event-level F1 dropped"
MULTIPLICITY = (
    "Verdicts (claim rule A-E) and unadjusted p-values are reported only for the three primary contrasts on "
    "AUC-ROC (M vs LGBM-ECG, P4 vs LGBM-ECG+belt, and the gap (P4 - M) - (LGBM-ECG+belt - LGBM-ECG), its "
    "index-paired per-seed and seed-averaged estimates); everything else (AUC-PR deltas, the gap's cross pairs, "
    "the short-hypopnoea endpoint, the ladder, the extended metrics) is estimation only (point estimate and "
    "95% CI), with no verdict, no p-value, no familywise adjustment and no directional claim."
)
# evaluate --split test output in each run folder; an earlier version is moved to superseded_<UTC>/
EVAL_RUN_OUTPUTS = ("metrics_extended.json", "bootstrap_aucs_subject.npy", "bootstrap_auprs_subject.npy")


def discover_runs(root: Path, which: str) -> dict[str, dict[int, Path]]:
    fname = "test_predictions.parquet" if which == "test" else "val_predictions.parquet"
    runs: dict[str, dict[int, Path]] = {}
    for p in sorted(root.glob(f"*/seed*/{fname}")):
        m = re.fullmatch(r"seed(\d+)", p.parent.name)
        if m:
            runs.setdefault(p.parent.parent.name, {})[int(m.group(1))] = p.parent
    return runs


def seed_dirs(cfg_dir: Path) -> list[int]:
    """Every seed<k> folder of a config, with or without predictions."""
    out = []
    for p in cfg_dir.iterdir():
        m = re.fullmatch(r"seed(\d+)", p.name)
        if p.is_dir() and m:
            out.append(int(m.group(1)))
    return sorted(out)


def load_ref(d: Path, which: str, primary: str | None = None, freeze_body: str | None = None) -> dict:
    """A LightGBM run folder holding ``<which>_predictions.parquet`` and subject-bootstrap arrays.

    With ``primary`` (a PRIMARY_CONTRASTS reference, test only), <d>/metrics.json must record the
    pinned model_sha256 of that cell and prereg.prereg_body_sha256 == ``freeze_body``.
    """
    d = Path(d)
    pinned = {}
    if primary is not None:  # checked before any prediction is read or any AUC computed
        want = PINNED_REF_MODEL_SHA256.get(primary)
        if want is None:
            raise click.ClickException(f"{primary} is not a pinned primary comparator ({sorted(PINNED_REF_MODEL_SHA256)})")
        mj = d / "metrics.json"
        if not mj.exists():
            raise click.ClickException(f"reference {primary} ({d}) has no metrics.json; the pinned model cannot be "
                                       "verified")
        mt = json.loads(mj.read_text())
        if mt.get("model_sha256") != want:
            raise click.ClickException(f"reference {primary} ({d}): model_sha256 {str(mt.get('model_sha256'))[:12]} "
                                       f"is not the pinned model {want[:12]}")
        body = (mt.get("prereg") or {}).get("prereg_body_sha256")
        if not freeze_body or body != freeze_body:
            raise click.ClickException(f"reference {primary} ({d}) was not scored under the frozen pre-registration "
                                       f"(prereg_body_sha256 {str(body)[:12]} != {str(freeze_body)[:12]})")
        pinned = {"model_sha256": mt["model_sha256"], "prereg_body_sha256": body}
    boots = {"auc": d / "bootstrap_aucs_subject.npy", "aupr": d / "bootstrap_auprs_subject.npy"}
    for p in boots.values():
        if not p.exists():
            raise click.ClickException(f"reference {d} lacks {p.name}")
    pred_path = d / f"{which}_predictions.parquet"
    pred = pd.read_parquet(pred_path)
    me = d / "metrics_extended.json"
    if me.exists():
        m = json.loads(me.read_text())
        auc, aupr = m["auc_roc"], m["auc_pr"]
    else:
        auc = float(roc_auc_score(pred["apnoea_label"], pred["pred_prob"]))
        aupr = float(average_precision_score(pred["apnoea_label"], pred["pred_prob"]))
    return {"dir": str(d), "auc": auc, "aupr": aupr, "boot_auc": np.load(boots["auc"]),
            "boot_aupr": np.load(boots["aupr"]), "key": pred[KEY_COLS],
            "pred_prob": pred["pred_prob"].to_numpy(np.float64),
            "predictions_sha256": sha256_file(pred_path), **pinned}


def test_provenance(run_dir: Path, cfg: str, seed: int, freeze_body: str) -> dict:
    """predict_test.json and training metrics.json of one run, checked against its files and the freeze."""
    def refuse(msg: str):
        raise click.ClickException(f"{run_dir}: {msg}")

    pj = run_dir / "predict_test.json"
    if not pj.exists():
        refuse("no predict_test.json; test predictions must come from predict_aim2_dl.py --split test")
    prov = json.loads(pj.read_text())
    if prov.get("split") != "test":
        refuse(f"predict_test.json split is {prov.get('split')!r}, not 'test'")
    ms = prov.get("model_seed")
    try:
        ms_int = int(ms)
    except (TypeError, ValueError):
        ms_int = None
    if prov.get("config") != cfg or ms_int != seed:
        refuse(f"predict_test.json is for config {prov.get('config')!r} seed {ms!r}, not this folder's "
               f"{cfg} seed{seed}")
    if not prov.get("prereg_body_sha256"):
        refuse("predict_test.json has no prereg_body_sha256 (a whole-file prereg hash alone is not accepted)")
    if prov["prereg_body_sha256"] != freeze_body:
        refuse(f"predictions were made under pre-registration body {str(prov['prereg_body_sha256'])[:12]}, "
               f"not the frozen body {freeze_body[:12]}")
    if prov.get("train_prereg_body_sha256") != freeze_body:
        refuse(f"training did not start under the frozen pre-registration (train_prereg_body_sha256 "
               f"{str(prov.get('train_prereg_body_sha256'))[:12]} != {freeze_body[:12]})")
    got = sha256_file(run_dir / "test_predictions.parquet")
    if prov.get("test_predictions_sha256") != got:
        refuse(f"test_predictions.parquet sha256 {got[:12]} != predict_test.json "
               f"{str(prov.get('test_predictions_sha256'))[:12]} (file replaced or edited)")
    sec_sha = prov.get("test_sec_probs_sha256")
    if sec_sha:
        npy = run_dir / "test_sec_probs.npy"
        if not npy.exists():
            refuse("predict_test.json records test_sec_probs_sha256 but test_sec_probs.npy is missing")
        got_sec = sha256_file(npy)
        if got_sec != sec_sha:
            refuse(f"test_sec_probs.npy sha256 {got_sec[:12]} != predict_test.json {str(sec_sha)[:12]} "
                   "(file replaced or edited)")
        sec_status = "test_sec_probs.npy saved; sha256 verified"
    else:
        sec_status = NO_SEC_PROBS
    parity = prov.get("parity")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        refuse("the validation parity gate did not pass on the test-inference device (parity.passed is not True)")
    if not prov.get("freeze_commit"):
        refuse("predict_test.json has no freeze_commit")
    tc = prov.get("train_git_commit")
    if not tc:
        refuse("predict_test.json has no train_git_commit")
    if prov.get("train_git_commits") != [tc]:
        refuse(f"the checkpoint's commit history {prov.get('train_git_commits')!r} is not exactly [train_git_commit "
               f"{str(tc)[:10]}]: every invocation of a run (resumes included) trains at one commit C_train")
    lin = prov.get("code_lineage")
    if (not isinstance(lin, Mapping) or lin.get("passed") is not True or lin.get("train_commit") != tc
            or lin.get("freeze_commit") != prov.get("freeze_commit")):
        refuse("predict_test.json has no passed code_lineage check (freeze commit descends from C_train with the "
               "model files byte-identical)")
    if prov.get("postproc_git_commit") != prov.get("freeze_commit"):
        refuse(f"the test threshold / parity gate was refit by code {prov.get('postproc_git_commit')!r}, not the "
               f"freeze commit {str(prov.get('freeze_commit'))[:10]}")
    if not prov.get("aggregator_lock_sha256") or prov.get("aggregator") != prov.get("aggregator_locked"):
        refuse("predict_test.json lacks a matching aggregator lock")

    mj = run_dir / "metrics.json"
    if not mj.exists():
        refuse("no training metrics.json (copy it from the run's training output folder)")
    tm = json.loads(mj.read_text())
    if not prov.get("ckpt_sha256") or tm.get("best_ckpt_sha256") != prov.get("ckpt_sha256"):
        refuse(f"training metrics.json best_ckpt_sha256 {str(tm.get('best_ckpt_sha256'))[:12]} != predict_test.json "
               f"ckpt_sha256 {str(prov.get('ckpt_sha256'))[:12]} (metrics of another checkpoint)")
    if tm.get("git_commit") != tc or tm.get("git_commits") != [tc]:
        refuse(f"training metrics.json git_commit {tm.get('git_commit')!r} / git_commits {tm.get('git_commits')!r} "
               f"is not exactly C_train {str(tc)[:10]}: the run was trained (or resumed or finalised) at another "
               "commit")
    recipe = tm.get("recipe")
    if not recipe:
        refuse("training metrics.json records no recipe")
    settings = tm.get("settings")
    if not isinstance(settings, Mapping):
        refuse("training metrics.json records no settings (the training options cannot be checked)")
    if tm.get("allow_unfrozen") is True or settings.get("allow_unfrozen"):
        refuse("training metrics.json says the run was trained with --allow-unfrozen (pilot only, never reportable)")
    pilot = {k: settings.get(k) for k in PILOT_ONLY_SETTINGS if settings.get(k) is not None}
    if pilot:
        refuse(f"training used PILOT ONLY options {pilot} (--max-passes / --max-batches-per-pass): the "
               "pre-registered pass cap is never extended and passes are never truncated")
    stop = tm.get("stop_reason")
    training = {
        "gpu_name": tm.get("gpu_name"),
        "train_seconds_total": tm.get("train_seconds_total"),
        "val_seconds_total": tm.get("val_seconds_total"),
        "passes_run": tm.get("passes_run"),
        "best_pass": tm.get("best_pass"),
        "stop_reason": stop,
        "cap_reached": "pass cap" in str(stop or ""),
        "recipe": recipe.get("name") if isinstance(recipe, Mapping) else recipe,
        "amp": tm.get("amp"),
        "nonfinite_losses": tm.get("nonfinite_losses"),
        "scaler_skipped_steps": tm.get("scaler_skipped_steps"),
    }
    return {"provenance": prov, "training": training, "recipe_full": recipe, "sec_probs": sec_status,
            "train_settings": {k: settings.get(k) for k in SHARED_TRAIN_SETTINGS},
            "train_amp": settings.get("amp", tm.get("amp"))}


def check_across_runs(checked: dict[str, dict[int, dict]]) -> dict:
    """One aggregator / lock / freeze commit / C_train / device type / recipe / training settings
    (TrainSettings.run_fields() minus config, model seed and amp); inference and training AMP
    each fixed per config."""
    flat = [r for seeds in checked.values() for r in seeds.values()]

    def one(what: str, fn) -> object:
        vals = sorted({fn(r) for r in flat}, key=str)
        if len(vals) != 1:
            raise click.ClickException(f"runs disagree on {what}: {vals}; it must be the same for every config "
                                       "and seed")
        return vals[0]

    out = {
        "aggregator": one("aggregator", lambda r: str(r["provenance"].get("aggregator"))),
        "aggregator_lock_sha256": one("aggregator_lock_sha256",
                                      lambda r: str(r["provenance"].get("aggregator_lock_sha256"))),
        "freeze_commit": one("freeze_commit", lambda r: str(r["provenance"]["freeze_commit"])),
        "train_git_commit": one("train_git_commit", lambda r: str(r["provenance"]["train_git_commit"])),
        "device_type": one("device type", lambda r: str(r["provenance"].get("device")).split(":")[0]),
        "recipe": json.loads(one("recipe", lambda r: json.dumps(r["recipe_full"], sort_keys=True))),
        "train_settings": json.loads(one("training settings",
                                         lambda r: json.dumps(r["train_settings"], sort_keys=True))),
        "amp_by_config": {},
        "train_amp_by_config": {},
    }
    for cfg, seeds in checked.items():
        amps = sorted({str(r["provenance"].get("amp")) for r in seeds.values()})
        if len(amps) != 1:
            raise click.ClickException(f"{cfg} runs disagree on amp: {amps}; precision is fixed within a config")
        out["amp_by_config"][cfg] = next(iter(seeds.values()))["provenance"].get("amp")
        tamps = sorted({str(r["train_amp"]) for r in seeds.values()})
        if len(tamps) != 1:
            raise click.ClickException(f"{cfg} runs disagree on training amp: {tamps}; the NaN fallback retrains "
                                       "every seed of a config")
        out["train_amp_by_config"][cfg] = next(iter(seeds.values()))["train_amp"]
    return out


def _flatten(d: Mapping, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (bool, np.bool_)):
            continue
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[key] = float(v)
    return out


def metrics_mean_sd(per_seed: Mapping[int, Mapping]) -> dict:
    """Mean +- SD (ddof 1) over seeds of every numeric leaf (dotted keys) present in every seed."""
    flats = {s: _flatten(m) for s, m in sorted(per_seed.items())}
    keys = sorted(set.intersection(*(set(f) for f in flats.values()))) if flats else []
    out = {}
    for k in keys:
        vals = np.array([flats[s][k] for s in flats])
        out[k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                  "per_seed": {s: flats[s][k] for s in flats}}
    return out


def delta_block(seeds: Mapping[int, dict], rd: dict, name: str, auc_p: bool = True) -> dict:
    """Per-seed and seed-averaged paired deltas (AUC-ROC and AUC-PR) of one config vs one reference.

    p-values only on AUC-ROC of a primary contrast (``auc_p``); AUC-PR deltas and ladder deltas are
    estimation only (point estimate and CI, no p-value), as the pre-registration's Multiplicity says.
    """
    order = sorted(seeds)
    rs = [seeds[s] for s in order]
    return {
        "reference": name,
        "seeds": order,
        "auc_per_seed": [paired_delta(r["boot_auc"], rd["boot_auc"], r["auc"], rd["auc"], with_p=auc_p)
                         for r in rs],
        "auc_seed_averaged": seed_averaged_delta([r["boot_auc"] for r in rs], rd["boot_auc"],
                                                 [r["auc"] for r in rs], rd["auc"], with_p=auc_p),
        "aupr_per_seed": [paired_delta(r["boot_aupr"], rd["boot_aupr"], r["aupr"], rd["aupr"], with_p=False)
                          for r in rs],
        "aupr_seed_averaged": seed_averaged_delta([r["boot_aupr"] for r in rs], rd["boot_aupr"],
                                                  [r["aupr"] for r in rs], rd["aupr"], with_p=False),
    }


def _ci(boot: np.ndarray) -> tuple[float | None, float | None]:
    v = boot[np.isfinite(boot)]
    if not len(v):
        return None, None
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _pairs(items, what: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise click.UsageError(f"{what} {it!r} is not NAME=DIR")
        name, d = it.split("=", 1)
        if name in out:
            raise click.UsageError(f"{what} {name} given twice")
        out[name] = Path(d)
    return out


def _inside(path: Path, root: Path) -> None:
    if root.resolve() not in path.resolve().parents and path.resolve() != root.resolve():
        raise click.ClickException(f"refusing to write outside {root}: {path}")


def archive_previous(folder: Path, names: tuple[str, ...], stamp: str, reason: dict) -> str | None:
    """Move the files ``names`` present in ``folder`` into folder/superseded_<stamp>/ with the reason
    (pre-registration: all versions are kept). Returns the archive folder, or None if nothing existed."""
    present = [n for n in names if (folder / n).exists()]
    if not present:
        return None
    dest, k = folder / f"superseded_{stamp}", 1
    while dest.exists():
        dest, k = folder / f"superseded_{stamp}_{k}", k + 1
    dest.mkdir()
    for n in present:
        (folder / n).rename(dest / n)
    (dest / "superseded_reason.json").write_text(json.dumps(
        {"archived_utc": stamp, **reason, "moved": present}, indent=2))
    return str(dest)


def _same_key(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    cast = {"subject_id": np.int32, "epoch_idx": np.int32, "apnoea_label": np.int8}
    return a[KEY_COLS].reset_index(drop=True).astype(cast).equals(b[KEY_COLS].reset_index(drop=True).astype(cast))


@click.command()
@click.option("--runs-root", type=click.Path(file_okay=False), default=str(DEFAULT_ROOT), show_default=True)
@click.option("--split", "which", type=click.Choice(["test", "val"]), default="test", show_default=True)
@click.option("--config", "configs", multiple=True,
              help="config to evaluate (repeatable; default every discovered config). Test: M and/or P4 only.")
@click.option("--key", type=click.Path(dir_okay=False), default=None,
              help="key parquet (default for test: canonical physio_only test predictions)")
@click.option("--ref", "refs", multiple=True,
              help="NAME=DIR of a LightGBM run with bootstrap arrays. Test: exactly the matched cells the "
                   "evaluated configs need (default models/aim2_matched_cells_v1/<name>/test)")
@click.option("--contrast", "contrasts", multiple=True, help="val only: CONFIG=REFNAME, e.g. M=ecg_only")
@click.option("--gap", "gaps", multiple=True, help="val only: HI,LO=REFHI,REFLO, e.g. P4,M=ecg_belt,ecg_only")
@click.option("--ladder", "ladder", multiple=True,
              help="NAME=DIR of a reference-ladder run (descriptive paired deltas only, no verdict)")
@click.option("--epoch-context", type=click.Path(dir_okay=False), default=None,
              help="test only: per-epoch event context (default models/aim2_analysis_v1/test_epoch_context.parquet; "
                   "sha256 pinned)")
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False), default=str(DEFAULT_CSV), show_default=True)
@click.option("--subject-metadata", type=click.Path(dir_okay=False), default=str(DEFAULT_SM), show_default=True)
@click.option("--n-bootstrap", type=int, default=1000, show_default=True)
@click.option("--prereg", type=click.Path(exists=True, dir_okay=False), default=None,
              help="pre-registration vault note (required for --split test; body checked against the freeze record)")
def main(runs_root, which, configs, key, refs, contrasts, gaps, ladder, epoch_context, csv_path, subject_metadata,
         n_bootstrap, prereg) -> None:
    """Evaluate the runs under --runs-root for the chosen split."""
    root = Path(runs_root)
    test = which == "test"
    if test and (contrasts or gaps):
        raise click.UsageError("--contrast and --gap are not accepted with --split test: the primary contrasts are "
                               f"fixed ({PRIMARY_CONTRASTS}, gap {PRIMARY_GAP})")
    if test and not prereg:
        raise click.UsageError("--split test requires --prereg (test metrics are tied to the pre-registration)")
    if not test and epoch_context:
        raise click.UsageError("--epoch-context is a test-only endpoint")
    prereg_check = None
    if prereg:
        try:
            prereg_check = check_prereg(prereg)
        except PreregMismatch as e:
            raise click.ClickException(str(e)) from e
    freeze_body = prereg_check["prereg_body_sha256"] if prereg_check else None

    # ---- runs
    found = discover_runs(root, which)
    if not found:
        raise click.ClickException(f"no {which} predictions under {root}")
    evaluated = list(dict.fromkeys(configs)) if configs else sorted(found)
    for cfg in evaluated:
        if cfg not in found:
            raise click.ClickException(f"no {which} predictions for config {cfg} under {root}")
    runs = {cfg: found[cfg] for cfg in evaluated}

    checked: dict[str, dict[int, dict]] = {}
    across: dict = {}
    ctx_path = None
    if test:  # all provenance is checked before any metric is computed
        for cfg in evaluated:
            if cfg not in PRIMARY_CONTRASTS:
                raise click.ClickException(f"config {cfg} is outside the pre-registration (only "
                                           f"{sorted(PRIMARY_CONTRASTS)} are evaluated on test); pass --config")
            on_disk = seed_dirs(root / cfg)
            if on_disk != list(REQUIRED_SEEDS) or sorted(runs[cfg]) != list(REQUIRED_SEEDS):
                raise click.ClickException(f"{cfg}: seed folders {on_disk} (with test predictions "
                                           f"{sorted(runs[cfg])}); the pre-registration needs exactly "
                                           f"{list(REQUIRED_SEEDS)}")
        for cfg in evaluated:
            for seed, d in sorted(runs[cfg].items()):
                checked.setdefault(cfg, {})[seed] = test_provenance(d, cfg, seed, freeze_body)
        across = check_across_runs(checked)
        pf = root / PARITY_FAILED_FILE.format(across["device_type"])
        if pf.exists():
            raise click.ClickException(
                f"{pf} records a failed parity gate on {across['device_type']}: the pre-registration abandons that "
                "device for every run, so these test predictions are not accepted (re-score on CPU in fp32)")
        ctx_path = Path(epoch_context) if epoch_context else DEFAULT_EPOCH_CONTEXT
        if not ctx_path.exists():
            raise click.ClickException(f"epoch context {ctx_path} not found (needed for the short-hypopnoea endpoint)")
        ctx_sha = sha256_file(ctx_path)
        if ctx_sha != EPOCH_CONTEXT_SHA256:
            raise click.ClickException(f"epoch context {ctx_path}: sha256 {ctx_sha[:12]} != pinned "
                                       f"{EPOCH_CONTEXT_SHA256[:12]}")

    # ---- references
    ref_paths = _pairs(refs, "--ref")
    if test:
        needed = [PRIMARY_CONTRASTS[c] for c in evaluated]
        if not ref_paths:
            ref_paths = {n: DEFAULT_REF_ROOT / n / "test" for n in needed}
        if set(ref_paths) != set(needed):
            raise click.ClickException(f"--ref names {sorted(ref_paths)} must be exactly the matched cells of the "
                                       f"evaluated configs: {sorted(needed)}")
    ref_data = {name: load_ref(d, which, primary=name if test else None, freeze_body=freeze_body)
                for name, d in ref_paths.items()}
    ladder_data = {name: load_ref(d, which) for name, d in _pairs(ladder, "--ladder").items()}
    for name, rd in {**ref_data, **ladder_data}.items():
        if len(rd["boot_auc"]) != n_bootstrap or len(rd["boot_aupr"]) != n_bootstrap:
            raise click.ClickException(f"reference {name} has {len(rd['boot_auc'])} bootstrap resamples, "
                                       f"not --n-bootstrap {n_bootstrap}")

    # ---- predictions: schema and key equality for every run before anything is written
    if key is None and test:
        key = str(DEFAULT_KEY)
    key_df = pd.read_parquet(key, columns=KEY_COLS) if key else None
    preds: dict[str, dict[int, pd.DataFrame]] = {}
    for cfg in evaluated:
        for seed, d in sorted(runs[cfg].items()):
            pred = pd.read_parquet(d / f"{which}_predictions.parquet")[list(PRED_DTYPES)]
            if key_df is None:
                key_df = pred[KEY_COLS].reset_index(drop=True)
            try:
                check_schema(pred, key_df)
            except ValueError as e:
                raise click.ClickException(f"{d}: {e}") from e
            preds.setdefault(cfg, {})[seed] = pred
    for name, rd in {**ref_data, **ladder_data}.items():
        if not _same_key(rd["key"], key_df):
            raise click.ClickException(f"reference {name} ({rd['dir']}) rows differ from the key / DL predictions")
    sh_mask = sh_counts = None
    if test:
        try:
            sh_mask, sh_counts = short_hypopnoea_mask(pd.read_parquet(ctx_path), key_df)
        except ValueError as e:
            raise click.ClickException(f"epoch context {ctx_path}: {e}") from e

    # ---- NSRR AHI completeness and output folders, before any metric is computed or written
    nsrr = load_nsrr_ahi(csv_path)
    sm = pd.read_parquet(subject_metadata) if Path(subject_metadata).exists() else None
    meta = metadata_with_ahi(sm, nsrr)
    subjects = np.unique(key_df["subject_id"].to_numpy(np.int64))  # every run's subjects (key equality above)
    ahi = pd.Index(subjects).map(meta.set_index("subject_id")["ahi_a0h3a"])
    no_ahi = subjects[pd.isna(np.asarray(ahi, dtype=float))]
    if len(no_ahi):
        raise click.ClickException(f"{len(subjects) - len(no_ahi)} of {len(subjects)} subjects have ahi_a0h3a "
                                   f"(missing: {no_ahi[:5].tolist()}); every subject needs the NSRR AHI")
    for cfg in evaluated:
        for d in runs[cfg].values():
            _inside(d if test else d / "val_eval", root)

    # ---- test: earlier evaluate outputs are moved aside (never overwritten) before anything is written
    superseded: dict = {"summary": None, "runs": {}}
    if test:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        why = "evaluate_aim2_dl.py --split test was run again; the earlier output is kept, not overwritten"
        _inside(root / f"evaluation_summary_{which}.json", root)
        superseded["summary"] = archive_previous(root, (f"evaluation_summary_{which}.json",), stamp,
                                                 {"reason": why})
        for cfg in evaluated:
            for seed, d in sorted(runs[cfg].items()):
                a = archive_previous(d, EVAL_RUN_OUTPUTS, stamp, {
                    "reason": why + " (computed from the predictions then in this folder)",
                    "predict_test_sha256": sha256_file(d / "predict_test.json")})
                if a:
                    superseded["runs"].setdefault(cfg, {})[seed] = a

    # ---- metrics
    sid = key_df["subject_id"].to_numpy()
    y = key_df["apnoea_label"].to_numpy(np.int8)

    def short_hyp(p: np.ndarray) -> dict:
        boot = subset_bootstrap_auc(sid, y, p, sh_mask, n_resamples=n_bootstrap, seed=42)
        lo, hi = _ci(boot)
        return {"auc": float(roc_auc_score(y[sh_mask], p[sh_mask])), "boot_auc": boot, "ci_low": lo, "ci_high": hi}

    per_run: dict[str, dict[int, dict]] = {}
    for cfg in evaluated:
        for seed, d in sorted(runs[cfg].items()):
            pred = preds[cfg][seed]
            out_dir = d if test else d / "val_eval"
            _inside(out_dir, root)
            m = write_extended_metrics(out_dir, pred, subject_metadata=meta, n_bootstrap_subj=n_bootstrap)
            n_subj = int(pred["subject_id"].nunique())
            if m["n_subjects_with_nsrr_ahi"] != n_subj:
                raise click.ClickException(f"{d}: {m['n_subjects_with_nsrr_ahi']} of {n_subj} subjects have ahi_a0h3a")
            rec = {"dir": str(d), "auc": m["auc_roc"], "aupr": m["auc_pr"],
                   "boot_auc": np.load(out_dir / "bootstrap_aucs_subject.npy"),
                   "boot_aupr": np.load(out_dir / "bootstrap_auprs_subject.npy"), "metrics": m}
            if test:
                c = checked[cfg][seed]
                rec |= {"provenance": c["provenance"], "training": c["training"], "sec_probs": c["sec_probs"],
                        "short_hyp": short_hyp(pred["pred_prob"].to_numpy(np.float64))}
            per_run.setdefault(cfg, {})[seed] = rec
            click.echo(f"{cfg} seed{seed}: AUC {m['auc_roc']:.4f} AUC-PR {m['auc_pr']:.4f}")

    summary: dict = {
        "split": which, "runs_root": str(root.resolve()), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "key": key, "key_sha256": sha256_file(key) if key else None, "n_bootstrap": n_bootstrap,
        "prereg": prereg_check, "prereg_sha256": prereg_check["prereg_sha256"] if prereg_check else None,
        "evaluated_configs": evaluated, "refs": {}, "configs": {}, "ladder": {},
    }
    if test:
        summary |= {"required_seeds": list(REQUIRED_SEEDS), "freeze_commits": [across["freeze_commit"]],
                    **across, "multiplicity": MULTIPLICITY, "superseded_previous": superseded}
    for name, rd in ref_data.items():
        summary["refs"][name] = {k: rd[k] for k in ("dir", "auc", "aupr", "predictions_sha256", "model_sha256",
                                                    "prereg_body_sha256") if k in rd}

    skip = ("boot_auc", "boot_aupr", "short_hyp")
    wanted: dict[str, list[str]] = {}
    for c in contrasts:
        cfg, rname = c.split("=", 1)
        wanted.setdefault(cfg, []).append(rname)
    for cfg, seeds in per_run.items():
        aucs = np.array([seeds[s]["auc"] for s in sorted(seeds)])
        auprs = np.array([seeds[s]["aupr"] for s in sorted(seeds)])
        cs = {
            "seeds": sorted(seeds),
            "per_seed": {s: {k: v for k, v in r.items() if k not in skip} for s, r in sorted(seeds.items())},
            "auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
            "aupr_mean": float(auprs.mean()), "aupr_sd": float(auprs.std(ddof=1)) if len(auprs) > 1 else None,
            "metrics_mean_sd": metrics_mean_sd({s: r["metrics"] for s, r in seeds.items()}),
            "contrasts": {},
        }
        if test:
            cs["training_cap_reached"] = {s: r["training"]["cap_reached"] for s, r in sorted(seeds.items())}
        names = [PRIMARY_CONTRASTS[cfg]] if test else wanted.get(cfg, list(ref_data))
        for rname in names:
            if rname not in ref_data:
                raise click.ClickException(f"contrast reference {rname} not given with --ref")
            blk = delta_block(seeds, ref_data[rname], rname)
            blk["claim_rule_auc"] = claim_rule(blk["auc_per_seed"], blk["auc_seed_averaged"])
            cs["contrasts"][rname] = blk
        summary["configs"][cfg] = cs

    # ---- gap (contrast 3)
    def gap(hi: str, lo: str, rhi: str, rlo: str) -> dict:
        for r in (rhi, rlo):
            if r not in ref_data:
                raise click.ClickException(f"gap reference {r} not given with --ref")
        for c in (hi, lo):
            if c not in per_run:
                raise click.ClickException(f"gap config {c} not evaluated")
        try:
            return gap_per_seed(per_run[hi], per_run[lo], ref_data[rhi], ref_data[rlo], hi_name=hi, lo_name=lo)
        except ValueError as e:
            raise click.ClickException(f"gap {hi},{lo}: {e}") from e

    if test:
        hi, lo, rhi, rlo = PRIMARY_GAP
        summary["gaps"] = ({f"{hi},{lo}={rhi},{rlo}": gap(hi, lo, rhi, rlo)} if {hi, lo} <= set(per_run)
                           else f"skipped: needs {lo} and {hi}")
    else:
        summary["gaps"] = {}
        for g in gaps:
            lhs, rhs = g.split("=", 1)
            hi, lo = lhs.split(",")
            rhi, rlo = rhs.split(",")
            summary["gaps"][g] = gap(hi, lo, rhi, rlo)

    # ---- ladder (descriptive only)
    for name, rd in ladder_data.items():
        summary["ladder"][name] = {
            "dir": rd["dir"], "auc": rd["auc"], "aupr": rd["aupr"], "predictions_sha256": rd["predictions_sha256"],
            "note": "descriptive only: no claim rule, no p-value",
            "configs": {cfg: delta_block(seeds, rd, name, auc_p=False) for cfg, seeds in per_run.items()},
        }

    # ---- key secondary: short-hypopnoea AUC (estimation only)
    if test:
        ref_sh = {name: short_hyp(rd["pred_prob"]) for name, rd in ref_data.items()}
        lad_sh = {name: short_hyp(rd["pred_prob"]) for name, rd in ladder_data.items()}
        sh_cfg = {}
        for cfg, seeds in per_run.items():
            order = sorted(seeds)
            vals = np.array([seeds[s]["short_hyp"]["auc"] for s in order])
            rname = PRIMARY_CONTRASTS[cfg]
            rs = ref_sh[rname]
            sh_cfg[cfg] = {
                "seeds": order,
                "auc_per_seed": {s: seeds[s]["short_hyp"]["auc"] for s in order},
                "ci_per_seed": {s: [seeds[s]["short_hyp"]["ci_low"], seeds[s]["short_hyp"]["ci_high"]] for s in order},
                "auc_mean": float(vals.mean()), "auc_sd": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                "reference": rname, "reference_auc": rs["auc"],
                "delta_per_seed": [paired_delta(seeds[s]["short_hyp"]["boot_auc"], rs["boot_auc"],
                                                seeds[s]["short_hyp"]["auc"], rs["auc"], with_p=False)
                                   for s in order],
                "delta_seed_averaged": seed_averaged_delta([seeds[s]["short_hyp"]["boot_auc"] for s in order],
                                                           rs["boot_auc"], list(vals), rs["auc"], with_p=False),
                "note": "estimation only: no verdict, no p-value",
            }

        def strip(d: dict) -> dict:
            return {k: {kk: vv for kk, vv in v.items() if kk != "boot_auc"} for k, v in d.items()}

        summary["short_hypopnoea"] = {
            "definition": SHORT_HYP_DEFINITION, **sh_counts, "context": str(ctx_path),
            "context_sha256": EPOCH_CONTEXT_SHA256, "configs": sh_cfg, "refs": strip(ref_sh), "ladder": strip(lad_sh),
        }
        summary["expectations"] = {}
        if "M" in sh_cfg:
            v = sh_cfg["M"]["auc_mean"]
            summary["expectations"]["E1"] = {"statistic": "M seed-averaged short-hypopnoea AUC", "value": v,
                                             "bound": E1_BOUND, "met": bool(v < E1_BOUND)}

    out = root / f"evaluation_summary_{which}.json"
    _inside(out, root)
    out.write_text(json.dumps(summary, indent=2, default=float))
    click.echo(f"summary -> {out}")


if __name__ == "__main__":
    main()
