"""Evaluate Aim 2 Olsen BiGRU runs: extended metrics, paired deltas, seed aggregation.

For every run folder ``<runs-root>/<config>/seed<k>/`` holding predictions:
  1. schema check (dtypes, order, no NaN) and ``DataFrame.equals`` of the key columns
     against the key file (test: the canonical physio_only test_predictions.parquet);
  2. ``thesis_pipeline.extended_metrics.write_extended_metrics`` with subject metadata
     merged with NSRR ``ahi_a0h3a`` from Dataset/shhs/csv/shhs1-dataset-0.15.0.csv
     (read-only), asserting every subject has an NSRR AHI;
  3. paired deltas against saved LightGBM bootstrap arrays (``--ref NAME=DIR``), valid
     because ``write_extended_metrics`` resamples the same subjects with the same RNG;
  4. per config: mean +- SD (ddof = 1) over seeds, the seed-averaged delta and the
     pre-declared claim rule; optional gap contrasts (``--gap P4,M=ecg_belt,ecg_only``).

Outputs stay under ``--runs-root`` (default Code/models/aim2_dl_olsen_v1). Test metrics
are computed only here and only after the training freeze; ``--split val`` runs the same
path on validation predictions (written to <run>/val_eval/, no pairing unless refs with
validation bootstrap arrays are given).

Test provenance (``--split test``): ``--prereg`` is required and its sha256 logged. Every
run folder must hold the predict_test.json written by ``predict_aim2_dl.py --split test``
with test_predictions_sha256 == sha256(test_predictions.parquet), a freeze_commit, a
prereg_sha256 equal to the given pre-registration and an aggregator equal to its
locked aggregator; the aggregator (and its lock file) must be the same for every run.
Anything else is refused. The per-run provenance is copied into the summary.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.dl_eval import (  # noqa: E402
    KEY_COLS,
    PRED_DTYPES,
    check_schema,
    claim_rule,
    gap_contrast,
    load_nsrr_ahi,
    metadata_with_ahi,
    paired_delta,
    seed_averaged_delta,
)
from thesis_pipeline.dl_stage_b import sha256_file  # noqa: E402
from thesis_pipeline.extended_metrics import write_extended_metrics  # noqa: E402

DEFAULT_ROOT = CODE_ROOT / "models" / "aim2_dl_olsen_v1"
DEFAULT_KEY = CODE_ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only" / \
    "test_predictions.parquet"
DEFAULT_CSV = CODE_ROOT.parent / "Dataset" / "shhs" / "csv" / "shhs1-dataset-0.15.0.csv"
DEFAULT_SM = CODE_ROOT / "features" / "subject_metadata.parquet"


def discover_runs(root: Path, which: str) -> dict[str, dict[int, Path]]:
    fname = "test_predictions.parquet" if which == "test" else "val_predictions.parquet"
    runs: dict[str, dict[int, Path]] = {}
    for p in sorted(root.glob(f"*/seed*/{fname}")):
        m = re.fullmatch(r"seed(\d+)", p.parent.name)
        if m:
            runs.setdefault(p.parent.parent.name, {})[int(m.group(1))] = p.parent
    return runs


def load_ref(d: Path, which: str) -> dict:
    boots = {"auc": d / "bootstrap_aucs_subject.npy", "aupr": d / "bootstrap_auprs_subject.npy"}
    for p in boots.values():
        if not p.exists():
            raise click.ClickException(f"reference {d} lacks {p.name}")
    pred = pd.read_parquet(d / f"{which}_predictions.parquet")
    me = d / "metrics_extended.json"
    if me.exists():
        m = json.loads(me.read_text())
        auc, aupr = m["auc_roc"], m["auc_pr"]
    else:
        auc = float(roc_auc_score(pred["apnoea_label"], pred["pred_prob"]))
        aupr = float(average_precision_score(pred["apnoea_label"], pred["pred_prob"]))
    return {"dir": str(d), "auc": auc, "aupr": aupr, "boot_auc": np.load(boots["auc"]),
            "boot_aupr": np.load(boots["aupr"]), "key": pred[KEY_COLS]}


PROV_KEYS = ("freeze_tag", "freeze_commit", "git_commit", "prereg_sha256", "ckpt_sha256", "postproc_sha256",
             "aggregator", "aggregator_lock_sha256", "threshold", "device", "amp", "split_sha256", "key_sha256",
             "data_manifest_sha256", "test_ids_sha256", "test_predictions_sha256", "rerun_reason",
             "replaced_previous", "started")


def test_provenance(run_dir: Path, prereg_sha: str) -> dict:
    """predict_test.json of a run, checked against its parquet and the pre-registration."""
    pj = run_dir / "predict_test.json"
    if not pj.exists():
        raise click.ClickException(f"{run_dir}: no predict_test.json; test predictions must come from "
                                   "predict_aim2_dl.py --split test")
    prov = json.loads(pj.read_text())
    got = sha256_file(run_dir / "test_predictions.parquet")
    if prov.get("test_predictions_sha256") != got:
        raise click.ClickException(f"{run_dir}: test_predictions.parquet sha256 {got[:12]} != predict_test.json "
                                   f"{str(prov.get('test_predictions_sha256'))[:12]} (file replaced or edited)")
    if not prov.get("freeze_commit"):
        raise click.ClickException(f"{run_dir}: predict_test.json has no freeze_commit")
    if prov.get("prereg_sha256") != prereg_sha:
        raise click.ClickException(f"{run_dir}: predictions were made under pre-registration "
                                   f"{str(prov.get('prereg_sha256'))[:12]}, not the given one {prereg_sha[:12]}")
    if prov.get("split") != "test" or not prov.get("aggregator_lock_sha256") \
            or prov.get("aggregator") != prov.get("aggregator_locked"):
        raise click.ClickException(f"{run_dir}: predict_test.json lacks a matching aggregator lock")
    return {k: prov.get(k) for k in PROV_KEYS}


def _inside(path: Path, root: Path) -> None:
    if root.resolve() not in path.resolve().parents and path.resolve() != root.resolve():
        raise click.ClickException(f"refusing to write outside {root}: {path}")


@click.command()
@click.option("--runs-root", type=click.Path(file_okay=False), default=str(DEFAULT_ROOT), show_default=True)
@click.option("--split", "which", type=click.Choice(["test", "val"]), default="test", show_default=True)
@click.option("--key", type=click.Path(dir_okay=False), default=None,
              help="key parquet (default for test: canonical physio_only test predictions)")
@click.option("--ref", "refs", multiple=True, help="NAME=DIR of a LightGBM run with bootstrap arrays")
@click.option("--contrast", "contrasts", multiple=True, help="CONFIG=REFNAME, e.g. M=ecg_only")
@click.option("--gap", "gaps", multiple=True, help="HI,LO=REFHI,REFLO, e.g. P4,M=ecg_belt,ecg_only")
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False), default=str(DEFAULT_CSV), show_default=True)
@click.option("--subject-metadata", type=click.Path(dir_okay=False), default=str(DEFAULT_SM), show_default=True)
@click.option("--n-bootstrap", type=int, default=1000, show_default=True)
@click.option("--prereg", type=click.Path(exists=True, dir_okay=False), default=None,
              help="pre-registration vault note (required for --split test; sha256 checked per run)")
def main(runs_root, which, key, refs, contrasts, gaps, csv_path, subject_metadata, n_bootstrap, prereg) -> None:
    """Evaluate every run under --runs-root for the chosen split."""
    root = Path(runs_root)
    if which == "test" and not prereg:
        raise click.UsageError("--split test requires --prereg (test metrics are tied to the pre-registration)")
    runs = discover_runs(root, which)
    if not runs:
        raise click.ClickException(f"no {which} predictions under {root}")
    prereg_sha = sha256_file(prereg) if prereg else None
    provs: dict[str, dict[int, dict]] = {}
    if which == "test":  # all provenance is checked before any metric is computed
        for cfg, seeds in sorted(runs.items()):
            for seed, d in sorted(seeds.items()):
                provs.setdefault(cfg, {})[seed] = test_provenance(d, prereg_sha)
        flat = [p for s in provs.values() for p in s.values()]
        for k in ("aggregator", "aggregator_lock_sha256"):
            vals = sorted({str(p[k]) for p in flat})
            if len(vals) > 1:
                raise click.ClickException(f"runs disagree on {k}: {vals}; the aggregator is frozen for every "
                                           "config and seed")
    if key is None and which == "test":
        key = str(DEFAULT_KEY)
    key_df = pd.read_parquet(key, columns=KEY_COLS) if key else None

    nsrr = load_nsrr_ahi(csv_path)
    sm = pd.read_parquet(subject_metadata) if Path(subject_metadata).exists() else None
    meta = metadata_with_ahi(sm, nsrr)

    ref_data = {}
    for r in refs:
        name, d = r.split("=", 1)
        ref_data[name] = load_ref(Path(d), which)

    summary: dict = {"split": which, "runs_root": str(root.resolve()), "key": key, "refs": {},
                     "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "configs": {},
                     "prereg": str(Path(prereg).resolve()) if prereg else None, "prereg_sha256": prereg_sha,
                     "key_sha256": sha256_file(key) if key else None}
    if provs:
        summary["freeze_commits"] = sorted({p["freeze_commit"] for s in provs.values() for p in s.values()})
    for name, rd in ref_data.items():
        summary["refs"][name] = {"dir": rd["dir"], "auc": rd["auc"], "aupr": rd["aupr"]}

    per_run: dict[str, dict[int, dict]] = {}
    for cfg, seeds in sorted(runs.items()):
        for seed, d in sorted(seeds.items()):
            pred = pd.read_parquet(d / f"{which}_predictions.parquet")
            pred = pred[list(PRED_DTYPES)]
            ref_key = key_df if key_df is not None else pred[KEY_COLS]
            if key_df is None and which == "val":
                key_df = pred[KEY_COLS]
            check_schema(pred, ref_key)
            out_dir = d if which == "test" else d / "val_eval"
            _inside(out_dir, root)
            m = write_extended_metrics(out_dir, pred, subject_metadata=meta, n_bootstrap_subj=n_bootstrap)
            n_subj = int(pred["subject_id"].nunique())
            if m["n_subjects_with_nsrr_ahi"] != n_subj:
                raise click.ClickException(
                    f"{d}: {m['n_subjects_with_nsrr_ahi']} of {n_subj} subjects have ahi_a0h3a")
            boot_auc = np.load(out_dir / "bootstrap_aucs_subject.npy")
            boot_aupr = np.load(out_dir / "bootstrap_auprs_subject.npy")
            rec = {"dir": str(d), "auc": m["auc_roc"], "aupr": m["auc_pr"], "boot_auc": boot_auc,
                   "boot_aupr": boot_aupr, "metrics": m, "deltas": {},
                   "provenance": provs.get(cfg, {}).get(seed)}
            for rname, rd in ref_data.items():
                if not rd["key"].reset_index(drop=True).astype(
                        {"subject_id": np.int32, "epoch_idx": np.int32, "apnoea_label": np.int8}).equals(
                        pred[KEY_COLS].reset_index(drop=True)):
                    raise click.ClickException(f"reference {rname} rows differ from {d}")
                rec["deltas"][rname] = {
                    "auc": paired_delta(boot_auc, rd["boot_auc"], m["auc_roc"], rd["auc"]),
                    "aupr": paired_delta(boot_aupr, rd["boot_aupr"], m["auc_pr"], rd["aupr"]),
                }
            per_run.setdefault(cfg, {})[seed] = rec
            click.echo(f"{cfg} seed{seed}: AUC {m['auc_roc']:.4f} AUC-PR {m['auc_pr']:.4f}")

    wanted = {}
    for c in contrasts:
        cfg, rname = c.split("=", 1)
        wanted.setdefault(cfg, []).append(rname)
    for cfg, seeds in per_run.items():
        aucs = np.array([r["auc"] for r in seeds.values()])
        auprs = np.array([r["aupr"] for r in seeds.values()])
        cs = {
            "seeds": sorted(seeds),
            "per_seed": {s: {k: v for k, v in r.items() if k not in ("boot_auc", "boot_aupr", "metrics")}
                         | {"metrics": r["metrics"]} for s, r in seeds.items()},
            "auc_mean": float(aucs.mean()), "auc_sd": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
            "aupr_mean": float(auprs.mean()), "aupr_sd": float(auprs.std(ddof=1)) if len(auprs) > 1 else None,
            "contrasts": {},
        }
        for rname in wanted.get(cfg, list(ref_data)):
            if rname not in ref_data:
                raise click.ClickException(f"contrast reference {rname} not given with --ref")
            rd = ref_data[rname]
            per_seed = [r["deltas"][rname]["auc"] for r in seeds.values()]
            cs["contrasts"][rname] = {
                "auc_per_seed": per_seed,
                "auc_seed_averaged": seed_averaged_delta(
                    [r["boot_auc"] for r in seeds.values()], rd["boot_auc"], list(aucs), rd["auc"]),
                "aupr_seed_averaged": seed_averaged_delta(
                    [r["boot_aupr"] for r in seeds.values()], rd["boot_aupr"], list(auprs), rd["aupr"]),
                "claim_rule_auc": claim_rule(per_seed),
            }
        summary["configs"][cfg] = cs
    summary["gaps"] = {}
    for g in gaps:
        lhs, rhs = g.split("=", 1)
        hi, lo = lhs.split(",")
        rhi, rlo = rhs.split(",")
        summary["gaps"][g] = gap_contrast(
            [r["boot_auc"] for r in per_run[hi].values()], [r["boot_auc"] for r in per_run[lo].values()],
            ref_data[rhi]["boot_auc"], ref_data[rlo]["boot_auc"])
    out = root / f"evaluation_summary_{which}.json"
    _inside(out, root)
    out.write_text(json.dumps(summary, indent=2, default=float))
    click.echo(f"summary -> {out}")


if __name__ == "__main__":
    main()
