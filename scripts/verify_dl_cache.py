"""Verify the stage-A DL cache: integrity, label parity with Aim 2, and a QC summary.

Label checks only; nothing here scores a model. For every cached subject:

  * the npz sha256 matches its sidecar (--verify-sha, default on);
  * n_epochs == subject_metadata.n_epochs;
  * sleep-epoch count == subject_metadata.tst_min * 2;
  * sleep-epoch apnoea label sum == subject_metadata.n_apnoea_epochs;
  * TEST subjects: the cached sleep rows (epoch_idx, apnoea_label) equal the
    canonical key file's rows exactly (the key columns are identical across
    the Aim 2 physio_only / full / aasm_only / exp2 prediction files).

Also prints per-split coverage (intact cached subjects / split size), checks
that the ledger's QC flags match the flags recomputed from the sidecars, and
summarises airflow picks, SpO2 / H.R. encodings, QC flags and disk use.

--require-complete val,test turns coverage into a pass condition (the design's
G1 / G3 gate: every val and test subject in stage A); without it a partial
cache can pass. --fix-ledger-flags appends corrected records for any ledger
flag drift (the only write this script ever makes).

Usage:
  python scripts/verify_dl_cache.py
  python scripts/verify_dl_cache.py --require-complete val,test
  python scripts/verify_dl_cache.py --report /private/tmp/claude-501/dl_build/parity.csv
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import click
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline import dl_cache, splits  # noqa: E402

DEFAULT_OUT = Path.home() / "thesis_dl_cache" / "raw_v1"
DEFAULT_SPLIT = CODE_ROOT / "splits" / "aim2_seed42.json"
DEFAULT_METADATA = CODE_ROOT / "features" / "subject_metadata.parquet"
DEFAULT_KEY = (CODE_ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only"
               / "test_predictions.parquet")


@click.command()
@click.option("--cache", type=click.Path(path_type=Path, exists=True), default=DEFAULT_OUT, show_default=True)
@click.option("--split", "split_path", type=click.Path(path_type=Path, exists=True), default=DEFAULT_SPLIT,
              show_default=True)
@click.option("--metadata", type=click.Path(path_type=Path, exists=True), default=DEFAULT_METADATA,
              show_default=True)
@click.option("--key", type=click.Path(path_type=Path, exists=True), default=DEFAULT_KEY, show_default=True)
@click.option("--verify-sha/--no-verify-sha", default=True, show_default=True)
@click.option("--report", type=click.Path(path_type=Path), default=None, help="Write the per-subject table (CSV).")
@click.option("--require-complete", default="", show_default=True,
              help="Comma-separated splits (e.g. val,test) whose every subject must be cached and intact.")
@click.option("--fix-ledger-flags", is_flag=True,
              help="Append corrected ledger records where the ledger's flags differ from the sidecars'.")
def main(cache: Path, split_path: Path, metadata: Path, key: Path, verify_sha: bool, report: Path | None,
         require_complete: str, fix_ledger_flags: bool) -> None:
    required = [x.strip() for x in require_complete.split(",") if x.strip()]
    if any(x not in splits.SPLIT_NAMES for x in required):
        raise click.UsageError(f"--require-complete: unknown split in {required}")
    split = splits.load_split(split_path)
    s2n = splits.subject_to_split(split)
    meta = splits.read_subject_metadata(metadata)
    keydf = pd.read_parquet(key, columns=["subject_id", "epoch_idx", "apnoea_label"])
    sids = sorted(int(p.stem.split("-")[1]) for p in cache.glob("shhs1-*.json"))
    click.echo(f"cache {cache}: {len(sids)} sidecars")

    bad_integrity = [s for s in sids if not dl_cache.is_complete(cache, s, verify=verify_sha)]
    rep = dl_cache.label_parity(cache, sids, meta, keydf)
    rep["split_expected"] = rep["subject_id"].map(s2n)
    rep["split_ok"] = rep["split"] == rep["split_expected"]

    side = []
    for s in sids:
        m = dl_cache.load_meta(cache, s)
        q = m["qc"]
        side.append({
            "subject_id": s,
            "npz_bytes": m["npz"]["bytes"],
            "airflow": m["airflow"]["chosen_label"],
            "airflow_ok": m["airflow"]["ok"],
            "n_airflow_candidates": len(m["airflow"]["candidates"]),
            "airflow_std_lsb": m["airflow"]["std_lsb"],
            "first_exact_match_dead": _first_match_dead(m),
            "spo2_encoding": m["channels"]["spo2"]["encoding"],
            "hr_encoding": m["channels"]["hr"]["encoding"],
            "ecg_encoding": m["channels"]["ecg"]["encoding"],
            "ecg_std_lsb": q.get("ecg_std_lsb"),
            "ecg_clip_frac": q.get("ecg_clip_frac"),
            "missing": ",".join(q.get("missing_channels", [])),
            "notes": " | ".join(q.get("notes", [])),
            "tail_s": m["tail_seconds_dropped"],
        })
    rep = rep.merge(pd.DataFrame(side), on="subject_id")

    click.secho("\n== integrity", bold=True)
    click.echo(f"  npz/sidecar complete{' + sha256' if verify_sha else ''}: {len(sids) - len(bad_integrity)}"
               f"/{len(sids)}" + (f"  BAD: {bad_integrity[:10]}" if bad_integrity else ""))
    click.echo(f"  sidecar split == frozen split: {int(rep['split_ok'].sum())}/{len(rep)}")

    click.secho("\n== coverage (intact cached subjects / split size)", bold=True)
    intact = np.asarray(sorted(set(sids) - set(bad_integrity)), dtype=np.int64)
    incomplete = []
    for name in splits.SPLIT_NAMES:
        want = np.asarray(split[name], dtype=np.int64)
        miss = want[~np.isin(want, intact)]
        req = name in required
        if req and miss.size:
            incomplete.append(name)
        click.echo(f"  {name:5s} {want.size - miss.size:5d}/{want.size:<5d}"
                   + (" [required]" if req else "")
                   + (f"  missing {miss.size}, e.g. {miss[:10].tolist()}" if miss.size else "  complete"))
    if incomplete:
        click.secho(f"  required split(s) incomplete: {incomplete}", fg="red")

    click.secho("\n== ledger flags vs sidecars", bold=True)
    ledger = dl_cache.Ledger(cache / "ledger.jsonl")
    drift = dl_cache.ledger_flag_drift(cache, ledger)
    click.echo(f"  latest ok records whose flags differ from the sidecar's: {len(drift)}"
               + (f" e.g. {[(d['subject_id'], d['ledger_flags'], d['sidecar_flags']) for d in drift[:3]]}"
                  if drift else ""))
    if drift and fix_ledger_flags:
        fixed = dl_cache.reflag_ledger(cache, ledger)
        drift = dl_cache.ledger_flag_drift(cache, ledger)
        click.echo(f"  appended {len(fixed)} corrected record(s); drift now {len(drift)}")
    elif drift:
        click.secho("  (audit trail only; rerun with --fix-ledger-flags to append corrected records)", fg="yellow")

    click.secho("\n== label parity (per split)", bold=True)
    for name in splits.SPLIT_NAMES:
        r = rep[rep["split_expected"] == name]
        if not len(r):
            continue
        line = (f"  {name:5s} n={len(r):5d}  n_epochs {int(r.n_epochs_ok.sum())}/{len(r)}  "
                f"sleep {int(r.n_sleep_ok.sum())}/{len(r)}  apnoea_sum {int(r.apnoea_sum_ok.sum())}/{len(r)}")
        if name == "test":
            k = r["key_rows_ok"].eq(True)
            nrows = int(r["n_key_rows"].sum())
            key_rows = int(keydf["subject_id"].isin(r["subject_id"]).sum())
            line += f"  key_rows {int(k.sum())}/{len(r)} ({nrows:,} rows; key has {key_rows:,} for these ids)"
        click.echo(line)
    fails = rep[~(rep.n_epochs_ok & rep.n_sleep_ok & rep.apnoea_sum_ok)
                | ((rep.split_expected == "test") & ~rep.key_rows_ok.eq(True))]
    click.echo(f"  subjects failing any label check: {len(fails)}"
               + (f" e.g. {fails.subject_id.head(10).tolist()}" if len(fails) else ""))
    nontest_in_key = set(rep.loc[rep.split_expected != "test", "subject_id"]) & set(keydf.subject_id.unique())
    click.echo(f"  non-test cached subjects present in the test key: {len(nontest_in_key)}")

    click.secho("\n== channels / QC", bold=True)
    click.echo(f"  airflow picked: {dict(Counter(rep.airflow.fillna('NONE')))}")
    click.echo(f"  airflow_ok: {int(rep.airflow_ok.sum())}/{len(rep)}; subjects with >1 candidate: "
               f"{int((rep.n_airflow_candidates > 1).sum())}; no live candidate: {int((~rep.airflow_ok).sum())}")
    click.echo(f"  first-exact-match picker (process_batch.py:58) would take a dead channel while a live one "
               f"exists: {int(rep.first_exact_match_dead.sum())}/{len(rep)}")
    click.echo(f"  spo2 encoding: {dict(Counter(rep.spo2_encoding))}; hr encoding: {dict(Counter(rep.hr_encoding))}; "
               f"ecg encoding: {dict(Counter(rep.ecg_encoding))}")
    click.echo(f"  ECG std < 5 LSB: {int((rep.ecg_std_lsb < 5).sum())}; ECG clip frac > 1%: "
               f"{int((rep.ecg_clip_frac > 0.01).sum())}; ECG std < 0.5 LSB (flat): {int((rep.ecg_std_lsb < 0.5).sum())}")
    click.echo(f"  missing channels: {dict(Counter(rep.missing.replace('', 'none')))}")
    click.echo(f"  header notes: {int((rep.notes != '').sum())} subject(s)"
               + (f" e.g. {rep.loc[rep.notes != '', 'notes'].head(3).tolist()}" if (rep.notes != '').any() else ""))

    click.secho("\n== disk", bold=True)
    npz = int(rep.npz_bytes.sum())
    js = sum(p.stat().st_size for p in cache.glob("shhs1-*.json"))
    click.echo(f"  npz {npz / 1e9:.3f} GB + json {js / 1e6:.1f} MB over {len(rep)} subjects "
               f"(mean {npz / max(len(rep), 1) / 1e6:.2f} MB/subject; x5,793 = {npz / max(len(rep), 1) * 5793 / 1e9:.1f} GB)")
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(report, index=False)
        click.echo(f"  per-subject table -> {report}")
    ok = not bad_integrity and len(fails) == 0 and rep.split_ok.all() and not nontest_in_key and not incomplete
    click.secho(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'}", fg="green" if ok else "red", bold=True)
    sys.exit(0 if ok else 1)


def _first_match_dead(m: dict) -> bool:
    """Would the Aim 1 picker (first of AIRFLOW, NEW AIR by label; case-sensitive) take a dead
    channel while a live candidate exists?"""
    cands = m["airflow"]["candidates"]
    live = [c for c in cands if c["std_lsb"] > 5]
    if not live:
        return False
    for alias in ("AIRFLOW", "NEW AIR", "nasal"):
        hit = [c for c in cands if c["label"] == alias]
        if len(hit) == 1:  # mne renames duplicated labels (AIRFLOW-0/-1), so a duplicate never matches
            return hit[0]["std_lsb"] <= 5
    return True  # the Aim 1 picker finds nothing (e.g. 'New Air', 'NEWAIR', 'AUX') although a live one exists


if __name__ == "__main__":
    main()
