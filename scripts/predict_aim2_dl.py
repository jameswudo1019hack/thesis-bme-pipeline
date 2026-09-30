"""Epoch predictions from a trained Aim 2 Olsen BiGRU, in the test_predictions schema.

``--prereg`` (the vault pre-registration note) is required for every run and its
sha256 is logged. Two modes:

--split val   exercises the whole path on validation data (allowed at any time). With
              ``--refit-postproc`` it refits the threshold on these predictions (same
              device as the later test inference, critique #3), keeping the frozen
              aggregator, and writes postproc_<device>.json.
--split test  run ONCE per (config, seed) after the code freeze, on the Mac: needs
              ``--freeze-tag`` naming a git tag equal to HEAD on a clean tree, opens
              the test folder explicitly, predicts all 1,159 test subjects, checks the
              key columns ``DataFrame.equals`` the canonical key file, and writes
              test_predictions.parquet plus predict_test.json provenance. It computes
              no metrics; ``evaluate_aim2_dl.py`` does that.

Postproc guards (every run): ``postproc["best_ckpt_sha256"]`` must equal the sha256 of
``--ckpt`` (no other seed's threshold). With ``--aggregator-lock`` (required for test)
the postproc aggregator must equal the locked one (AGGREGATOR_LOCK.json, chosen once on
M seed-42 validation). For test the postproc must have been fitted on this device type
(the postproc_<device>.json written by ``--split val --refit-postproc``).

Once-only guard (test): any earlier test output in ``--out-dir`` (test_predictions.parquet,
predict_test.json, a superseded_* archive, or an attempt that reached inference in
predict_test_attempts.jsonl) means test inference already ran, and a new run needs
``--rerun-reason``. The key and subject checks run before inference; predictions go to a
temporary file, and only after success are the previous parquet and its predict_test.json
moved together into superseded_<UTC>/ with the reason. Every attempt is logged.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.dl_aggregate import AGGREGATORS  # noqa: E402
from thesis_pipeline.dl_data import PackedSplit, assert_ids_in_split, load_split  # noqa: E402
from thesis_pipeline.dl_eval import assemble_predictions, threshold_f1max  # noqa: E402
from thesis_pipeline.dl_stage_b import git_commit, sha256_file  # noqa: E402
from thesis_pipeline.dl_train import load_model_from_ckpt, resolve_device, score_split  # noqa: E402

DEFAULT_SPLIT = CODE_ROOT / "splits" / "aim2_seed42.json"
DEFAULT_KEY = CODE_ROOT / "models" / "recovery_2026-05-03" / "aim2_v85_taxonomy" / "physio_only" / \
    "test_predictions.parquet"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(CODE_ROOT), *args], capture_output=True, text=True).stdout.strip()


def check_freeze(tag: str) -> str:
    """HEAD must equal the tag's commit and the tracked tree must be clean."""
    head = _git("rev-parse", "HEAD")
    tagc = _git("rev-list", "-n", "1", tag)
    if not tagc:
        raise click.ClickException(f"git tag {tag!r} not found")
    if head != tagc:
        raise click.ClickException(f"HEAD {head[:10]} != tag {tag} ({tagc[:10]}); check out the frozen code")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise click.ClickException("tracked files modified; test inference needs a clean frozen tree")
    untracked = [ln for ln in _git("status", "--porcelain", "--untracked-files=all", "--",
                                   "thesis_pipeline", "scripts", "splits").splitlines()
                 if ln.startswith("??") and ln.rstrip().endswith((".py", ".json"))]
    if untracked:
        raise click.ClickException(f"untracked code not in the frozen tag: {untracked[:5]}")
    return head


ATTEMPTS = "predict_test_attempts.jsonl"


def prior_test_outputs(out: Path) -> list[str]:
    """Evidence that test inference already ran in ``out`` (see the module docstring)."""
    found = [p.name for p in (out / "test_predictions.parquet", out / "predict_test.json") if p.exists()]
    found += sorted(p.name for p in out.glob("superseded_*"))
    found += sorted(p.name for p in out.glob("test_predictions.2*.parquet"))  # archives of the first build
    log = out / ATTEMPTS
    if log.exists() and any(json.loads(ln).get("event") == "inference_start"
                            for ln in log.read_text().splitlines() if ln.strip()):
        found.append(f"{ATTEMPTS} (an earlier attempt reached inference)")
    return found


def _log_attempt(out: Path, rec: dict) -> None:
    with open(out / ATTEMPTS, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")


def load_aggregator_lock(path: str) -> dict:
    lock = json.loads(Path(path).read_text())
    if lock.get("aggregator") not in AGGREGATORS:
        raise click.ClickException(f"{path}: 'aggregator' must be one of {AGGREGATORS}")
    return lock


def check_postproc(pp: dict, ckpt_sha: str, lock: dict | None, dev_type: str | None) -> None:
    """Refuse a postproc from another checkpoint, another aggregator, or (test) another device."""
    if pp.get("best_ckpt_sha256") != ckpt_sha:
        raise click.ClickException(
            f"postproc best_ckpt_sha256 {str(pp.get('best_ckpt_sha256'))[:12]} != sha256(--ckpt) {ckpt_sha[:12]}: "
            "this postproc belongs to another checkpoint (seed or run)")
    if lock is not None and pp.get("aggregator") != lock["aggregator"]:
        raise click.ClickException(
            f"postproc aggregator {pp.get('aggregator')!r} != locked {lock['aggregator']!r}; re-finalise the run "
            "with train_aim2_dl.py --aggregator <locked> (no retraining) and refit on this device")
    if dev_type is not None and str(pp.get("device", "")).split(":")[0] != dev_type:
        raise click.ClickException(
            f"postproc was fitted on device {pp.get('device')!r}, test inference runs on {dev_type!r}; "
            "use the postproc_<device>.json from --split val --refit-postproc on this device")


@click.command()
@click.option("--ckpt", type=click.Path(exists=True, dir_okay=False), required=True, help="best.pt")
@click.option("--postproc", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), required=True,
              help="packed split folder (val or test)")
@click.option("--split", "which", type=click.Choice(["val", "test"]), required=True)
@click.option("--prereg", type=click.Path(exists=True, dir_okay=False), required=True,
              help="pre-registration vault note (sha256 logged)")
@click.option("--out-dir", type=click.Path(file_okay=False), required=True)
@click.option("--split-json", type=click.Path(exists=True, dir_okay=False), default=str(DEFAULT_SPLIT),
              show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--amp/--no-amp", default=True, show_default=True)
@click.option("--eval-batch", type=int, default=256, show_default=True)
@click.option("--refit-postproc", is_flag=True, help="val only: refit the threshold on these predictions")
@click.option("--freeze-tag", default=None, help="test only: git tag of the frozen code (must equal HEAD)")
@click.option("--key", type=click.Path(dir_okay=False), default=str(DEFAULT_KEY), show_default=True,
              help="test only: canonical key parquet (subject_id, epoch_idx, apnoea_label)")
@click.option("--rerun-reason", default=None, help="test only: logged bug reason to replace a prior run")
@click.option("--aggregator-lock", type=click.Path(exists=True, dir_okay=False), default=None,
              help="AGGREGATOR_LOCK.json (required for test; checked for val when given)")
def main(ckpt, postproc, data_dir, which, prereg, out_dir, split_json, device, amp, eval_batch,
         refit_postproc, freeze_tag, key, rerun_reason, aggregator_lock) -> None:
    """Predict epochs for the val or test split (see module docstring)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split = load_split(split_json)
    prov = {
        "split": which,
        "prereg": str(Path(prereg).resolve()),
        "prereg_sha256": sha256_file(prereg),
        "ckpt": str(Path(ckpt).resolve()),
        "ckpt_sha256": sha256_file(ckpt),
        "postproc_sha256": sha256_file(postproc),
        "split_json": split["path"],
        "split_sha256": split["sha256"],
        "git_commit": git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    prior: list[str] = []
    if which == "val" and rerun_reason:
        raise click.UsageError("--rerun-reason is test-only")
    if which == "test":
        if refit_postproc:
            raise click.UsageError("--refit-postproc is validation-only")
        if not freeze_tag:
            raise click.UsageError("--split test requires --freeze-tag (test inference runs after the code freeze)")
        if not aggregator_lock:
            raise click.UsageError("--split test requires --aggregator-lock (the frozen AGGREGATOR_LOCK.json)")
        prov["freeze_commit"] = check_freeze(freeze_tag)
        prov["freeze_tag"] = freeze_tag
        prior = prior_test_outputs(out)
        if prior and not rerun_reason:
            raise click.ClickException(
                f"test inference already ran in {out} (found {prior}); it runs once. Pass --rerun-reason "
                "'<logged bug reason>' to replace it (the old outputs are kept under superseded_<UTC>/).")
        if rerun_reason and not prior:
            raise click.UsageError(f"--rerun-reason given but {out} holds no earlier test inference")
        if rerun_reason:
            prov["rerun_reason"] = rerun_reason

    pp = json.loads(Path(postproc).read_text())
    lock = load_aggregator_lock(aggregator_lock) if aggregator_lock else None
    if lock is not None:
        prov.update({"aggregator_lock": str(Path(aggregator_lock).resolve()),
                     "aggregator_lock_sha256": sha256_file(aggregator_lock), "aggregator_locked": lock["aggregator"]})
    dev = resolve_device(device)
    check_postproc(pp, prov["ckpt_sha256"], lock, dev.type if which == "test" else None)
    prov["postproc_checks"] = ("ckpt sha256 matched" + (", aggregator == lock" if lock else "")
                               + (", fitted on this device type" if which == "test" else ""))
    model, ck = load_model_from_ckpt(ckpt, dev)
    channels = tuple(ck["channels"])
    ps = PackedSplit(data_dir, channels, allow_test=(which == "test"))
    if ps.split != which:
        raise click.ClickException(f"{data_dir} manifest split is {ps.split!r}, not {which!r}")
    assert_ids_in_split(ps.subject_ids, split, which)
    prov["data_manifest_sha256"] = sha256_file(Path(data_dir) / "MANIFEST.json")
    use_amp = bool(amp and dev.type in ("cuda", "mps"))
    prov.update({"device": str(dev), "amp": use_amp, "channels": list(channels),
                 "config": ck["settings"]["config"], "model_seed": ck["settings"]["model_seed"]})
    if which == "test":  # every check that does not need predictions runs before inference
        if not np.array_equal(np.sort(ps.subject_ids), split["test"]):
            raise click.ClickException("test cache does not hold exactly the split's test subjects")
        key_df = pd.read_parquet(key, columns=["subject_id", "epoch_idx", "apnoea_label"])
        if not np.array_equal(np.unique(key_df["subject_id"].to_numpy(np.int64)), split["test"]):
            raise click.ClickException(f"key {key} does not hold exactly the split's test subjects")
        _log_attempt(out, {"event": "inference_start", "rerun_reason": rerun_reason, "prior": prior,
                           "freeze_commit": prov["freeze_commit"], "ckpt_sha256": prov["ckpt_sha256"]})
    t0 = time.perf_counter()
    df = score_split(model, ps, dev, use_amp, batch=eval_batch)
    prov["predict_seconds"] = round(time.perf_counter() - t0, 1)
    agg = pp["aggregator"]
    thr = float(pp["threshold"])

    if which == "val":
        y = df["apnoea_label"].to_numpy()
        aucs = {c[2:]: float(roc_auc_score(y, df[c])) for c in df.columns if c.startswith("p_")}
        if refit_postproc:
            thr, f1 = threshold_f1max(y, df[f"p_{agg}"].to_numpy())
            new_pp = {**pp, "threshold": thr, "val_f1_at_threshold": f1, "device": str(dev), "amp": use_amp,
                      "refit_from": str(Path(postproc).resolve()),
                      "refit_from_sha256": prov["postproc_sha256"],
                      "aggregator_lock_sha256": prov.get("aggregator_lock_sha256"),
                      "refit_note": "threshold refit on validation predictions recomputed on this device; "
                                    "aggregator kept frozen"}
            (out / f"postproc_{dev.type}.json").write_text(json.dumps(new_pp, indent=2))
        key_df = df[["subject_id", "epoch_idx", "apnoea_label"]]
        pred = assemble_predictions(df.rename(columns={f"p_{agg}": "pred_prob"}), key_df, thr)
        pred.to_parquet(out / f"val_predictions_{dev.type}.parquet", index=False)
        prov.update({"val_auc_by_aggregator": aucs, "aggregator": agg, "threshold": thr,
                     "n_rows": int(len(pred)), "n_subjects": int(pred["subject_id"].nunique())})
        (out / f"predict_val_{dev.type}.json").write_text(json.dumps(prov, indent=2))
        click.echo(json.dumps({k: prov[k] for k in ("val_auc_by_aggregator", "aggregator", "threshold",
                                                   "n_rows", "predict_seconds")}))
        return

    # ---- test: predictions + provenance only, no metrics ----
    prov["key"] = str(Path(key).resolve())
    prov["key_sha256"] = sha256_file(key)
    prov["test_ids_sha256"] = hashlib.sha256(
        json.dumps(sorted(int(i) for i in split["test"])).encode()).hexdigest()
    declared = json.loads(Path(split_json).read_text()).get("ids_sha256")
    if isinstance(declared, dict):
        prov["split_declared_test_ids_sha256"] = declared.get("test")
    pred = assemble_predictions(df.rename(columns={f"p_{agg}": "pred_prob"}), key_df, thr)
    target, tmp = out / "test_predictions.parquet", out / "test_predictions.tmp.parquet"
    pred.to_parquet(tmp, index=False)
    prov.update({"aggregator": agg, "threshold": thr, "n_rows": int(len(pred)),
                 "n_subjects": int(pred["subject_id"].nunique()),
                 "test_predictions_sha256": sha256_file(tmp)})
    if prior:  # archive the previous parquet together with its provenance, only now that this run succeeded
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = out / f"superseded_{stamp}"
        dest.mkdir()
        moved = []
        for name in ("test_predictions.parquet", "predict_test.json"):
            if (out / name).exists():
                (out / name).rename(dest / name)
                moved.append(name)
        (dest / "superseded_reason.json").write_text(json.dumps(
            {"archived_utc": stamp, "rerun_reason": rerun_reason, "moved": moved, "prior": prior}, indent=2))
        prov["replaced_previous"] = str(dest)
    os.replace(tmp, target)
    ptmp = out / "predict_test.json.tmp"
    ptmp.write_text(json.dumps(prov, indent=2))
    os.replace(ptmp, out / "predict_test.json")
    _log_attempt(out, {"event": "done", "test_predictions_sha256": prov["test_predictions_sha256"]})
    click.echo(f"wrote {target} ({len(pred):,} rows); no metrics computed here")


if __name__ == "__main__":
    main()
