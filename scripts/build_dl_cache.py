"""Build the stage-A raw-signal cache for the Aim 2 DL arm (one npz + json per subject).

Two modes (exactly one is required):

  --only-local   Extract every subject whose EDF is already materialised on
                 this Mac. Never downloads an EDF: dataless EDFs are skipped,
                 and the EDF reader itself refuses placeholders. Missing NSRR
                 XMLs (40-115 KB) are fetched unless --no-fetch-xml.
  --download     Also materialise placeholder EDFs, one subject per unit of
                 work, in the order --order (default val,test,train; train in a
                 seeded random order so a partial train set is a random
                 sample). Retries EPERM / timeouts with exponential backoff,
                 pauses below --min-free-gb, stops after --max-download-gb
                 (then run Finder > Free Up Space on the extracted subjects and
                 rerun). Never evicts anything itself.

Both modes are resumable: subjects whose npz + json exist and whose sha256
matches are skipped; subjects whose last ledger entry is 'failed' are skipped
unless --retry-failed, except val / test subjects whose failure was transient
(EPERM, timeout, placeholder, reader error: dl_cache.TRANSIENT_ERROR_TYPES),
which are retried automatically. Every attempt is appended to
<out>/ledger.jsonl and failures also to <out>/failures.jsonl; every run's
config to <out>/run_configs.jsonl (run_config.json holds the latest).

--out may not lie in a cloud-synced folder (~/Desktop, ~/Documents, iCloud
Drive, ~/Library/CloudStorage, ...): stage A holds test-subject signals.

Exit codes: 0 done; 2 stopped early (disk floor / --max-download-gb; rerun to
continue); 3 some val / test subject (in --order) is not in stage A because it
failed or its file is missing: the G1 / G3 completeness gate (takes precedence
over 2). Train failures never change the exit code.

Examples:
  python scripts/build_dl_cache.py --only-local --dry-run
  python scripts/build_dl_cache.py --only-local --workers 4
  python scripts/build_dl_cache.py --download --order val,test,train --workers 3 --max-download-gb 50
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import click
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline import dl_cache, shhs, splits  # noqa: E402

DEFAULT_SPLIT = CODE_ROOT / "splits" / "aim2_seed42.json"
DEFAULT_OUT = Path.home() / "thesis_dl_cache" / "raw_v1"  # home root: outside iCloud Desktop sync
GIB = 1024 ** 3
GATED_SPLITS = ("val", "test")  # must reach 100 % (design G1 / G3); transient failures auto-retried
BLOCKING_STATES = ("file_missing", "failed_before_skipped")
EXIT_STOPPED_EARLY, EXIT_GATED_INCOMPLETE = 2, 3


def order_subjects(split: dict, order: list[str], train_seed: int) -> list[tuple[int, str]]:
    """Subjects in processing order: each split in ``order``; train shuffled with ``train_seed``."""
    out: list[tuple[int, str]] = []
    for name in order:
        ids = np.asarray(split[name], dtype=np.int64)
        if name == "train":
            ids = ids[np.random.default_rng(train_seed).permutation(ids.size)]
        out.extend((int(s), name) for s in ids)
    return out


def free_gib(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / GIB


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.glob("shhs1-*.*") if f.suffix in (".npz", ".json"))


def plan(
    subjects: list[tuple[int, str]],
    root: Path,
    out: Path,
    ledger: dl_cache.Ledger,
    mode: str,
    retry_failed: bool,
    fetch_xml: bool,
    verify_existing: bool,
) -> tuple[list[dict], Counter, dict[str, list[tuple[int, str]]]]:
    """Classify every subject.

    Returns the to-do tasks, a (split, state) counter, and ``blocked``: per split,
    the ``(subject_id, state)`` of subjects in a BLOCKING_STATES state (they will
    not be in stage A after this run). Val / test subjects whose last failure was
    transient (dl_cache.is_transient_failure) are re-queued without --retry-failed.
    """
    latest = ledger.latest()
    todo: list[dict] = []
    states: Counter = Counter()
    blocked: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for sid, split_name in subjects:
        sp = shhs.paths_for(sid, cohort="shhs1")
        edf, xml = root / sp.edf.name, root / sp.nsrr_xml.name
        if dl_cache.is_complete(out, sid, verify=verify_existing):
            states[(split_name, "cached")] += 1
            continue
        if not edf.exists() or not xml.exists():
            states[(split_name, "file_missing")] += 1
            blocked[split_name].append((sid, "file_missing"))
            continue
        edf_local = not dl_cache.is_dataless(edf)
        xml_local = not dl_cache.is_dataless(xml)
        if mode == "only-local" and not edf_local:
            states[(split_name, "edf_dataless_skipped")] += 1
            continue
        if not xml_local and not fetch_xml:
            states[(split_name, "xml_dataless_skipped")] += 1
            continue
        last = latest.get(sid)
        if last is not None and last.get("status") == "failed" and not retry_failed:
            if split_name in GATED_SPLITS and dl_cache.is_transient_failure(last):
                states[(split_name, "todo_transient_auto_retry")] += 1
            else:
                states[(split_name, "failed_before_skipped")] += 1
                blocked[split_name].append((sid, "failed_before_skipped"))
                continue
        states[(split_name, "todo")] += 1
        if not xml_local:
            states[(split_name, "todo_needs_xml_fetch")] += 1
        if not edf_local:
            states[(split_name, "todo_needs_edf_download")] += 1
        todo.append({
            "subject_id": sid,
            "split": split_name,
            "edf": str(edf),
            "xml": str(xml),
            "edf_bytes": edf.stat().st_size,
            "edf_local": edf_local,
        })
    return todo, states, dict(blocked)


def gate_report(blocked: dict[str, list[tuple[int, str]]], failed_now: dict[str, list[int]],
                order: list[str]) -> int:
    """Print val / test subjects that are not in stage A; return how many."""
    n = 0
    for s in (x for x in order if x in GATED_SPLITS):
        by_state = Counter(st for _, st in blocked.get(s, []))
        ids = [sid for sid, _ in blocked.get(s, [])] + list(failed_now.get(s, []))
        if failed_now.get(s):
            by_state["failed_this_run"] = len(failed_now[s])
        if ids:
            n += len(ids)
            click.secho(f"  GATE {s}: {len(ids)} subject(s) not in stage A {dict(by_state)} e.g. {ids[:10]} "
                        "(--retry-failed after fixing the cause; see failures.jsonl)", fg="red")
    return n


def print_states(states: Counter, order: list[str]) -> None:
    keys = sorted({k for (_, k) in states})
    click.echo("  " + "state".ljust(28) + "".join(s.rjust(8) for s in order) + "total".rjust(8))
    for k in keys:
        row = [states.get((s, k), 0) for s in order]
        click.echo("  " + k.ljust(28) + "".join(str(v).rjust(8) for v in row) + str(sum(row)).rjust(8))


@click.command()
@click.option("--only-local", "mode", flag_value="only-local", help="Extract already-materialised EDFs only.")
@click.option("--download", "mode", flag_value="download", help="Also materialise placeholder EDFs.")
@click.option("--out", type=click.Path(path_type=Path), default=DEFAULT_OUT, show_default=True)
@click.option("--split", "split_path", type=click.Path(path_type=Path, exists=True), default=DEFAULT_SPLIT,
              show_default=True, help="Frozen split JSON (make_aim2_split.py).")
@click.option("--shhs-root", type=click.Path(path_type=Path, exists=True), default=shhs.SHHS_ROOT,
              show_default=False, help="Folder holding shhs1-<id>.edf and shhs1-<id>-nsrr.xml.")
@click.option("--order", default="val,test,train", show_default=True,
              help="Splits to process, in this order.")
@click.option("--train-order-seed", type=int, default=42, show_default=True,
              help="Seed of the random train order (download campaigns).")
@click.option("--workers", type=int, default=3, show_default=True)
@click.option("--min-free-gb", type=float, default=25.0, show_default=True,
              help="Never start a subject that would leave less free disk (GiB) than this.")
@click.option("--max-download-gb", type=float, default=50.0, show_default=True,
              help="--download: stop after materialising this many GiB of EDF in this run.")
@click.option("--low-disk-wait-min", type=float, default=0.0, show_default=True,
              help="Below --min-free-gb, poll this long for space (Free Up Space) before stopping.")
@click.option("--retry-failed", is_flag=True, help="Retry subjects whose last ledger status is 'failed'.")
@click.option("--fetch-xml/--no-fetch-xml", default=True, show_default=True,
              help="Fetch placeholder NSRR XMLs (small) for subjects being extracted.")
@click.option("--timeout-edf-s", type=float, default=900.0, show_default=True)
@click.option("--timeout-xml-s", type=float, default=120.0, show_default=True)
@click.option("--backoff", default="30,60,120,240,480", show_default=True,
              help="Retry delays (s) for EPERM / timeouts; one retry per entry.")
@click.option("--ids", default=None, help="Comma-separated subject ids to restrict to (within --order splits).")
@click.option("--limit", type=int, default=None, help="Process at most this many subjects (smoke runs).")
@click.option("--no-verify-existing", is_flag=True, help="Skip sha256 of existing npz when checking completeness.")
@click.option("--dry-run", is_flag=True, help="Print the plan and exit; touches no file contents.")
def main(mode, out, split_path, shhs_root, order, train_order_seed, workers, min_free_gb, max_download_gb,
         low_disk_wait_min, retry_failed, fetch_xml, timeout_edf_s, timeout_xml_s, backoff, ids, limit,
         no_verify_existing, dry_run) -> None:
    if mode is None:
        raise click.UsageError("choose exactly one of --only-local / --download")
    out = Path(out).expanduser()
    synced = dl_cache.cloud_synced_root(out)
    if synced is not None:
        raise click.UsageError(f"refusing to write the cache under {synced} (cloud-synced: ~/Desktop, ~/Documents, "
                               "iCloud Drive, ~/Library/CloudStorage, ...); stage A holds test-subject signals")
    order_l = [s.strip() for s in order.split(",") if s.strip()]
    bad = [s for s in order_l if s not in splits.SPLIT_NAMES]
    if bad:
        raise click.UsageError(f"unknown split(s) in --order: {bad}")
    delays = [float(x) for x in backoff.split(",") if x.strip()]

    split = splits.load_split(split_path)
    subjects = order_subjects(split, order_l, train_order_seed)
    if ids:
        keep = {int(x) for x in ids.split(",") if x.strip()}
        subjects = [(s, n) for s, n in subjects if s in keep]
    ledger = dl_cache.Ledger(out / "ledger.jsonl")
    failures = dl_cache.Ledger(out / "failures.jsonl")

    t_plan = time.time()
    todo, states, blocked = plan(subjects, Path(shhs_root), out, ledger, mode, retry_failed, fetch_xml,
                                 verify_existing=not no_verify_existing)
    if limit is not None:
        todo = todo[:limit]
    click.secho(f"mode={mode}  out={out}  split={split_path.name} ({split['sha256'][:12]})  "
                f"plan {time.time() - t_plan:.1f}s", fg="cyan")
    print_states(states, order_l)
    dl_bytes = sum(t["edf_bytes"] for t in todo if not t["edf_local"])
    click.echo(f"  to process now: {len(todo)}"
               + (f" (EDF to download: {dl_bytes / GIB:.1f} GiB)" if mode == "download" else "")
               + f"; free disk {free_gib(out):.1f} GiB (floor {min_free_gb:g})")
    n_blocked = gate_report(blocked, {}, order_l)
    if dry_run:
        return
    if not todo:
        if n_blocked:
            sys.exit(EXIT_GATED_INCOMPLETE)
        return

    out.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git": dl_cache.git_commit(CODE_ROOT),
        "split_file": split_path.name,
        "split_sha256": split["sha256"],
        "extractor_version": dl_cache.EXTRACTOR_VERSION,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "mode": mode,
    }
    run_cfg = {**provenance, "argv": sys.argv, "n_todo": len(todo)}
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=1) + "\n")  # the latest run
    dl_cache.Ledger(out / "run_configs.jsonl").append(run_cfg)  # every run, with its start time
    common = {
        "out_dir": str(out),
        "provenance": provenance,
        "fetch_xml": fetch_xml,
        "fetch_edf": mode == "download",
        "timeout_xml_s": timeout_xml_s,
        "timeout_edf_s": timeout_edf_s,
        "backoff": delays,
        "mode": mode,
    }

    t0 = time.time()
    bytes_start = dir_bytes(out)
    done: Counter = Counter()
    by_split: dict[str, Counter] = defaultdict(Counter)
    failed_now: dict[str, list[int]] = defaultdict(list)
    downloaded = 0
    stop_reason = None
    queue = list(todo)
    in_flight: dict = {}
    max_in_flight = max(1, workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        try:
            while queue or in_flight:
                while queue and len(in_flight) < max_in_flight and stop_reason is None:
                    nxt = queue[0]
                    need_gib = (0 if nxt["edf_local"] else nxt["edf_bytes"] / GIB) + 0.01
                    if mode == "download" and not nxt["edf_local"] and \
                            (downloaded + nxt["edf_bytes"]) / GIB > max_download_gb:
                        stop_reason = f"--max-download-gb {max_download_gb:g} reached"
                        break
                    if free_gib(out) - need_gib < min_free_gb:
                        if in_flight:
                            break  # let running subjects finish, then re-check
                        waited = 0.0
                        while free_gib(out) - need_gib < min_free_gb and waited < low_disk_wait_min * 60:
                            click.secho(f"  free {free_gib(out):.1f} GiB < floor {min_free_gb:g}: Finder > "
                                        "Free Up Space on extracted subjects; waiting...", fg="yellow")
                            time.sleep(60)
                            waited += 60
                        if free_gib(out) - need_gib < min_free_gb:
                            stop_reason = f"free disk below --min-free-gb {min_free_gb:g}"
                            break
                    queue.pop(0)
                    if not nxt["edf_local"]:
                        downloaded += nxt["edf_bytes"]
                    fut = pool.submit(dl_cache.build_one, {**common, **nxt})
                    in_flight[fut] = nxt
                if not in_flight:
                    break
                finished, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for fut in finished:
                    task = in_flight.pop(fut)
                    rec = fut.result()
                    rec["edf_bytes"] = task["edf_bytes"]
                    ledger.append(rec)
                    if rec["status"] != "ok":
                        failures.append(rec)
                        failed_now[task["split"]].append(int(task["subject_id"]))
                        click.secho(f"  ! {task['subject_id']} ({task['split']}): {rec.get('error')}", fg="red")
                    done[rec["status"]] += 1
                    by_split[task["split"]][rec["status"]] += 1
                    n = sum(done.values())
                    if n % 50 == 0 or n == len(todo):
                        el = time.time() - t0
                        click.echo(f"  {n}/{len(todo)}  ok={done['ok']} failed={done['failed']}  "
                                   f"{el / 60:.1f} min  ({n / el:.2f} subj/s)  free {free_gib(out):.1f} GiB")
        except KeyboardInterrupt:
            click.secho("interrupted: cancelling queued work (running subjects finish)", fg="yellow")
            pool.shutdown(wait=True, cancel_futures=True)
            raise

    el = time.time() - t0
    added = dir_bytes(out) - bytes_start
    click.secho(f"\ndone in {el / 60:.1f} min: {dict(done)}", fg="green")
    for s in order_l:
        if by_split.get(s):
            click.echo(f"  {s}: {dict(by_split[s])}")
    click.echo(f"  cache grew by {added / 1e9:.2f} GB; cache total {dir_bytes(out) / 1e9:.2f} GB; "
               f"free disk {free_gib(out):.1f} GiB")
    n_blocked = gate_report(blocked, failed_now, order_l)
    if stop_reason:
        click.secho(f"  stopped early: {stop_reason}; {len(queue)} subject(s) not started", fg="yellow")
    if n_blocked:
        sys.exit(EXIT_GATED_INCOMPLETE)
    if stop_reason:
        sys.exit(EXIT_STOPPED_EARLY)


if __name__ == "__main__":
    main()
