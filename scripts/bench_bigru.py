"""Throughput benchmark for the Aim 2 Olsen BiGRU (the Colab gate, design section 5).

Uses the production model (``thesis_pipeline.dl_models.OlsenBiGRU``: SoftMinMax in fp32,
BiGRU x2 -> MaxPool -> BatchNorm -> ReLU -> Dropout, x2 blocks, Linear 512 -> Linear 1)
and the production optimiser (AdamW lr 1e-3, weight decay 1e-4), with the masked
per-second BCE loss. Input is synthetic unless ``--cache-dir`` points at a packed
stage-B split, in which case every step also does the host-side mmap CPU gather.

Gate rule (locked before any production run):
    R-A: batch 128 single process >= 800 windows/s, or 3 processes >= 1,500 in total
    R-B: 400-800 windows/s  -> batch 256, cap 20
    R-C: < 400 windows/s (or T4 only) -> batch 256, 50 % subsample, patience 2/4, cap 15

Examples
--------
    python scripts/bench_bigru.py --device cuda --amp --batch 128 --batch 256 --batch 512
    python scripts/bench_bigru.py --device cuda --amp --batch 128 --procs 3
    python scripts/bench_bigru.py --device mps --amp --batch 128 --cache-dir ~/thesis_dl_cache/model_v1/val
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.dl_models import OlsenBiGRU, count_parameters  # noqa: E402

CHANNELS = ("RR", "EDR", "THOR", "ABDO", "AIRFLOW", "SPO2")


def _sync(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def gate_recipe(single_128: float | None, multi_total: float | None, t4: bool = False) -> str:
    """Pre-declared recipe rule (design section 5)."""
    if t4:
        return "C"
    if (single_128 is not None and single_128 >= 800) or (multi_total is not None and multi_total >= 1500):
        return "A"
    if single_128 is not None and single_128 >= 400:
        return "B"
    return "C"


def run_one(device: str, batch: int, cin: int, iters: int, warmup: int, amp: bool,
            cache_dir: str | None, seed: int = 0) -> dict:
    dev = torch.device(device)
    torch.manual_seed(seed)
    amp = bool(amp and dev.type in ("cuda", "mps"))
    model = OlsenBiGRU(c_in=cin).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)

    ps = win = None
    if cache_dir:
        from thesis_pipeline.dl_data import PackedSplit, train_window_starts

        ps = PackedSplit(cache_dir, CHANNELS[:cin])
        win = train_window_starts(ps, seed, 0)
    x_syn = torch.randn(batch, 1200, cin, device=dev).half()
    v_syn = torch.ones(batch, 1200, dtype=torch.bool, device=dev)
    y_syn = (torch.rand(batch, 300, device=dev) < 0.28).float()
    m_syn = torch.rand(batch, 300, device=dev) < 0.8
    rng = np.random.default_rng(seed)

    def batch_tensors():
        if ps is None:
            return x_syn, v_syn, y_syn, m_syn
        sel = rng.integers(0, len(win), size=batch)
        bd = ps.gather(win, sel)
        return (torch.from_numpy(bd["x"]).to(dev), torch.from_numpy(bd["valid"]).to(dev),
                torch.from_numpy(bd["y"]).to(dev).float(), torch.from_numpy(bd["mask"]).to(dev))

    def step():
        x, v, y, m = batch_tensors()
        model.train()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            logits = model(x, v)
        le = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
        loss = (le * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    for _ in range(warmup):
        step()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(dev)
    dt = (time.perf_counter() - t0) / iters

    model.eval()
    with torch.no_grad():
        x, v, _, _ = batch_tensors()
        for _ in range(2):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                model(x, v)
        _sync(dev)
        t1 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                model(x, v)
        _sync(dev)
        dti = (time.perf_counter() - t1) / iters
    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "batch": batch, "c_in": cin, "amp": amp, "data": "mmap-gather" if ps is not None else "synthetic",
        "n_params": count_parameters(model),
        "train_sec_per_batch": round(dt, 4),
        "train_windows_per_sec": round(batch / dt, 1),
        "infer_windows_per_sec": round(batch / dti, 1),
        "torch": torch.__version__, "threads": torch.get_num_threads(),
    }


def parse_child_result(stdout: str) -> dict:
    """The per-process result line from a ``--child`` run (ignores any other JSON line)."""
    for line in reversed(stdout.strip().splitlines()):
        if '"train_windows_per_sec"' in line:
            return json.loads(line)
    raise RuntimeError(f"benchmark child printed no result line:\n{stdout[-2000:]}")


@click.command()
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", show_default=True)
@click.option("--batch", "batches", type=int, multiple=True, default=(128,), show_default=True)
@click.option("--cin", type=int, default=2, show_default=True)
@click.option("--iters", type=int, default=10, show_default=True)
@click.option("--warmup", type=int, default=3, show_default=True)
@click.option("--amp/--no-amp", default=True, show_default=True)
@click.option("--procs", type=int, default=1, show_default=True, help="concurrent processes (same GPU)")
@click.option("--cache-dir", default=None, help="packed train/val split folder for real mmap gather")
@click.option("--threads", type=int, default=None, help="torch.set_num_threads (CPU runs)")
@click.option("--json-out", type=click.Path(dir_okay=False), default=None)
@click.option("--child", is_flag=True, hidden=True, help="internal: a --procs worker (prints only its result)")
def main(device, batches, cin, iters, warmup, amp, procs, cache_dir, threads, json_out, child) -> None:
    """Measure training and inference windows/s; print one JSON line per setting."""
    if threads:
        torch.set_num_threads(threads)
    results = []
    for b in batches:
        if procs <= 1:
            r = run_one(device, b, cin, iters, warmup, amp, cache_dir)
        else:
            cmd = [sys.executable, __file__, "--device", device, "--batch", str(b), "--cin", str(cin),
                   "--iters", str(iters), "--warmup", str(warmup), "--amp" if amp else "--no-amp",
                   "--procs", "1", "--child"]
            if cache_dir:
                cmd += ["--cache-dir", cache_dir]
            ps = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=os.environ.copy())
                  for _ in range(procs)]
            outs = [parse_child_result(p.communicate()[0]) for p in ps]
            r = {**outs[0], "procs": procs,
                 "per_proc_train_windows_per_sec": [o["train_windows_per_sec"] for o in outs],
                 "total_train_windows_per_sec": round(sum(o["train_windows_per_sec"] for o in outs), 1)}
        results.append(r)
        click.echo(json.dumps(r))
    single = next((r["train_windows_per_sec"] for r in results if r["batch"] == 128 and "procs" not in r), None)
    multi = next((r["total_train_windows_per_sec"] for r in results if r.get("procs", 1) > 1), None)
    if not child and device.startswith("cuda") and (single is not None or multi is not None):
        gpu = results[0].get("gpu") or ""
        rec = gate_recipe(single, multi, t4="T4" in gpu)
        click.echo(json.dumps({"gate_recipe": rec, "single_128": single, "multi_total": multi, "gpu": gpu}))
    if json_out:
        Path(json_out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
