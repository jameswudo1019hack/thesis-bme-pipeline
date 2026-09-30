"""Train one (config, model seed) of the Aim 2 Olsen BiGRU on packed stage-B data.

Accepts only a train folder and a val folder (their MANIFEST split must be "train" and
"val"), checks every subject against the frozen split JSON (no test id may appear) and
resumes automatically from ``--ckpt-dir/last.pt`` or ``--mirror-dir/last.pt``.

Examples
--------
    # production (Colab), recipe chosen by the benchmark gate
    python scripts/train_aim2_dl.py --config M --model-seed 42 --recipe A \
        --train-dir /content/dl_cache/model_v1/train --val-dir /content/dl_cache/model_v1/val \
        --ckpt-dir /content/ckpt/M/seed42 --out-dir /content/out/M/seed42 \
        --mirror-dir /content/drive/MyDrive/Thesis/results/aim2_dl_olsen_v1/M/seed42
    # (checkpoints and outputs are written locally and mirrored to Drive; a Drive I/O
    #  error is retried on the next save instead of stopping the run)

    # Mac smoke test: 50 batches on MPS, loss must fall
    python scripts/train_aim2_dl.py --config M --model-seed 42 --smoke-batches 50 --device mps \
        --train-dir ~/thesis_dl_cache/model_v1/train --out-dir /tmp/smoke

Seeds: 42, 43, 44 (``--model-seed``); the split seed stays 42 and is never changed here.
F6 is deferred (Aim 1 airflow picker defect) and needs ``--allow-deferred``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from thesis_pipeline.dl_aggregate import AGGREGATORS  # noqa: E402
from thesis_pipeline.dl_train import CONFIGS, RECIPES, Trainer, TrainSettings  # noqa: E402

DEFAULT_SPLIT = CODE_ROOT / "splits" / "aim2_seed42.json"


@click.command()
@click.option("--config", type=click.Choice(sorted(CONFIGS)), required=True)
@click.option("--model-seed", type=int, required=True, help="42, 43 or 44")
@click.option("--recipe", type=click.Choice(sorted(RECIPES)), default="A", show_default=True)
@click.option("--train-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--val-dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--split-json", type=click.Path(exists=True, dir_okay=False), default=str(DEFAULT_SPLIT),
              show_default=True)
@click.option("--ckpt-dir", type=click.Path(file_okay=False), default=None)
@click.option("--mirror-dir", type=click.Path(file_okay=False), default=None,
              help="second copy of checkpoints and outputs (Drive); resume picks the most advanced "
                   "and refuses a mirror written by another config / seed")
@click.option("--out-dir", type=click.Path(file_okay=False), required=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--amp/--no-amp", default=True, show_default=True)
@click.option("--aggregator", type=click.Choice(("auto",) + AGGREGATORS), default="auto", show_default=True,
              help="'auto' applies the pre-declared rule (M seed 42 only); others pass the frozen choice")
@click.option("--ckpt-every-steps", type=int, default=0, show_default=True, help="mid-pass checkpoints (T4)")
@click.option("--stop-after-passes", type=int, default=None, help="pause after N passes this invocation")
@click.option("--max-passes", type=int, default=None, help="PILOT ONLY: override the recipe pass cap")
@click.option("--max-batches-per-pass", type=int, default=None, help="PILOT ONLY: truncate passes")
@click.option("--eval-batch", type=int, default=512, show_default=True)
@click.option("--prefetch", type=int, default=2, show_default=True)
@click.option("--num-threads", type=int, default=None)
@click.option("--allow-deferred", is_flag=True, help="allow F6 (deferred by decision 2026-09-30)")
@click.option("--smoke-batches", type=int, default=None, help="only run N training batches and report loss")
def main(config, model_seed, recipe, train_dir, val_dir, split_json, ckpt_dir, mirror_dir, out_dir, device,
         amp, aggregator, ckpt_every_steps, stop_after_passes, max_passes, max_batches_per_pass, eval_batch,
         prefetch, num_threads, allow_deferred, smoke_batches) -> None:
    """Train, validate each pass, checkpoint, then write val predictions + postproc + metrics."""
    s = TrainSettings(
        config=config, model_seed=model_seed, recipe=recipe, amp=amp, device=device,
        eval_batch=eval_batch, prefetch=prefetch, ckpt_every_steps=ckpt_every_steps,
        stop_after_passes=stop_after_passes, max_passes=max_passes,
        max_batches_per_pass=max_batches_per_pass, aggregator=aggregator,
        allow_deferred=allow_deferred, num_threads=num_threads,
    )
    out = Path(out_dir)
    if smoke_batches:
        tr = Trainer(s, train_dir, None, split_json, ckpt_dir or out / "smoke_ckpt", out, log=click.echo)
        res = tr.smoke(smoke_batches)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"smoke_{tr.device.type}.json").write_text(json.dumps(res, indent=2))
        click.echo(json.dumps({k: v for k, v in res.items() if k != "losses"}))
        return
    if val_dir is None or ckpt_dir is None:
        raise click.UsageError("--val-dir and --ckpt-dir are required for training")
    tr = Trainer(s, train_dir, val_dir, split_json, ckpt_dir, out, mirror_dir=mirror_dir, log=click.echo)
    click.echo(f"config {config} channels {tr.channels} seed {model_seed} recipe {recipe} "
               f"device {tr.device} amp {tr.amp} params {tr.n_params:,} hash {tr.config_hash}")
    click.echo(f"train {tr.train.n_subjects} subjects, val {tr.val.n_subjects} subjects")
    status = tr.run()
    click.echo(f"status: {status}")


if __name__ == "__main__":
    main()
