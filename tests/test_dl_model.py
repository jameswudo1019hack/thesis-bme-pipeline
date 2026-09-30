"""Tests for the Olsen BiGRU, SoftMinMax, aggregators and the trainer
(design section 8, tests 7, 9, 10, 12, 13), plus the predict / evaluate CLIs on
validation data (no test inference anywhere)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dl_synthetic import build_packed_dataset, write_split_json  # noqa: E402
from thesis_pipeline import dl_train  # noqa: E402
from thesis_pipeline.dl_aggregate import choose_aggregator, sec_to_epoch  # noqa: E402
from thesis_pipeline.dl_data import TestDataRefused  # noqa: E402
from thesis_pipeline.dl_eval import threshold_f1max  # noqa: E402
from thesis_pipeline.dl_models import OlsenBiGRU, SoftMinMax, count_parameters, masked_quantiles  # noqa: E402

# ---------------------------------------------------------------- model


def test_parameter_count_and_output_shape():
    """Design test 9."""
    m = OlsenBiGRU(c_in=2)
    assert count_parameters(m) == 1_123_841
    assert count_parameters(OlsenBiGRU(c_in=4)) == 1_125_377
    n_bn = sum(b.numel() for n, b in m.named_buffers() if "running" in n)
    assert count_parameters(m) + n_bn + 513 == 1_125_378  # paper figure: + BN stats + 2nd output unit
    y = m(torch.randn(3, 1200, 2))
    assert y.shape == (3, 300)


def test_batchnorm_sits_after_maxpool():
    m = OlsenBiGRU(c_in=2, hidden=16).eval()
    seen = []
    m.blocks[0].bn.register_forward_hook(lambda mod, inp, out: seen.append(tuple(inp[0].shape)))
    m(torch.randn(2, 1200, 2))
    assert seen == [(2, 32, 600)]  # 1200 samples pooled to 600 before BN


def test_initialisation():
    m = OlsenBiGRU(c_in=2, hidden=16)
    for name, p in m.named_parameters():
        if "bias" in name and "bn" not in name:
            assert torch.all(p == 0), name
        if "weight_hh" in name:
            h = p.shape[1]
            for g in range(3):
                w = p[g * h:(g + 1) * h]
                assert torch.allclose(w @ w.T, torch.eye(h), atol=1e-5), name


# ---------------------------------------------------------------- SoftMinMax (design test 12)


def _ref_softminmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(x, [0.05, 0.95], axis=0)
    rng = hi - lo
    out = (x - lo) / np.where(rng < 1e-6, 1, rng)
    out[:, rng < 1e-6] = 0
    return out


def test_softminmax_excludes_padding_by_index():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 1200, 2)).astype(np.float32)
    x[0, 1000:] = 1e6  # padding with extreme values
    x[0, 100:200, 0] = 0.0  # genuine zeros inside the night must count
    valid = np.zeros((1, 1200), bool)
    valid[0, :1000] = True
    y = SoftMinMax()(torch.from_numpy(x), torch.from_numpy(valid)).numpy()
    ref = _ref_softminmax(x[0, :1000].astype(np.float64))
    assert np.allclose(y[0, :1000], ref, atol=1e-5)
    assert not y[0, 1000:].any()


def test_masked_quantiles_match_nanquantile():
    torch.manual_seed(0)
    x = torch.randn(6, 300, 3)
    x[:, :50] = 0.0
    valid = torch.rand(6, 300) > 0.3
    Q, n = masked_quantiles(x, valid, (0.05, 0.5, 0.95))
    ref = torch.nanquantile(torch.where(valid[..., None], x, torch.nan), torch.tensor([0.05, 0.5, 0.95]), dim=1)
    assert torch.allclose(Q, ref, atol=1e-6)
    Q0, n0 = masked_quantiles(x, torch.zeros(6, 300, dtype=torch.bool), (0.5,))
    assert torch.all(Q0 == 0) and torch.all(n0 == 0)


def test_softminmax_flat_channel_is_zero_and_spo2_bypassed():
    x = torch.randn(2, 1200, 3)
    x[:, :, 1] = 7.0  # flat
    x[:, :, 2] = 0.2  # SpO2 affine value
    valid = torch.ones(2, 1200, dtype=torch.bool)
    valid[1, :240] = False
    y = SoftMinMax(skip_channels=(2,))(x, valid)
    assert torch.all(y[:, :, 1] == 0)
    assert torch.allclose(y[0, :, 2], x[0, :, 2]) and torch.all(y[1, :240, 2] == 0)
    assert torch.allclose(y[1, 240:, 2], x[1, 240:, 2])


def test_softminmax_fp16_input_and_autocast():
    """torch.nanquantile rejects fp16 (the crash in critique #8); SoftMinMax must not."""
    with pytest.raises(RuntimeError, match="float or double"):
        torch.nanquantile(torch.randn(2, 10, dtype=torch.float16), 0.5, dim=1)
    x = torch.randn(2, 1200, 2).half()
    v = torch.ones(2, 1200, dtype=torch.bool)
    ref = SoftMinMax()(x.float(), v)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = SoftMinMax()(x, v)
    assert y.dtype == torch.float32 and torch.allclose(y, ref)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_softminmax_on_mps_fp16_autocast_matches_cpu():
    x = torch.randn(4, 1200, 2).half()
    v = torch.rand(4, 1200) > 0.1
    ref = SoftMinMax()(x, v)
    with torch.autocast("mps", dtype=torch.float16):
        y = SoftMinMax()(x.to("mps"), v.to("mps"))
    assert y.dtype == torch.float32 and torch.allclose(y.cpu(), ref, atol=1e-5)


# ---------------------------------------------------------------- aggregators (design test 13)


def test_aggregators_on_hand_made_vectors():
    p = np.zeros((4, 30))
    p[0, 5:15] = 0.9  # 10 consecutive seconds
    p[1, 5:14] = 0.9
    p[1, 20] = 0.9  # 9 consecutive + 1 separate: 10 high seconds but not consecutive
    p[2] = np.linspace(0.0, 0.29, 30)
    p[3, :] = 0.4
    p[3, 0] = 1.0
    assert np.allclose(sec_to_epoch(p, "mean"), p.mean(axis=1))
    assert np.allclose(sec_to_epoch(p, "max"), [0.9, 0.9, 0.29, 1.0])
    assert np.allclose(sec_to_epoch(p, "k10"), [0.9, 0.9, 0.20, 0.4])
    assert np.allclose(sec_to_epoch(p, "c10"), [0.9, 0.0, np.linspace(0, 0.29, 30)[20], 0.4])
    with pytest.raises(ValueError):
        sec_to_epoch(p[:, :29], "mean")


def test_choose_aggregator_rule():
    base = {"mean": 0.800, "max": 0.790, "k10": 0.795, "c10": 0.8015}
    assert choose_aggregator(base)["chosen"] == "mean"  # within 0.002 of best
    assert choose_aggregator({**base, "c10": 0.8030})["chosen"] == "c10"
    assert choose_aggregator({**base, "mean": 0.81})["chosen"] == "mean"


# ---------------------------------------------------------------- trainer


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return build_packed_dataset(tmp_path_factory.mktemp("dltrain"))


@pytest.fixture()
def tiny_recipe(monkeypatch):
    monkeypatch.setitem(dl_train.RECIPES, "T", dl_train.Recipe("T", 4, 1, 2, 3, 1.0))
    return "T"


def _settings(recipe, **kw):
    base = dict(config="M", model_seed=42, recipe=recipe, amp=False, device="cpu", hidden=8,
                eval_batch=16, prefetch=2)
    base.update(kw)
    return dl_train.TrainSettings(**base)


def _trainer(ds, tmp, recipe, **kw):
    return dl_train.Trainer(_settings(recipe, **kw), ds["out"] / "train", ds["out"] / "val",
                            ds["split_json"], tmp / "ckpt", tmp / "out", log=lambda *a: None)


def _state(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_resume_is_deterministic_on_cpu(ds, tmp_path, tiny_recipe):
    """Design test 10: uninterrupted == paused-and-resumed == killed-mid-pass-and-resumed."""
    torch.set_num_threads(1)
    a = _trainer(ds, tmp_path / "a", tiny_recipe)
    assert a.run() == "done"
    ref = _state(tmp_path / "a" / "ckpt" / "last.pt")
    assert ref["state"]["pass_idx"] == 3 and len(ref["state"]["history"]) == 3

    for _ in range(3):  # one pass per invocation
        status = _trainer(ds, tmp_path / "b", tiny_recipe, stop_after_passes=1).run()
    assert status == "done"
    got = _state(tmp_path / "b" / "ckpt" / "last.pt")

    c = _trainer(ds, tmp_path / "c", tiny_recipe, ckpt_every_steps=1)
    calls = {"n": 0}
    orig = c._step

    def dying_step(bd):
        calls["n"] += 1
        if calls["n"] == 5:  # killed inside pass 1 after a mid-pass checkpoint
            raise KeyboardInterrupt
        return orig(bd)

    c._step = dying_step
    with pytest.raises(KeyboardInterrupt):
        c.run()
    mid = _state(tmp_path / "c" / "ckpt" / "last.pt")["state"]
    assert mid["pass_idx"] == 1 and mid["step_in_pass"] > 0
    assert _trainer(ds, tmp_path / "c", tiny_recipe, ckpt_every_steps=1).run() == "done"
    got_c = _state(tmp_path / "c" / "ckpt" / "last.pt")

    for other in (got, got_c):
        for k, v in ref["model"].items():
            assert torch.equal(v, other["model"][k]), k
        h_ref = [h["val_auc_mean"] for h in ref["state"]["history"]]
        assert [h["val_auc_mean"] for h in other["state"]["history"]] == h_ref
    va = pd.read_parquet(tmp_path / "a" / "out" / "val_predictions.parquet")
    vb = pd.read_parquet(tmp_path / "b" / "out" / "val_predictions.parquet")
    pd.testing.assert_frame_equal(va, vb)


def test_trainer_outputs_postproc_fitted_on_validation(ds, tmp_path, tiny_recipe):
    """Design test 7 (fit side): aggregator + threshold come from validation predictions."""
    t = _trainer(ds, tmp_path, tiny_recipe)
    t.run()
    out = tmp_path / "out"
    pp = json.loads((out / "postproc.json").read_text())
    vp = pd.read_parquet(out / "val_predictions.parquet")
    assert set(vp["subject_id"]) == set(ds["split"]["val"])
    assert pp["fitted_on"] == "validation sleep epochs"
    thr, _ = threshold_f1max(vp["apnoea_label"].to_numpy(), vp[f"p_{pp['aggregator']}"].to_numpy())
    assert thr == pp["threshold"]
    m = json.loads((out / "metrics.json").read_text())
    assert m["scale_pos_weight"] == 1.0 and m["n_val_sleep_epochs"] == len(vp)
    assert "Reading (a)" in m["schedule_note"] and "Reading (b)" in m["schedule_note"]
    hist = pd.read_csv(out / "history.csv")
    assert len(hist) == m["passes_run"] and {"val_auc_mean", "val_auc_c10", "lr"} <= set(hist.columns)


def test_trainer_refuses_test_data_and_test_ids(ds, tmp_path, tiny_recipe):
    """Design test 7 (guard side)."""
    s = _settings(tiny_recipe)
    with pytest.raises(TestDataRefused):
        dl_train.Trainer(s, ds["out"] / "test", ds["out"] / "val", ds["split_json"], tmp_path / "k", tmp_path / "o")
    with pytest.raises(TestDataRefused):
        dl_train.Trainer(s, ds["out"] / "train", ds["out"] / "test", ds["split_json"], tmp_path / "k", tmp_path / "o")
    with pytest.raises(TestDataRefused):  # a train folder passed as val
        dl_train.Trainer(s, ds["out"] / "train", ds["out"] / "train", ds["split_json"], tmp_path / "k",
                         tmp_path / "o")
    sp = ds["split"]
    leaky = write_split_json(tmp_path / "leak.json", sp["train"][1:], sp["val"], sp["test"] + sp["train"][:1])
    with pytest.raises(TestDataRefused):
        dl_train.Trainer(s, ds["out"] / "train", ds["out"] / "val", leaky, tmp_path / "k", tmp_path / "o")
    with pytest.raises(ValueError, match="deferred"):
        dl_train.Trainer(_settings(tiny_recipe, config="F6"), ds["out"] / "train", ds["out"] / "val",
                         ds["split_json"], tmp_path / "k", tmp_path / "o")


def test_resume_refuses_changed_configuration(ds, tmp_path, tiny_recipe):
    _trainer(ds, tmp_path, tiny_recipe, stop_after_passes=1).run()
    with pytest.raises(ValueError, match="config_hash"):
        _trainer(ds, tmp_path, tiny_recipe, model_seed=43).run()


def test_mirror_resume_picks_most_advanced(ds, tmp_path, tiny_recipe):
    s = _settings(tiny_recipe, stop_after_passes=1)
    mk = lambda ck: dl_train.Trainer(s, ds["out"] / "train", ds["out"] / "val", ds["split_json"],  # noqa: E731
                                     ck, tmp_path / "out", mirror_dir=tmp_path / "drive", log=lambda *a: None)
    mk(tmp_path / "ck1").run()
    assert (tmp_path / "drive" / "last.pt").exists() and (tmp_path / "drive" / "history.csv").exists()
    t2 = mk(tmp_path / "ck_fresh")  # fresh runtime: local ckpt dir empty
    t2.maybe_resume()
    assert t2.state["pass_idx"] == 1


def test_mirror_of_another_run_is_refused_before_copying(ds, tmp_path, tiny_recipe):
    """Relaunching seed 43 with seed 42's --mirror-dir must not overwrite seed 43's checkpoints."""
    import hashlib

    def mk(seed, ck, mirror, stop):
        return dl_train.Trainer(_settings(tiny_recipe, model_seed=seed, stop_after_passes=stop), ds["out"] / "train",
                                ds["out"] / "val", ds["split_json"], ck, tmp_path / f"out{seed}",
                                mirror_dir=mirror, log=lambda *a: None)

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    mk(43, tmp_path / "ck43", tmp_path / "mirror43", 1).run()
    mk(42, tmp_path / "ck42", tmp_path / "mirror42", 2).run()
    before = {n: digest(tmp_path / "ck43" / n) for n in ("last.pt", "best.pt")}
    mirror_before = {n: digest(tmp_path / "mirror42" / n) for n in ("last.pt", "best.pt")}
    with pytest.raises(ValueError, match="wrong --mirror-dir"):
        mk(43, tmp_path / "ck43", tmp_path / "mirror42", 1).run()
    assert {n: digest(tmp_path / "ck43" / n) for n in before} == before
    assert {n: digest(tmp_path / "mirror42" / n) for n in mirror_before} == mirror_before
    t = mk(43, tmp_path / "ck43", tmp_path / "mirror43", 1)  # the right mirror still resumes
    assert t.maybe_resume() and t.state["pass_idx"] == 1


def test_finalize_rewrites_a_stale_best_pt_from_last_pt(ds, tmp_path, tiny_recipe):
    """best.pt and last.pt reach Drive separately; a stale best.pt must never be scored."""
    import shutil

    def mk(ck, out):
        return dl_train.Trainer(_settings(tiny_recipe), ds["out"] / "train", ds["out"] / "val", ds["split_json"],
                                ck, out, mirror_dir=tmp_path / "drive", log=lambda *a: None)

    mk(tmp_path / "ck1", tmp_path / "out1").run()
    m1 = json.loads((tmp_path / "out1" / "metrics.json").read_text())
    assert m1["best_ckpt_rewritten_from_last"] is False and m1["best_auc_recheck_abs_diff"] == 0.0
    last = _state(tmp_path / "drive" / "last.pt")
    assert last["best_model"] is not None
    stale = _state(tmp_path / "drive" / "best.pt")  # e.g. the newest best.pt never reached Drive
    stale["model"] = {k: v + 0.25 if v.is_floating_point() else v for k, v in stale["model"].items()}
    stale["state"]["best_pass"] = -7
    torch.save(stale, tmp_path / "drive" / "best.pt")
    shutil.rmtree(tmp_path / "ck1")  # the VM-local checkpoints are lost
    mk(tmp_path / "ck2", tmp_path / "out2").run()
    m2 = json.loads((tmp_path / "out2" / "metrics.json").read_text())
    assert m2["best_ckpt_rewritten_from_last"] is True
    assert m2["val_auc_by_aggregator_best_ckpt"] == m1["val_auc_by_aggregator_best_ckpt"]
    assert m2["best_val_auc_mean_agg"] == m2["val_auc_by_aggregator_best_ckpt"]["mean"]
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "out1" / "val_predictions.parquet"),
                                  pd.read_parquet(tmp_path / "out2" / "val_predictions.parquet"))
    healed = _state(tmp_path / "ck2" / "best.pt")
    assert healed["state"]["best_pass"] == m2["best_pass"]
    for k, v in last["best_model"].items():
        assert torch.equal(healed["model"][k], v), k
    assert (tmp_path / "drive" / "best.pt").read_bytes() == (tmp_path / "ck2" / "best.pt").read_bytes()
    pp = json.loads((tmp_path / "out2" / "postproc.json").read_text())
    import hashlib
    assert pp["best_ckpt_sha256"] == hashlib.sha256((tmp_path / "ck2" / "best.pt").read_bytes()).hexdigest()


@pytest.mark.parametrize("recipe,lrs,stop_after", [
    ("A", [1e-3] * 5 + [1e-4] * 2, 7),  # cut on the 4th non-improving pass, stop on the 6th
    ("C", [1e-3] * 4 + [1e-4] * 1, 5),  # cut on the 3rd, stop on the 4th
])
def test_lr_cut_and_early_stop_pass_numbers(ds, tmp_path, recipe, lrs, stop_after):
    """Pins torch's ReduceLROnPlateau semantics (cut when bad passes > patience)."""
    t = _trainer(ds, tmp_path, recipe)
    t.train_pass = lambda: None
    t.validate = lambda: (0.8, {"mean": 0.8, "max": 0.8, "k10": 0.8, "c10": 0.8})  # pass 0 improves, then flat
    while not t.state["done"]:
        t.train_pass()
        t.end_pass()
    hist = t.state["history"]
    assert [h["lr"] for h in hist] == pytest.approx(lrs)
    assert len(hist) == stop_after and t.state["stop_reason"].startswith("early stop")
    f = dl_train.schedule_facts(dl_train.RECIPES[recipe])
    # pass 0 improves; pass k >= 1 is the k-th non-improving pass; the cut takes effect on the next pass
    assert f["lr_cut_on_consecutive_nonimproving_pass"] == lrs.index(1e-4) - 1
    assert f["stop_on_consecutive_nonimproving_pass"] == stop_after - 1
    assert f["max_passes_at_reduced_lr_before_stop"] == lrs.count(1e-4)
    assert "4th non-improving pass" in dl_train.SCHEDULE_NOTE and "at most 2 passes" in dl_train.SCHEDULE_NOTE


def test_nonfinite_steps_are_skipped_without_amp(ds, tmp_path, tiny_recipe):
    """--no-amp / CPU: a NaN loss or an overflowing fp32 gradient must not update the weights."""
    t = _trainer(ds, tmp_path, tiny_recipe, ckpt_every_steps=1)
    assert not t.amp
    calls = {"fwd": 0}

    def nan_logits(mod, inp, out):
        calls["fwd"] += 1
        return out * float("nan") if calls["fwd"] == 2 else out

    def inf_grad(g):
        return torch.full_like(g, float("inf")) if calls["fwd"] == 3 else g

    h1 = t.model.register_forward_hook(nan_logits)
    h2 = t.model.fc2.weight.register_hook(inf_grad)
    t.train_pass()
    h1.remove()
    h2.remove()
    assert calls["fwd"] == 3  # the pass has 3 batches: ok, NaN loss, inf gradient
    assert t.state["nonfinite"] == 2  # step 2 (NaN loss, no backward) and step 3 (inf gradient)
    assert dl_train._all_finite(p for p in t.model.parameters())
    ck = _state(tmp_path / "ckpt" / "last.pt")
    assert dl_train._all_finite(ck["model"].values()) and ck["state"]["nonfinite"] == 2
    with torch.no_grad():
        next(t.model.parameters())[0].fill_(float("nan"))
    before = (tmp_path / "ckpt" / "last.pt").read_bytes()
    with pytest.raises(FloatingPointError, match="not written"):
        t._save("last.pt", t._payload())
    assert (tmp_path / "ckpt" / "last.pt").read_bytes() == before


def test_mirror_io_errors_are_retried_not_fatal(ds, tmp_path, tiny_recipe, monkeypatch):
    drive = tmp_path / "drive"
    fail = {"on": True, "n": 0}
    orig = dl_train._copy_atomic

    def flaky(src, dst):
        if fail["on"] and Path(dst).parent == drive:
            fail["n"] += 1
            raise OSError(5, "Input/output error")
        return orig(src, dst)

    monkeypatch.setattr(dl_train, "_copy_atomic", flaky)
    logs = []

    def mk(stop):
        t = dl_train.Trainer(_settings(tiny_recipe, stop_after_passes=stop), ds["out"] / "train", ds["out"] / "val",
                             ds["split_json"], tmp_path / "ck", tmp_path / "out", mirror_dir=drive, log=logs.append)
        t.mirror_retry_wait_s = 0.0
        return t

    assert mk(1).run() == "paused"  # Drive failing: training continues, local checkpoints intact
    assert fail["n"] > 0 and any("retrying on the next save" in m for m in logs)
    assert (tmp_path / "ck" / "last.pt").exists() and not (drive / "last.pt").exists()
    t = mk(None)
    with pytest.raises(RuntimeError, match="still fails"):  # finalised locally, but Drive never recovered
        t.run()
    assert (tmp_path / "out" / "metrics.json").exists() and t.state["done"]
    fail["on"] = False
    assert mk(None).run() == "done"  # re-run: resumes done=True, re-finalises and mirrors everything
    for n in ("last.pt", "best.pt", "history.csv", "postproc.json", "metrics.json", "val_predictions.parquet"):
        assert (drive / n).exists(), n


def test_smoke_reports_loss_trend(ds, tmp_path, tiny_recipe):
    t = dl_train.Trainer(_settings(tiny_recipe), ds["out"] / "train", None, ds["split_json"], tmp_path / "k",
                         tmp_path / "o")
    r = t.smoke(12)
    assert r["n_batches"] == 12 and len(r["losses"]) == 12 and np.isfinite(r["losses"]).all()


# ---------------------------------------------------------------- predict / evaluate CLIs (validation only)


def test_predict_val_and_evaluate_val_cli(ds, tmp_path, tiny_recipe):
    from click.testing import CliRunner

    import evaluate_aim2_dl
    import predict_aim2_dl

    run_dir = tmp_path / "runs" / "M" / "seed42"
    t = dl_train.Trainer(_settings(tiny_recipe), ds["out"] / "train", ds["out"] / "val", ds["split_json"],
                         tmp_path / "ckpt", run_dir, log=lambda *a: None)
    t.run()
    prereg = tmp_path / "prereg.md"
    prereg.write_text("# pre-registration (test)\n")
    runner = CliRunner()
    base = ["--ckpt", str(tmp_path / "ckpt" / "best.pt"), "--postproc", str(run_dir / "postproc.json"),
            "--prereg", str(prereg), "--out-dir", str(tmp_path / "pred"), "--split-json", str(ds["split_json"]),
            "--device", "cpu", "--no-amp", "--eval-batch", "16"]
    r = runner.invoke(predict_aim2_dl.main, base + ["--data-dir", str(ds["out"] / "val"), "--split", "val",
                                                    "--refit-postproc"], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    re_pred = pd.read_parquet(tmp_path / "pred" / "val_predictions_cpu.parquet")
    trained = pd.read_parquet(run_dir / "val_predictions.parquet")
    assert np.array_equal(re_pred["pred_prob"].to_numpy(), trained["pred_prob"].to_numpy())
    prov = json.loads((tmp_path / "pred" / "predict_val_cpu.json").read_text())
    assert len(prov["prereg_sha256"]) == 64 and (tmp_path / "pred" / "postproc_cpu.json").exists()

    # the test path is refused without a freeze tag, and a test folder cannot be read as val
    r = runner.invoke(predict_aim2_dl.main, base + ["--data-dir", str(ds["out"] / "test"), "--split", "test"])
    assert r.exit_code != 0 and "freeze-tag" in r.output
    r = runner.invoke(predict_aim2_dl.main, base + ["--data-dir", str(ds["out"] / "test"), "--split", "val"])
    assert r.exit_code != 0 and isinstance(r.exception, TestDataRefused)
    r = runner.invoke(predict_aim2_dl.main, [a for a in base if a not in ("--prereg", str(prereg))]
                      + ["--data-dir", str(ds["out"] / "val"), "--split", "val"])
    assert r.exit_code != 0 and "prereg" in r.output

    # evaluate on validation predictions, with a fake LightGBM reference
    csv = tmp_path / "nsrr.csv"
    pd.DataFrame({"nsrrid": ds["split"]["val"], "ahi_a0h3a": [4.0, 22.0]}).to_csv(csv, index=False)
    ref_dir = tmp_path / "lgbm_ref"
    ref_dir.mkdir()
    rng = np.random.default_rng(0)
    ref_pred = trained[["subject_id", "epoch_idx", "apnoea_label"]].assign(
        pred_prob=rng.random(len(trained)), pred_label=np.int64(0))
    ref_pred.to_parquet(ref_dir / "val_predictions.parquet", index=False)
    from thesis_pipeline.extended_metrics import write_extended_metrics

    write_extended_metrics(ref_dir, ref_pred, n_bootstrap_subj=50)
    r = runner.invoke(evaluate_aim2_dl.main, ["--runs-root", str(tmp_path / "runs"), "--split", "val",
                                              "--ref", f"lgbm={ref_dir}", "--contrast", "M=lgbm",
                                              "--csv", str(csv), "--subject-metadata", str(tmp_path / "none.parquet"),
                                              "--n-bootstrap", "50"], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    summ = json.loads((tmp_path / "runs" / "evaluation_summary_val.json").read_text())
    c = summ["configs"]["M"]["contrasts"]["lgbm"]
    assert c["claim_rule_auc"]["verdict"].startswith("insufficient")  # one seed only
    me = json.loads((run_dir / "val_eval" / "metrics_extended.json").read_text())
    assert me["n_subjects_with_nsrr_ahi"] == 2
    assert not (run_dir / "metrics_extended.json").exists()  # val metrics kept apart from test outputs
