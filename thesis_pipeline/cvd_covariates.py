"""Baseline covariates for Aim 4, built from the NSRR SHHS-1 CSV (read-only).

Every PSG summary except sleeping heart rate is NSRR's own scoring. NSRR has
no whole-sleep heart-rate summary (its savb*/havb*/aavb* averages are heart
rate during desaturation or arousal events), so sleeping HR comes from the
Aim 1 pipeline: the median of the per-epoch R-peak heart rate over all sleep
epochs (`sleeping_hr_from_features`). Nothing comes from subject_metadata.parquet.
Column choices and cleaning rules follow the Aim 4 design spec (2026-09-29,
section 4) and are documented next to each term.

`build_covariates` returns un-imputed, un-scaled model terms. Imputation and
scaling are fitted on training subjects only, in the fit script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLINICAL = ["ln_age", "male", "race_black", "race_other", "ln_bmi", "smk", "ln_chol", "ln_hdl",
            "lipidmed", "dm", "ln_sbp", "trt", "betablk"]
PSG = ["l_rdi4p", "l_ahi_a0h3a", "l_pctlt90", "sao2nrem", "l_cai", "l_ai_all", "tst_h", "l_waso",
       "times34p", "timeremp", "hr_sleep"]
BINARY = ["male", "race_black", "race_other", "smk", "lipidmed", "dm", "trt", "betablk"]

# D'Agostino et al. 2008 Circulation 117:743-53, general CVD, lipid model, 10-year.
# Verified against the paper's worked examples (tests/test_cvd_covariates.py).
FRS = {
    "female": {"ln_age": 2.32888, "ln_chol": 1.20904, "ln_hdl": -0.70833, "ln_sbp_untreated": 2.76157,
               "ln_sbp_treated": 2.82263, "smk": 0.52873, "dm": 0.69154, "s0": 0.95012, "mean": 26.1931},
    "male": {"ln_age": 3.06117, "ln_chol": 1.12370, "ln_hdl": -0.93263, "ln_sbp_untreated": 1.93303,
             "ln_sbp_treated": 1.99881, "smk": 0.65451, "dm": 0.57367, "s0": 0.88936, "mean": 23.9802},
}


def _binary(s: pd.Series) -> pd.Series:
    return s.where(s.isin([0, 1])).astype(float)


def build_covariates(shhs1: pd.DataFrame) -> pd.DataFrame:
    """Model terms per nsrrid. `shhs1` must have lower-cased columns and nsrrid as index."""
    h = shhs1
    x = pd.DataFrame(index=h.index)
    x["ln_age"] = np.log(h["age_s1"])
    x["male"] = (h["gender"] == 1).astype(float)
    x["race_black"] = (h["race"] == 2).astype(float)
    x["race_other"] = (h["race"] == 3).astype(float)
    x["ln_bmi"] = np.log(h["bmi_s1"])
    # current smoker; smokstat_s1: 0 never, 1 current, 2 former; fill gaps from smknow15 (0/1)
    smk = (h["smokstat_s1"] == 1).astype(float).where(h["smokstat_s1"].notna())
    x["smk"] = smk.fillna(_binary(h["smknow15"]))
    x["ln_chol"] = np.log(h["chol"].where(h["chol"] >= 80))
    x["ln_hdl"] = np.log(h["hdl"].where(h["hdl"] >= 10))
    x["lipidmed"] = _binary(h["lipid1"])
    # diabetes: any of self-report / oral hypoglycaemic / insulin; 0 only when all three are 0
    dm_cols = h[["parrptdiab", "ohga1", "insuln1"]]
    x["dm"] = np.where((dm_cols == 1).any(axis=1), 1.0, np.where((dm_cols == 0).all(axis=1), 0.0, np.nan))
    # SBP: mean of the 2nd and 3rd SHHS-1 readings (SHHS protocol), fallback to the 1st
    sbp = h[["syst220", "syst320"]].mean(axis=1, skipna=False).fillna(h["syst120"])
    x["sbp"] = sbp
    x["ln_sbp"] = np.log(sbp)
    x["trt"] = _binary(h["htnmed1"])
    bb = h[["beta1", "betad1"]]
    x["betablk"] = np.where((bb == 1).any(axis=1), 1.0, np.where((bb == 0).all(axis=1), 0.0, np.nan))

    # NSRR PSG summaries
    x["rdi4p"] = h["rdi4p"]  # apnoeas + hypopnoeas, each with >=4 % desaturation (Punjabi 2009, Gottlieb 2010)
    x["l_rdi4p"] = np.log1p(h["rdi4p"])
    x["l_ahi_a0h3a"] = np.log1p(h["ahi_a0h3a"])  # sensitivity S4 only
    x["l_pctlt90"] = np.log1p(h["pctlt90"])
    x["sao2nrem"] = h["sao2nrem"]
    x["l_cai"] = np.log1p((h["ahi_a0h4"] - h["ahi_o0h4"]).clip(lower=0))  # derived central apnoea index
    x["l_ai_all"] = np.log1p(h["ai_all"])
    x["tst_h"] = h["slpprdp"] / 60.0
    x["l_waso"] = np.log1p(h["waso"])
    x["times34p"] = h["times34p"]
    x["timeremp"] = h["timeremp"]
    x["hrqual"] = h["hrqual"]  # NSRR HR-signal quality; 1 = < 2 h artefact-free, used to gate hr_sleep
    return x


def event_heart_rate(h: pd.DataFrame) -> pd.Series:
    """Mean heart rate DURING desaturation events (NOT sleeping HR; not used as a covariate).

    savb{n,r}{b,o}h are NSRR's "Average Heart Rate (NREM/REM, supine/other,
    all oxygen desaturations)"; the {nrem,rem}ep{b,o}p weights are time in each
    state (seconds). Undefined for subjects without events. Kept only to
    document why the design spec's "hr_sleep" was replaced (critic review
    2026-09-29).
    """
    pairs = [("savbnbh", "nremepbp"), ("savbnoh", "nremepop"), ("savbrbh", "remepbp"), ("savbroh", "remepop")]
    num = pd.Series(0.0, index=h.index)
    den = pd.Series(0.0, index=h.index)
    for hr, w in pairs:
        ok = h[hr].notna() & h[w].notna() & (h[w] > 0)
        num += np.where(ok, h[hr] * h[w], 0.0)
        den += np.where(ok, h[w], 0.0)
    out = (num / den).where(den > 0)
    return out.where((h["hrqual"] != 1) & out.between(35, 120))


def _sleep_hr_one(table, min_valid_epochs: int) -> float:
    t = table.to_pandas()
    hr = t.loc[t["sleep_stage"].isin(["N1", "N2", "N3", "REM"]), "hr_mean"]
    hr = hr[(hr >= 30) & (hr <= 150)]
    return float(hr.median()) if len(hr) >= min_valid_epochs else np.nan


def sleeping_hr_from_features(source, ids, min_valid_epochs: int = 240) -> pd.Series:
    """Median per-epoch R-peak heart rate over sleep epochs (N1-REM), per subject.

    `source` is the per-subject parquet directory or a tar archive of it
    (members `features/shhs1-<id>.parquet`, streamed without extracting).
    Epoch HR outside 30-150 bpm is treated as artefact. NaN when fewer than
    `min_valid_epochs` (default 240 = 2 h) plausible sleep epochs remain,
    mirroring NSRR's hrqual == 1 threshold of < 2 h artefact-free signal.
    """
    import io
    import re
    import tarfile
    from pathlib import Path

    import pyarrow.parquet as pq

    cols = ["sleep_stage", "hr_mean"]
    want = {int(i) for i in ids}
    out = {}
    source = Path(source)
    if source.is_file() and tarfile.is_tarfile(source):
        with tarfile.open(source) as tf:
            for m in tf:
                hit = re.fullmatch(r"(?:.*/)?shhs1-(\d+)\.parquet", m.name)
                if not (m.isfile() and hit) or int(hit.group(1)) not in want:
                    continue
                table = pq.read_table(io.BytesIO(tf.extractfile(m).read()), columns=cols)
                out[int(hit.group(1))] = _sleep_hr_one(table, min_valid_epochs)
    else:
        for sid in sorted(want):
            out[sid] = _sleep_hr_one(pq.read_table(source / f"shhs1-{sid}.parquet", columns=cols), min_valid_epochs)
    missing = want - set(out)
    if missing:
        raise FileNotFoundError(f"{len(missing)} subjects not found in {source}, e.g. {sorted(missing)[:5]}")
    return pd.Series(out, name="hr_sleep", dtype=float).sort_index()


def self_reported_cvd(shhs1: pd.DataFrame) -> pd.Series:
    """Any self-reported physician-diagnosed MI, stroke, HF, CABG, angioplasty or angina.

    1 if any item is 1; 0 if at least one item is answered and none is 1;
    NaN if all six are missing (15 NY subjects), to be imputed.
    """
    items = shhs1[["mi15", "stroke15", "hf15", "cabg15", "ca15", "angina15"]]
    return pd.Series(np.where((items == 1).any(axis=1), 1.0, np.where(items.notna().any(axis=1), 0.0, np.nan)),
                     index=shhs1.index, name="prev_cvd_sr")


def frs_linear_predictor(age, chol, hdl, sbp, trt, smk, dm, male) -> np.ndarray:
    """Sex-specific D'Agostino 2008 sum beta*x, centred on the published sex-specific mean."""
    age, chol, hdl, sbp = (np.asarray(v, dtype=float) for v in (age, chol, hdl, sbp))
    trt, smk, dm, male = (np.asarray(v, dtype=float) for v in (trt, smk, dm, male))
    out = np.empty_like(age)
    for sex, mask in (("male", male == 1), ("female", male == 0)):
        c = FRS[sex]
        lp = (c["ln_age"] * np.log(age[mask]) + c["ln_chol"] * np.log(chol[mask]) + c["ln_hdl"] * np.log(hdl[mask])
              + np.where(trt[mask] == 1, c["ln_sbp_treated"], c["ln_sbp_untreated"]) * np.log(sbp[mask])
              + c["smk"] * smk[mask] + c["dm"] * dm[mask])
        out[mask] = lp - c["mean"]
    return out


def frs_risk10(lp_centred: np.ndarray, male) -> np.ndarray:
    """Published 10-year general-CVD risk from the centred linear predictor."""
    male = np.asarray(male, dtype=float)
    s0 = np.where(male == 1, FRS["male"]["s0"], FRS["female"]["s0"])
    return 1.0 - s0 ** np.exp(np.asarray(lp_centred, dtype=float))
