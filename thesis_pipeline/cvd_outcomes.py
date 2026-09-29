"""SHHS-1 cardiovascular outcome and cohort construction for Aim 4.

Reads the NSRR CSVs read-only and returns one row per subject with a parent-
cohort label, prevalent-CVD flags and (event, time-in-days) pairs for every
outcome used in the thesis. Every count the design relies on is asserted in
`audit()`; the dataset is frozen at NSRR release 0.15.0, so a changed count
means the data or the code changed and the run must stop.

Parent cohort (`coh`)
---------------------
NSRR withholds the site variable. The label is inferred from which prevalent-
history columns each parent study supplied (design spec 2026-09-29, section 1.1,
amended after the critic review the same day):

    NY    prev_mi is NaN (no CVD surveillance at all)            760
    TUC   shhs1_tcvd == 1, or prev_ang and prev_revpro both NaN  911
    ARIC  prev_ang NaN and prev_revpro present                  1915
    CHS   prev_ang present and prev_revpro present              1229
    FHS-A prev_ang present, prev_revpro NaN, no SF-36 date,
          white (NSRR lists 688 Framingham Offspring)            687
    FHS-B same prevalence pattern but SF-36 form date present or
          non-white: a separate site with its own forms           300

The NY group matches the 760 NYU-Cornell subjects that Gottlieb et al. 2010
excluded for lack of outcome data. Use the label as a Cox stratum and for
stratified censoring weights only; it is not a biological covariate.

Outcomes (days since the SHHS-1 PSG; non-events censored at `censdate`)
-----------------------------------------------------------------------
    cvd       mi | stroke | chf | revasc_proc | ptca | cabg | cvd_death  (primary)
    chd       mi | revasc_proc | ptca | cabg | chd_death                  (Gottlieb 2010)
    hf        chf
    stroke    stroke
    cvd_death cvd_death
    death     vital == 0 (all-cause)
    any_cvd   NSRR any_cvd flag (sensitivity only: inconsistent in ARIC)

    hard_cvd  mi | stroke | chf | cvd_death  (co-primary: no procedures; in ARIC
              about half of composite first events are PCI/CABG)

Event time is the earliest dated component. An event with no dated component
falls back to `censdate`. An event dated AFTER `censdate` is censored at
`censdate` (E=0) by default, because only event subjects could otherwise
gain person-time beyond their recorded follow-up; `keep_post_censdate=True`
keeps such events at their own date (sensitivity analysis).

Prevalent CVD
-------------
    P2 (sensitivity) any of prev_mi, prev_mip, prev_stk, prev_chf, prev_revpro,
                     prev_ang > 0 (NaN = not recorded = 0; prev_ang is absent by
                     design in ARIC/TUC and prev_revpro in FHS/TUC)
    P3 (primary)     P2, or self-reported mi15, stroke15, hf15, cabg15, ca15
                     (coronary angioplasty), angina15 == 1 (8 = don't know is
                     not an exclusion). P3 is primary because the adjudicated
                     columns are missing by design in some sites (prev_ang in
                     ARIC/TUC, prev_revpro in FHS/TUC), so P2 alone excludes
                     angina in CHS/FHS only; self-report makes the exclusion
                     symmetric, as in Gottlieb 2010.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY_FILE = "shhs-cvd-summary-dataset-0.15.0.csv"
SHHS1_FILE = "shhs1-dataset-0.15.0.csv"

OUTCOMES: dict[str, tuple[list[str], list[str]]] = {
    # name: (event-indicator expressions, date columns)
    "cvd": (["mi", "stroke", "chf", "revasc_proc", "ptca", "cabg", "cvd_death"],
            ["mi_date", "stk_date", "chf_date", "revpro_date", "ptca_date", "cabg_date", "cvd_dthdt"]),
    "chd": (["mi", "revasc_proc", "ptca", "cabg", "chd_death"],
            ["mi_date", "revpro_date", "ptca_date", "cabg_date", "chd_dthdt"]),
    "hf": (["chf"], ["chf_date"]),
    "stroke": (["stroke"], ["stk_date"]),
    "cvd_death": (["cvd_death"], ["cvd_dthdt"]),
    "hard_cvd": (["mi", "stroke", "chf", "cvd_death"], ["mi_date", "stk_date", "chf_date", "cvd_dthdt"]),
}
PREV_P2 = ["prev_mi", "prev_mip", "prev_stk", "prev_chf", "prev_revpro", "prev_ang"]
PREV_SELF_REPORT = ["mi15", "stroke15", "hf15", "cabg15", "ca15", "angina15"]


def load_tables(csv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary and SHHS-1 tables indexed by nsrrid, with lower-cased column names."""
    summary = pd.read_csv(csv_dir / SUMMARY_FILE)
    shhs1 = pd.read_csv(csv_dir / SHHS1_FILE, low_memory=False)
    for df in (summary, shhs1):
        df.columns = df.columns.str.lower()
    return summary.set_index("nsrrid"), shhs1.set_index("nsrrid")


def cohort_label(summary: pd.DataFrame, shhs1: pd.DataFrame) -> pd.Series:
    h = shhs1.reindex(summary.index)
    tcvd = h["shhs1_tcvd"].fillna(0) == 1
    ang, rev = summary["prev_ang"].notna(), summary["prev_revpro"].notna()
    coh = pd.Series("UNASSIGNED", index=summary.index, dtype=object)
    coh[summary["prev_mi"].isna()] = "NY"
    rest = coh == "UNASSIGNED"
    coh[rest & (tcvd | (~ang & ~rev))] = "TUC"
    rest = coh == "UNASSIGNED"
    coh[rest & ~ang & rev] = "ARIC"
    coh[rest & ang & rev] = "CHS"
    fhs = rest & ang & ~rev
    second_site = h["date25"].notna() | (h["race"] != 1)
    coh[fhs & ~second_site] = "FHS-A"
    coh[fhs & second_site] = "FHS-B"
    assert not (coh == "UNASSIGNED").any(), "cohort label left subjects unassigned"
    return coh


def _event_time(summary: pd.DataFrame, indicators: list[str], dates: list[str], keep_post_censdate: bool):
    s = summary
    flags = []
    for c in indicators:
        flags.append(s[c] == 1 if c.endswith("death") else s[c].fillna(0) > 0)
    event = np.logical_or.reduce(flags)
    first = s[dates].min(axis=1, skipna=True)
    time = pd.Series(np.where(event, first, s["censdate"]), index=s.index, dtype=float)
    undated = event & first.isna()
    time[undated] = s.loc[undated, "censdate"]
    after = event & (time > s["censdate"])
    event = event.astype(int)
    if not keep_post_censdate:
        event = np.where(after, 0, event)
        time[after] = s.loc[after, "censdate"]
    return pd.Series(event, index=s.index), time, undated, after


def build(csv_dir: Path, keep_post_censdate: bool = False) -> pd.DataFrame:
    """One row per summary subject: cohort, prevalence flags, and event/time per outcome."""
    summary, shhs1 = load_tables(csv_dir)
    out = pd.DataFrame(index=summary.index)
    out["coh"] = cohort_label(summary, shhs1)
    out["surveillance"] = summary["prev_mi"].notna()
    out["prev_p2"] = (summary[PREV_P2].fillna(0) > 0).any(axis=1)
    sr = shhs1[PREV_SELF_REPORT].reindex(summary.index)
    out["prev_self_report"] = (sr == 1).any(axis=1)
    out["prev_p3"] = out["prev_p2"] | out["prev_self_report"]
    out["censdate"] = summary["censdate"].astype(float)
    for name, (ind, dates) in OUTCOMES.items():
        e, t, undated, after = _event_time(summary, ind, dates, keep_post_censdate)
        out[f"E_{name}"], out[f"T_{name}"] = e, t
        out[f"undated_{name}"], out[f"after_censdate_{name}"] = undated, after
    out["E_death"] = (summary["vital"] == 0).astype(int)
    out["T_death"] = summary["censdate"].astype(float)
    # NSRR's own flag, timed like the derived composite (sensitivity S1)
    out["E_any_cvd"] = (summary["any_cvd"] == 1).astype(int)
    out["T_any_cvd"] = np.where(out["E_any_cvd"] == 1, out["T_cvd"], out["censdate"])
    out.attrs["keep_post_censdate"] = keep_post_censdate
    return out


def select(out: pd.DataFrame, ids_with_features: np.ndarray | None, definition: str = "P3") -> pd.DataFrame:
    """Incident-CVD analysis cohort: surveillance, free of prevalent CVD, censdate > 0."""
    df = out
    if ids_with_features is not None:
        df = df[df.index.isin(ids_with_features)]
    prev = {"P2": "prev_p2", "P3": "prev_p3"}[definition]
    return df[df["surveillance"] & ~df[prev] & (df["censdate"] > 0)]


def audit(out: pd.DataFrame) -> dict:
    """Assert every count the Aim 4 design depends on (NSRR 0.15.0, default build). Returns the counts."""
    assert len(out) == 5802, len(out)
    coh = out["coh"].value_counts().to_dict()
    assert coh == {"ARIC": 1915, "CHS": 1229, "FHS-A": 687, "FHS-B": 300, "TUC": 911, "NY": 760}, coh
    counts = {"censdate_zero_removed": int(((out["censdate"] <= 0) & out["surveillance"] & ~out["prev_p2"]).sum())}
    for d in ("P2", "P3"):
        sel = select(out, None, d)
        counts[f"n_{d}"] = len(sel)
        for k in ("cvd", "hard_cvd", "any_cvd", "chd", "hf", "stroke", "cvd_death", "death"):
            counts[f"events_{d}_{k}"] = int(sel[f"E_{k}"].sum())
        counts[f"cvd_post_censdate_{d}"] = int(sel["after_censdate_cvd"].sum())
        counts[f"hard_cvd_post_censdate_{d}"] = int(sel["after_censdate_hard_cvd"].sum())
        for k in OUTCOMES:
            assert (sel[f"T_{k}"] > 0).all() and sel[f"T_{k}"].notna().all(), f"bad times for {k} ({d})"
            assert (sel[f"T_{k}"] <= sel["censdate"]).all() or out.attrs.get("keep_post_censdate"),                 f"event time beyond censdate for {k} ({d})"
    expected = {
        "censdate_zero_removed": 15,
        "n_P2": 4343, "events_P2_cvd": 897, "events_P2_hard_cvd": 711, "events_P2_any_cvd": 830,
        "events_P2_chd": 572, "events_P2_hf": 391, "events_P2_stroke": 184, "events_P2_cvd_death": 215,
        "events_P2_death": 839, "cvd_post_censdate_P2": 6, "hard_cvd_post_censdate_P2": 4,
        "n_P3": 4080, "events_P3_cvd": 773, "events_P3_hard_cvd": 611, "events_P3_any_cvd": 721,
        "events_P3_chd": 487, "events_P3_hf": 328, "events_P3_stroke": 161, "events_P3_cvd_death": 187,
        "events_P3_death": 739, "cvd_post_censdate_P3": 6, "hard_cvd_post_censdate_P3": 4,
    }
    if out.attrs.get("keep_post_censdate"):
        return counts
    bad = {k: (counts[k], v) for k, v in expected.items() if counts[k] != v}
    assert not bad, f"cohort/outcome counts differ from the verified design (got, expected): {bad}"
    return counts
