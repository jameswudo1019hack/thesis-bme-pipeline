"""4 Hz model-input signals for the Aim 2 raw-signal DL arm (Olsen 2020 front-end).

All outputs are sampled on the grid t_m = m / 4 s, m = 0 .. 4 * n_sec - 1, with t = 0 at
the EDF start, so 30-s epoch k covers samples [120 k, 120 k + 120).

ECG -> RR and EDR (``ecg_to_rr_edr``)
    1. Polyphase resample 125 -> 256 Hz (Olsen interpolates to 256 Hz).
    2. 0.5 Hz high-pass (Butterworth order 2, zero phase) and 60 Hz notch (Q = 30).
       Janbakhshi 2018 uses a 60 Hz notch plus median-filter baseline removal; the
       phase-space area is translation-invariant, so the high-pass is an equivalent,
       cheaper baseline step (declared).
    3. R peaks with the Aim 1 detector (``features._r_peaks_neurokit``: NeuroKit2
       ``ecg_peaks(method="neurokit")``, Pan-Tompkins fallback, same plausibility floor
       as ``features.r_peaks``).
    4. The peak train is cut at raw gaps > ``gap_s`` (3 s) and NeuroKit2
       ``signal_fixpeaks(method="Kubios", iterative=True)`` runs on each segment, so
       Kubios never invents beats inside a signal dropout. QC counts the correction
       extent directly (raw peaks removed or moved, peaks added or moved to). NeuroKit's
       per-type ``info`` lists only the LAST iteration, so those counts are logged as
       ``kubios_lastiter_<type>`` and are not a correction fraction.
    5. RR_k = t_k - t_{k-1} at t_k. Intervals > ``gap_s`` are gaps, not RR samples.
       RR is clipped to [0.3, 2.0] s.
    6. Resampling to 4 Hz: cubic spline inside each clean run (>= 4 samples), linear
       interpolation across gaps, hold before the first and after the last sample. The
       spline can overshoot, so the 4 Hz RR is clipped to [0.3, 2.0] s again (the clipped
       fraction is logged as ``rr4_clipped_frac``). The per-sample gap mask is returned
       and summarised in QC.
EDR
    "psa" (default): Janbakhshi & Shamsollahi, IRBM 2018;39(3):206-218, section 2.2.2.
        For each R peak take 40 ms before and after it (+-10 samples at 256 Hz),
        embed in 2-D with delay tau = 8 ms (2 samples; dm = 2), and take the area of the
        polygon traced by the trajectory (shoelace, closed). Beat values are corrected
        with a width-5 median filter: a value more than 2x or less than 0.5x the
        median-filter output is replaced by it (our reading of "variations larger than
        2 times from output of median filter").
    "ramp": R-wave amplitude of the filtered ECG at each R peak (declared fallback),
        same correction.
    Resampling to 4 Hz uses the same runs, gaps and holds as RR but a shape-preserving
    PCHIP interpolant inside each run instead of the cubic spline: EDR has no
    physiological clip range, and the cubic spline overshot far below 0 next to
    outlier-beat clusters that survive the width-5 median correction (frontend 1.0:
    impossible negative areas in 78 of 80 pilot subjects). PCHIP stays inside the range
    of neighbouring beat values, so a phase-space area stays >= 0 (also enforced).

Respiratory effort (10 Hz) -> 4 Hz: ``resample_poly(x, 2, 5)`` (Kaiser anti-alias).
SpO2 (1 Hz, percent): dropouts (< 50 % or non-finite) interpolated across gaps
<= 60 s; longer gaps set to 0 after the fixed affine (pct - 95) / 5.
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
from scipy import signal as sps
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import median_filter

FS_OUT = 4
FS_WORK = 256
RR_CLIP = (0.3, 2.0)
GAP_S = 3.0
PSA_HALF_S = 0.040
PSA_TAU_S = 0.008
FRONTEND_VERSION = "dl-frontend-1.1"  # 1.1: PCHIP EDR, Kubios extent QC
KUBIOS_TYPES = ("ectopic", "missed", "extra", "longshort")

__all__ = [
    "FS_OUT",
    "FRONTEND_VERSION",
    "ecg_to_rr_edr",
    "edr_beats",
    "interp_runs",
    "resp_to_4hz",
    "spo2_to_4hz",
    "hr_agreement",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    f = Fraction(int(fs_out), int(fs_in))
    if f == 1:
        return np.asarray(x, dtype=np.float64)
    return sps.resample_poly(np.asarray(x, dtype=np.float64), f.numerator, f.denominator)


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.full(n - len(x), x[-1] if len(x) else 0.0)])


def interp_runs(
    t: np.ndarray,
    v: np.ndarray,
    seg: np.ndarray,
    t_out: np.ndarray,
    min_spline: int = 4,
    method: str = "cubic",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample irregular samples (t, v) onto ``t_out``.

    ``seg`` labels contiguous clean runs (non-decreasing integers). Inside a run with
    >= ``min_spline`` samples a ``method`` interpolant is used ("cubic": not-a-knot
    cubic spline, can overshoot; "pchip": shape-preserving, stays within the range of
    the neighbouring samples); elsewhere linear interpolation between neighbouring
    samples, and a hold outside the sampled range.

    Returns (values, gap_mask) where gap_mask is True for output samples outside
    every run (inside a gap or beyond the first / last sample).
    """
    interp = {"cubic": CubicSpline, "pchip": PchipInterpolator}.get(method)
    if interp is None:
        raise ValueError(f"unknown interpolation method {method!r}")
    t = np.asarray(t, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    seg = np.asarray(seg)
    out = np.zeros(len(t_out), dtype=np.float64)
    gap = np.ones(len(t_out), dtype=bool)
    if len(t) == 0:
        return out, gap
    out[:] = np.interp(t_out, t, v)
    bounds = np.flatnonzero(np.diff(seg)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(t)]])
    for a, b in zip(starts, ends):
        lo = int(np.searchsorted(t_out, t[a], side="left"))
        hi = int(np.searchsorted(t_out, t[b - 1], side="right"))
        if hi <= lo:
            continue
        gap[lo:hi] = False
        if b - a >= min_spline:
            out[lo:hi] = interp(t[a:b], v[a:b], extrapolate=False)(t_out[lo:hi])
    return out, gap


def _median5_correct(e: np.ndarray) -> tuple[np.ndarray, int]:
    """Janbakhshi 2018 correction: replace values > 2x or < 0.5x the width-5 median."""
    if len(e) < 5:
        return e, 0
    med = median_filter(e, size=5, mode="nearest")
    pos = med > 0
    bad = pos & ((e > 2.0 * med) | (e < 0.5 * med))
    out = np.where(bad, med, e)
    return out, int(bad.sum())


def edr_beats(x: np.ndarray, peaks: np.ndarray, fs: int, method: str = "psa") -> np.ndarray:
    """Per-beat EDR values; NaN where the beat window leaves the signal."""
    peaks = np.asarray(peaks, dtype=np.int64)
    out = np.full(len(peaks), np.nan)
    if method == "ramp":
        ok = (peaks >= 0) & (peaks < len(x))
        out[ok] = x[peaks[ok]]
        return out
    if method != "psa":
        raise ValueError(f"unknown EDR method {method!r}")
    half = int(round(PSA_HALF_S * fs))
    tau = max(1, int(round(PSA_TAU_S * fs)))
    ok = (peaks - half >= 0) & (peaks + half < len(x))
    if not ok.any():
        return out
    idx = peaks[ok, None] + np.arange(-half, half + 1)[None, :]
    w = x[idx]
    X = w[:, :-tau]
    Y = w[:, tau:]
    area = 0.5 * np.abs(np.sum(X * np.roll(Y, -1, axis=1) - np.roll(X, -1, axis=1) * Y, axis=1))
    out[ok] = area
    return out


def _detect_peaks(x: np.ndarray, fs: int) -> tuple[np.ndarray, str]:
    """Aim 1 detector logic (features.r_peaks), also reporting which branch ran."""
    from .features import _r_peaks_neurokit, _r_peaks_pantompkins

    min_expected = max(10, int(x.size / fs / 2.5))
    try:
        peaks = _r_peaks_neurokit(x, fs)
        if peaks.size >= min_expected:
            return peaks, "neurokit"
    except Exception:  # noqa: BLE001 - mirror features.r_peaks
        pass
    return _r_peaks_pantompkins(x, fs), "pantompkins"


def _fixpeaks_segments(peaks: np.ndarray, fs: int, gap_s: float) -> tuple[np.ndarray, dict]:
    """Kubios fixpeaks on each run of peaks separated by raw gaps > gap_s.

    The correction extent is measured directly per segment: ``removed_or_moved`` counts
    raw peaks absent from the corrected train (setdiff raw - clean) and
    ``added_or_moved_to`` counts corrected peaks absent from the raw train (setdiff
    clean - raw). A moved beat appears once in each. NeuroKit2's ``info`` holds only the
    artifacts of the last iteration it applied (``artifacts = new_artifacts`` in
    ``_signal_fixpeaks_kubios``), so its per-type counts are kept as ``lastiter_<type>``.
    """
    import neurokit2 as nk

    counts = {f"lastiter_{k}": 0 for k in KUBIOS_TYPES}
    extent = {"removed_or_moved": 0, "added_or_moved_to": 0}
    if len(peaks) < 2:
        return peaks, {**counts, **extent, "n_segments": int(len(peaks) > 0), "n_fix_failed": 0}
    cut = np.flatnonzero(np.diff(peaks) > gap_s * fs) + 1
    pieces = np.split(peaks, cut)
    fixed = []
    n_fail = 0
    for seg in pieces:
        if len(seg) < 10:
            fixed.append(seg)
            continue
        try:
            info, clean = nk.signal_fixpeaks(seg, sampling_rate=fs, iterative=True, method="Kubios")
            clean = np.asarray(clean, dtype=np.int64)
            for k in KUBIOS_TYPES:
                counts[f"lastiter_{k}"] += int(len(info.get(k, [])))
            extent["removed_or_moved"] += int(len(np.setdiff1d(seg, clean)))
            extent["added_or_moved_to"] += int(len(np.setdiff1d(clean, seg)))
            fixed.append(clean)
        except Exception:  # noqa: BLE001 - keep raw peaks, count the failure
            n_fail += 1
            fixed.append(seg)
    out = np.unique(np.concatenate(fixed).astype(np.int64))
    return out, {**counts, **extent, "n_segments": len(pieces), "n_fix_failed": n_fail}


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------


def ecg_to_rr_edr(
    ecg: np.ndarray,
    fs: int = 125,
    n_sec: int | None = None,
    edr: str = "psa",
    fs_out: int = FS_OUT,
    rr_clip: tuple[float, float] = RR_CLIP,
    gap_s: float = GAP_S,
    fs_work: int = FS_WORK,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """ECG (physical units, or sign-correct digital units) -> (rr4 [s], edr4 [a.u.], qc).

    ``rr4`` and ``edr4`` have length ``n_sec * fs_out``. If the ECG is flat or too few
    beats are found both are zeros and ``qc["ok"]`` is False.
    """
    t0 = time.perf_counter()
    ecg = np.asarray(ecg, dtype=np.float64).ravel()
    if n_sec is None:
        n_sec = len(ecg) // fs
    n_out = int(n_sec) * fs_out
    zeros = np.zeros(n_out, dtype=np.float32)
    qc: dict = {"ok": False, "reason": "", "edr_method": edr, "frontend": FRONTEND_VERSION}
    finite = np.isfinite(ecg)
    if not finite.all():
        ecg = np.where(finite, ecg, 0.0)
    qc["nonfinite_frac"] = float(1.0 - finite.mean()) if len(ecg) else 1.0
    if len(ecg) < fs * 60 or float(np.std(ecg)) == 0.0 or len(np.unique(ecg[: fs * 600])) < 3:
        qc["reason"] = "flat_or_short_ecg"
        return zeros, zeros.copy(), qc

    x = _resample(ecg, fs, fs_work)
    sos = sps.butter(2, 0.5, btype="highpass", fs=fs_work, output="sos")
    x = sps.sosfiltfilt(sos, x)
    if fs_work > 2 * 61:
        b, a = sps.iirnotch(60.0, 30.0, fs=fs_work)
        x = sps.filtfilt(b, a, x)
    t1 = time.perf_counter()

    peaks_raw, detector = _detect_peaks(x, fs_work)
    peaks_raw = np.unique(np.asarray(peaks_raw, dtype=np.int64))
    peaks_raw = peaks_raw[(peaks_raw >= 0) & (peaks_raw < len(x))]
    qc["detector"] = detector
    qc["n_peaks_raw"] = int(len(peaks_raw))
    t2 = time.perf_counter()
    if len(peaks_raw) < max(10, n_sec / 10.0):
        qc["reason"] = "too_few_peaks"
        return zeros, zeros.copy(), qc

    peaks, fix = _fixpeaks_segments(peaks_raw, fs_work, gap_s)
    peaks = peaks[(peaks >= 0) & (peaks < len(x))]
    t3 = time.perf_counter()
    qc.update({f"kubios_{k}": v for k, v in fix.items()})
    qc["n_peaks_fixed"] = int(len(peaks))
    n_raw = max(len(peaks_raw), 1)
    qc["kubios_removed_or_moved_frac"] = float(fix["removed_or_moved"] / n_raw)
    qc["kubios_added_or_moved_to_frac"] = float(fix["added_or_moved_to"] / n_raw)

    tb = peaks / float(fs_work)
    rr = np.diff(tb)
    is_gap = rr > gap_s
    seg = np.concatenate([[0], np.cumsum(is_gap)])  # segment id per beat
    rr_ok = ~is_gap
    rr_t = tb[1:][rr_ok]
    rr_v = rr[rr_ok]
    rr_seg = seg[1:][rr_ok]
    clipped = (rr_v < rr_clip[0]) | (rr_v > rr_clip[1])
    qc["rr_clip_frac"] = float(clipped.mean()) if len(rr_v) else 1.0
    rr_v = np.clip(rr_v, *rr_clip)
    qc["n_gaps"] = int(is_gap.sum())

    t_out = np.arange(n_out, dtype=np.float64) / fs_out
    rr4, gap4 = interp_runs(rr_t, rr_v, rr_seg, t_out)
    qc["rr4_clipped_frac"] = float(np.mean((rr4 < rr_clip[0]) | (rr4 > rr_clip[1]))) if n_out else 0.0
    rr4 = np.clip(rr4, *rr_clip)
    qc["rr_gap_frac"] = float(gap4.mean()) if n_out else 1.0
    qc["hr_median_bpm"] = float(60.0 / np.median(rr_v)) if len(rr_v) else float("nan")

    e = edr_beats(x, peaks, fs_work, method=edr)
    good = np.isfinite(e)
    e_t, e_v, e_seg = tb[good], e[good], seg[good]
    e_v, n_corr = _median5_correct(e_v)
    qc["edr_corrected_frac"] = float(n_corr / max(len(e_v), 1))
    if len(e_v):
        med_e = float(np.median(e_v))
        qc["edr_beat_max_over_median"] = float(np.max(e_v) / med_e) if med_e > 0 else float("nan")
    edr4, _ = interp_runs(e_t, e_v, e_seg, t_out, method="pchip")
    if edr == "psa":
        edr4 = np.maximum(edr4, 0.0)  # an area is >= 0; PCHIP already guarantees it
    t4 = time.perf_counter()

    mid = x[len(x) // 4: 3 * len(x) // 4]
    if len(mid) > 10 and np.std(mid) > 0:
        qc["skew"] = float(np.mean(((mid - mid.mean()) / mid.std()) ** 3))
    qc["t_filter_s"] = round(t1 - t0, 3)
    qc["t_peaks_s"] = round(t2 - t1, 3)
    qc["t_fixpeaks_s"] = round(t3 - t2, 3)
    qc["t_rr_edr_s"] = round(t4 - t3, 3)
    qc["t_total_s"] = round(t4 - t0, 3)
    qc["ok"] = bool(len(rr_v) >= 10)
    if not qc["ok"]:
        qc["reason"] = "too_few_clean_rr"
        return zeros, zeros.copy(), qc
    return rr4.astype(np.float32), edr4.astype(np.float32), qc


def resp_to_4hz(x: np.ndarray, fs: int = 10, n_out: int | None = None) -> np.ndarray:
    """Effort / airflow channel -> 4 Hz by polyphase resampling (10 Hz: up 2, down 5)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = np.where(np.isfinite(x), x, 0.0)
    y = _resample(x, fs, FS_OUT)
    if n_out is None:
        n_out = len(x) * FS_OUT // fs
    return _fit_length(y, int(n_out)).astype(np.float32)


def spo2_to_4hz(
    pct: np.ndarray,
    n_out: int,
    fs_in: int = 1,
    max_gap_s: float = 60.0,
    dropout_below: float = 50.0,
) -> tuple[np.ndarray, dict]:
    """SpO2 percent -> (pct - 95) / 5 at 4 Hz with dropout handling. Returns (x4, qc)."""
    pct = np.asarray(pct, dtype=np.float64).ravel()
    n = len(pct)
    qc: dict = {"ok": False}
    bad = ~np.isfinite(pct) | (pct < dropout_below) | (pct > 100.5)
    qc["spo2_dropout_frac"] = float(bad.mean()) if n else 1.0
    if n == 0 or bad.all():
        qc["reason"] = "no_valid_spo2"
        return np.zeros(int(n_out), dtype=np.float32), qc
    t_in = np.arange(n, dtype=np.float64) / fs_in
    good = ~bad
    filled = np.interp(t_in, t_in[good], pct[good])
    # runs of bad samples
    edges = np.diff(np.concatenate([[0], bad.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    long_mask = np.zeros(n, dtype=bool)
    n_long = 0
    for a, b in zip(starts, ends):
        if (b - a) / fs_in > max_gap_s:
            long_mask[a:b] = True
            n_long += 1
    scaled = (filled - 95.0) / 5.0
    scaled[long_mask] = 0.0
    qc["spo2_long_gap_frac"] = float(long_mask.mean())
    qc["spo2_n_long_gaps"] = int(n_long)
    t_out = np.arange(int(n_out), dtype=np.float64) / FS_OUT
    x4 = np.interp(t_out, t_in, scaled)
    qc["ok"] = True
    return x4.astype(np.float32), qc


def hr_agreement(rr4: np.ndarray, hr_bpm: np.ndarray, fs_out: int = FS_OUT) -> dict:
    """Compare 60 / RR (sampled at each second) with the oximeter pulse rate (1 Hz)."""
    rr1 = np.asarray(rr4, dtype=np.float64)[::fs_out]
    hr = np.asarray(hr_bpm, dtype=np.float64).ravel()
    n = min(len(rr1), len(hr))
    rr1, hr = rr1[:n], hr[:n]
    ok = (rr1 > 0) & np.isfinite(hr) & (hr >= 30) & (hr <= 200)
    if ok.sum() < 60:
        return {"hr_check_n": int(ok.sum())}
    hr_ecg = 60.0 / rr1[ok]
    d = np.abs(hr_ecg - hr[ok])
    r = float(np.corrcoef(hr_ecg, hr[ok])[0, 1]) if np.std(hr[ok]) > 0 else float("nan")
    return {
        "hr_check_n": int(ok.sum()),
        "hr_check_median_abs_bpm": float(np.median(d)),
        "hr_check_frac_within_5bpm": float((d <= 5).mean()),
        "hr_check_r": r,
    }
