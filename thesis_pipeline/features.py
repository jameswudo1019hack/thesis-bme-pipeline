"""Per-epoch feature extractors.

Each function takes a 1-D signal + sampling frequency and returns a dict of
1-D arrays (length ``n_epochs``). Feature dicts are merged into the epoch
frame by ``scripts/process_batch.process_subject``.

Features implemented:
    SpO2:         mean, min, max, std, ODI 3% + 4% event counts, desat depth
    ECG / HRV:    mean HR, SDNN, RMSSD, pNN50
                  R-peak detection: NeuroKit2 primary, Pan-Tompkins fallback
    EEG band power: absolute + relative δ/θ/α/σ/β, total power, spectral edge 95%
    Respiratory:  airflow/thor/abdo RMS, thor-abdo correlation, breath rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps

from .epochs import EPOCH_SECONDS

# Bumped whenever feature extraction logic changes. Written into every parquet
# so downstream code can filter mixed-version cohorts cleanly.
FEATURES_VERSION = "2026-05-01-phase1batch-v1"


# Base columns used as inputs for contextual_features. Anything in the epoch
# frame matching one of these gets a set of lag/lead/rolling derivatives.
ROLLING_BASE_COLS = (
    "spo2_mean", "spo2_min", "spo2_max", "spo2_std",
    "spo2_sampen",  # Phase 1 Exp 1 — sample entropy on per-epoch SpO2 (1 Hz)
    "spo2_psd_apnea_band", "spo2_psd_total", "spo2_psd_apnea_ratio",  # Phase 1 Exp 2 — Welch PSD on 2-min window
    "odi3_count", "odi4_count", "desat_depth",
    "hypoxic_burden_epoch",  # NEW T4 — Azarbarzin 2019 hypoxic burden
    "hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
    "hrv_sampen",  # Phase 1 Exp 1 — sample entropy on per-epoch RR intervals
    "cpc_amp_cv", "cpc_amp_iqr",  # Phase 1 Exp 3 — CPC-proxy via R-peak amplitude variability
    "hrv_lf_power", "hrv_hf_power", "hrv_lf_hf_ratio", "hrv_total_power_freq",  # NEW T3
    "eeg_delta_power", "eeg_theta_power", "eeg_alpha_power",
    "eeg_sigma_power", "eeg_beta_power", "eeg_total_power", "eeg_spectral_edge95",
    "eeg_delta_rel", "eeg_theta_rel", "eeg_alpha_rel", "eeg_sigma_rel", "eeg_beta_rel",
    "ecg_band_low_power", "ecg_band_mid_low_power", "ecg_band_mid_power",  # Phase 1 Exp 4
    "ecg_band_mid_high_power", "ecg_band_high_power", "ecg_band_total_power",
    "ecg_band_low_rel", "ecg_band_mid_low_rel", "ecg_band_mid_rel",
    "ecg_band_mid_high_rel", "ecg_band_high_rel",
    "resp_airflow_rms", "resp_airflow_std", "resp_thor_rms", "resp_abdo_rms",
    "resp_thor_abdo_corr", "resp_breath_rate_bpm",
    "position_right_frac", "position_left_frac", "position_supine_frac",  # NEW T5
    "position_prone_frac", "position_upright_frac",  # NEW T5
)


def _sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Sample entropy (SampEn) for a 1-D series.

    Phase 1 Exp 1 (added 2026-05-01). Sample entropy quantifies time-series
    irregularity / complexity. For apnoea-detection contexts:
      - Healthy / regular breathing → SpO₂ is smooth, RR intervals stable → low SampEn
      - Apnoeic / arousal-driven breathing → irregular SpO₂ recovery, RR scatter → higher SampEn

    Parameters
    ----------
    x : np.ndarray
        1-D series. NaNs are dropped before computation.
    m : int
        Embedding dimension (default 2 — Pincus & Goldberger 1994 standard).
    r : float | None
        Tolerance. If None, defaults to 0.2 × std of the (NaN-dropped) series
        (Richman & Moorman 2000 standard for physiological signals).

    Returns
    -------
    float
        SampEn value, or NaN if the series is too short / pathological.

    Notes
    -----
    Uses the textbook O(N²) definition (Richman & Moorman 2000). For our
    typical inputs (30-sample SpO₂ epochs, 20–50-sample RR-interval epochs)
    the cost is negligible — ~1 ms per epoch. NeuroKit2's
    ``nk.entropy_sample`` is a drop-in alternative but adds dependency
    overhead and behaves identically on these short series.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    # Need at least m+2 points to form one (m+1)-template comparison
    if n < m + 2:
        return float("nan")
    std = float(arr.std(ddof=0))
    if std <= 0.0:
        # Constant signal — SampEn is conventionally 0 (or undefined; NaN is safer)
        return float("nan")
    if r is None:
        r = 0.2 * std

    # Build embedding matrices for length m and m+1
    def _phi(mm: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(arr, mm)  # (n-mm+1, mm)
        # Chebyshev distance (max abs diff) between every pair of templates
        # Vectorized: distance matrix is (T, T) where T = n-mm+1
        # For our short series (n ~ 30–50), T ~ 28–48, so the (T, T) matrix
        # is ~50×50 — fine in memory.
        diffs = np.abs(templates[:, None, :] - templates[None, :, :]).max(axis=2)
        # Exclude self-matches (Richman 2000: i != j)
        T = templates.shape[0]
        within = (diffs <= r).sum() - T  # subtract self-matches on diagonal
        return int(within // 2)  # symmetric, count unordered pairs

    A = _phi(m + 1)  # (m+1)-template matches
    B = _phi(m)      # m-template matches
    if B == 0 or A == 0:
        # No template matches — undefined log; conventionally NaN here
        # (some implementations return log(N) as upper bound; we use NaN
        # so downstream median-imputation handles it consistently)
        return float("nan")
    return float(-np.log(A / B))


def _spo2_psd_per_epoch(
    spo2_1hz: np.ndarray,
    n_epochs: int,
    window_sec: int = 120,
    apnea_band: tuple[float, float] = (0.01, 0.067),
) -> dict[str, np.ndarray]:
    """Per-epoch Welch PSD on a sliding 2-min window centred at each epoch.

    Phase 1 Exp 2. Captures the apnoea-cycle frequency content of SpO₂.

    Periodic apnoea (Cheyne-Stokes-like or OSA-cluster) creates a characteristic
    oscillation in SpO₂ at the period of the apnoea-recovery cycle, typically
    20–100 s (= 0.01–0.05 Hz). A 30-s epoch is too short to resolve this band
    (frequency resolution would be ~0.033 Hz, coarser than the band itself), so
    we use a 2-min window centred on the epoch midpoint, giving ~0.008 Hz
    resolution.

    Returns
    -------
    dict with three keys:
      "apnea_band":   power in the apnoea-cycle band (0.01–0.067 Hz default).
      "total":        total spectral power across the resolvable band.
      "apnea_ratio":  apnea_band / total (dimensionless, 0–1).

    All three arrays length ``n_epochs``. NaN for epochs with too few finite
    samples in the window or zero variance.
    """
    out_band = np.full(n_epochs, np.nan, dtype=np.float32)
    out_total = np.full(n_epochs, np.nan, dtype=np.float32)
    out_ratio = np.full(n_epochs, np.nan, dtype=np.float32)

    half_win = window_sec // 2
    half_epoch = EPOCH_SECONDS // 2
    n_signal = spo2_1hz.size

    for ep_idx in range(n_epochs):
        center = ep_idx * EPOCH_SECONDS + half_epoch
        start = max(0, center - half_win)
        end = min(n_signal, center + half_win)
        seg = spo2_1hz[start:end]
        seg = seg[np.isfinite(seg)]
        if seg.size < 30:  # too few samples for reliable PSD
            continue
        if seg.std(ddof=0) == 0.0:  # constant signal — no spectral content
            continue
        # nperseg ≤ segment length; use min(64, seg.size) for short segments
        nperseg = min(64, seg.size)
        try:
            f, psd = sps.welch(seg, fs=1.0, nperseg=nperseg, detrend="linear")
        except Exception:
            continue
        if psd.size == 0:
            continue
        # Frequency bin width Δf — needed to integrate power over a band
        df = f[1] - f[0] if f.size > 1 else 1.0 / nperseg
        band_mask = (f >= apnea_band[0]) & (f <= apnea_band[1])
        total_power = float(psd.sum() * df)
        if band_mask.any():
            band_power = float(psd[band_mask].sum() * df)
        else:
            band_power = 0.0
        out_band[ep_idx] = band_power
        out_total[ep_idx] = total_power
        if total_power > 0:
            out_ratio[ep_idx] = band_power / total_power

    return {"apnea_band": out_band, "total": out_total, "apnea_ratio": out_ratio}


def _epoch_view(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Reshape a 1-D signal into (n_epochs, samples_per_epoch). Drops the
    trailing partial epoch if recording length isn't a multiple of 30 s."""
    samples_per_epoch = int(round(EPOCH_SECONDS * sfreq))
    n_epochs = signal.size // samples_per_epoch
    usable = signal[: n_epochs * samples_per_epoch]
    return usable.reshape(n_epochs, samples_per_epoch)


def _n_epochs(signal_size: int, sfreq: float) -> int:
    return signal_size // int(round(EPOCH_SECONDS * sfreq))


# ─────────────────────────────────────────────────────────────────────────
# SpO2
# ─────────────────────────────────────────────────────────────────────────

# Values below this are treated as sensor artifacts (disconnected probe etc.).
# Real clinical desaturations rarely go below 50% and 0/low values dominate
# when the sensor loses contact.
SPO2_MIN_VALID = 50.0


def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling maximum over a one-sided look-back window — **NaN-aware**.

    Returns array of same length as ``x``. Within each contiguous finite run
    (segments separated by NaN samples), each position is the max of up to
    ``window`` preceding finite samples *from the same run*. NaN samples
    themselves return NaN; the rolling window never carries pre-NaN data into
    a post-NaN run.

    Why this shape: SpO₂ artefacts (sensor disconnect / motion) are NaN-masked
    upstream. A naive trailing rolling-max would skip the NaN samples but still
    use pre-gap values as the baseline, registering the first post-gap sample
    as a phantom desaturation. Resetting the rolling window at every NaN gap
    avoids this.

    History:
      audit-v1/v2: scipy.ndimage.maximum_filter1d with NaN padding — was actually
        centred, leaked future samples into the baseline. /codex:adversarial-review caught.
      audit-v3: pd.Series().rolling().max() — proper trailing, but skipped NaNs and
        propagated stale baselines across gaps. Codex caught.
      audit-v4: any-NaN-in-100s-window mask → over-rejected (1 s blip suppressed
        99 s of detection). Codex caught.
      audit-v5: 5-sample recovery mask → still leaked pre-gap baseline for 95 s
        post-recovery because rolling().max() skipped NaN. Codex caught.
      audit-v6: groupby on NaN cumsum so each post-NaN run gets its own baseline.
    """
    s = pd.Series(x)
    is_nan = s.isna()
    # Each NaN bumps the group id by 1; within a group the rolling().max()
    # only sees that group's samples (the NaN itself rolls in as NaN and is
    # skipped, but no pre-NaN values are in the group either).
    group_id = is_nan.cumsum()
    max_ = s.groupby(group_id, sort=False).transform(
        lambda g: g.rolling(window=window, min_periods=1).max()
    )
    max_[is_nan] = np.nan
    return max_.to_numpy()


def _desaturation_events(
    spo2_1hz: np.ndarray, threshold: float = 3.0, min_duration: int = 10
) -> list[tuple[int, int, float, float]]:
    """Find desaturation events in a 1-Hz SpO2 trace.

    An event starts when SpO2 drops by ≥ ``threshold`` % from a rolling
    100-second baseline (max of preceding 100 s) and lasts ≥ ``min_duration``
    seconds. Returns a list of (start_sec, duration_sec, nadir_percent,
    baseline_percent) where ``baseline_percent`` is the rolling-max baseline
    value at event start — consistent with the ODI detection baseline.
    """
    if spo2_1hz.size == 0:
        return []
    baseline = _rolling_max(spo2_1hz, window=100)
    drop = baseline - spo2_1hz
    is_desat = drop >= threshold

    events: list[tuple[int, int, float, float]] = []
    i = 0
    n = is_desat.size
    while i < n:
        if is_desat[i]:
            j = i
            while j < n and is_desat[j]:
                j += 1
            if (j - i) >= min_duration:
                nadir = float(np.nanmin(spo2_1hz[i:j]))
                baseline_at_event_start = float(baseline[i])  # rolling-max at start
                events.append((i, j - i, nadir, baseline_at_event_start))
            i = j
        else:
            i += 1
    return events


def spo2_features(
    signal: np.ndarray,
    sfreq: float,
    sleep_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Per-epoch SpO2 statistics and desaturation counts.

    ``signal`` is expected in % (0–100). SHHS stores SpO2 at 125 Hz but the
    underlying update rate is ~1 Hz (piecewise-constant). We downsample to
    1 Hz for event detection but compute stats on the full-rate signal.

    Parameters
    ----------
    signal : np.ndarray
        SpO2 signal at the EDF native rate.
    sfreq : float
        EDF native sampling rate.
    sleep_mask : np.ndarray | None
        Optional epoch-level boolean array (length ``n_epochs``; True for
        N1/N2/N3/REM sleep epochs). When provided, ODI count features
        (``odi3_count``, ``odi4_count``) are zeroed and ``desat_depth`` is
        set to NaN at non-sleep epoch indices, so that per-night summaries
        derived from these counts are sleep-only. Per-epoch SpO2 statistics
        (mean/min/max/std) are unaffected — they are intrinsically per-epoch.
        Default ``None`` keeps the original behaviour (all epochs counted).
    """
    # mne may or may not scale — if values look like fractions (0–1), rescale
    s = np.asarray(signal, dtype=float)
    if np.isfinite(s).any() and np.nanmax(np.abs(s[np.isfinite(s)])) <= 2.0:
        s = s * 100.0

    # Mask artifacts: set them to NaN so stats ignore them
    s_clean = np.where(s >= SPO2_MIN_VALID, s, np.nan)

    ep = _epoch_view(s_clean, sfreq)
    n = ep.shape[0]

    with np.errstate(invalid="ignore"):
        mean_ = np.nanmean(ep, axis=1)
        min_ = np.nanmin(ep, axis=1)
        max_ = np.nanmax(ep, axis=1)
        std_ = np.nanstd(ep, axis=1)

    # Downsample to 1 Hz for event detection
    step = int(round(sfreq))
    spo2_1hz = s_clean[::step]

    odi3_per_epoch = np.zeros(n, dtype=np.int16)
    odi4_per_epoch = np.zeros(n, dtype=np.int16)
    deepest_desat = np.full(n, np.nan)  # max drop from per-event baseline within epoch

    for start, dur, nadir, base in _desaturation_events(spo2_1hz, threshold=3.0):
        ep_idx = start // EPOCH_SECONDS
        if 0 <= ep_idx < n:
            odi3_per_epoch[ep_idx] += 1

    for start, dur, nadir, base in _desaturation_events(spo2_1hz, threshold=4.0):
        ep_idx = start // EPOCH_SECONDS
        if 0 <= ep_idx < n:
            odi4_per_epoch[ep_idx] += 1
            # Depth uses per-event baseline (rolling-max at event start),
            # matching the ODI detection baseline — avoids underestimating
            # depth for subjects with chronic nocturnal hypoxaemia.
            depth = (base - nadir) if (np.isfinite(nadir) and np.isfinite(base)) else np.nan
            prev = deepest_desat[ep_idx]
            if not np.isfinite(prev) or (np.isfinite(depth) and depth > prev):
                deepest_desat[ep_idx] = depth

    # Apply sleep_mask: zero ODI counts and NaN desat_depth at non-sleep epochs.
    if sleep_mask is not None:
        mask = np.asarray(sleep_mask, dtype=bool)
        # Align mask length to n (clip or pad)
        if mask.size < n:
            pad = np.zeros(n - mask.size, dtype=bool)
            mask = np.concatenate([mask, pad])
        elif mask.size > n:
            mask = mask[:n]
        non_sleep = ~mask
        odi3_per_epoch[non_sleep] = 0
        odi4_per_epoch[non_sleep] = 0
        deepest_desat[non_sleep] = np.nan

    # Phase 1 Exp 1 — sample entropy on 1 Hz SpO2 per epoch (~30 samples).
    # Captures breathing-cycle irregularity that statistical moments miss.
    spo2_1hz_ep = _epoch_view(spo2_1hz, 1.0)
    spo2_sampen = np.array(
        [_sample_entropy(spo2_1hz_ep[i], m=2) for i in range(spo2_1hz_ep.shape[0])],
        dtype=np.float32,
    )
    # Pad/clip to align with n epochs from full-rate epoch_view
    if spo2_sampen.size < n:
        spo2_sampen = np.concatenate(
            [spo2_sampen, np.full(n - spo2_sampen.size, np.nan, dtype=np.float32)]
        )
    elif spo2_sampen.size > n:
        spo2_sampen = spo2_sampen[:n]

    # Phase 1 Exp 2 — Welch PSD on 2-min sliding window centred per epoch.
    # Captures apnoea-cycle frequency content (typical OSA cycles 20-100s →
    # 0.01-0.05 Hz band). 30-sample epochs are too short for useful spectral
    # resolution; a 2-min window gives ~0.008 Hz frequency bins.
    psd_features = _spo2_psd_per_epoch(spo2_1hz, n_epochs=n)

    return {
        "spo2_mean": mean_,
        "spo2_min": min_,
        "spo2_max": max_,
        "spo2_std": std_,
        "spo2_sampen": spo2_sampen,
        "spo2_psd_apnea_band": psd_features["apnea_band"],
        "spo2_psd_total": psd_features["total"],
        "spo2_psd_apnea_ratio": psd_features["apnea_ratio"],
        "odi3_count": odi3_per_epoch.astype(np.float32),
        "odi4_count": odi4_per_epoch.astype(np.float32),
        "desat_depth": deepest_desat,
    }


def _derive_search_window(
    spo2_1hz: np.ndarray,
    events: list,
    apnoea_kinds: set,
) -> tuple[int, int]:
    """Derive per-subject search window per Azarbarzin 2019.

    Stacks event-aligned SpO₂ traces (60 s before to 120 s after event end)
    into a matrix, takes the column-wise nanmean, then finds the argmax in
    each half to locate the pre-event baseline peak and post-event recovery
    peak. The search window is defined as [PRE - pre_peak, post_peak - PRE]
    seconds relative to event end.

    Returns (lo_sec, hi_sec) — offsets from event end (both positive, meaning
    lo_sec seconds before event end and hi_sec seconds after). Falls back to
    (30, 90) if fewer than 5 qualifying events are found or the derived window
    falls outside sanity bounds.
    """
    PRE = 60    # seconds before event end to capture
    POST = 120  # seconds after event end to capture
    DEFAULT = (30, 90)

    segments = []
    for ev in events:
        if ev.kind not in apnoea_kinds:
            continue
        ev_end = int(round(float(ev.start_sec + ev.duration_sec)))
        lo, hi = ev_end - PRE, ev_end + POST
        if lo < 0 or hi > spo2_1hz.size:
            continue
        seg = spo2_1hz[lo:hi]
        if seg.size == PRE + POST and np.isfinite(seg).any():
            segments.append(seg)

    if len(segments) < 5:
        return DEFAULT

    stacked = np.stack(segments)
    avg = np.nanmean(stacked, axis=0)
    # avg[0..PRE-1]: pre-event region; avg[PRE..PRE+POST-1]: post-event region.
    # Per-column support: how many events provide a finite value at this offset?
    # Codex audit-v5 finding: the previous gate used ``isfinite(avg)`` which
    # counted a column as "covered" if even one event contributed. Need actual
    # support counts on the stacked segments to reject sparse / disjoint cases.
    # audit-v6: ceil rounding so "majority" is genuinely majority for 5/7-event
    # subjects (was int(0.5*5)=2 i.e. minority).
    import math
    support = np.isfinite(stacked).sum(axis=0)
    min_support = max(2, math.ceil(0.5 * len(segments)))
    pre_well_supported = float((support[:PRE] >= min_support).mean())
    post_well_supported = float((support[PRE:] >= min_support).mean())
    if pre_well_supported < 0.5 or post_well_supported < 0.5:
        return DEFAULT
    # argmax of each region gives the SpO₂ peak (highest = least desaturated).
    pre_peak_idx = int(np.nanargmax(avg[:PRE]))
    post_peak_idx = int(np.nanargmax(avg[PRE:])) + PRE
    # Verify the chosen peak columns themselves have adequate support — a
    # well-covered half can still pick a poorly-supported peak column.
    if support[pre_peak_idx] < min_support or support[post_peak_idx] < min_support:
        return DEFAULT

    lo_sec = PRE - pre_peak_idx   # seconds before event end
    hi_sec = post_peak_idx - PRE  # seconds after event end

    # Sanity bounds: clamp to physiologically plausible values
    if lo_sec < 5 or hi_sec < 5 or lo_sec > 90 or hi_sec > 120:
        return DEFAULT

    return (lo_sec, hi_sec)


def hypoxic_burden_features(
    spo2_signal: np.ndarray,
    sfreq: float,
    events: list,  # list[RespiratoryEvent] from io.read_nsrr_xml
    n_epochs: int,
) -> dict[str, np.ndarray]:
    """Per-epoch hypoxic burden (Azarbarzin 2019 §Methods).

    Hypoxic burden quantifies the depth × duration of SpO₂ desaturations
    associated with respiratory events. The Azarbarzin et al. 2019
    *Eur Heart J* paper defines this as the cleanest CVD-mortality predictor
    on SHHS+MrOS (Q5/Q1 fully-adjusted HR = 1.96 [1.11–3.43] on SHHS).

    Implementation (per Azarbarzin 2019):

    1. **Event list**: every annotated apnoea + hypopnoea (obstructive,
       central, mixed, hypopnea) — regardless of desaturation threshold.
       The event list is supplied by ``read_nsrr_xml``.
    2. **Baseline**: max SpO₂ in the 100 s **prior to event END** (NOT
       event start — this is the documented gotcha in the paper).
    3. **Search window** — derived per subject from the average event-aligned
       SpO₂ trace (``_derive_search_window``): stack 60 s pre/120 s post
       segments, take column-wise nanmean, find pre/post argmax peaks to
       define the window. Falls back to (-30 s, +90 s) if < 5 qualifying
       events exist or the derived window falls outside sanity bounds.
    4. **Burden = AUC of (baseline − SpO₂)** over the search window where
       the deficit is positive. At 1 Hz integration, the AUC numerically
       equals sum of deficits in %·s.
    5. **Per-epoch attribution**: each event's burden is added to the
       epoch index containing the event END, using the unrounded event-end
       time to avoid off-by-one epoch attribution.

    Parameters
    ----------
    spo2_signal : np.ndarray
        SpO₂ signal at the EDF native rate (typically 125 Hz, but
        underlying is ~1 Hz so we downsample to 1 Hz internally).
    sfreq : float
        EDF native sampling rate of ``spo2_signal``.
    events : list[RespiratoryEvent]
        Annotated respiratory events from ``read_nsrr_xml``. Filtered
        internally to apnoeas + hypopnoeas.
    n_epochs : int
        Number of 30-s epochs in the recording.

    Returns
    -------
    dict[str, np.ndarray]
        Single column ``hypoxic_burden_epoch`` of length ``n_epochs``
        (units: %·s).

    Reference
    ---------
    Azarbarzin A et al. 2019. "The hypoxic burden of sleep apnoea
    predicts cardiovascular disease-related mortality: the Osteoporotic
    Fractures in Men Study and the Sleep Heart Health Study."
    Eur Heart J 40(14):1149–1157. DOI 10.1093/eurheartj/ehy624.
    """
    # Downsample SpO₂ to 1 Hz (mirrors the convention in spo2_features)
    s = np.asarray(spo2_signal, dtype=float)
    if np.isfinite(s).any() and np.nanmax(np.abs(s[np.isfinite(s)])) <= 2.0:
        s = s * 100.0
    s_clean = np.where(s >= SPO2_MIN_VALID, s, np.nan)
    step = max(1, int(round(sfreq)))
    spo2_1hz = s_clean[::step]
    n_samples_1hz = spo2_1hz.size

    burden = np.zeros(n_epochs, dtype=np.float32)
    if n_samples_1hz == 0 or not events:
        return {"hypoxic_burden_epoch": burden}

    # Constants (per Azarbarzin 2019 §Methods)
    BASELINE_WINDOW_SEC = 100  # max SpO₂ in 100 s before event END

    APNOEA_EVENT_KINDS = {
        "Obstructive Apnea",
        "Central Apnea",
        "Mixed Apnea",
        "Hypopnea",
        "Obstructive apnea",
        "Central apnea",
        "Mixed apnea",
    }

    # Derive per-subject search window; fallback (30, 90) for sparse subjects.
    SEARCH_LO_SEC, SEARCH_HI_SEC = _derive_search_window(
        spo2_1hz, events, APNOEA_EVENT_KINDS
    )

    for ev in events:
        if ev.kind not in APNOEA_EVENT_KINDS:
            continue
        ev_end_sec = float(ev.start_sec + ev.duration_sec)
        ev_end_idx = int(round(ev_end_sec))
        if ev_end_idx <= 0 or ev_end_idx > n_samples_1hz:
            continue

        # Baseline: max SpO₂ in 100 s prior to event END
        baseline_lo = max(0, ev_end_idx - BASELINE_WINDOW_SEC)
        if baseline_lo >= ev_end_idx:
            continue
        baseline_segment = spo2_1hz[baseline_lo:ev_end_idx]
        with np.errstate(invalid="ignore"):
            baseline = float(np.nanmax(baseline_segment))
        if not np.isfinite(baseline):
            continue

        # Search window: subject-specific offsets from event end
        sw_lo = max(0, ev_end_idx - SEARCH_LO_SEC)
        sw_hi = min(n_samples_1hz, ev_end_idx + SEARCH_HI_SEC)
        if sw_hi <= sw_lo:
            continue
        spo2_seg = spo2_1hz[sw_lo:sw_hi]
        # AUC of positive deficit at 1 Hz
        with np.errstate(invalid="ignore"):
            deficit = np.clip(baseline - spo2_seg, 0.0, None)
        auc = float(np.nansum(deficit))  # units: %·s

        # Attribute the AUC to the epoch containing the event END.
        # Use unrounded ev_end_sec for epoch attribution to avoid the int(round())
        # shifting ~3 % of fractional-second events into the next epoch.
        ep_idx = int(ev_end_sec // EPOCH_SECONDS)
        if 0 <= ep_idx < n_epochs:
            burden[ep_idx] += auc

    return {"hypoxic_burden_epoch": burden}


# ─────────────────────────────────────────────────────────────────────────
# ECG / HRV
# ─────────────────────────────────────────────────────────────────────────


def _r_peaks_pantompkins(ecg: np.ndarray, sfreq: float) -> np.ndarray:
    """Legacy R-peak detector (Pan–Tompkins 1985 style).

    Bandpass 5–15 Hz → differentiate → square → moving-average envelope →
    adaptive-threshold peak finding. Kept as a fallback for subjects where
    NeuroKit2 fails.
    """
    ecg = np.asarray(ecg, dtype=float)
    finite = np.isfinite(ecg)
    if not finite.all():
        ecg = np.where(finite, ecg, np.nanmedian(ecg[finite]) if finite.any() else 0.0)

    nyq = sfreq / 2.0
    low, high = 5.0 / nyq, 15.0 / nyq
    if not (0 < low < high < 1):
        return np.array([], dtype=int)
    sos = sps.butter(4, [low, high], btype="band", output="sos")
    filt = sps.sosfiltfilt(sos, ecg)
    diff = np.diff(filt, prepend=filt[0])
    squared = diff * diff
    win = max(3, int(0.15 * sfreq))
    envelope = np.convolve(squared, np.ones(win) / win, mode="same")

    thresh = np.percentile(envelope, 90) * 0.3
    min_dist = max(1, int(0.2 * sfreq))
    peaks, _ = sps.find_peaks(envelope, distance=min_dist, height=thresh)
    return peaks


def _r_peaks_neurokit(ecg: np.ndarray, sfreq: float) -> np.ndarray:
    """R-peak detector using NeuroKit2's default hybrid method.

    NeuroKit2's ``ecg_peaks(method='neurokit')`` pipeline: clean signal
    (bandpass + powerline filter) → QRS detection via the neurokit2-native
    hybrid that chains Engelse–Zeelenberg + custom refinement. Published
    benchmarks (Makowski et al. 2021) show it outperforms Pan–Tompkins on
    noisy clinical ECG, which matches the apnoea subjects in SHHS.
    """
    import neurokit2 as nk  # deferred import — heavy

    ecg = np.asarray(ecg, dtype=float)
    finite = np.isfinite(ecg)
    if not finite.any():
        return np.array([], dtype=int)
    if not finite.all():
        ecg = np.where(finite, ecg, float(np.nanmedian(ecg[finite])))

    _, info = nk.ecg_peaks(ecg, sampling_rate=int(sfreq), method="neurokit")
    peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    return peaks


def r_peaks(ecg: np.ndarray, sfreq: float) -> np.ndarray:
    """Detect R-peak sample indices. Primary: NeuroKit2 hybrid. Fallback: Pan–Tompkins.

    The fallback triggers if NeuroKit2 errors out (rare — happens on
    pathological signals or size mismatches) or returns fewer than 3 peaks
    on a signal we'd expect to have hundreds. Logged via a warning-free
    exception swallow; callers handle NaN-on-insufficient-peaks downstream.
    """
    try:
        peaks = _r_peaks_neurokit(ecg, sfreq)
        # Plausibility: for an 8-hour ECG at 60-80 bpm we expect ~30,000 peaks
        min_expected = max(10, int(ecg.size / sfreq / 2.5))  # HR floor ~24 bpm
        if peaks.size >= min_expected:
            return peaks
    except Exception:
        pass
    return _r_peaks_pantompkins(ecg, sfreq)


def _robust_rr(rr_ms: np.ndarray) -> np.ndarray:
    """de Chazal 2003 §III.C two-heuristic Robust-RR correction.

    Returns a corrected RR-interval array. Two passes:
      (1) Spurious-merge: any RR < 0.5× local-median is treated as a
          spurious detection — dropped and combined with the next RR.
      (2) Missed-subdivide: any RR ≥ 1.8× local-median is treated as
          one or more missed beats — subdivided into round(RR / median)
          equal intervals.
    Local-median is computed over a centred 5-beat window. Output length
    differs from input length by the net effect of merges and subdivides.

    Reference:
        de Chazal et al. 2003, "Automated processing of the single-lead
        electrocardiogram for the detection of obstructive sleep apnoea",
        IEEE Trans. Biomed. Eng. 50(6):686–696, §III.C.

    Parameters
    ----------
    rr_ms : np.ndarray
        RR intervals in milliseconds (length N).

    Returns
    -------
    np.ndarray
        Corrected RR intervals in milliseconds (length may differ from N).
    """
    if rr_ms.size < 3:
        return rr_ms
    rr_in = np.asarray(rr_ms, dtype=float)
    # Centred 5-beat local median (handles boundary by truncated window)
    local_med = np.empty_like(rr_in)
    for i in range(rr_in.size):
        lo, hi = max(0, i - 2), min(rr_in.size, i + 3)
        local_med[i] = float(np.median(rr_in[lo:hi]))
    # Single forward pass that applies both heuristics in order
    out: list[float] = []
    skip_next = False
    for i, rr in enumerate(rr_in):
        if skip_next:
            skip_next = False
            continue
        med_i = local_med[i] if local_med[i] > 0 else 1.0
        ratio = rr / med_i
        if ratio < 0.5 and i + 1 < rr_in.size:
            # Spurious — merge with next interval
            out.append(rr + rr_in[i + 1])
            skip_next = True
        elif ratio >= 1.8:
            # Missed beat(s) — subdivide into N equal intervals
            n_sub = max(2, int(round(ratio)))
            sub = rr / n_sub
            out.extend([sub] * n_sub)
        else:
            out.append(rr)
    return np.asarray(out, dtype=float)


def hrv_features(ecg: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """Per-epoch HRV: mean HR (bpm), SDNN (ms), RMSSD (ms), pNN50.

    RR intervals outside [300, 2000] ms are rejected as artefacts before
    aggregation.
    """
    n_epochs = _n_epochs(ecg.size, sfreq)
    nan_out = {
        k: np.full(n_epochs, np.nan)
        for k in (
            "hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50", "hrv_sampen",
            "cpc_amp_cv", "cpc_amp_iqr",  # Phase 1 Exp 3
        )
    }

    peaks = r_peaks(ecg, sfreq)
    if peaks.size < 3:
        return nan_out

    rr_sec = np.diff(peaks) / sfreq
    rr_ms = rr_sec * 1000.0
    # de Chazal 2003 §III.C Robust-RR: spurious-merge + missed-subdivide
    rr_ms = _robust_rr(rr_ms)
    # Physiological range: 40–150 bpm
    valid = (rr_ms >= 400.0) & (rr_ms <= 1500.0)
    # Lightweight ratio check on corrected RRs (now strict; missed beats
    # have already been subdivided so legitimate jumps should be rare)
    with np.errstate(invalid="ignore"):
        ratio = np.concatenate(
            [[1.0], rr_ms[1:] / np.clip(rr_ms[:-1], 1e-6, None)]
        )
    valid &= (ratio > 0.6) & (ratio < 1.6)  # widened slightly post Robust-RR
    # Time-index each RR by its later beat. Cumulative time of corrected RRs
    # starts from the first detected R-peak (peaks[0]).
    cumulative_sec = peaks[0] / sfreq + np.cumsum(rr_ms / 1000.0)
    rr_epoch = (cumulative_sec // EPOCH_SECONDS).astype(int)

    hr_mean = np.full(n_epochs, np.nan)
    sdnn = np.full(n_epochs, np.nan)
    rmssd = np.full(n_epochs, np.nan)
    pnn50 = np.full(n_epochs, np.nan)
    sampen = np.full(n_epochs, np.nan)  # Phase 1 Exp 1
    cpc_amp_cv = np.full(n_epochs, np.nan)   # Phase 1 Exp 3 — R-peak amplitude CV
    cpc_amp_iqr = np.full(n_epochs, np.nan)  # Phase 1 Exp 3 — robust amplitude variability

    # Pre-compute R-peak amplitudes (raw ECG values at peak indices) and their epoch assignment.
    # NOTE: amplitudes are aligned to the original `peaks` array, NOT to robust-RR-corrected
    # `rr_ms` (whose length differs after merge/subdivide). For per-epoch amplitude statistics
    # we use peaks directly — robust-RR correction affects rate/timing, not waveform amplitude.
    peak_amplitudes = ecg[peaks].astype(float)
    peak_epoch = (peaks // (sfreq * EPOCH_SECONDS)).astype(int)

    for i in range(n_epochs):
        mask = valid & (rr_epoch == i)
        if mask.sum() < 3:
            continue
        epoch_rr = rr_ms[mask]
        hr_mean[i] = 60000.0 / epoch_rr.mean()
        sdnn[i] = epoch_rr.std(ddof=1)
        diffs = np.diff(epoch_rr)
        if diffs.size > 0:
            rmssd[i] = float(np.sqrt(np.mean(diffs * diffs)))
            pnn50[i] = float((np.abs(diffs) > 50.0).mean())
        # Phase 1 Exp 1 — SampEn on per-epoch RR intervals (typically 20–50 RRs)
        # Need ≥ m+2 = 4 points minimum; gracefully NaN otherwise.
        if epoch_rr.size >= 4:
            sampen[i] = _sample_entropy(epoch_rr, m=2)

        # Phase 1 Exp 3 — CPC-proxy via R-peak amplitude variability.
        # Respiration modulates R-peak amplitude via thoracic impedance changes
        # (the EDR — ECG-derived respiration). Strong modulation = healthy
        # coupling; flat amplitudes = decoupling, often during apnoea/arousal.
        # This is a SIMPLIFIED proxy for full Thomas 2005 cross-spectral CPC —
        # captures the amplitude side of cardiopulmonary coupling without the
        # spectral machinery. Two features:
        #   cpc_amp_cv  = std/mean of amplitudes  (sensitive but outlier-prone)
        #   cpc_amp_iqr = IQR/median  (robust)
        ep_peak_mask = (peak_epoch == i)
        ep_amps = peak_amplitudes[ep_peak_mask]
        ep_amps = ep_amps[np.isfinite(ep_amps)]
        if ep_amps.size >= 4:
            mean_amp = float(np.mean(ep_amps))
            if abs(mean_amp) > 1e-9:
                cpc_amp_cv[i] = float(np.std(ep_amps, ddof=0) / abs(mean_amp))
            median_amp = float(np.median(ep_amps))
            if abs(median_amp) > 1e-9:
                q1, q3 = np.percentile(ep_amps, [25, 75])
                cpc_amp_iqr[i] = float((q3 - q1) / abs(median_amp))

    return {
        "hr_mean": hr_mean,
        "hrv_sdnn": sdnn,
        "hrv_rmssd": rmssd,
        "hrv_pnn50": pnn50,
        "hrv_sampen": sampen,
        "cpc_amp_cv": cpc_amp_cv,
        "cpc_amp_iqr": cpc_amp_iqr,
    }


def hrv_freq_features(
    ecg: np.ndarray,
    sfreq: float,
    window_sec: int = 120,
    min_peaks: int = 80,
) -> dict[str, np.ndarray]:
    """Per-epoch frequency-domain HRV (LF, HF, LF/HF ratio, total power).

    Uses NeuroKit2's ``nk.hrv_frequency()`` (Pelidisi et al. 2022,
    *Frontiers in Neuroscience*) which implements Task Force ESC/NASPE 1996
    standards for HRV spectral analysis.

    A ``window_sec``-second sliding window centred on each 30-s epoch is fed
    to ``nk.hrv_frequency`` per epoch. Default 120 s = 2 min satisfies the
    NeuroKit2 / Task Force recommended minima for both LF (≥ 2 min) and
    HF (≥ 1 min). LF/HF ratio strictly recommends ≥ 5 min, so the per-epoch
    LF/HF here is at the lower end of the recommended range — a known
    trade-off when reporting per-epoch resolution; cohort-level distributions
    remain interpretable (see post-T8 validation in the Sprint 1 plan).

    Pipeline (per epoch):
        1. Locate R-peaks within ``[centre - window/2, centre + window/2]``.
        2. If ≥ ``min_peaks`` peaks: pass to ``nk.hrv_frequency`` (Welch PSD).
        3. Integrate over LF (0.04–0.15 Hz), HF (0.15–0.4 Hz), total (0.04–0.4 Hz).
        4. Otherwise: NaN for that epoch.

    Returns
    -------
    dict[str, np.ndarray]
        4 columns of length ``n_epochs``:
            hrv_lf_power           ms² · Hz⁻¹ in LF band
            hrv_hf_power           ms² · Hz⁻¹ in HF band
            hrv_lf_hf_ratio        dimensionless
            hrv_total_power_freq   ms² · Hz⁻¹, 0.04–0.4 Hz

    Reference
    ---------
    Pelidisi N et al. 2022. "Comprehensive HRV estimation pipeline in Python
    using Neurokit2." Frontiers in Neuroscience. PMID 35880142.
    """
    import neurokit2 as nk

    n_epochs = _n_epochs(ecg.size, sfreq)
    nan_out = {
        k: np.full(n_epochs, np.nan, dtype=np.float32)
        for k in (
            "hrv_lf_power",
            "hrv_hf_power",
            "hrv_lf_hf_ratio",
            "hrv_total_power_freq",
        )
    }

    peaks = r_peaks(ecg, sfreq)
    if peaks.size < min_peaks:
        return nan_out

    half_window = window_sec / 2.0
    peak_times_sec = peaks / sfreq
    epoch_centres = np.arange(n_epochs) * EPOCH_SECONDS + EPOCH_SECONDS / 2.0
    ecg_duration_sec = ecg.size / sfreq

    out = {k: np.full(n_epochs, np.nan, dtype=np.float32) for k in nan_out}

    # Implausible-power clamp: with normalize=False on degraded ECG (sparse /
    # missed-beat detection), nk.hrv_frequency produces artefactual high LF
    # because the FFT picks up slow oscillations from the spurious sparse
    # peaks. Healthy adult LF tops out around 5–10 k ms²; we clamp anything
    # above 50 k ms² as implausible per Task Force ESC/NASPE 1996 norms.
    CLAMP_MS2 = 5e4

    for i, centre in enumerate(epoch_centres):
        # At recording boundaries the centred 2-min window is truncated. Scale
        # the required peak count by the effective window duration so that
        # low-HR subjects (e.g. < 64 bpm) at boundary epochs aren't NaN'd
        # purely due to truncation (Codex audit-v4 finding).
        lo, hi = centre - half_window, centre + half_window
        lo_clip = max(0.0, lo)
        hi_clip = min(ecg_duration_sec, hi)
        effective_sec = hi_clip - lo_clip
        if effective_sec <= 0:
            continue
        effective_min_peaks = max(20, int(min_peaks * effective_sec / window_sec))
        mask = (peak_times_sec >= lo_clip) & (peak_times_sec <= hi_clip)
        if int(mask.sum()) < effective_min_peaks:
            continue
        window_peaks = peaks[mask]
        # Sanity-check the window's RR distribution before computing freq-HRV.
        # Catches degraded ECG where peaks are detected but mostly noise-driven
        # (median RR outside physiological range).
        rr_ms_window = np.diff(window_peaks) / sfreq * 1000.0
        if rr_ms_window.size == 0:
            continue
        med_rr = float(np.median(rr_ms_window))
        if med_rr < 400.0 or med_rr > 1500.0:
            continue
        try:
            results = nk.hrv_frequency(
                {"ECG_R_Peaks": window_peaks},
                sampling_rate=int(sfreq),
                psd_method="welch",
                normalize=False,  # F3: keep absolute ms² power; True (default) normalises
                show=False,
                silent=True,
            )
            lf = float(results["HRV_LF"].iloc[0]) if "HRV_LF" in results.columns else np.nan
            hf = float(results["HRV_HF"].iloc[0]) if "HRV_HF" in results.columns else np.nan
            tp = (
                float(results["HRV_TP"].iloc[0])
                if "HRV_TP" in results.columns
                else (lf + hf if np.isfinite(lf) and np.isfinite(hf) else np.nan)
            )
            # Post-hoc clamp on implausible NK output. v3 clamped only LF/HF
            # but missed TP; v4 nulled all three on any one violation, dropping
            # valid LF/HF outputs when only TP was high (Codex audit-v5
            # finding). v5: clamp each independently. TP can legitimately
            # equal ~LF+HF, so its own bound is 2× the per-band threshold.
            if np.isfinite(lf) and lf > CLAMP_MS2:
                lf = np.nan
            if np.isfinite(hf) and hf > CLAMP_MS2:
                hf = np.nan
            if np.isfinite(tp) and tp > 2 * CLAMP_MS2:
                tp = np.nan
            out["hrv_lf_power"][i] = lf
            out["hrv_hf_power"][i] = hf
            out["hrv_total_power_freq"][i] = tp
            if np.isfinite(lf) and np.isfinite(hf) and hf > 0:
                out["hrv_lf_hf_ratio"][i] = lf / hf
        except Exception:
            # NeuroKit2 occasionally fails on pathological windows; leave NaN.
            continue

    return out


# ─────────────────────────────────────────────────────────────────────────
# EEG spectral
# ─────────────────────────────────────────────────────────────────────────

# AASM 2007 canonical band edges.  Alpha/sigma share 11–13 Hz; sigma/beta
# share 13–16 Hz — this intentional overlap is the AASM convention.
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),   # AASM: 8–13 Hz (was 8–12)
    "sigma": (11.0, 16.0),  # AASM: 11–16 Hz, overlaps alpha (11–13) (was 12–16)
    "beta":  (13.0, 30.0),  # AASM: 13–30 Hz, overlaps sigma (13–16) (was 16–30)
}


def eeg_band_power(signal: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """Per-epoch absolute + relative EEG band powers via Welch PSD.

    Pipeline:
        * Bandpass 0.3–35 Hz to trim DC drift and line noise
        * Welch PSD per epoch, 4-second segments
        * Integrate PSD over canonical bands
        * Total power (0.5–30 Hz) and per-band relative power
    Returns ``eeg_<band>_power`` and ``eeg_<band>_rel`` for each band, plus
    ``eeg_total_power`` and ``eeg_spectral_edge95`` (frequency below which
    95% of total power lies).
    """
    sig = np.asarray(signal, dtype=float)
    finite = np.isfinite(sig)
    if not finite.all():
        sig = np.where(finite, sig, np.nanmedian(sig[finite]) if finite.any() else 0.0)

    nyq = sfreq / 2.0
    lo, hi = 0.3 / nyq, 35.0 / nyq
    if 0 < lo < hi < 1:
        sos = sps.butter(4, [lo, hi], btype="band", output="sos")
        sig = sps.sosfiltfilt(sos, sig)

    ep = _epoch_view(sig, sfreq)
    n_epochs = ep.shape[0]
    nperseg = int(4 * sfreq)

    bands_keys = list(EEG_BANDS.keys())
    out: dict[str, np.ndarray] = {
        **{f"eeg_{b}_power": np.full(n_epochs, np.nan) for b in bands_keys},
        "eeg_total_power": np.full(n_epochs, np.nan),
        "eeg_spectral_edge95": np.full(n_epochs, np.nan),
    }

    for i in range(n_epochs):
        freqs, psd = sps.welch(ep[i], fs=sfreq, nperseg=nperseg)
        in_range = (freqs >= 0.5) & (freqs <= 30.0)
        total = np.trapezoid(psd[in_range], freqs[in_range])
        out["eeg_total_power"][i] = total
        for b, (flo, fhi) in EEG_BANDS.items():
            mask = (freqs >= flo) & (freqs < fhi)
            out[f"eeg_{b}_power"][i] = np.trapezoid(psd[mask], freqs[mask])
        # Spectral edge 95% — use trapezoid-based cumulative integral to be
        # consistent with the band-power integration above (np.trapezoid).
        if total > 0:
            psd_in = psd[in_range]
            freqs_in = freqs[in_range]
            # Cumulative trapezoid: cumul[k] = trapezoid(psd_in[0:k+1], freqs_in[0:k+1])
            # scipy.integrate.cumulative_trapezoid gives length-(n-1) array; prepend 0.
            from scipy.integrate import cumulative_trapezoid
            cumul = np.concatenate([[0.0], cumulative_trapezoid(psd_in, freqs_in)])
            edge_idx = np.searchsorted(cumul, 0.95 * total)
            if 0 <= edge_idx < len(freqs_in):
                out["eeg_spectral_edge95"][i] = freqs_in[edge_idx]

    # Relative powers (avoid divide-by-zero)
    total = out["eeg_total_power"]
    safe = np.where(total > 1e-20, total, np.nan)
    for b in bands_keys:
        out[f"eeg_{b}_rel"] = out[f"eeg_{b}_power"] / safe
    return out


# Phase 1 Exp 4 — multi-scale ECG band power.
#
# Hypothesis: multi-scale frequency content of ECG adds signal beyond time-/
# freq-domain HRV features (which summarise inter-beat interval dynamics, not
# the waveform shape itself). The 5 bands roughly correspond to a 5-level
# Daubechies wavelet decomposition at 125 Hz sampling — without taking on a
# pywt dependency. Captures:
#   low (0–2 Hz)        — baseline drift, very-slow respiration coupling
#   mid_low (2–8 Hz)    — T-wave / late repolarisation
#   mid (8–16 Hz)       — primary QRS energy band (apnoeic events alter QRS shape)
#   mid_high (16–32 Hz) — fast QRS components, motion artifact
#   high (32–62 Hz)     — high-freq noise / muscle artifact (interpret with care)
#
# References:
#   Khandoker, Karmakar, Palaniswami 2009 — wavelet ECG features for OSA detection
#   Almazaydeh, Faezipour, Elleithy 2012 — neural net on ECG-derived features for SA
ECG_BANDS: dict[str, tuple[float, float]] = {
    "low":      (0.5,  2.0),
    "mid_low":  (2.0,  8.0),
    "mid":      (8.0,  16.0),
    "mid_high": (16.0, 32.0),
    "high":     (32.0, 62.0),
}


def ecg_band_power(ecg: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """Per-epoch ECG band power across 5 wavelet-like bands.

    Phase 1 Exp 4. Welch PSD on each 30-s epoch, integrated over each band
    (numerical trapezoid) to give absolute power. Plus relative power per band
    (band / total). Total of 11 features per epoch (5 absolute + 5 relative
    + 1 total).
    """
    sig = np.asarray(ecg, dtype=float)
    ep = _epoch_view(sig, sfreq)
    n_epochs = ep.shape[0]
    # Use shorter Welch segments since epoch is only 30s; 4-s segs give
    # frequency resolution ~0.25 Hz which is plenty for the band integrations.
    nperseg = min(int(4 * sfreq), ep.shape[1])

    bands_keys = list(ECG_BANDS.keys())
    out: dict[str, np.ndarray] = {
        **{f"ecg_band_{b}_power": np.full(n_epochs, np.nan, dtype=np.float32) for b in bands_keys},
        "ecg_band_total_power": np.full(n_epochs, np.nan, dtype=np.float32),
    }

    for i in range(n_epochs):
        epoch = ep[i]
        if not np.isfinite(epoch).any() or np.nanstd(epoch) == 0:
            continue
        try:
            freqs, psd = sps.welch(epoch, fs=sfreq, nperseg=nperseg)
        except Exception:
            continue
        # Total power across the union of bands (0.5 – Nyquist or 62 Hz)
        nyquist = sfreq / 2.0
        upper = min(nyquist, 62.0)
        in_range = (freqs >= 0.5) & (freqs <= upper)
        total = float(np.trapezoid(psd[in_range], freqs[in_range]))
        out["ecg_band_total_power"][i] = total
        for b, (flo, fhi) in ECG_BANDS.items():
            fhi_eff = min(fhi, nyquist)  # in case sfreq < 124
            if fhi_eff <= flo:
                continue
            mask = (freqs >= flo) & (freqs < fhi_eff)
            if mask.any():
                out[f"ecg_band_{b}_power"][i] = float(np.trapezoid(psd[mask], freqs[mask]))

    # Relative powers (avoid divide-by-zero)
    total = out["ecg_band_total_power"]
    safe = np.where(total > 1e-20, total, np.nan)
    for b in bands_keys:
        out[f"ecg_band_{b}_rel"] = (out[f"ecg_band_{b}_power"] / safe).astype(np.float32)
    return out



# ─────────────────────────────────────────────────────────────────────────
# Respiratory effort
# ─────────────────────────────────────────────────────────────────────────


def _bandpass(signal: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    nyq = sfreq / 2.0
    if not (0 < lo / nyq < hi / nyq < 1):
        return signal
    sos = sps.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, signal)


def _breath_rate_per_epoch(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Dominant respiratory frequency per epoch, via FFT peak in 0.1–0.5 Hz."""
    ep = _epoch_view(signal, sfreq)
    n = ep.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        x = ep[i] - ep[i].mean()
        if np.allclose(x, 0):
            continue
        freqs = np.fft.rfftfreq(x.size, 1.0 / sfreq)
        mag = np.abs(np.fft.rfft(x))
        mask = (freqs >= 0.1) & (freqs <= 0.5)
        if mask.any():
            peak = mag[mask].argmax()
            out[i] = freqs[mask][peak] * 60.0  # breaths per minute
    return out


def respiratory_features(
    thor: np.ndarray | None,
    abdo: np.ndarray | None,
    airflow: np.ndarray | None,
    sfreq: float,
) -> dict[str, np.ndarray]:
    """Per-epoch respiratory effort + airflow summaries.

    Emitted columns (NaN where the channel is missing):
        resp_airflow_rms      amplitude of airflow in respiratory band
        resp_airflow_std      variability within epoch
        resp_thor_rms         thoracic belt amplitude
        resp_abdo_rms         abdominal belt amplitude
        resp_thor_abdo_corr   Pearson correlation per epoch (drops on paradox)
        resp_breath_rate_bpm  dominant respiratory frequency (FFT peak, 0.1–0.5 Hz)
    """
    # Use any available channel to size the output
    template = next((s for s in (thor, abdo, airflow) if s is not None), None)
    if template is None:
        return {}
    n = _n_epochs(template.size, sfreq)

    out: dict[str, np.ndarray] = {
        k: np.full(n, np.nan)
        for k in (
            "resp_airflow_rms",
            "resp_airflow_std",
            "resp_thor_rms",
            "resp_abdo_rms",
            "resp_thor_abdo_corr",
            "resp_breath_rate_bpm",
        )
    }

    if airflow is not None:
        af = _bandpass(np.asarray(airflow, dtype=float), sfreq, 0.05, 3.0)
        ep = _epoch_view(af, sfreq)
        out["resp_airflow_rms"] = np.sqrt((ep * ep).mean(axis=1))
        out["resp_airflow_std"] = ep.std(axis=1)
        out["resp_breath_rate_bpm"] = _breath_rate_per_epoch(af, sfreq)

    if thor is not None:
        th = _bandpass(np.asarray(thor, dtype=float), sfreq, 0.05, 3.0)
        ep = _epoch_view(th, sfreq)
        out["resp_thor_rms"] = np.sqrt((ep * ep).mean(axis=1))

    if abdo is not None:
        ab = _bandpass(np.asarray(abdo, dtype=float), sfreq, 0.05, 3.0)
        ep = _epoch_view(ab, sfreq)
        out["resp_abdo_rms"] = np.sqrt((ep * ep).mean(axis=1))

    if thor is not None and abdo is not None:
        th_ep = _epoch_view(_bandpass(np.asarray(thor, dtype=float), sfreq, 0.05, 3.0), sfreq)
        ab_ep = _epoch_view(_bandpass(np.asarray(abdo, dtype=float), sfreq, 0.05, 3.0), sfreq)
        corr = np.full(n, np.nan)
        for i in range(min(th_ep.shape[0], ab_ep.shape[0])):
            a, b = th_ep[i], ab_ep[i]
            a_std, b_std = a.std(), b.std()
            if a_std > 1e-12 and b_std > 1e-12:
                corr[i] = float(np.mean((a - a.mean()) * (b - b.mean())) / (a_std * b_std))
        out["resp_thor_abdo_corr"] = corr

    return out


# ─────────────────────────────────────────────────────────────────────────
# Body position
# ─────────────────────────────────────────────────────────────────────────


# SHHS Compumedics POSITION channel encoding (per SHHS Manual of Operations).
# Verified against shhs1-200001 in T5 of Sprint 1 (2026-04-26).
POSITION_CODES = {
    0: "right",
    1: "left",
    2: "supine",
    3: "prone",
    4: "upright",
}


def position_features(signal: np.ndarray, sfreq: float) -> dict[str, np.ndarray]:
    """Per-epoch body-position fractions.

    Body position is a strong AHI modifier — supine apnoea is a well-
    documented clinical phenomenon. Per-epoch position fractions allow
    downstream features to capture supine-OSA stratification.

    SHHS Compumedics POSITION channel encoding:
        0 = Right, 1 = Left, 2 = Back/Supine, 3 = Front/Prone, 4 = Up/Sit
    Verified at run time from shhs1-200001 (Sprint 1 T5, 2026-04-26).

    Returns one column per position code with the fraction of samples in
    that position per epoch. Plus a ``position_supine_frac`` alias-pointer
    explicitly named for the AHI-relevant aggregate.

    Parameters
    ----------
    signal : np.ndarray
        POSITION signal at the EDF native rate (typically 125 Hz; piecewise-
        constant since position changes slowly).
    sfreq : float
        EDF native sampling rate.

    Returns
    -------
    dict[str, np.ndarray]
        5 columns of length ``n_epochs``:
            position_right_frac, position_left_frac, position_supine_frac,
            position_prone_frac, position_upright_frac
    """
    sig = np.asarray(signal, dtype=float)
    finite = np.isfinite(sig)
    # Round to integer code; NaN samples become -1 sentinel
    sig_int = np.where(finite, np.round(sig).astype(int), -1)
    ep = _epoch_view(sig_int.astype(float), sfreq)

    out: dict[str, np.ndarray] = {}
    for code, name in POSITION_CODES.items():
        out[f"position_{name}_frac"] = (ep == code).mean(axis=1).astype(np.float32)

    return out


# ─────────────────────────────────────────────────────────────────────────
# Contextual / temporal features
# ─────────────────────────────────────────────────────────────────────────


def contextual_features(
    frame: pd.DataFrame, cols: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Add lag/lead/rolling-window aggregates for each base feature column.

    For every column in ``cols`` (default = ``ROLLING_BASE_COLS`` intersected
    with the frame's columns), append:

      <col>_lag1          previous epoch's value          (NaN at first epoch)
      <col>_lead1         next epoch's value              (NaN at last epoch)
      <col>_roll5_mean    centered 5-epoch (2.5 min) mean
      <col>_roll5_std     centered 5-epoch (2.5 min) std
      <col>_roll11_mean   centered 11-epoch (5.5 min) mean
      <col>_roll11_std    centered 11-epoch (5.5 min) std

    Per-subject computation; never crosses subject boundaries (the function
    is called once per subject's epoch frame).
    """
    if cols is None:
        cols = tuple(c for c in ROLLING_BASE_COLS if c in frame.columns)
    if not cols:
        return frame

    new_cols: dict[str, pd.Series] = {}
    for c in cols:
        s = frame[c]
        new_cols[f"{c}_lag1"] = s.shift(1)
        new_cols[f"{c}_lead1"] = s.shift(-1)
        roll5 = s.rolling(window=5, center=True, min_periods=1)
        new_cols[f"{c}_roll5_mean"] = roll5.mean()
        new_cols[f"{c}_roll5_std"] = roll5.std()
        roll11 = s.rolling(window=11, center=True, min_periods=1)
        new_cols[f"{c}_roll11_mean"] = roll11.mean()
        new_cols[f"{c}_roll11_std"] = roll11.std()

    return pd.concat([frame, pd.DataFrame(new_cols, index=frame.index)], axis=1)
