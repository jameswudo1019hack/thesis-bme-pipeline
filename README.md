# Thesis pipeline — SHHS PSG processing + classical-ML / DL benchmarks

> **Comparing Traditional ML and Deep Learning Approaches for PSG-Based Sleep Event Detection and Cardiovascular Risk Profiling**
> James Wu · BME Honours thesis · University of Sydney · 2026
> Supervised by Prof. Philip de Chazal

This repo contains the data pipeline + experiment code for an honours-thesis methodological comparison of classical ML and deep learning on the [Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs) cohort. The companion vault (Obsidian) holds the literature review, methodology notes, and experiment logs and is **not** part of this repo.

**Status (2026-04-26):**
- ✅ Pipeline complete on full SHHS1 cohort (5,793 / 5,793 subjects).
- ✅ Classical-ML SOTA in hand: XGBoost 0.8486 AUC (`v8a`, see [Aim 2 baselines](#aim-2-baselines)).
- ✅ Sprint 1 feature engineering integrated (LF/HF HRV, hypoxic burden, position channel, Robust-RR, wake-handling, subject metadata). Schema: ~280 columns / epoch.
- 📋 Next: full-cohort regen with Sprint 1 features, then v8.5 LightGBM and v9 DL baselines on Colab.

## Directory layout

```
Code/
├── README.md                     # this file
├── pyproject.toml                # Python dependencies
├── pilot_subjects.json           # 100-subject pilot sample (seed=42)
├── features/                     # per-subject parquet outputs (created at runtime)
├── thesis_pipeline/              # importable package
│   ├── __init__.py
│   ├── shhs.py                   # SHHS paths, subject discovery, dataless/local detection
│   ├── io.py                     # EDF + NSRR XML readers
│   ├── epochs.py                 # 30-second epoch extraction + labelling
│   └── features.py               # per-epoch feature extractors
└── scripts/
    └── process_batch.py          # CLI: process all locally-materialised subjects
```

## Data source

SHHS1 EDFs and annotation XMLs are in Prof. de Chazal's OneDrive shared folder:

```
~/Library/CloudStorage/OneDrive-SharedLibraries-TheUniversityofSydney(Staff)/Philip de Chazal - SHHS/
```

Files are OneDrive placeholders by default — `ls` shows them but they have no local bytes. `thesis_pipeline.shhs.is_local()` distinguishes materialised files from placeholders using the macOS `UF_DATALESS` flag.

## Workflow — per batch

Philip's guidance (email, 2026-04-23): *"You'll have to load the signals down in batches. Try batches of 500."*

For each batch:

1. **Pin** — in Finder, select 500 EDFs + their XMLs → right-click → **Always Keep on This Device**. OneDrive downloads them (~18 GB).
2. **Process** — `python scripts/process_batch.py`. The script finds only the subjects that are currently local, extracts features, writes `features/<subject>.parquet`.
3. **Free Up Space** — in Finder, select the same files → right-click → **Free Up Space**. OneDrive reverts them to placeholders, reclaiming ~18 GB.
4. **Next batch.**

Max disk footprint at any time: ~18–20 GB. Over 12 batches the full 5,793-subject cohort is processed. Output (features) totals well under 1 GB.

## Setup

```bash
cd Code
python3 -m venv .venv
source .venv/bin/activate
pip install -e .                 # uses pyproject.toml
```

Key deps: `mne` (EDF), `lxml` (XML), `numpy`/`scipy` (signal processing), `pandas`/`pyarrow` (parquet), `tqdm`, `click`.

## Running on the pilot first

Two ways to get the pilot subjects onto your disk:

```bash
# Option 1 — scripted download (easier, ~3.6 GB pulled via OneDrive API):
python scripts/pin_pilot.py --workers 8

# Option 2 — manual pin via Finder:
# select the pilot subjects, right-click → Always Keep on This Device
```

Then process them:

```bash
python scripts/process_batch.py --pilot-only
```

Only subjects listed in `pilot_subjects.json` AND currently materialised locally will be processed. This gives a fast dev loop for the pipeline code itself.

## Per-subject outputs

Each subject produces `features/shhs1-<id>.parquet` with one row per 30-second epoch:

| Column | Description |
|---|---|
| `subject_id` | SHHS subject identifier |
| `epoch_idx` | 0-based epoch index within the recording |
| `sleep_stage` | W / N1 / N2 / N3 / REM / Unknown (from NSRR XML hypnogram) |
| `apnoea_label` | 1 if ≥10 s of the epoch overlaps an annotated apnoea/hypopnoea event, else 0 |
| *(feature columns)* | Per-modality features — see `thesis_pipeline/features.py` |

## Feature schema (post-Sprint-1, ~280 columns)

| Group | Columns | Reference |
|---|---|---|
| **SpO₂** | mean, min, max, std, ODI3/4 counts, desat depth, **hypoxic burden** | Azarbarzin 2019 *Eur Heart J* |
| **HRV (time-domain)** | mean HR, SDNN, RMSSD, pNN50 — with NeuroKit2 R-peak detection + de Chazal 2003 §III.C **Robust-RR** missed-beat correction | de Chazal 2003 |
| **HRV (frequency-domain)** | LF (0.04–0.15 Hz), HF (0.15–0.4 Hz), LF/HF ratio, total power — via `nk.hrv_frequency()` on 2-min sliding windows | Pelidisi 2022, Task Force ESC/NASPE 1996 |
| **EEG band power** | δ/θ/α/σ/β absolute + relative, total power, spectral edge 95% — via Welch PSD | standard AASM bands |
| **Respiratory** | airflow/THOR/ABDO RMS + std, thor-abdo Pearson correlation (paradox indicator), breath rate from FFT peak | Mostafa 2019 review |
| **Body position** | per-epoch fractions of right / left / supine / prone / upright | SHHS Compumedics encoding |
| **Contextual derivatives** | lag-1, lead-1, rolling-5 (mean+std), rolling-11 (mean+std) wrapped around each base feature | per Sprint 0 — v6 |

Per-subject summary metrics (TST, sleep efficiency, WASO, AHI proxy, hypoxic burden per night) live in `Code/features/subject_metadata.parquet`, built via `scripts/build_subject_metadata.py`.

`FEATURES_VERSION` (top of `thesis_pipeline/features.py`) is bumped on every schema change so downstream code can filter mixed-version cohorts cleanly.

## <a name="aim-2-baselines"></a>Aim 2 baselines (apnoea detection)

Patient-level GroupKFold(5) + Optuna HPO + bootstrap CI — see `scripts/fit_aim2_cv.py`.

| v | N subj | Features | Method | CV AUC | Test AUC |
|---|---|---|---|---|---|
| 1 | 100 | 11 | hand-set LightGBM | — | 0.713 |
| 2 | 100 | 29 | hand-set LightGBM | — | 0.737 |
| 3 | 600 | 29 | hand-set LightGBM | — | 0.760 |
| 4 | 668 | 29 | LightGBM + GroupKFold + Optuna | 0.769 ± 0.006 | 0.759 |
| 5 | 4,985 | 29 (NeuroKit2) | same protocol | 0.770 ± 0.002 | 0.774 |
| 7 | 5,793 | 29 | LightGBM + extended Optuna | 0.769 ± 0.003 | 0.779 |
| 6 | 5,789 | 203 | LightGBM + contextual features | 0.846 ± 0.003 | 0.846 |
| **8a** | **5,789** | **203** | **XGBoost (Colab GPU)** | **0.8486 ± 0.0035** | **0.8486** ← SOTA |
| 8b | 5,789 | 203 | CatBoost (Colab GPU) | 0.8462 ± 0.0035 | 0.8460 |
| 8c | 5,789 | — | Ensemble (3 GBMs) | — | 0.8482–0.8483 (no lift) |

## Data access

SHHS data is governed by the [NSRR Data Use Agreement](https://sleepdata.org/data/requests). The DUA prohibits redistribution of the raw data and per-subject derived data. **This repo does not include any NSRR data, raw or derived** — only the code that processes it. To reproduce, you need:

1. NSRR account + approved DAR for SHHS-1.
2. SHHS1 EDFs + NSRR XMLs in the OneDrive folder (or other local mirror); update `thesis_pipeline/shhs.py:SHHS_ROOT` accordingly.
3. CITI HIPAA training (prerequisite for DAR submission).

## Citation

If you use this code, please cite the thesis (placeholder — replaced after submission):

```bibtex
@phdthesis{wu2026shhs,
  author = {Wu, James},
  title  = {Comparing Traditional ML and Deep Learning Approaches for PSG-Based Sleep Event Detection and Cardiovascular Risk Profiling},
  school = {The University of Sydney},
  year   = {2026},
  type   = {Honours thesis},
  note   = {Supervised by Prof. Philip de Chazal}
}
```

## Acknowledgements

- Data provided by the [National Sleep Research Resource (NSRR)](https://sleepdata.org/) — Sleep Heart Health Study (SHHS).
- Supervised by Prof. Philip de Chazal (BMET, USyd). Methodology lineage descends in part from de Chazal et al. 2003, *IEEE Trans. Biomed. Eng.* — see citations in `thesis_pipeline/features.py`.

## License

(TBD — likely MIT or Apache-2.0 for the code; the *outputs* and trained models remain under NSRR DUA terms.)
