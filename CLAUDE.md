# CLAUDE.md — Code/

Project-level context lives in `../CLAUDE.md`. This file covers Code/-specific operational detail.

## Package layout

```
Code/
├── pyproject.toml                 # hatchling build, deps pinned in [project.dependencies]
├── README.md                      # human-facing onboarding
├── pilot_subjects.json            # 100-subject pilot sample (seed=42)
├── thesis_pipeline/               # importable package (PEP 517 build target)
│   ├── shhs.py                    # OneDrive paths, subject discovery, UF_DATALESS-aware local detection
│   ├── io.py                      # EDF (mne) + NSRR XML readers
│   ├── epochs.py                  # 30-s epoch extraction + ≥10-s overlap apnoea labelling
│   └── features.py                # SpO₂ / HRV / EEG band power / respiratory effort feature extractors
├── scripts/                       # Click-based CLI entry points
│   ├── pin_pilot.py               # trigger OneDrive download of the 100-subject pilot
│   ├── process_batch.py           # process all locally-materialised subjects
│   ├── scale_process.py           # resumable scale driver — pin → process → evict → next batch
│   ├── scale_process_parallel.py  # multiprocessing variant (3-4× throughput)
│   ├── verify_parallel_features.py  # sanity-check parquet outputs after a parallel run
│   ├── fit_aim2_baseline.py       # single-split LightGBM baseline (legacy, kept for reproducibility)
│   ├── fit_aim2_cv.py             # GroupKFold(5) + Optuna + bootstrap — current canonical baseline
│   └── watch_optuna.py            # live monitor for an in-flight Optuna study
├── features/                      # per-subject parquet outputs (created at runtime, gitignore'd)
└── models/                        # trained model artefacts + Optuna studies
```

## Setup

```bash
cd Code
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # picks up pytest, ruff, ipykernel
```

Python ≥ 3.10. Key runtime deps: `mne`, `lxml`, `numpy`, `scipy`, `pandas`, `pyarrow`, `tqdm`, `click`.

## OneDrive batch loop — important

SHHS1 EDFs (~214 GB) live in Prof. de Chazal's OneDrive shared folder:

```
~/Library/CloudStorage/OneDrive-SharedLibraries-TheUniversityofSydney(Staff)/Philip de Chazal - SHHS/
```

Files default to OneDrive **placeholders** — `ls` shows them but they have no local bytes. `thesis_pipeline.shhs.is_local()` distinguishes materialised files from placeholders via the macOS `UF_DATALESS` flag.

Per Philip's email (2026-04-23): *"Load the signals down in batches. Try batches of 500."* Workflow:

1. **Pin** — Finder → select 500 EDFs + their XMLs → right-click → **Always Keep on This Device**. (~18 GB downloaded.)
2. **Process** — `python scripts/process_batch.py` — finds locally-materialised subjects only, writes `features/<subject>.parquet`.
3. **Free Up Space** — Finder → same files → right-click → **Free Up Space**. Reverts to placeholders, reclaims disk.
4. **Next batch.**

Or use `scripts/scale_process.py` (resumable, downloads + processes in 25-subject sub-batches, pauses at 20 GB free-disk threshold). For full-cohort runs use `scripts/scale_process_parallel.py`.

Max disk footprint at any time: ~18–20 GB.

## Where to put new things

| New thing | Goes in |
|---|---|
| New CLI entry-point | `scripts/<name>.py`, Click-based, single `if __name__ == "__main__":` block at the bottom |
| New importable module / class / function | `thesis_pipeline/<name>.py` (or extend an existing module if the responsibility fits) |
| New feature extractor | extend `thesis_pipeline/features.py`, register in the feature dictionary so it lands in the parquet |
| New model training script | `scripts/fit_<aim>_<variant>.py` — the `fit_aim2_cv.py` script is the current template |
| Trained model artefacts | `models/<run-id>/` — never commit large binaries; rely on local files + `models/<run-id>/study.pkl` for the Optuna study |
| Tests | `tests/` (does not yet exist; create on first test) using pytest |
| Notebooks | `notebooks/` (does not yet exist; create on first notebook); commit cleared outputs only |

## Conventions

- All scripts are Click-based CLIs with `--help`. Don't add bare `argparse` scripts.
- Per-subject parquet outputs live in `features/<subject>.parquet`. One row per 30-s epoch. Columns: `subject_id`, `epoch_idx`, `sleep_stage`, `apnoea_label`, then feature columns.
- Patient-level splits only. Use `scripts/fit_aim2_cv.py` as the reference for GroupKFold + Optuna + bootstrap CI.
- Random seed: 42 unless specifically varying it. For DL multi-seed runs, use `[42, 43, 44]` at minimum.
- Lint with `ruff check .` before committing.

## When you finish a code change that produced an experiment result

Don't just commit and move on. Log it in the vault — see the workflow in `../CLAUDE.md` ("Workflow — when you finish an experiment").

## Don't do

- Don't write to `../Dataset/` — it's raw cohort data, treat as read-only.
- Don't commit `features/` or `models/` artefacts.
- Don't bypass the OneDrive pin/evict loop by trying to download all 214 GB at once.
- Don't hard-code subject IDs or paths into `thesis_pipeline/`; pass them as arguments.
- Don't rename `fit_aim2_cv.py` outputs without updating the running log table in `../Thesis Vault/Aims/Aim 2 - Baseline Experiments.md`.
