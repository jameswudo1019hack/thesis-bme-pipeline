"""Live unified status of the v8 ablation: which models have started, how many
Optuna trials each is into, the running best CV AUC, and which (if any) is
currently running.

Usage (from anywhere):
    python Code/scripts/watch_v8.py            # one-shot snapshot
    python Code/scripts/watch_v8.py --watch    # refresh every 30s
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
MODELS = ["xgboost", "catboost", "rf", "logreg"]


def _study_status(db: Path) -> tuple[int, int, int, float] | None:
    """Return (complete, running, pruned, best_auc) or None if DB missing."""
    if not db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        c = conn.cursor()
        done = c.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'").fetchone()[0]
        running = c.execute("SELECT COUNT(*) FROM trials WHERE state='RUNNING'").fetchone()[0]
        pruned = c.execute("SELECT COUNT(*) FROM trials WHERE state='PRUNED'").fetchone()[0]
        best = c.execute("SELECT MAX(value) FROM trial_values").fetchone()[0] or 0.0
        conn.close()
        return done, running, pruned, float(best)
    except Exception:
        return None


def _final_metrics(model_dir: Path) -> dict | None:
    f = model_dir / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def _log_tail(log: Path, n: int = 1) -> str:
    if not log.exists():
        return ""
    try:
        lines = log.read_text(errors="replace").splitlines()
        return lines[-n] if lines else ""
    except Exception:
        return ""


def snapshot() -> str:
    rows = []
    rows.append(f"=== v8 ablation status @ {datetime.now().strftime('%H:%M:%S')} ===")

    # Reference: v6 LightGBM (the SOTA we're trying to match/beat)
    v6 = _final_metrics(CODE_ROOT / "models" / "aim2_cv_v6")
    if v6:
        rows.append(
            f"  [v6  ] LightGBM     ✓ DONE  CV {v6['cv_mean_auc']:.4f} ± {v6['cv_std_auc']:.4f}  "
            f"Test {v6['test_auc_roc']:.4f} [{v6['test_auc_ci_low']:.4f}, {v6['test_auc_ci_high']:.4f}]"
        )

    for model in MODELS:
        model_dir = CODE_ROOT / "models" / f"aim2_v8_{model}"
        log = CODE_ROOT / "models" / "v8_logs" / f"{model}.log"
        final = _final_metrics(model_dir)
        study = _study_status(model_dir / "optuna_study.db")

        if final:
            status = (
                f"✓ DONE  CV {final['cv_mean_auc']:.4f} ± {final['cv_std_auc']:.4f}  "
                f"Test {final['test_auc_roc']:.4f} [{final['test_auc_ci_low']:.4f}, {final['test_auc_ci_high']:.4f}]"
            )
        elif study is not None:
            done, running, pruned, best = study
            status = f"… RUNNING  trials done={done} running={running} pruned={pruned}  best_CV={best:.4f}"
        elif log.exists():
            status = f"… STARTED  (log exists, no Optuna DB yet — loading cohort?)"
        else:
            status = "  pending"

        rows.append(f"  [v8  ] {model:11s} {status}")

    # Ensemble (only after all 4 base models done)
    ens = _final_metrics(CODE_ROOT / "models" / "aim2_v8_ensemble")
    if ens:
        best_ens = max(ens["ensemble"], key=lambda r: r["auc"])
        rows.append(
            f"  [ens ] best={best_ens['name']:10s} ✓ DONE  Test {best_ens['auc']:.4f} "
            f"[{best_ens['auc_ci_low']:.4f}, {best_ens['auc_ci_high']:.4f}]"
        )
    else:
        rows.append(f"  [ens ] {'mean/weighted/rank':11s}   pending (runs after all 4 base models)")

    # Last log line of whichever model appears to be currently running
    running_model = next((m for m in MODELS
                          if (CODE_ROOT / "models" / f"aim2_v8_{m}" / "optuna_study.db").exists()
                          and not (CODE_ROOT / "models" / f"aim2_v8_{m}" / "metrics.json").exists()), None)
    if running_model:
        last = _log_tail(CODE_ROOT / "models" / "v8_logs" / f"{running_model}.log")
        if last:
            rows.append(f"\n  last log line ({running_model}): {last[:120]}")

    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Refresh every 30s")
    args = parser.parse_args()

    if not args.watch:
        print(snapshot())
        return

    while True:
        print("\n" + snapshot(), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
