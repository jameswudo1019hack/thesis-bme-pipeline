"""Tiny live monitor for an Optuna study DB. Usage:

    python scripts/watch_optuna.py models/aim2_cv_v7/optuna_study.db
"""

import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

db = Path(sys.argv[1] if len(sys.argv) > 1 else "models/aim2_cv_v7/optuna_study.db")

while True:
    try:
        conn = sqlite3.connect(db)
        c = conn.cursor()
        done = c.execute("SELECT COUNT(*) FROM trials WHERE state='COMPLETE'").fetchone()[0]
        running = c.execute("SELECT COUNT(*) FROM trials WHERE state='RUNNING'").fetchone()[0]
        pruned = c.execute("SELECT COUNT(*) FROM trials WHERE state='PRUNED'").fetchone()[0]
        best = c.execute("SELECT MAX(value) FROM trial_values").fetchone()[0] or 0.0
        conn.close()
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"{ts}  done={done}  running={running}  pruned={pruned}  best_AUC={best:.4f}", flush=True)
    except Exception as e:
        print(f"(error: {type(e).__name__}: {e})", flush=True)
    time.sleep(30)
