#!/usr/bin/env bash
# v8 ablation — GBM-only path (XGBoost → CatBoost → Ensemble).
# RF and LogReg deferred (they need either a memory upgrade or Colab Pro+ — sklearn
# pipelines copy the X matrix during fit, blew past the 16 GB ceiling on 2026-04-26).
# Logs go to Code/models/v8_logs/<model>.log (line-buffered, full stdout+stderr).
# Stops on the first failure so you can investigate before burning more wall time.
#
# Usage (from Code/):
#   bash scripts/run_v8_ablation.sh

set -euo pipefail

PY=/opt/anaconda3/bin/python
LOG_DIR=models/v8_logs
mkdir -p "$LOG_DIR"

run_one() {
    local model="$1"
    local trials="$2"
    local timeout="$3"
    local log="$LOG_DIR/${model}.log"
    echo "==> [$(date '+%H:%M:%S')] starting $model (trials=$trials, timeout=${timeout}s) → $log"
    $PY -u scripts/fit_aim2_v8_ablation.py \
        --model "$model" \
        --trials "$trials" \
        --timeout "$timeout" \
        > "$log" 2>&1
    local cv=$(python -c "import json; m=json.load(open('models/aim2_v8_${model}/metrics.json')); print(f'CV {m[\"cv_mean_auc\"]:.4f} ± {m[\"cv_std_auc\"]:.4f} / Test {m[\"test_auc_roc\"]:.4f} [{m[\"test_auc_ci_low\"]:.4f}, {m[\"test_auc_ci_high\"]:.4f}]')" 2>/dev/null || echo "metrics not parseable")
    echo "    $model done — $cv"
}

START_TS=$(date '+%H:%M:%S')
echo "=== v8 ablation (GBM-only) started at $START_TS ==="

# Budgets revised after smoke testing showed ~25 min per trial with tightened
# search space (max_depth/depth ≤ 7). 2 hr → ~5-6 trials per model with TPE +
# MedianPruner, which is sufficient for the simple/tight search spaces here.
run_one xgboost  10 7200     # 2 hr  (~5-6 trials, max_depth ≤ 7)
run_one catboost 10 9000     # 2.5 hr (~5-6 trials, depth ≤ 7)

echo "==> [$(date '+%H:%M:%S')] starting ensemble (v6 LightGBM + XGBoost + CatBoost)"
$PY -u scripts/fit_aim2_v8_ensemble.py > "$LOG_DIR/ensemble.log" 2>&1
echo "    ensemble done"

echo "=== v8 ablation finished at $(date '+%H:%M:%S') (started $START_TS) ==="
