#!/usr/bin/env bash
# Final closure (28 Sep 2026): local focused learned runs + downstream recompute.
# Run from the analysis/ folder in a macOS terminal (the runner refuses other
# systems). Resumable: completed runs are verified and skipped.
#   cd analysis && bash reproducibility/run_final_closure_compute.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
OUT=audit/verified_results/final_closure_2026-09-28
mkdir -p "$OUT"
LOG="$OUT/compute_console.log"
exec > >(tee -a "$LOG") 2>&1
echo "== $(date '+%Y-%m-%dT%H:%M:%S%z') final-closure compute, PY=$PY"
[ "$(uname -s)" = "Darwin" ] || { echo "must run on macOS"; exit 1; }
"$PY" -c 'import sys,platform,numpy,pandas,xgboost,torch,optuna;print(sys.executable,platform.system(),platform.machine(),sys.version.split()[0],"xgboost",xgboost.__version__,"torch",torch.__version__,"optuna",optuna.__version__)'
# keep the Mac awake while the runs execute (about 7 h on an M1)
CAF=""; command -v caffeinate >/dev/null && CAF="caffeinate -dimsu"
"$PY" ../experiments/tests/test_xgb_alignment.py
[ -f "$OUT/preflight.json" ] || $CAF "$PY" audit/final_closure_local_runs.py --stage preflight
$CAF "$PY" audit/final_closure_local_runs.py --stage xgb
$CAF "$PY" audit/final_closure_local_runs.py --stage lstm
"$PY" audit/final_closure_local_runs.py --stage verify
if [ ! -f "$OUT/downstream/summary.json" ]; then
  $CAF "$PY" audit/recompute_final_closure_downstream.py --write
fi
"$PY" audit/recompute_final_closure_downstream.py --check
echo "== done $(date '+%Y-%m-%dT%H:%M:%S%z')."
