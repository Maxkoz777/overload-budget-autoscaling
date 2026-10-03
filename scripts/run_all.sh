#!/usr/bin/env bash
# run_all.sh — original focused-tier experiment pipeline (phases 4 -> 9)
# Run from the repository root:  bash scripts/run_all.sh
#
# Steps:
#   Step 1  Phase 4  build_service_timeseries.py   - stratified service set
#   Step 2  Phase 6  run_baseline_replay.py         - baseline policies (oracle, static, ...)
#   Step 3  Phase 8  run_risk_policy_replay.py      - risk-aware policy (persistence forecast)
#   Step 4  Phase 7  train_arima.py                 — ARIMA + Fourier
#   Step 5  Phase 7  train_xgb.py                   — XGBoost
#   Step 6  Phase 7  train_lstm.py                  - LSTM (longest step)
#   Step 7  Phase 7  evaluate_advanced_forecasting.py - forecast accuracy comparison
#   Step 8  Phase 9  run_advanced_policy_replay.py  - cost-risk with learned forecasts

set -e   # stop at the first error

SCRIPTS=(
    "experiments/src/build_service_timeseries.py"
    "experiments/src/run_baseline_replay.py"
    "experiments/src/run_risk_policy_replay.py"
    "experiments/src/train_arima.py"
    "experiments/src/train_xgb.py"
    "experiments/src/train_lstm.py"
    "experiments/src/evaluate_advanced_forecasting.py"
    "experiments/src/run_advanced_policy_replay.py"
)

DESCRIPTIONS=(
    "Phase 4  · Build stratified service timeseries (20 services, 5 groups)"
    "Phase 6  · Baseline policy replay (oracle / static / persistence / ...)"
    "Phase 8  · Risk-aware conformal policy (persistence forecast)"
    "Phase 7a · Train ARIMA  (Fourier terms, K-selection CV)"
    "Phase 7b · Train XGBoost (Optuna 200 trials, 5-fold CV)"
    "Phase 7c · Train LSTM   (Optuna TPE, MPS/CPU mode)"
    "Phase 7d · Evaluate forecasting accuracy (ARIMA vs XGBoost vs LSTM)"
    "Phase 9  · Advanced policy replay (cost-risk table with ML forecasts)"
)

TOTAL=${#SCRIPTS[@]}
START_ALL=$(date +%s)

echo ""
echo "████████████████████████████████████████████████████████████████████████"
echo "  Full Experiment Pipeline  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Steps: $TOTAL  |  Services: 20 (5 groups × 4)"
echo "████████████████████████████████████████████████████████████████████████"
echo ""

for i in "${!SCRIPTS[@]}"; do
    SCRIPT="${SCRIPTS[$i]}"
    DESC="${DESCRIPTIONS[$i]}"
    STEP=$((i + 1))
    T0=$(date +%s)

    echo "════════════════════════════════════════════════════════════════════════"
    printf "  Step %d/%d  ·  %s\n" "$STEP" "$TOTAL" "$DESC"
    echo "  Script:  $SCRIPT"
    echo "  Started: $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""

    python3 "$SCRIPT"

    T1=$(date +%s)
    ELAPSED=$((T1 - T0))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))

    WALL=$((T1 - START_ALL))
    WALL_MINS=$((WALL / 60))
    WALL_SECS=$((WALL % 60))

    echo ""
    echo "  ✓ Step $STEP/$TOTAL done in ${MINS}m${SECS}s  (wall ${WALL_MINS}m${WALL_SECS}s)"
    echo ""
done

END_ALL=$(date +%s)
TOTAL_ELAPSED=$((END_ALL - START_ALL))
TOTAL_H=$((TOTAL_ELAPSED / 3600))
TOTAL_M=$(( (TOTAL_ELAPSED % 3600) / 60 ))
TOTAL_S=$((TOTAL_ELAPSED % 60))

echo "████████████████████████████████████████████████████████████████████████"
echo "  ALL DONE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total time: ${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s"
echo "████████████████████████████████████████████████████████████████████████"
echo ""
echo "  Key results:"
echo "    experiments/results/baseline_replay_summary.csv"
echo "    experiments/results/risk_policy_replay_summary.csv"
echo "    experiments/results/arima_forecasts.parquet"
echo "    experiments/results/xgb_forecasts.parquet"
echo "    experiments/results/lstm_forecasts.parquet"
echo "    experiments/results/advanced_forecasting_comparison.csv"
echo "    experiments/results/advanced_policy_replay_summary.csv"
echo ""
echo "  Reports:"
echo "    experiments/reports/service_selection_stratified_report.md"
echo "    experiments/reports/arima_training_report.md"
echo "    experiments/reports/xgb_training_report.md"
echo "    experiments/reports/lstm_training_report.md"
echo "    experiments/reports/forecasting_advanced_report.md"
echo ""
