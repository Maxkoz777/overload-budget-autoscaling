# Script catalogue

Every script in the repository, grouped by stage. The run order is given in the [README](../README.md#reproducing-the-results). Paths are relative to the repository root.

## Data download and preprocessing

Run from the repository root unless the path starts with `analysis/`.

| Script | Purpose |
|---|---|
| `fetchData.sh` | Downloads the hourly Alibaba v2022 archives; called by the pipeline. |
| `pipeline/run_hourly_pipeline.py` | Alibaba: download, hour-by-hour aggregation to service level, atomic Parquet commits to `processed/final/`. |
| `experiments/src/define_splits.py` | Writes the chronological split (`experiments/data/splits/split_definition.json`): days 0–7 history, 8–9 calibration, 10–12 test. |
| `experiments/src/build_service_timeseries.py` | Builds regular one-minute series for the 20 focused services. |
| `experiments/src/select_services_200.py` | Selects the fixed 200-service suite by burstiness stratum (seed 42). |
| `analysis/audit/rebuild_verified_service_series.py` | Rebuilds the 200 service series from `processed/final/` and checks them against the selection file; compares with the project's historical series only if those files exist. |
| `analysis/audit/verify_phase8_upstream.py` | Read-only check of the 200 series against `processed/final/`. |
| `analysis/audit/prep_external_traces_2026_09_30.py` | Huawei 2023 (days 28–60) and Azure 2019 minute series and unit selection for the external replication. |
| `analysis/audit/prep_huawei_confirmatory_2026_10_02.py` | Huawei 2023 series for the three untouched confirmatory periods (frozen by SHA-256). |

## Core library (`experiments/src/`)

Imported by the experiment and analysis scripts.

| Script | Purpose |
|---|---|
| `experiments/src/simulator.py` | Split handling and the per-service capacity replay of the baseline policies. |
| `experiments/src/policies.py` | Capacity rules of the baseline replay: observed, oracle, static quantile, reactive threshold, persistence, and moving-average prediction. |
| `experiments/src/risk_policy.py` | Risk-aware policy: positive persistence residuals, one-sided conformal margin, optional offset. |
| `experiments/src/metrics.py` | Cost model, overload metrics, and overload run lengths. |
| `experiments/src/forecasting.py` | Persistence, moving-average, seasonal-naive, and autoregressive-ridge forecasters. |
| `experiments/src/exp_core.py` | Shared replay primitives for the focused-tier supplementary experiments. |
| `experiments/src/exp_core_ls.py` | Loaders and adapters for the 200-service suite. |
| `experiments/src/exp_validator_fixes.py` | Additional baselines B3–B5, persistent demand shift (EXP-10b), paired tests, and overhead frontier. |

## Forecasters

| Script | Purpose |
|---|---|
| `experiments/src/train_arima.py` | ARIMA with Fourier terms (original pipeline; the ARIMA results in the article come from `analysis/audit/recompute_arima_causal.py`). |
| `experiments/src/train_xgb.py` | XGBoost with an Optuna search (seed 42, expanding-window CV). |
| `experiments/src/train_lstm.py` | LSTM with an Optuna search (seed 42). |
| `experiments/src/evaluate_forecasting_models.py` | Accuracy of the simple forecasters. |
| `experiments/src/evaluate_advanced_forecasting.py` | Accuracy of ARIMA, XGBoost, and LSTM against the simple forecasters. |
| `experiments/tests/test_xgb_alignment.py` | Behavioural test: XGBoost features use only past values. |
| `analysis/audit/final_closure_local_runs.py` | Canonical focused-tier XGBoost and LSTM fits; records the environment, macOS only (`--smoke` elsewhere). |
| `analysis/audit/refit_focused_causal_fill.py` | Refits one focused learned forecaster on an explicitly chosen input series. |
| `analysis/audit/refit_arima_fill_check.py` | ARIMA check for the past-only fill of one focused service. |
| `analysis/audit/recompute_arima_causal.py` | Focused-tier ARIMA under a verified one-step causal protocol. |

## Focused tier (20 services)

| Script | Purpose |
|---|---|
| `scripts/run_all.sh` | Runs the original focused-tier pipeline in order: series, replays, forecasters, replay with learned forecasts. |
| `experiments/src/run_baseline_replay.py` | Baseline policies with persistence forecasts. |
| `experiments/src/run_risk_policy_replay.py` | Risk-aware policy with persistence forecasts. |
| `experiments/src/run_advanced_policy_replay.py` | Policies with ARIMA, XGBoost, and LSTM forecasts. |
| `experiments/src/run_supplemental_experiments.py` | Runs EXP-2 to EXP-10 (window, α, guard, run lengths, costs, penalties, recovery). |
| `experiments/src/run_exp3.py` | EXP-3: window-size sensitivity. |
| `experiments/src/run_exp4.py` | EXP-4: conservativeness α sweep. |
| `experiments/src/run_exp5.py` | EXP-5: guard-rail ablation. |
| `experiments/src/run_exp7.py` | EXP-7: overload run lengths (violation clustering). |
| `experiments/src/run_exp8.py` | EXP-8: cost decomposition. |
| `experiments/src/run_exp9.py` | EXP-9: penalty sensitivity. |
| `experiments/src/run_exp10.py` | EXP-10: recovery time. |
| `experiments/src/run_c1_adaptive_conformal.py` | Adaptive conformal (B10) and weighted conformal (B11) against fixed-δ B6. |
| `experiments/src/run_c4_exchangeability.py` | Residual exchangeability diagnostics against compliance. |
| `experiments/src/run_c5_frontier_density.py` | Denser focused cost–risk frontier. |
| `experiments/src/run_p3_penalty_optimizer.py` | Whether a cost-optimising predictive agent under-provisions at weak penalties. |
| `experiments/src/run_synthetic_dkw.py` | Controlled i.i.d. experiment for the DKW localisation of the conformal quantile. |
| `analysis/audit/recompute_final_closure_downstream.py` | Installs the canonical focused forecasts and recomputes every focused-tier result; history-only checks are skipped when their files are absent. |
| `analysis/audit/recompute_focused_learned_downstream.py` | Installs one past-only refit and recomputes the focused outputs. |
| `analysis/audit/phase8_focused_fill_sensitivity.py` | Imputation sensitivity of the focused series. |
| `analysis/audit/recompute_overhead_accounting.py` | Predictor timing logs reconciled with replay-only costs. |
| `analysis/audit/recompute_timeline_sensitivity.py` | Forecasting/calibration timeline and warm-start sensitivity. |

## 200-service suite, immediate actuation

| Script | Purpose |
|---|---|
| `experiments/src/run_ls_frontier.py` | Cost–risk frontier and coverage for the 200 services. |
| `experiments/src/run_ls_baselines.py` | Baselines on the 200 services. |
| `experiments/src/run_ls_stats.py` | Aggregation and significance tests for the 200 services. |
| `experiments/src/run_ls_c3_conditional.py` | Conditional (at-risk) compliance by re-aggregation. |
| `experiments/src/run_ls_actuation_delay.py` | Earlier actuation-delay sensitivity on the 200 services. |
| `experiments/src/run_c2_delay_alpha.py` | Actuation delay × conservativeness α. |
| `experiments/src/run_c6_cost_ci.py` | Paired bootstrap intervals for relative-cost differences. |
| `analysis/audit/recompute_margin_comparison.py` | Conformal against Gaussian margins (matched factorial). |
| `analysis/audit/recompute_reactive_baseline.py` | Reactive comparator tuned on days 8–9 without test leakage. |
| `analysis/audit/recompute_verified_baselines.py` | Non-learned baselines on the verified 200 series. |
| `analysis/audit/recompute_verified_rank_ablation.py` | Matched margin/rank ablation (corrected against inverse-CDF rank). |
| `analysis/audit/recompute_verified_effects.py` | Matched policy effects. |
| `analysis/audit/recompute_verified_dependence.py` | Dependence-aware uncertainty for the finite suite. |
| `analysis/audit/statistical_robustness.py` | Hierarchical moving-block bootstrap and Wilcoxon tests. |
| `analysis/audit/recompute_preprocessing_sensitivity.py` | Sensitivity to gap filling in the 200 series. |
| `analysis/audit/recompute_guardrail_actuation.py` | Offset mechanism, recovery, and closed-loop delay. |
| `analysis/audit/recompute_external_validity.py` | Service characteristics used to scope external-validity claims. |
| `analysis/review/readiness_checks_2026_09_23.py` | Independent re-implementation of the strict-budget replay rows. |
| `analysis/audit/strict_budget_summary.py` | Strict 1% budget comparison (Table 3). |
| `analysis/audit/budget_utilisation.py` | Budget utilisation of the frozen 1% replays. |
| `analysis/audit/final_extension_2026_09_29.py` | Resource axis, scale strata, cost decomposition (protocol 2026-09-29). |
| `analysis/audit/exact_envelope_2026_09_29.py` | Exact resource–overload envelope. |

## External replication (Huawei 2023, Azure 2019)

| Script | Purpose |
|---|---|
| `analysis/audit/external_replication_2026_09_30.py` | Replication under the protocol of 2026-09-30; `regress-alibaba` checks the Alibaba results first. |
| `analysis/audit/external_replication_summary_2026_09_30.py` | Summaries, called by `external_replication_2026_09_30.py summarise`. |

## Actuation delay, PAC-h, confirmatory replay (protocol 2026-10-02, addenda R1–R9)

| Script | Purpose |
|---|---|
| `analysis/audit/revision_2026_10_02.py` | R1 delay at a strict budget, R2 extended reactive grid, R3 horizon-aligned scores; shared helpers. |
| `analysis/audit/revision_subset_2026_10_02.py` | Pre-test non-trivial subset of 95 services. |
| `analysis/audit/revision_external_delay_2026_10_02.py` | R4: closed-loop delay on Huawei and Azure. |
| `analysis/audit/revision_alibaba_delay_2026_10_02.py` | R5: closed-loop delay on Alibaba with the same policies. |
| `analysis/audit/revision_pac_rank_2026_10_02.py` | R6: PAC (training-conditional) rank on three traces. |
| `analysis/audit/revision_round2_2026_10_02.py` | R7: PAC-h window and η grid, one-step ablation, pre-test window rule, conformal test martingales. |
| `analysis/audit/revision_confirmatory_2026_10_02.py` | R8: confirmatory replay on the untouched Huawei periods (frozen by SHA-256). |
| `analysis/audit/revision_round3_2026_10_02.py` | R9: stride-(τ+1) PAC rank and cost-aware window rule. |

## Canonical capacity units, matched-rank ablation, clustered sensitivity (2026-10-03, R10–R12)

| Script | Purpose |
|---|---|
| `analysis/audit/canonical_inputs_2026_10_03.py` | Canonical reading of capacity units: full-precision parsing of `units.csv` and a documented relative boundary tolerance of 1e-12 for exact ties. |
| `analysis/audit/test_canonical_inputs_2026_10_03.py` | Unit tests of the canonical rule: exact tie, value inside the tolerance, value outside it, consistency of the capacity mapping. |
| `analysis/audit/run_canonical.py` | Runs any analysis script with the canonical rule installed, without modifying the script. |
| `analysis/audit/revision_canonical_2026_10_03.py` | R10 re-run of all Huawei/Azure analyses under the canonical rule with a list of changed values; R11 matched-rank ablation; R12 function-clustered confirmatory sign test. |
| `analysis/audit/verify_confirmatory_inputs_2026_10_03.py` | Checks the frozen code hashes and the content of the prepared confirmatory inputs. |

## Figures and table rows

| Script | Purpose |
|---|---|
| `analysis/audit/plot_delay_three_traces_2026_10_02.py` | Fig. 2: compliance against actuation delay on three traces (reads the canonical results when present). |
| `analysis/audit/plot_cost_compliance_tau1_2026_10_02.py` | Fig. 3: cost against compliance at τ = 1 (reads the canonical results when present). |
| `analysis/audit/plot_external_replication_2026_09_30.py` | External-replication figure and LaTeX table rows. |
| `analysis/audit/plot_final_extension_2026_09_29.py` | Resource-envelope and guard-characterisation figures. |
| `analysis/audit/tables_final_extension_2026_09_29.py` | LaTeX rows: scale strata, cost decomposition, pre-test-selected points. |
| `analysis/audit/plot_verified_phase2.py` | 200-service coverage, strata, margin-factorial, and percentile figures. |
| `analysis/audit/plot_phase6_figures.py` | Focused frontier, calibration, and window-sensitivity figures. |
| `analysis/audit/plot_phase7_dkw.py` | DKW localisation figures. |
| `experiments/src/generate_figures.py` | Original focused-tier figures (exploratory; needs seaborn). |
| `experiments/src/generate_ls_figures.py` | Original 200-service figures and table fragments. |
| `experiments/src/replot_exp3.py` | Re-plots EXP-3 from its saved CSV. |

## Reproducibility helpers

| Script | Purpose |
|---|---|
| `analysis/reproducibility/run_final_closure_compute.sh` | Resumable runner for the canonical focused-tier fits and their downstream recomputation. |
| `analysis/reproducibility/SEEDS.md` | Every random seed and where it is set. |
