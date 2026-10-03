# Random seeds

| Component | Seed | Set in |
|---|---:|---|
| Stratified fill of the 200-service suite | 42 | `experiments/src/select_services_200.py` |
| XGBoost training and Optuna search | 42 | `experiments/src/train_xgb.py`, `analysis/audit/final_closure_local_runs.py` |
| LSTM training (NumPy, PyTorch, Optuna) | 42 | `experiments/src/train_lstm.py` |
| Paired cost bootstrap (C6) | 42 | `experiments/src/run_c6_cost_ci.py` |
| Controlled i.i.d. DKW experiment | 20260530 | `experiments/src/run_synthetic_dkw.py` |
| Margin-comparison bootstrap | 20260920 plus deterministic row offsets | `analysis/audit/recompute_margin_comparison.py` |
| Reactive-comparison bootstrap | 20260920 plus deterministic family offsets | `analysis/audit/recompute_reactive_baseline.py` |
| Paired-effect bootstrap | 20260920 plus deterministic comparison offsets | `analysis/audit/recompute_verified_effects.py` |
| Dependence-aware intervals | 20260920 | `analysis/audit/recompute_verified_dependence.py` |
| Hierarchical moving-block bootstrap | 20260920 unless overridden by `--seed` | `analysis/audit/statistical_robustness.py` |
| Random tie-breaking in conformal test martingales (R7) | 20261002 | `analysis/audit/revision_round2_2026_10_02.py` |
| Unit tests | 0, 1, 20261003 | `experiments/tests/test_xgb_alignment.py`, `analysis/audit/test_canonical_inputs_2026_10_03.py` |

The trace replays are deterministic and use no random numbers. Every stochastic analysis either uses the constant listed above or records its command-line seed in its `methodology*.json` output.
