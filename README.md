# How Much Headroom Is Enough? Overload Budgets for Predictive Autoscaling under Actuation Delay (code)

[![DOI](https://zenodo.org/badge/1403471325.svg)](https://doi.org/10.5281/zenodo.23125234)

Code for the article

> M. Kozhinov, M. Mazzara. *How Much Headroom Is Enough? Overload Budgets for Predictive Autoscaling under Actuation Delay.* Submitted to the *Journal of Cloud Computing* (2026).

This repository contains **scripts only**. It includes no trace data, no preprocessed series, and no precomputed results. All three traces used in the article are public. Download them from their owners ([`docs/DATA.md`](docs/DATA.md)), and the scripts rebuild every number, table, and figure from them.

## Contents

- [What the code does](#what-the-code-does)
- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Where to put the data](#where-to-put-the-data)
- [Reproducing the results](#reproducing-the-results)
- [Where each result in the article comes from](#where-each-result-in-the-article-comes-from)
- [Protocols and integrity checks](#protocols-and-integrity-checks)
- [Citation and licence](#citation-and-licence)

A one-line description of every script is in [`docs/SCRIPTS.md`](docs/SCRIPTS.md).

## What the code does

An operator declares an **overload budget** `δ`: the fraction of minutes in which demand may exceed provisioned capacity. A predictive autoscaler sets the replica count for a future minute in three steps:

1. forecast demand;
2. add a one-sided conformal margin, an order statistic of the last `W` positive forecast errors;
3. round the protected demand up to whole replicas.

The proposed controller, **PAC-h**, makes two changes to this rule:

- **Horizon alignment.** A decision applied `τ` minutes after it is made is calibrated on `(τ+1)`-step errors. Without this, the margin covers an error the decision no longer faces.
- **Training-conditional (PAC) rank.** For one fixed window of i.i.d. scores, the rank guarantees coverage of that window with confidence `1−η`, not only on average. The controller applies it to overlapping, dependent multi-step scores and selects the window from data; this implementation is evaluated empirically, not covered by the guarantee.

The code replays this controller and its comparators against three public production traces with a closed-loop actuation queue. The comparators are:

- reactive threshold control (HPA-style);
- pure prediction;
- fixed, Gaussian, and empirical-percentile margins;
- the standard conformal margin with and without an offset.

For each policy, the replay measures realised overload, the number of services within budget, and relative replay cost.

| Trace | Units | Role in the article |
|---|---|---|
| Alibaba Microservices Trace v2022 | 20 focused services (four forecasters); fixed suite of 200 services (persistence) | development and main analyses |
| Huawei Cloud 2023, private serverless functions | 64 functions (days 28–60); 195 unit-runs in three untouched periods | external replication; confirmatory replay |
| Azure Functions 2019 | 2,839 applications | external replication |

## Repository layout

```text
overload-budget-autoscaling/
├── README.md                     this file
├── LICENSE                       MIT
├── CITATION.cff
├── requirements.txt              pinned versions used for the article
├── requirements-optional.txt     optional accelerators
├── matplotlibrc                  TrueType fonts in PDF figures (also in analysis/)
├── docs/
│   ├── DATA.md                   where to download each trace and where to put it
│   └── SCRIPTS.md                one-line description of every script, grouped by stage
├── fetchData.sh                  downloads the hourly Alibaba v2022 archives (called by the pipeline)
├── pipeline/
│   └── run_hourly_pipeline.py    Alibaba: download + hour-by-hour aggregation → processed/final/
├── experiments/
│   ├── src/                      core library and the original replay experiments:
│   │                             simulator, policies, forecasters (ARIMA, XGBoost, LSTM),
│   │                             service selection, focused-tier and 200-service replays
│   ├── tests/                    behavioural test of the XGBoost feature alignment
│   └── data/splits/              study configuration: chronological split, list of the 200 services
├── analysis/
│   ├── audit/                    canonical replays, statistics, revision analyses, figures, table rows
│   │   └── verified_results/     protocol files written before the corresponding runs
│   ├── review/                   independent re-implementation used by one summary step
│   └── reproducibility/          random seeds; runner for the canonical learned-forecaster fits
└── scripts/
    └── run_all.sh                focused-tier pipeline: series → forecasters → replay
```

Notes on the layout:

- **Folder structure.** The scripts keep the relative folder structure of the original project, because several analyses import each other and locate their inputs relative to their own position. `analysis/audit/` is therefore one flat folder; [`docs/SCRIPTS.md`](docs/SCRIPTS.md) groups its contents by stage.
- **Working directory.** Run the commands for `pipeline/`, `experiments/`, and `scripts/` from the repository root, and the commands for `analysis/` from inside `analysis/`.
- **Outputs.** Generated data and results are written next to the code and are excluded by `.gitignore`:
  - `processed/` — preprocessed Alibaba partitions;
  - `experiments/data/` — service series, apart from `splits/`;
  - `experiments/results/` and `experiments/figures/` — focused-tier outputs;
  - `analysis/audit/*_results/` and `analysis/audit/verified_results/` — analysis outputs, apart from the protocol files;
  - `analysis/audit/verified_results_canonical/` — re-run with the canonical capacity-unit rule (step 7b);
  - `analysis/figures/` — figures.
- **`experiments/data/splits/` is study configuration, not data.** It holds the day boundaries of the split and the list of the 200 selected service IDs with the summary statistics used to stratify and verify them. `experiments/src/select_services_200.py` regenerates the list from the trace (seed 42).

## Environment

- **Pinned environment.** `requirements.txt` pins the main environment: Python 3.14.7, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1, statsmodels 0.14.6, pmdarima 2.1.1, scikit-learn 1.8.0, XGBoost 3.2.0, PyTorch 2.12.0, Optuna 4.8.0, Matplotlib 3.10.9, PyArrow 24.0.0.
- **Recorded environments.** Each analysis records the environment it ran in, in a `methodology*.json` file next to its outputs:
  - the canonical learned-forecaster fits record macOS 15.5 (Apple M1);
  - the revision analyses R1–R12 record Python 3.13.15, NumPy 2.5.3, and pandas 3.0.5.
- **Verified for reproduction.** The clean-folder check below used Python 3.13.15, NumPy 2.5.3, and pandas 3.0.5. The unit tests of the canonical rule also pass with pandas 2.3.3 and NumPy 2.2.6.
- **Operating system.** The replay and analysis scripts use NumPy, pandas, and SciPy only and do not depend on the operating system.
- **Learned forecasters.** The canonical fits (`analysis/audit/final_closure_local_runs.py`) record the executing environment and refuse to run outside macOS; `--smoke` runs a short test on other systems.
- **Randomness.** Seeds are listed in [`analysis/reproducibility/SEEDS.md`](analysis/reproducibility/SEEDS.md). Deterministic replays use no random numbers.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional: numba, polars, psutil, seaborn
```

## Where to put the data

```text
workspace/
├── shared-data/                      raw Huawei 2023 and Azure 2019 traces (see docs/DATA.md)
│   ├── huawei_2023/private/{requests_minute,instances_minute}/day_XXX.csv
│   └── azure_functions_2019/raw/invocations_per_function_md.anon.dXX.csv
└── overload-budget-autoscaling/      this repository
    └── processed/final/              Alibaba partitions, written by pipeline/run_hourly_pipeline.py
```

The external-trace scripts look for `shared-data/` **next to** the repository folder. The Alibaba pipeline writes inside the repository (`--project-root .`). Details, sizes, and the trace days required are in [`docs/DATA.md`](docs/DATA.md).

## Reproducing the results

The steps follow the order in which the results were produced. Each step reads the outputs of the previous ones, and many scripts first check that they reproduce the outputs of earlier steps exactly.

**Three levels of reproduction.**

1. **Integrity only (seconds).** Check the frozen code hashes of the confirmatory replay ([below](#protocols-and-integrity-checks)).
2. **Replay with persistence forecasts (no model training).** Steps 1, 2 (series only), 4–7, and 7b rebuild every delay, PAC-h, strict-budget, and external-replication result, including Tables 1–4 and Figs. 2–3. "Series only" means building the service series without training any model:

   ```bash
   python3 experiments/src/build_service_timeseries.py              # focused 20 services (from the repository root)
   cd analysis && python3 audit/rebuild_verified_service_series.py  # fixed 200-service suite
   ```

   For the Huawei confirmatory replay alone (Table 2), the raw Huawei trace and a few scripts suffice. The article's Table 2 uses the canonical capacity-unit rule, so run the frozen script through the wrapper:

   ```bash
   cd analysis
   python3 audit/prepare_confirmatory_periods.py
   python3 audit/verify_confirmatory_inputs.py
   python3 audit/run_canonical.py audit/confirmatory_replay.py   # published (canonical) Table 2
   ```

   Running `python3 audit/confirmatory_replay.py` without the wrapper reproduces the frozen run instead. It gives the same counts and differs in four cost entries by 0.01. Both variants write to `audit/verified_results/delay_pac_study/`, so use separate copies of the repository if you want to keep both.

3. **Learned forecasters.** Steps 2 (`run_all.sh`) and 3 retrain ARIMA, XGBoost, and LSTM for the focused 20-service tier only. The canonical fits run on macOS and take about 7 hours.

**Project-history checks.** A few scripts also compare their outputs with files from the project's own history, for example the earlier service series or superseded forecasts. These comparisons are provenance evidence for the article, not reproduction steps: `rebuild_verified_service_series.py` and `recompute_final_closure_downstream.py` skip them automatically when the historical files are absent.

**Canonical capacity units.** The numbers in the article come from the canonical rule in `analysis/audit/canonical_inputs.py`:

- capacity units are read from `units.csv` at full precision;
- a documented relative boundary tolerance of 1e-12 (each unit is scaled by 1 + 1e-12) makes exact ties, where demand equals a whole number of units, resolve as in exact arithmetic, consistently for every policy, the overload count, and the clairvoyant cost reference.

Like any tolerance, the rule would also treat a strict inequality closer than a relative 1e-12 to the boundary as a tie. `python3 audit/test_canonical_inputs.py` tests an exact tie, a value inside the tolerance, a value outside it, and the consistency of the capacity mapping. The rule also makes the results independent of the pandas version's default float parser.

Step 7b re-runs every analysis on the Huawei and Azure traces with this rule and lists every number that changes. The frozen scripts are not modified: `analysis/audit/run_canonical.py <script>` runs any of them with the rule installed.

| Step | What | Cost |
|---|---|---|
| 1 | Alibaba download and preprocessing | ≈172 GB, several days |
| 2 | Service series; focused-tier pipeline | hours |
| 3 | Canonical focused-tier forecasts | ≈7 h on an M1 (macOS) |
| 4 | Fixed 200-service persistence comparison | < 1 h |
| 5 | Resource axis and scale strata | minutes |
| 6 | External replication (Huawei, Azure) | ≈1 h, plus trace download |
| 7 | Actuation delay, PAC-h, confirmatory replay | ≈1–2 h |
| 8 | Supplementary analyses | minutes to hours |

### 1. Alibaba trace: download and preprocessing

```bash
python3 pipeline/run_hourly_pipeline.py --start 0d0 --end 13d0 --project-root . \
    --cleanup-on-success true --disk-abort-gb 30 --disk-warn-gb 50
```

The pipeline is resumable and commits each hour atomically to `processed/final/`. NodeMetrics windows are empty for four hours of the trace; re-run those hours with `--sources MSMetrics,MSRTMCR` (see `docs/DATA.md`).

### 2. Service series and the focused-tier pipeline

```bash
bash scripts/run_all.sh
cd analysis && python3 audit/rebuild_verified_service_series.py
```

`run_all.sh` does four things:

- builds the 20 focused series;
- replays the baseline and risk-aware policies with persistence forecasts;
- trains ARIMA, XGBoost, and LSTM;
- replays the policies with the learned forecasts.

`rebuild_verified_service_series.py` rebuilds the 200 service series from `processed/final/` and checks them against the configuration in `experiments/data/splits/`.

### 3. Canonical focused-tier forecasts (macOS)

```bash
cd analysis
bash reproducibility/run_final_closure_compute.sh
python3 audit/recompute_arima_causal.py
```

The runner does three things, and it is resumable:

- repeats the XGBoost search for the 20 focused services (seed 42, 200 trials, expanding-window cross-validation);
- refits one LSTM service on its past-only series;
- recomputes every focused-tier result that uses these forecasts.

### 4. Fixed 200-service persistence comparison (immediate actuation)

```bash
cd analysis
python3 audit/recompute_margin_comparison.py --verified
python3 audit/recompute_reactive_baseline.py --verified
python3 audit/recompute_verified_baselines.py
python3 audit/recompute_verified_rank_ablation.py
python3 audit/recompute_verified_effects.py
python3 audit/recompute_preprocessing_sensitivity.py --verified
python3 audit/recompute_timeline_sensitivity.py --verified
python3 audit/recompute_guardrail_actuation.py --verified
python3 audit/recompute_external_validity.py --verified
python3 audit/recompute_verified_dependence.py
python3 audit/statistical_robustness.py
python3 review/strict_budget_crosscheck.py
python3 audit/strict_budget_summary.py
python3 audit/budget_utilisation.py
python3 audit/plot_verified_suite.py
```

### 5. Resource axis and scale strata

Protocol: `analysis/audit/verified_results/resource_axis_study/protocol.json`.

```bash
cd analysis
python3 audit/resource_axis_extension.py
python3 audit/resource_overload_envelope.py
python3 audit/plot_resource_axis.py
python3 audit/tables_resource_axis.py
```

### 6. External replication on Huawei Cloud 2023 and Azure Functions 2019

Protocol: `analysis/audit/verified_results/external_replication_study/protocol.json`.

```bash
cd analysis
python3 audit/prepare_external_traces.py --family huawei2023
python3 audit/prepare_external_traces.py --family azure2019 --stage days --days 1-7
python3 audit/prepare_external_traces.py --family azure2019 --stage days --days 8-14
python3 audit/prepare_external_traces.py --family azure2019 --stage build
python3 audit/external_replication.py regress-alibaba
for r in "huawei2023 primary" "huawei2023 p90" "azure2019 primary" "azure2019 K5" "azure2019 K20"; do
  set -- $r; python3 audit/external_replication.py run --family $1 --mu $2; done
(cd audit && python3 external_replication.py summarise)
python3 audit/plot_external_replication.py
```

`regress-alibaba` checks that the replication code reproduces the step-4 Alibaba results exactly. The Azure preparation writes per-day intermediates to `~/ext_cache_overload_budget`; set `EXT_CACHE` to change the location.

### 7. Actuation delay, the proposed controller, and the confirmatory replay

Protocol with addenda R1–R9: `analysis/audit/verified_results/delay_pac_study/protocol.json`. This step needs the outputs of steps 4–6.

```bash
cd analysis
python3 audit/delay_and_reactive_grid.py r1 r2 r3          # R1–R3: delay at 1%, extended reactive grid, horizon-aligned scores
python3 audit/pretest_subset.py            # pre-test non-trivial subset (95 services)
python3 audit/external_delay.py    # R4: delay on Huawei and Azure
python3 audit/alibaba_delay.py     # R5: delay on Alibaba, same policies
python3 audit/pac_rank_replay.py          # R6: PAC rank on three traces
python3 audit/pac_window_grid.py            # R7: PAC-h window/eta grid, ablations, pre-test window rule, martingales
python3 audit/prepare_confirmatory_periods.py   # R8 data: three untouched Huawei periods
python3 audit/confirmatory_replay.py      # R8: confirmatory replay (frozen protocol)
python3 audit/pac_stride_sensitivity.py            # R9: stride PAC rank, cost-aware window rule
```

The figures are drawn after step 7b, from the canonical results.

### 7b. Canonical capacity units, matched-rank ablation, clustered confirmatory sensitivity (R10–R12)

```bash
cd analysis
python3 audit/canonical_rerun.py          # all stages; outputs in audit/verified_results_canonical/
```

The stages are:

- **R10** re-runs the external replication, the extended reactive grid, R4, and R6–R9 on Huawei and Azure with the canonical rule. It writes `r10_changed_rows.csv`, which lists every summary value that differs from the frozen outputs, and `r10_check_log.csv`, the built-in regression checks.
- **R11** is the matched-rank ablation: PAC-h with the standard conformal rank, the same window, and no offset.
- **R12** gives the confirmatory sign test by period and with each function as one cluster.

Run single stages by name, for example `... canonical_rerun.py r11 r12 compare`. Then draw Figs. 2 and 3; the plotting scripts read the canonical results when they are present:

```bash
python3 audit/test_canonical_inputs.py      # unit tests of the canonical rule
python3 audit/plot_delay_compliance.py    # Fig. 2
python3 audit/plot_cost_vs_compliance.py  # Fig. 3
```

### 8. Supplementary analyses

```bash
python3 experiments/src/run_synthetic_dkw.py              # controlled i.i.d. DKW experiment
python3 experiments/src/run_supplemental_experiments.py   # alpha sweep and other focused-tier sensitivities
cd analysis
python3 audit/plot_focused_figures.py
python3 audit/plot_dkw.py
python3 audit/recompute_overhead_accounting.py
python3 audit/focused_fill_sensitivity.py
python3 audit/verify_series_upstream.py                   # 200 series against processed/final
```

## Where each result in the article comes from

Script paths are relative to `analysis/audit/` unless noted.

**Main text**

| Result | Script |
|---|---|
| Fig. 1, control loop of PAC-h | schematic, no data |
| Fig. 2 and Table 1, actuation delay on three traces | `delay_and_reactive_grid.py` (r1, r3), `external_delay.py`, `alibaba_delay.py`, `pac_window_grid.py`, `plot_delay_compliance.py` |
| Fig. 3, cost against compliance at τ = 1 | `plot_cost_vs_compliance.py` (reads the outputs behind Table 1) |
| Matched-rank ablation (Sect. 6.2, Sect. S10); canonical capacity units; clustered confirmatory sensitivity (Sect. 6.3) | `canonical_rerun.py` (R11, R10, R12) |
| Sect. 6.2, window/η ablations, stride variant, cost-aware rule, exchangeability martingales | `pac_window_grid.py`, `pac_stride_sensitivity.py` |
| Table 2, confirmatory replay | `prepare_confirmatory_periods.py`, `confirmatory_replay.py` |
| Table 3, strict 1% budget on the 200-service suite | `strict_budget_summary.py` (inputs from step 4) |
| Table 4, external replication | `external_replication.py`, `plot_external_replication.py` |
| Sect. 6.6, forecasters, offset, resource use | `recompute_final_closure_downstream.py`, `resource_axis_extension.py`, `resource_overload_envelope.py` |

**Additional file 1**

| Section | Script |
|---|---|
| S5, adaptive conformal extensions | `experiments/src/run_c1_adaptive_conformal.py` |
| S6.1, focused-tier protocol and tables; reactive frontier | `scripts/run_all.sh`, `recompute_final_closure_downstream.py`, `recompute_arima_causal.py`, `recompute_reactive_baseline.py` |
| S6.2, design-parameter sensitivity (window, DKW localisation, α sweep) | `plot_focused_figures.py`, `experiments/src/run_synthetic_dkw.py`, `plot_dkw.py`, `experiments/src/run_supplemental_experiments.py` (EXP-4) |
| S6.3, contribution of each layer; guard characterisation | `experiments/src/run_supplemental_experiments.py` (EXP-5), `recompute_guardrail_actuation.py`, `plot_resource_axis.py` |
| S6.4, predictor agnosticism and overhead | `recompute_overhead_accounting.py` |
| S6.5, violation clustering | `experiments/src/run_supplemental_experiments.py` (EXP-7) |
| S6.6, matched margins, 200-service coverage and strata figures, strict-budget sensitivity, rank ablation | `recompute_margin_comparison.py`, `plot_verified_suite.py`, `recompute_verified_rank_ablation.py`, `delay_and_reactive_grid.py` (r2), `pretest_subset.py` |
| S6.7, budget utilisation, cost-constrained comparisons | `budget_utilisation.py`, `recompute_verified_effects.py`, `recompute_verified_dependence.py`, `statistical_robustness.py` |
| S6.8, resource axis, cost decomposition, scale strata | `resource_axis_extension.py`, `tables_resource_axis.py`, `plot_resource_axis.py` |
| S7, external replication | `prepare_external_traces.py`, `external_replication.py`, `external_replication_summary.py` |
| S8, actuation delay and horizon-aligned calibration | `recompute_guardrail_actuation.py`, `experiments/src/run_ls_actuation_delay.py`, `delay_and_reactive_grid.py`, `external_delay.py`, `alibaba_delay.py`, `pac_rank_replay.py` |
| S9, residual exchangeability diagnostics | `experiments/src/run_c4_exchangeability.py`, `pac_window_grid.py` |
| S10, round-two analyses and moved figures (focused frontier and calibration, resource envelope, external replication) | `pac_window_grid.py`, `pac_stride_sensitivity.py`, `confirmatory_replay.py`, `plot_focused_figures.py`, `plot_resource_axis.py`, `plot_external_replication.py` |

## Protocols and integrity checks

- **Protocol files.** `analysis/audit/verified_results/*/protocol.json` record the hypotheses, parameters, and decision rules of the external replication, the resource-axis extension, and the revision analyses (addenda R1–R9). They are the only files that the repository keeps in `verified_results/`.
- **Frozen confirmatory replay.** Addendum `R8_confirmatory` stores two sets of SHA-256 hashes, both written before the replay was run:
  - the hashes of the six scripts it uses (`confirmatory_replay.py`, `pac_window_grid.py`, `external_delay.py`, `delay_and_reactive_grid.py`, `external_replication.py`, `prepare_confirmatory_periods.py`);
  - the hashes of the three prepared data periods.

  The protocol lists these scripts under the file names they had when it was frozen (for example `revision_confirmatory_2026_10_02.py` for `confirmatory_replay.py`). Release v1.0.0 ([10.5281/zenodo.23125235](https://doi.org/10.5281/zenodo.23125235)) preserves them byte for byte under those names. From v1.1.0 on, all scripts and protocol folders have descriptive names; in the six frozen scripts, the only edits are the module, file, and folder names in their text. `verify_confirmatory_inputs.py` maps the current names back to the frozen ones and checks that the reconstructed text has the stored hash:

```bash
cd analysis
python3 audit/verify_confirmatory_inputs.py      # code hashes; data hashes too once the periods are prepared
```

- **Data hashes.** The protocol stores the SHA-256 of each period's `series.npz`. A `.npz` file is a zip archive, and its container bytes need not match across environments even when every array does. In our rebuild, all entries had the same CRC-32, compressed data, and timestamp; only local-header fields written by the zip library differed. After `prepare_confirmatory_periods.py` has run, `python3 audit/verify_confirmatory_inputs.py` checks three things:
  - the frozen code hashes;
  - a content hash of the arrays in each period;
  - the SHA-256 of each `units.csv`.
- **Tested from a clean folder.** In a fresh copy of this repository with only the raw Huawei trace next to it, the scripts rebuilt all three confirmatory periods and the Huawei replication series. All arrays were identical, and so were the `units.csv` files. The frozen confirmatory replay reproduced the frozen run exactly (7,020 per-unit rows). Run through `run_canonical.py`, it reproduced the canonical results reported in the article exactly as well. The external replication and the PAC-h grid for Huawei also matched, to floating-point rounding (largest difference 4e-15).

  Not tested from a clean folder:
  - the Alibaba pipeline, about 172 GB of downloads;
  - the macOS forecaster fits.

## Citation and licence

- **Citation.** See [`CITATION.cff`](CITATION.cff) or the "Cite this repository" button. The code version cited in the article is v1.1.0, archived at https://doi.org/10.5281/zenodo.23125637; v1.0.0 (https://doi.org/10.5281/zenodo.23125235) keeps the original file names, and https://doi.org/10.5281/zenodo.23125234 always points to the latest version. The article's DOI will be added on publication.
- **Code licence.** The code is released under the MIT License ([`LICENSE`](LICENSE)).
- **Trace data.** The traces are not covered by this licence: they are published by their owners under their own terms, and this repository redistributes none of them ([`docs/DATA.md`](docs/DATA.md)).
