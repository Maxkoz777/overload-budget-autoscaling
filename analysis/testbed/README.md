# Kubernetes testbed: analysis after the freeze

This folder holds everything about the Kubernetes testbed of the article (Sect. 7 and Additional file 1, Sect. S11) that was written **after** the inputs were frozen:

- the analysis that builds the article's tables and figures from the raw run data;
- two operator scripts used during the campaign;
- the list of deviations from the protocol;
- the SHA-256 of every file of the raw data archive.

The frozen inputs (code, protocol, plan, calibration, actuator model) are in [`testbed/`](../../testbed/). They are unchanged since the freeze, so this folder lives outside `testbed/`, and the check below still passes:

```bash
python3 testbed/freeze.py --verify      # from the repository root: "123 inputs identical to FREEZE.json"
```

## Campaign

| Step | Date (2026, UTC+3) | Record |
|---|---|---|
| Capacity calibration, phases 1–3 and amendment A1 | 21 Oct | `testbed/capacity.json`, `testbed/runs/capacity_phase*` |
| Pilot (2 h, fits the actuator model) | 21 Oct | `testbed/actuator_model.json`, `testbed/runs/pilot` |
| Calibration and pilot committed | 21 Oct, 20:17 | commit `0971244` |
| Freeze (input hashes) | 21 Oct, 20:17 | commit `1211dc1`, `testbed/FREEZE.json` |
| Runs 1–4 (days 10, 11, 10, 11) | 21–26 Oct | `testbed/runs/run1` … `run4` (rendered manifests) |
| Combination of the four runs | 26 Oct | `results/combined.json` in the data archive |

The policies rotated over the four workers so that each ran once on every worker: run 1 `pach` w1, `conf` w2, `b1` w3, `hpa` w4, and each later run shifts the assignment by one worker.

**Hypothesis names.** The protocol names the hypotheses H1–H3 and the HPA comparison H4. The article calls them K1–K3, to distinguish them from the hypotheses of the confirmatory replay; the HPA comparison is descriptive in both.

## Data

The raw data of all calibration phases, the pilot, the four runs, and every failed or superseded attempt are archived separately:

- **Archive:** `testbed-results-2026-10-26.zip`, https://doi.org/10.5281/zenodo.23818617. 428 files; 0.77 GB packed, 5.2 GB unpacked.
- **Licence:** CC BY 4.0. The replayed requests are scaled per-minute counts of 12 functions of the Huawei Cloud 2023 data release, itself under CC BY 4.0 ([`docs/DATA.md`](../../docs/DATA.md)).
- **Also in the deposit:**
  - the two archives of the local engineering stage of 18 October 2026, `testbed/local_stage/*.tar.gz`, whose checksums are in `testbed/local_stage/SHA256SUMS`;
  - `RESULTS_SHA256.txt`, the same file as here;
  - `SHA256SUMS.txt`, with the checksums of the uploaded files.

Unpack it into `testbed/`, so that it creates `testbed/results/`, and check it from the repository root:

```bash
(cd testbed && shasum -a 256 -c ../analysis/testbed/RESULTS_SHA256.txt)
```

The check succeeds if it exits with code 0 and prints `OK` for all 428 files. The archive contains:

- the raw files of each run (`load.csv`, `counts.csv`, `pods.csv`, `served.csv`, `cpu.csv`, `hpa.csv`, `decisions_*.csv`);
- the outputs of the frozen `testbed/analyze.py` and `testbed/combine.py`;
- the container logs and the operator logs (`ops/`).

## Rebuilding the tables and figures

Python 3 with NumPy, pandas, and Matplotlib. Run from any folder; outputs go to `analysis/testbed/output/` (set `TESTBED_ANALYSIS_OUT` to change it):

```bash
cd analysis/testbed
python3 extract.py                 # raw run files -> output/data/ (about 30 s)
python3 attribution_fix.py         # corrected attribution of removed pods (see below)
python3 build_tables.py            # output/tables/*.csv and draft *.tex
python3 build_figures.py           # output/figures/*.pdf and *.png
PAPER=1 python3 build_figures.py   # output/figures_paper/: the figures of the article
diff -r reference_tables output/tables -x '*.tex' && echo "tables identical"
```

[`reference_tables/`](reference_tables/) holds the tables behind the numbers of the article. We rebuilt them from the archive with Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, and Matplotlib 3.10.9:

- **Tables.** All 22 were identical byte for byte. Independent rebuilds with pandas 2.2.3 and, from v1.3.1 on, with the pinned Python 3.14.7, NumPy 2.4.6, and pandas 3.0.3 gave the same 22 tables. In v1.3.0, `extract.py` failed with pandas 3: splitting the categorical request targets did not return the policy names.
- **Derived data files.** In our rebuild, the decompressed contents of all 41 files in `output/data/` were identical. In another environment, the gzip headers differed and five files differed in the last digits of floating-point values (at most about 1e-12). The derived data are therefore numerically, not byte for byte, reproducible.

**T and X tables.** Tables **T** report quantities that the protocol defines; their K1–K3 counts equal `combined.json`. Tables **X** are diagnostics added after the runs (synchrony of bad requests, scale events, drift, hourly CPU per request, node snapshot): they exclude no request and establish no cause. There is no T11.

| Article | Source |
|---|---|
| Table 4 | `T04_h1`, `T05_h2`, `T07_cost`, `T08_users` |
| Fig. 4, actuation delays | `F01_actuation_delays` (`T09_delays`) |
| Fig. 5, overload per function-run | `F02_overload_by_function` (`T05_h2`, `function_runs.csv`) |
| Sect. 7.2, K3 sign tests and bootstrap | `T06_h3`, `T06b_h3_by_function` |
| Sect. 7.2, resources and the HPA | `T07_cost`, `T10b_cpu_by_node`, `T13_hpa` |
| Sect. 7.2, user-visible errors | `T08_users`, `T14_proxy_vs_users_curve`, `X01_bad_second_synchrony`, `X04_drift` |
| S11.1–S11.4, functions, runs, validity | `T01_campaign`, `T02_parameters`, `T03_functions` |
| S11.5, actuation delays | `T09_delays`, `T09b_delays_by_policy`, `F01` |
| S11.6, K1 | `T04_h1`, `F04_h1_model_vs_observed` |
| S11.7, K2, K3, resources | `T05_h2`, `T06_h3`, `T06b_h3_by_function`, `T07_cost`, `T12_repeatability`, `F03`, `F05`, `F06` |
| S11.8, user-visible errors | `T08_users`, `T14_proxy_vs_users_curve`, `F09_proxy_vs_users` |
| S11.9, workers and the HPA | `T10_cpu_per_request`, `T10b_cpu_by_node`, `T13_hpa`, `F07_node_effect` |
| S11.10, post hoc diagnostics | `X01`–`X06`, `F08`, `F10`, `F11` |

## Corrections after the freeze

1. **Attribution of removed pods.**
   - **The flaw.** The frozen `analyze.py` attributes a removed pod to a scale-down call if the removal time lies in `[call − 1 s, next call)`. With removal times at 1 s resolution, consecutive windows overlap by one second. One pod (run 2, B1, f9) was therefore attributed to two calls.
   - **The fix.** `attribution_fix.py` first reproduces the frozen records exactly. It then applies the non-overlapping rule that the frozen code already uses for scale-ups; the only difference is that the duplicate record is dropped (`output/data/attribution_correction_diff.csv`).
   - **The effect.** The scale-down delays in `T01`, `T09`, `T09b`, and `F01` use the corrected records. No reported quantile changes, and K1–K3 do not use these records.
2. **Denominator of a post hoc share.** In `X01`, `residual_contribution_pct` is the contribution of bad requests outside synchronous seconds to the overall share (denominator: all requests). `bad_pct_among_requests_outside_ge3` is the share among the requests outside those seconds.

The frozen outputs of `analyze.py` and `combine.py` in the archive are unchanged.

## Operator scripts used during the campaign

The main operator script, `testbed/ops/cloud_stage.sh`, is frozen. Two further scripts ran outside the frozen inputs. Both read the cluster credentials from `~/.testbed-cloud.env`, which is not part of the repository. In each script, only the line that locates the repository was changed, to the folder layout used here.

- **[`ops/finish_fetch.sh`](ops/finish_fetch.sh)** recovered runs 1–3, after `kubectl cp` broke off while copying their results.
  - It computes SHA-256 inside the driver pod, copies each file again until it matches, and verifies all files.
  - Only then does it delete the namespaces and run the frozen `analyze.py`.
  - Every failed copy is kept in the archive (`run*_fetch_failed`, `run*_superseded`).
- **[`ops/node_snapshot.sh`](ops/node_snapshot.sh)** recorded, after run 4 and before the cluster was deleted:
  - the CPU model of each node;
  - steal time since boot;
  - a single-thread SHA-256 benchmark.

  This snapshot is descriptive and outside the protocol.

## Cluster

| Item | Value |
|---|---|
| Provider | VK Cloud Managed Kubernetes, one availability zone. Cloud.ru refused the quota and Selectel lacked hosts. |
| Kubernetes | 1.35.6 (server), CRI-O 1.35.5, AlmaLinux 9.7, Calico; full record in `testbed/cluster/cluster.json` |
| Nodes | five STD3-4-8 nodes (4 vCPU, 8 GiB), no node autoscaling: `load` and workers w1–w4 |
| Images | `ghcr.io`, pulled by digest; digests in `testbed/cluster/images.json`; build from `testbed/app`, `testbed/loadgen`, `testbed/tools` |
| API access | a service-account token; no credential is stored in this repository or in the archive |

Deviations from the protocol: [`DEVIATIONS.md`](DEVIATIONS.md).
