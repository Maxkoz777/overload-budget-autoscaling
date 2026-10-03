# Data

This repository contains no trace data, no preprocessed series, and no results. All three trace families are public. Download them from their owners and follow each owner's terms of use.

## Folder layout

```text
workspace/
├── shared-data/                         raw external traces (you download them)
│   ├── huawei_2023/private/
│   │   ├── requests_minute/day_000.csv …
│   │   └── instances_minute/day_000.csv …
│   └── azure_functions_2019/raw/
│       └── invocations_per_function_md.anon.d01.csv … d14.csv
└── overload-budget-autoscaling/         this repository
    ├── processed/final/                 Alibaba partitions (written by the pipeline)
    └── experiments/data/                series written by the scripts
```

The Huawei and Azure preparation scripts look for `shared-data/` in the folder that contains the repository.

## Alibaba Microservices Trace v2022

- **Source.** Alibaba cluster trace programme, `cluster-trace-microservices-v2022` (https://github.com/alibaba/clusterdata).
- **Used.**
  - The first 13 days at one-minute resolution.
  - MSMetrics (per-instance CPU and memory utilisation), MSRTMCR (call rates and response times), and NodeMetrics. CallGraph is not used.
- **Download and preprocessing.** `pipeline/run_hourly_pipeline.py` calls `fetchData.sh` for each hour, aggregates the archives to service level, and commits Parquet partitions atomically to `processed/final/`:

  ```bash
  python3 pipeline/run_hourly_pipeline.py --start 0d0 --end 13d0 --project-root . \
      --cleanup-on-success true --disk-abort-gb 30 --disk-warn-gb 50
  ```

- **Disk.** About 172 GB for `processed/final/` (312 hourly partitions). Raw archives are deleted after each hour unless `--keep-raw` is given.
- **Missing NodeMetrics.** NodeMetrics windows are missing or empty for four hours. Re-run those hours with `--sources MSMetrics,MSRTMCR`. The service-level tables the article uses (`service_resource`, `service_rtmcr`, `joined_service_features`) are then complete for all 312 hours.
- **Units.** Utilisation and call-rate values are min–max normalised by Alibaba. The article uses the summed normalised CPU of a service as a demand proxy, not as physical cores.
- **Services.** The 200 services are listed in `experiments/data/splits/selected_services_200.csv`. `experiments/src/select_services_200.py` regenerates the list (seed 42).

## Huawei Cloud 2023 (private-cloud serverless functions)

- **Source.** Huawei Cloud data release 2023 (https://github.com/sir-lab/data-release, file `README_data_release_2023.md`).
- **Used.** Per-minute `requests` and `instances` of the serverless functions in the private-cloud release.
- **Trace days needed.**

  | Trace days | Purpose | Script |
  |---|---|---|
  | 28–60 (the longest contiguous run) | external replication | `analysis/audit/prep_external_traces_2026_09_30.py --family huawei2023` |
  | 0–18, 147–165, 168–184 | confirmatory replay, not touched by any earlier analysis | `analysis/audit/prep_huawei_confirmatory_2026_10_02.py` |

- **Expected layout.** `shared-data/huawei_2023/private/requests_minute/day_DDD.csv` and `shared-data/huawei_2023/private/instances_minute/day_DDD.csv`, with three-digit day numbers.

## Azure Functions 2019

- **Source.** Azure Public Dataset, Azure Functions Dataset 2019 (https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md).
- **Used.** Per-minute invocation counts per function for days 1–14, aggregated to applications.
- **Expected layout.** After extracting the archive: `shared-data/azure_functions_2019/raw/invocations_per_function_md.anon.d01.csv` … `d14.csv`.
- **Intermediate files.** `analysis/audit/prep_external_traces_2026_09_30.py` writes per-day aggregates to `~/ext_cache_2026_09_30` (override with the environment variable `EXT_CACHE`) and the final minute series to `experiments/data/external_traces/`.

## Integrity records

The preparation scripts write an `inputs_sha256.json` (or an equivalent record) next to each prepared series. It lists the SHA-256 of every raw file read, by its path relative to the workspace folder. The confirmatory protocol (`analysis/audit/verified_results/revision_2026-10-02/protocol.json`, addendum `R8_confirmatory`) stores the hashes of the prepared periods used in the article.
