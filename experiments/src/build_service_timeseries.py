"""
build_service_timeseries.py
============================
Build regular per-service time-series files for the stratified
experimental subset.

The default 20-service set covers 5 volatility regimes × 4 services each:

  Group 1 — Stable          burstiness 1.2–1.6   ARIMA domain
  Group 2 — Moderate        burstiness 1.5–2.4   ARIMA / XGBoost competitive
  Group 3 — High variance   burstiness 2.5–3.5   XGBoost / LSTM domain
  Group 4 — Bursty          burstiness 8–26       LSTM domain
  Group 5 — Highly bursty   burstiness 28–50      stress test

Usage (20-service default):
    python3 experiments/src/build_service_timeseries.py

Usage (200-service large-scale):
    python3 experiments/src/build_service_timeseries.py \
        --services-csv experiments/data/splits/selected_services_200.csv \
        --output-dir experiments/data/service_timeseries_200/

Outputs (default mode):
    experiments/data/service_timeseries/<msname>.parquet  (20 files)
    experiments/data/service_timeseries/_quality_summary.csv
    experiments/reports/service_selection_stratified_report.md

Outputs (large-scale mode):
    experiments/data/service_timeseries_200/<msname>.parquet  (200 files)
    experiments/data/service_timeseries_200/_quality_summary.csv
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


INPUT_DIR   = Path("processed/final/joined_service_features")
OUTPUT_DIR  = Path("experiments/data/service_timeseries")
SUMMARY_PATH = OUTPUT_DIR / "_quality_summary.csv"
REPORTS_DIR = Path("experiments/reports")
REPORT_PATH = REPORTS_DIR / "service_selection_stratified_report.md"

# Number of services scanned per batch in large-scale mode
SCAN_BATCH_SIZE = 20

# Stratified service set (5 groups x 4 services); burstiness = cpu_max / cpu_mean
STRATIFIED_GROUPS: dict[str, list[str]] = {
    # Regular daily pattern, low CV.  ARIMA with Fourier terms should excel.
    "G1_stable": [
        "MS_31285",   # burst=1.18  mean=433  cv=0.16
        "MS_6298",    # burst=1.50  mean=435  cv=0.16
        "MS_63525",   # burst=1.22  mean=310  cv=0.23
        "MS_8458",    # burst=1.52  mean=305  cv=0.32
    ],
    # Moderate variance, clear daily cycle.  ARIMA / XGBoost competitive.
    "G2_moderate": [
        "MS_70053",   # burst=1.46  mean=736  cv=0.30
        "MS_29860",   # burst=1.74  mean=702  cv=0.26
        "MS_42222",   # burst=1.76  mean=409  cv=0.43
        "MS_66711",   # burst=2.36  mean=370  cv=0.43
    ],
    # High absolute variance; occasional large deviations.  XGBoost domain.
    "G3_high_variance": [
        "MS_491",     # burst=3.11  mean=378  cv=0.56
        "MS_48534",   # burst=2.67  mean=307  cv=0.45
        "MS_19988",   # burst=2.84  mean=285  cv=0.46
        "MS_21035",   # burst=2.50  mean=282  cv=0.51
    ],
    # Bursty: frequent spikes well above mean.  LSTM should handle best.
    "G4_bursty": [
        "MS_2024",    # burst=8.07   mean=117  cv=0.51
        "MS_49699",   # burst=9.44   mean=52   cv=0.56
        "MS_14526",   # burst=19.24  mean=39   cv=1.39
        "MS_12652",   # burst=25.53  mean=15   cv=0.58
    ],
    # Highly bursty / irregular.  Stress-test for conformal margin.
    "G5_highly_bursty": [
        "MS_51028",   # burst=27.66  mean=17   cv=1.49
        "MS_5201",    # burst=35.97  mean=6    cv=2.20
        "MS_345",     # burst=40.90  mean=11   cv=1.79
        "MS_25320",   # burst=49.75  mean=41   cv=2.57
    ],
}

SELECTED_SERVICES: list[str] = [
    svc for svcs in STRATIFIED_GROUPS.values() for svc in svcs
]

# Group membership lookup (for reporting)
SERVICE_GROUP: dict[str, str] = {
    svc: grp for grp, svcs in STRATIFIED_GROUPS.items() for svc in svcs
}

COLUMNS = [
    "timestamp", "msname", "replica_count", "node_count",
    "cpu_sum", "cpu_mean", "cpu_p95", "cpu_max",
    "memory_sum", "memory_mean", "memory_p95", "memory_max",
]
CAPACITY_COLUMNS = ["replica_count", "node_count"]
RESOURCE_COLUMNS = [
    "cpu_sum", "cpu_mean", "cpu_p95", "cpu_max",
    "memory_sum", "memory_mean", "memory_p95", "memory_max",
]

TIMESTAMP_START     = 0
TIMESTAMP_END       = 1_123_140_000
TIMESTAMP_STEP      = 60_000
EXPECTED_TIMESTAMPS = 18_720


def adjacent_max_fill(series: pd.Series) -> pd.Series:
    """Fill NaN resource values with max of forward-fill and backward-fill."""
    forward     = series.ffill()
    backward    = series.bfill()
    fill_values = pd.concat([forward, backward], axis=1).max(axis=1)
    result      = series.copy()
    result[result.isna()] = fill_values[result.isna()]
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-service time-series parquets.")
    p.add_argument(
        "--services-csv",
        default=None,
        help="CSV with service_id column to process. If not given, uses hardcoded 20-service set.",
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory to write output parquets (default: experiments/data/service_timeseries/).",
    )
    return p.parse_args()


def process_service_batch(
    services_batch: list[str],
    output_dir: Path,
    grid: pd.Index,
) -> list[dict]:
    """Scan and process a batch of services. Returns summary rows."""
    dataset = ds.dataset(str(INPUT_DIR), format="parquet", partitioning="hive")
    table   = dataset.to_table(
        columns=COLUMNS,
        filter=ds.field("msname").isin(services_batch),
    )
    df = table.to_pandas()

    summary_rows = []
    for service in services_batch:
        service_df = df[df["msname"] == service].copy().sort_values("timestamp")

        observed_rows  = len(service_df)
        duplicate_rows = int(service_df["timestamp"].duplicated().sum())
        if duplicate_rows:
            numeric_cols = [c for c in COLUMNS if c not in {"timestamp", "msname"}]
            service_df = (
                service_df.groupby("timestamp", as_index=False)[numeric_cols]
                .mean()
                .assign(msname=service)
            )
            service_df = service_df[["timestamp", "msname", *numeric_cols]]

        diffs = service_df["timestamp"].diff().dropna()
        non_step_intervals = int((diffs != TIMESTAMP_STEP).sum())

        regular      = service_df.set_index("timestamp").reindex(grid)
        was_missing  = regular["msname"].isna()
        missing_count = int(was_missing.sum())

        regular["msname"]      = service
        regular["was_missing"] = was_missing.astype(bool)

        for col in CAPACITY_COLUMNS:
            regular[col] = regular[col].ffill().bfill().round().astype("int64")
        for col in RESOURCE_COLUMNS:
            regular[col] = adjacent_max_fill(regular[col]).astype("float64")

        regular = regular.reset_index()
        output_path = output_dir / f"{service}.parquet"
        regular.to_parquet(output_path, index=False)

        remaining_nulls = int(regular[COLUMNS + ["was_missing"]].isna().sum().sum())
        group = SERVICE_GROUP.get(service, "n/a")
        print(
            f"  {service:10s}  group={group:20s}  "
            f"missing={missing_count:3d}  nulls_remaining={remaining_nulls}",
            flush=True,
        )
        summary_rows.append({
            "msname":              service,
            "group":               group,
            "observed_rows":       observed_rows,
            "expected_rows":       EXPECTED_TIMESTAMPS,
            "missing_rows":        missing_count,
            "coverage":            observed_rows / EXPECTED_TIMESTAMPS,
            "duplicate_rows":      duplicate_rows,
            "non_60000_intervals": non_step_intervals,
            "output_rows":         len(regular),
            "remaining_nulls":     remaining_nulls,
            "output_file":         str(output_path),
        })
    return summary_rows


def main() -> None:
    args = parse_args()
    large_scale_mode = args.services_csv is not None

    if large_scale_mode:
        csv_path = Path(args.services_csv)
        svc_df = pd.read_csv(csv_path)
        services_to_process = svc_df["service_id"].tolist()
        output_dir = Path(args.output_dir)
        write_report = False
        print(f"Large-scale mode: {len(services_to_process)} services from {csv_path}", flush=True)
    else:
        services_to_process = SELECTED_SERVICES
        output_dir = OUTPUT_DIR
        write_report = True
        print(f"Default mode: {len(services_to_process)} services (hardcoded stratified set)", flush=True)

    summary_path = output_dir / "_quality_summary.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Default mode: remove timeseries files not in the current set.
    removed = []
    if not large_scale_mode:
        for old_file in sorted(output_dir.glob("MS_*.parquet")):
            if old_file.stem not in services_to_process:
                old_file.unlink()
                removed.append(old_file.stem)
                print(f"  Removed old timeseries: {old_file.name}")

    # Resumable: skip services that already have output.
    services_todo = []
    services_skipped = []
    for svc in services_to_process:
        out_path = output_dir / f"{svc}.parquet"
        if out_path.exists():
            services_skipped.append(svc)
        else:
            services_todo.append(svc)

    if services_skipped:
        print(f"  Resuming: {len(services_skipped)} already done, {len(services_todo)} remaining.", flush=True)

    grid = pd.Index(
        range(TIMESTAMP_START, TIMESTAMP_END + TIMESTAMP_STEP, TIMESTAMP_STEP),
        name="timestamp",
    )
    assert len(grid) == EXPECTED_TIMESTAMPS, f"Unexpected grid length: {len(grid)}"

    # Reuse existing quality summary rows for skipped services.
    existing_summary: dict[str, dict] = {}
    if summary_path.exists():
        try:
            ex_df = pd.read_csv(summary_path)
            for _, row in ex_df.iterrows():
                existing_summary[str(row["msname"])] = row.to_dict()
        except Exception:
            pass

    summary_rows: list[dict] = []

    for service in services_skipped:
        if service in existing_summary:
            summary_rows.append(existing_summary[service])

    if services_todo:
        if large_scale_mode:
            # Process in batches so each batch is independent and resumable
            batch_size = SCAN_BATCH_SIZE
            batches = [
                services_todo[i:i + batch_size]
                for i in range(0, len(services_todo), batch_size)
            ]
            total_batches = len(batches)
            print(f"\nProcessing {len(services_todo)} services in {total_batches} batches of {batch_size} ...", flush=True)

            for batch_idx, batch in enumerate(batches, 1):
                print(f"\n--- Batch {batch_idx}/{total_batches}: {batch} ---", flush=True)
                try:
                    rows = process_service_batch(batch, output_dir, grid)
                    summary_rows.extend(rows)
                    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
                    done_count = len(services_skipped) + len(summary_rows) - len(services_skipped)
                    print(f"  Batch {batch_idx}/{total_batches} done. "
                          f"Total written: {len([r for r in summary_rows if r not in [existing_summary.get(s,{}) for s in services_skipped]])}",
                          flush=True)
                except Exception as e:
                    import traceback
                    print(f"  Batch {batch_idx} FAILED: {e}", flush=True)
                    traceback.print_exc()
        else:
            # Default mode: single scan.
            print(f"\nLoading source data for {len(services_todo)} services ...", flush=True)
            rows = process_service_batch(services_todo, output_dir, grid)
            summary_rows.extend(rows)
    else:
        print("  All services already processed.", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    total_written = len([s for s in services_to_process if (output_dir / f"{s}.parquet").exists()])
    print(f"\n  Quality summary: {summary_path}")
    print(f"  Total parquets present: {total_written}/{len(services_to_process)}")
    print(f"  Services written this run: {len(services_todo)}")
    print(f"  Services skipped (already done): {len(services_skipped)}")
    if removed:
        print(f"  Old services removed: {removed}")

    if write_report:
        _write_report(summary, removed)


def _write_report(summary: pd.DataFrame, removed: list[str]) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Service Selection — Stratified Report (Phase 4 update)",
        "",
        f"Generated: {now}",
        "",
        "## Motivation",
        "",
        "The original Phase 3 recommended subset selected services using a combined",
        "rank of (cpu_mean + cpu_variance + burstiness). This produced 20 services",
        "all with burstiness > 8 — biasing every experiment toward the highly-bursty",
        "regime and preventing fair evaluation of ARIMA and the conformal margin",
        "across the full volatility spectrum.",
        "",
        "The updated set uses **stratified sampling**: 5 volatility groups × 4 services.",
        "This enables stratified reporting in the paper and a fair predictor comparison.",
        "",
        "## Stratification Design",
        "",
        "| Group | Label | Burstiness range | Expected winner | Services |",
        "|---|---|---|---|---|",
        "| G1 | Stable | 1.2 – 1.6 | ARIMA | MS_31285, MS_6298, MS_63525, MS_8458 |",
        "| G2 | Moderate | 1.5 – 2.4 | ARIMA / XGBoost | MS_70053, MS_29860, MS_42222, MS_66711 |",
        "| G3 | High variance | 2.5 – 3.5 | XGBoost | MS_491, MS_48534, MS_19988, MS_21035 |",
        "| G4 | Bursty | 8 – 26 | LSTM | MS_2024, MS_49699, MS_14526, MS_12652 |",
        "| G5 | Highly bursty | 28 – 50 | LSTM / stress | MS_51028, MS_5201, MS_345, MS_25320 |",
        "",
        "## Service Profiles",
        "",
        "| Service | Group | Coverage | Missing | Output rows | Nulls |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['msname']}` | {row['group']} "
            f"| {row['coverage']:.4f} | {row['missing_rows']} "
            f"| {row['output_rows']} | {row['remaining_nulls']} |"
        )

    lines += [""]
    if removed:
        lines += [
            "## Removed Services",
            "",
            "The following services from the original Phase 3 set were removed",
            "because they do not contribute to stratification coverage:",
            "",
        ]
        for svc in removed:
            lines.append(f"- `{svc}`")
        lines += [""]

    lines += [
        "## Notes",
        "",
        "- `MS_50707` (original set, 256 missing rows) removed — unreliable coverage.",
        "- Groups 4–5 reuse 8 services from the original set (already processed).",
        "- Groups 1–3 are new additions (12 services, high-coverage in Phase 3 scan).",
        "- The same split_definition.json timestamps apply to all services.",
        "",
    ]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
