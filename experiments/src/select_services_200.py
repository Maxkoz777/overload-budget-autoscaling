"""
select_services_200.py
======================
Select 200 services for the large-scale forecasting experiment.

Steps:
  A) Stream all 312 partitions of processed/final/joined_service_features/
     accumulating per-service stats (n_intervals, cpu_sum_total, cpu_sumsq,
     cpu_max, n_zeros, ts_min, ts_max). Checkpoint every 30 partitions.
  B) Compute derived stats: coverage, mean_demand, std_demand, cv, burstiness,
     zero_rate.
  C) Eligibility filters with logging.
  D) Stratify by burstiness into 5 quantile groups, force-include 20 focused
     services, fill remaining slots (40 per stratum) with random sample seed=42.
  E) Write outputs.

Outputs:
    experiments/data/splits/selected_services_200.csv
    experiments/data/splits/service_eligibility_full.csv
    experiments/data/splits/eligibility_checkpoint.json
    experiments/reports/selected_services_200_summary.md
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

BASE_DIR         = Path("processed/final/joined_service_features")
SPLITS_DIR       = Path("experiments/data/splits")
REPORTS_DIR      = Path("experiments/reports")
CHECKPOINT_PATH  = SPLITS_DIR / "eligibility_checkpoint.json"
SELECTION_PATH   = SPLITS_DIR / "selected_services_200.csv"
ELIGIBILITY_PATH = SPLITS_DIR / "service_eligibility_full.csv"
SUMMARY_PATH     = REPORTS_DIR / "selected_services_200_summary.md"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TIMESTAMPS = 18720
COVERAGE_THRESHOLD  = 0.95
ZERO_RATE_THRESHOLD = 0.80
MIN_INTERVALS       = 14400
TARGET_N            = 200
STRATA_N            = 5
PER_STRATUM         = TARGET_N // STRATA_N
CHECKPOINT_EVERY    = 30
RANDOM_SEED         = 42

FOCUSED_20 = [
    "MS_31285", "MS_6298",  "MS_63525", "MS_8458",  "MS_70053",
    "MS_29860", "MS_42222", "MS_66711", "MS_491",   "MS_48534",
    "MS_19988", "MS_21035", "MS_2024",  "MS_49699", "MS_14526",
    "MS_12652", "MS_51028", "MS_5201",  "MS_345",   "MS_25320",
]


def stream_partitions() -> dict[str, dict]:
    """Stream all partitions; resume from checkpoint if available."""
    acc: dict[str, dict] = {}
    done_partitions: set[str] = set()

    if CHECKPOINT_PATH.exists():
        try:
            ckpt = json.loads(CHECKPOINT_PATH.read_text())
            acc = ckpt.get("acc", {})
            done_partitions = set(ckpt.get("done_partitions", []))
            print(f"Resumed checkpoint: {len(done_partitions)} partitions already done, "
                  f"{len(acc)} services seen.", flush=True)
        except Exception as e:
            print(f"Checkpoint load failed ({e}), starting fresh.", flush=True)
            acc = {}
            done_partitions = set()

    def ensure(name: str) -> dict:
        if name not in acc:
            acc[name] = {
                "n_intervals": 0,
                "cpu_sum_total": 0.0,
                "cpu_sumsq": 0.0,
                "cpu_max": 0.0,
                "n_zeros": 0,
                "ts_min": None,
                "ts_max": None,
            }
        return acc[name]

    partition_dirs = sorted([
        p for p in BASE_DIR.iterdir()
        if p.is_dir() and p.name.startswith("hour_id=")
    ])
    total_partitions = len(partition_dirs)
    print(f"Total partitions: {total_partitions}", flush=True)

    batch_count = 0
    for part_dir in partition_dirs:
        part_key = part_dir.name
        if part_key in done_partitions:
            continue

        try:
            part_ds = ds.dataset(str(part_dir), format="parquet")
            scanner = part_ds.scanner(
                columns=["msname", "timestamp", "cpu_sum"],
                batch_size=1_048_576,
                use_threads=True,
            )
            for batch in scanner.to_batches():
                if batch.num_rows == 0:
                    continue
                tbl = pa.Table.from_batches([batch])

                cpu_col  = tbl["cpu_sum"]
                cpu_sq   = pc.multiply(cpu_col, cpu_col)
                # cpu_sum <= 1e-6 counts as a zero interval
                is_zero  = pc.cast(pc.less_equal(cpu_col, pa.scalar(1e-6)), pa.int64())
                tbl = (tbl
                       .append_column("cpu_sumsq", cpu_sq)
                       .append_column("is_zero", is_zero))

                grouped = tbl.group_by("msname").aggregate([
                    ("timestamp", "count"),
                    ("timestamp", "min"),
                    ("timestamp", "max"),
                    ("cpu_sum",   "sum"),
                    ("cpu_sumsq", "sum"),
                    ("cpu_sum",   "max"),
                    ("is_zero",   "sum"),
                ]).to_pydict()

                names = grouped["msname"]
                for i, name in enumerate(names):
                    item = ensure(name)
                    item["n_intervals"]   += int(grouped["timestamp_count"][i] or 0)
                    item["cpu_sum_total"] += float(grouped["cpu_sum_sum"][i] or 0.0)
                    item["cpu_sumsq"]     += float(grouped["cpu_sumsq_sum"][i] or 0.0)
                    item["n_zeros"]       += int(grouped["is_zero_sum"][i] or 0)

                    raw_max = grouped["cpu_sum_max"][i]
                    if raw_max is not None:
                        item["cpu_max"] = max(item["cpu_max"], float(raw_max))

                    ts_min = grouped["timestamp_min"][i]
                    ts_max = grouped["timestamp_max"][i]
                    if ts_min is not None:
                        item["ts_min"] = (min(item["ts_min"], ts_min)
                                          if item["ts_min"] is not None else ts_min)
                    if ts_max is not None:
                        item["ts_max"] = (max(item["ts_max"], ts_max)
                                          if item["ts_max"] is not None else ts_max)

        except Exception as e:
            print(f"  WARNING: partition {part_key} failed: {e}", flush=True)

        done_partitions.add(part_key)
        batch_count += 1

        if batch_count % CHECKPOINT_EVERY == 0 or batch_count == total_partitions - len(done_partitions) + batch_count:
            _save_checkpoint(acc, list(done_partitions))
            print(f"  Checkpoint at {len(done_partitions)}/{total_partitions} partitions, "
                  f"{len(acc)} services seen.", flush=True)

    _save_checkpoint(acc, list(done_partitions))
    print(f"\nStreaming complete: {len(done_partitions)} partitions, {len(acc)} services.", flush=True)
    return acc


def _save_checkpoint(acc: dict, done_partitions: list) -> None:
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"acc": acc, "done_partitions": done_partitions}, f)
    tmp.replace(CHECKPOINT_PATH)


def compute_derived(acc: dict[str, dict]) -> list[dict]:
    records = []
    for name, item in acc.items():
        n = item["n_intervals"]
        if n == 0:
            continue
        mean  = item["cpu_sum_total"] / n
        var   = max(0.0, item["cpu_sumsq"] / n - mean * mean)
        std   = math.sqrt(var)
        cv    = (std / mean) if mean > 0 else float("inf")
        burst = (item["cpu_max"] / mean) if mean > 0 else 0.0
        zrate = item["n_zeros"] / n

        records.append({
            "service_id":    name,
            "n_intervals":   n,
            "cpu_sum_total": item["cpu_sum_total"],
            "cpu_sumsq":     item["cpu_sumsq"],
            "cpu_max":       item["cpu_max"],
            "n_zeros":       item["n_zeros"],
            "ts_min":        item["ts_min"],
            "ts_max":        item["ts_max"],
            "coverage":      n / EXPECTED_TIMESTAMPS,
            "mean_demand":   mean,
            "std_demand":    std,
            "cv":            cv,
            "burstiness":    burst,
            "zero_rate":     zrate,
        })
    return records


def apply_filters(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    counts = {"total": len(records)}

    after_cov = [r for r in records if r["coverage"] >= COVERAGE_THRESHOLD]
    counts["after_coverage_095"] = len(after_cov)
    print(f"  After coverage >= 0.95:       {len(after_cov):>6,}", flush=True)

    def is_valid(r):
        if r["mean_demand"] <= 0:
            return False
        for key in ("mean_demand", "std_demand", "cv", "burstiness", "zero_rate"):
            v = r[key]
            if math.isnan(v) or math.isinf(v):
                return False
        return True

    after_valid = [r for r in after_cov if is_valid(r)]
    counts["after_valid_stats"] = len(after_valid)
    print(f"  After valid stats (mean>0, no NaN/Inf): {len(after_valid):>6,}", flush=True)

    after_zero = [r for r in after_valid if r["zero_rate"] <= ZERO_RATE_THRESHOLD]
    counts["after_zero_rate"] = len(after_zero)
    print(f"  After zero_rate <= 0.80:      {len(after_zero):>6,}", flush=True)

    after_n = [r for r in after_zero if r["n_intervals"] >= MIN_INTERVALS]
    counts["after_min_intervals"] = len(after_n)
    print(f"  After n_intervals >= 14400:   {len(after_n):>6,}", flush=True)

    return after_n, counts


def stratify_and_sample(eligible: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(eligible).sort_values("burstiness").reset_index(drop=True)

    labels = ["G1", "G2", "G3", "G4", "G5"]
    df["stratum"] = pd.qcut(df["burstiness"], q=5, labels=labels).astype(str)

    focused_set = set(FOCUSED_20)
    focused_df  = df[df["service_id"].isin(focused_set)].copy()

    missing_focused = [s for s in FOCUSED_20 if s not in df["service_id"].values]
    if missing_focused:
        print(f"  WARNING: {len(missing_focused)} focused services NOT in eligible list: {missing_focused}", flush=True)

    rng = random.Random(RANDOM_SEED)
    selected_rows = []

    for stratum in labels:
        stratum_df   = df[df["stratum"] == stratum]
        forced_in_stratum = stratum_df[stratum_df["service_id"].isin(focused_set)]
        forced_ids   = set(forced_in_stratum["service_id"].tolist())
        n_forced     = len(forced_ids)
        n_fill       = PER_STRATUM - n_forced

        for _, row in forced_in_stratum.iterrows():
            selected_rows.append(row)

        pool = stratum_df[~stratum_df["service_id"].isin(forced_ids)]
        if len(pool) < n_fill:
            print(f"  WARNING: stratum {stratum} has only {len(pool)} non-forced eligible services "
                  f"(need {n_fill}), taking all.", flush=True)
            n_fill = len(pool)

        sample_ids = rng.sample(pool["service_id"].tolist(), n_fill)
        sample_rows = pool[pool["service_id"].isin(sample_ids)]
        for _, row in sample_rows.iterrows():
            selected_rows.append(row)

        print(f"  Stratum {stratum}: {n_forced} forced + {n_fill} sampled = "
              f"{n_forced + n_fill} total  (pool size: {len(stratum_df)})", flush=True)

    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    result["is_focused_20"] = result["service_id"].isin(focused_set)
    result["peak_demand"]   = result["cpu_max"]

    output_cols = [
        "service_id", "stratum", "burstiness", "mean_demand", "peak_demand",
        "cv", "zero_rate", "n_intervals", "coverage", "is_focused_20",
    ]
    return result[output_cols]


def write_summary_report(selected: pd.DataFrame, filter_counts: dict,
                         full_eligible: list[dict]) -> None:
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Selected Services 200 — Summary Report",
        "",
        f"Generated: {now}",
        "",
        "## Filter Step Counts",
        "",
        "| Step | Services |",
        "|---|---:|",
    ]
    for step, count in filter_counts.items():
        lines.append(f"| {step} | {count:,} |")

    lines += [
        "",
        "## Per-Stratum Selection",
        "",
        "| Stratum | Count | Burstiness Min | Burstiness Median | Burstiness Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for stratum in ["G1", "G2", "G3", "G4", "G5"]:
        sub = selected[selected["stratum"] == stratum]["burstiness"]
        lines.append(
            f"| {stratum} | {len(sub)} | {sub.min():.4f} | {sub.median():.4f} | {sub.max():.4f} |"
        )

    focused_present = selected[selected["is_focused_20"] == True]["service_id"].tolist()
    missing         = [s for s in FOCUSED_20 if s not in focused_present]

    lines += [
        "",
        "## Focused-20 Confirmation",
        "",
        f"Present: {len(focused_present)}/20",
        "",
        "| Service | Stratum | Burstiness |",
        "|---|---|---:|",
    ]
    for _, row in selected[selected["is_focused_20"] == True].iterrows():
        lines.append(f"| {row['service_id']} | {row['stratum']} | {row['burstiness']:.4f} |")

    if missing:
        lines += ["", f"**MISSING from selection:** {missing}"]
    else:
        lines += ["", "All 20 focused services confirmed present."]

    lines += [""]

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Summary report: {SUMMARY_PATH}", flush=True)


def main() -> None:
    t0 = time.perf_counter()

    print("\n" + "="*72, flush=True)
    print("  select_services_200.py", flush=True)
    print("="*72 + "\n", flush=True)

    print("Phase A: Streaming partitions ...", flush=True)
    acc = stream_partitions()

    print("\nPhase B: Computing derived stats ...", flush=True)
    records = compute_derived(acc)
    print(f"  Total services with data: {len(records):,}", flush=True)

    print("\nPhase C: Applying eligibility filters ...", flush=True)
    print(f"  Total services: {len(records):,}", flush=True)
    eligible, filter_counts = apply_filters(records)
    filter_counts = {"total": len(records), **{k: v for k, v in filter_counts.items()}}

    elig_df = pd.DataFrame(eligible)
    elig_df.to_csv(ELIGIBILITY_PATH, index=False)
    print(f"\nEligibility CSV: {ELIGIBILITY_PATH}  ({len(elig_df):,} rows)", flush=True)

    print("\nPhase D: Stratifying and sampling ...", flush=True)
    selected = stratify_and_sample(eligible)
    print(f"\nSelected: {len(selected)} services", flush=True)

    present_focused = selected[selected["is_focused_20"] == True]["service_id"].tolist()
    print(f"Focused-20 present: {len(present_focused)}/20", flush=True)
    missing = [s for s in FOCUSED_20 if s not in present_focused]
    if missing:
        print(f"  MISSING: {missing}", flush=True)
    else:
        print("  All 20 focused services confirmed.", flush=True)

    selected.to_csv(SELECTION_PATH, index=False)
    print(f"\nSelection CSV: {SELECTION_PATH}  ({len(selected)} rows)", flush=True)

    write_summary_report(selected, filter_counts, eligible)

    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s", flush=True)
    print("="*72 + "\n", flush=True)


if __name__ == "__main__":
    main()
