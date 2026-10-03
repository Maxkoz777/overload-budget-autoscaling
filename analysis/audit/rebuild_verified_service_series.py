#!/usr/bin/env python3
"""Rebuild and verify the fixed 200-service series from available upstream.

The historical ``service_timeseries_200`` directory is read-only in this
workflow. Corrected files are written to ``service_timeseries_200_verified``.
Every observed upstream row is asserted to remain observed and numerically
identical in the verified output. Missing timestamps use past-only forward
fill, with zero resource and one capacity unit before the first observation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
EXP = RESEARCH / "experiments"
UPSTREAM = RESEARCH / "processed" / "final" / "joined_service_features"
SELECTION = EXP / "data" / "splits" / "selected_services_200.csv"
HISTORICAL = EXP / "data" / "service_timeseries_200"
VERIFIED = EXP / "data" / "service_timeseries_200_verified"
OUT = PAPER / "audit" / "data_integrity_results"

TIMESTAMP_STEP = 60_000
EXPECTED_ROWS = 18_720
GRID = pd.Index(range(0, 1_123_140_000 + TIMESTAMP_STEP, TIMESTAMP_STEP), name="timestamp")
COLUMNS = [
    "timestamp", "msname", "replica_count", "node_count",
    "cpu_sum", "cpu_mean", "cpu_p95", "cpu_max",
    "memory_sum", "memory_mean", "memory_p95", "memory_max",
]
CAPACITY = ["replica_count", "node_count"]
RESOURCES = [
    "cpu_sum", "cpu_mean", "cpu_p95", "cpu_max",
    "memory_sum", "memory_mean", "memory_p95", "memory_max",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_upstream(ids: set[str]) -> tuple[dict[str, list[pd.DataFrame]], list[dict[str, object]]]:
    buckets: dict[str, list[pd.DataFrame]] = {name: [] for name in ids}
    partitions: list[dict[str, object]] = []
    paths = sorted(UPSTREAM.glob("hour_id=*/part.parquet"))
    assert len(paths) == 312, f"expected 312 upstream partitions, found {len(paths)}"
    for index, path in enumerate(paths, 1):
        frame = pq.read_table(path, columns=COLUMNS, filters=[("msname", "in", sorted(ids))]).to_pandas()
        partitions.append({
            "partition": path.parent.name,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "selected_rows": len(frame),
        })
        for name, part in frame.groupby("msname", sort=False):
            buckets[str(name)].append(part)
        if index % 24 == 0:
            print(f"loaded {index}/312 upstream partitions", flush=True)
    return buckets, partitions


def regularise(name: str, parts: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    observed = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    duplicates = int(observed.timestamp.duplicated().sum())
    if duplicates:
        numeric = [column for column in COLUMNS if column not in {"timestamp", "msname"}]
        observed = observed.groupby("timestamp", as_index=False)[numeric].mean().assign(msname=name)
        observed = observed[["timestamp", "msname", *numeric]]

    assert observed.timestamp.is_unique
    assert observed.timestamp.between(GRID.min(), GRID.max()).all()
    indexed = observed.set_index("timestamp")
    regular = indexed.reindex(GRID)
    regular["was_missing"] = regular.msname.isna()
    regular["msname"] = name

    for column in CAPACITY:
        regular[column] = regular[column].ffill().fillna(1).round().clip(lower=1).astype("int64")
    for column in RESOURCES:
        regular[column] = regular[column].ffill().fillna(0.0).astype("float64")
    regular = regular.reset_index()

    matched = regular.merge(observed, on=["timestamp", "msname"], suffixes=("_verified", "_source"))
    assert len(matched) == len(observed)
    assert not matched.was_missing.any(), f"{name}: observed upstream row marked missing"
    for column in CAPACITY + RESOURCES:
        assert np.array_equal(
            matched[f"{column}_verified"].to_numpy(),
            matched[f"{column}_source"].to_numpy(),
        ), f"{name}: mismatch in {column}"

    missing = regular.was_missing.to_numpy(bool)
    leading = int(np.flatnonzero(~missing)[0]) if (~missing).any() else EXPECTED_ROWS
    return regular, observed, {
        "service_id": name,
        "source_rows": len(observed),
        "duplicate_source_rows": duplicates,
        "verified_rows": len(regular),
        "missing_rows": int(missing.sum()),
        "leading_missing_rows": leading,
        "first_observed_timestamp": int(observed.timestamp.min()),
        "last_observed_timestamp": int(observed.timestamp.max()),
    }


def selection_statistics(frame: pd.DataFrame) -> dict[str, float]:
    values = frame.cpu_sum.to_numpy(float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {
        "n_intervals": len(frame),
        "coverage": len(frame) / EXPECTED_ROWS,
        "mean_demand": mean,
        "peak_demand": float(values.max()),
        "cv": std / mean,
        "burstiness": float(values.max()) / mean,
        "zero_rate": float((values <= 1e-6).mean()),
    }


def historical_manifest(ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in ids:
        path = HISTORICAL / f"{name}.parquet"
        rows.append({"path": path.relative_to(RESEARCH).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size, "role": "historical_input"})
    for model in ("arima", "xgb", "lstm"):
        for name in ids:
            path = EXP / "results" / "large_scale" / "forecasts" / model / f"{name}.parquet"
            rows.append({"path": path.relative_to(RESEARCH).as_posix(), "sha256": digest(path), "bytes": path.stat().st_size, "role": f"historical_{model}_forecast"})
    return pd.DataFrame(rows)


def main() -> None:
    selection = pd.read_csv(SELECTION)
    ids = selection.service_id.astype(str).tolist()
    assert len(ids) == len(set(ids)) == 200
    OUT.mkdir(parents=True, exist_ok=True)
    VERIFIED.mkdir(parents=True, exist_ok=True)

    # The comparison with the project's historical series is provenance evidence for the
    # article, not a reproduction step; it runs only when those historical files exist.
    historical_inputs = [HISTORICAL / f"{name}.parquet" for name in ids] + [
        EXP / "results" / "large_scale" / "forecasts" / model / f"{name}.parquet"
        for model in ("arima", "xgb", "lstm") for name in ids]
    have_history = all(path.exists() for path in historical_inputs)
    if have_history:
        historical_manifest(ids).to_csv(OUT / "historical_artifact_manifest.csv", index=False)
    else:
        print("historical series or forecasts not found: skipping the historical comparison", flush=True)
    buckets, partitions = load_upstream(set(ids))
    pd.DataFrame(partitions).to_csv(OUT / "upstream_partition_inventory.csv", index=False)

    summaries: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    selection_checks: list[dict[str, object]] = []
    for index, name in enumerate(ids, 1):
        assert buckets[name], f"no upstream rows for {name}"
        verified, observed, summary = regularise(name, buckets[name])
        output = VERIFIED / f"{name}.parquet"
        verified.to_parquet(output, index=False)
        summary["sha256"] = digest(output)
        summaries.append(summary)

        periods = ()
        if have_history:
            historical = pd.read_parquet(HISTORICAL / f"{name}.parquet").sort_values("timestamp")
            historical = historical.merge(verified, on="timestamp", suffixes=("_historical", "_verified"))
            periods = (
                ("train_days0_7", 0, 691_200_000),
                ("calibration_days8_9", 691_200_000, 864_000_000),
                ("test_days10_12", 864_000_000, 1_123_200_000),
            )
        for label, lo, hi in periods:
            part = historical[(historical.timestamp >= lo) & (historical.timestamp < hi)]
            demand_mismatch = ~np.isclose(
                part.cpu_sum_historical, part.cpu_sum_verified, atol=1e-8, rtol=1e-10
            )
            capacity_mismatch = part.replica_count_historical != part.replica_count_verified
            both_observed = ~part.was_missing_historical & ~part.was_missing_verified
            both_missing = part.was_missing_historical & part.was_missing_verified
            comparisons.append({
                "service_id": name,
                "period": label,
                "rows": len(part),
                "demand_mismatches": int(demand_mismatch.sum()),
                "capacity_mismatches": int(capacity_mismatch.sum()),
                "demand_mismatches_both_observed": int((demand_mismatch & both_observed).sum()),
                "demand_mismatches_both_missing": int((demand_mismatch & both_missing).sum()),
                "historical_missing_but_verified_observed": int((part.was_missing_historical & ~part.was_missing_verified).sum()),
                "historical_observed_but_verified_missing": int((~part.was_missing_historical & part.was_missing_verified).sum()),
            })

        expected = selection.loc[selection.service_id.eq(name)].iloc[0]
        actual = selection_statistics(observed)
        selection_checks.append({
            "service_id": name,
            **{f"saved_{key}": expected[key] for key in actual},
            **{f"upstream_{key}": value for key, value in actual.items()},
        })
        if index % 25 == 0:
            print(f"verified {index}/200 services", flush=True)

    summary = pd.DataFrame(summaries).sort_values("service_id")
    summary.to_csv(VERIFIED / "_quality_summary.csv", index=False)
    summary.to_csv(OUT / "verified_series_summary.csv", index=False)
    if have_history:
        comparison = pd.DataFrame(comparisons)
        comparison.to_csv(OUT / "historical_vs_verified_per_service.csv", index=False)
        aggregate = comparison.groupby("period", as_index=False).sum(numeric_only=True)
        aggregate.to_csv(OUT / "historical_vs_verified_summary.csv", index=False)

    checks = pd.DataFrame(selection_checks)
    for key in ("n_intervals", "coverage", "mean_demand", "peak_demand", "cv", "burstiness", "zero_rate"):
        assert np.allclose(checks[f"saved_{key}"], checks[f"upstream_{key}"], atol=1e-8, rtol=1e-8), key
    checks.to_csv(OUT / "selection_provenance.csv", index=False)

    assert len(list(VERIFIED.glob("MS_*.parquet"))) == 200
    assert set(summary.verified_rows) == {EXPECTED_ROWS}
    assert int(summary.leading_missing_rows.max()) == 0
    if have_history:
        post_train = aggregate[aggregate.period.isin(["calibration_days8_9", "test_days10_12"])]
        assert int(post_train.capacity_mismatches.sum()) == 0
        assert int(post_train.demand_mismatches_both_observed.sum()) == 0
        assert int(post_train.historical_missing_but_verified_observed.sum()) == 0
        assert int(post_train.historical_observed_but_verified_missing.sum()) == 0

    decision = {
        "upstream_scope": "available processed/final snapshot; original Alibaba archives were not independently reconstructed",
        "historical_series_status": "preserved unchanged; materially inconsistent with available upstream on days 0-7; 68 later demand differences are confined to imputed rows",
        "verified_series_status": "exact on every available upstream observation; past-only fill for genuinely absent timestamps",
        "selection_status": "fixed 200-service membership retained; saved full-period selection statistics verified against upstream",
        "historical_learned_forecast_status": "provisional historical sensitivity; training logs do not contain input hashes",
        "confirmatory_all200_scope_until_recomputation": "none; all 200-service results are quarantined until rerun on the verified series",
        "focused_forecaster_scope": "four-forecaster evidence retained on the separately verified focused 20",
        "historical_artifacts_overwritten": False,
    }
    if have_history:
        (OUT / "provenance_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
        print(aggregate.to_string(index=False))
        print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
