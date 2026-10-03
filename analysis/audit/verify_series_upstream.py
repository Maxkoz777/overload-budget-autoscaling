#!/usr/bin/env python3
"""Read-only, direct check of the fixed 200 series against upstream.

This deliberately does not trust the provenance CSV or re-create any
derived file. It checks every available observed upstream row in all 312
hourly partitions against the current verified series, including the missing
indicator and every saved resource/capacity column used by preprocessing.
The upstream is ``processed/final``, not the original Alibaba raw archive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
EXPERIMENTS = RESEARCH / "experiments"
UPSTREAM = RESEARCH / "processed/final/joined_service_features"
VERIFIED = EXPERIMENTS / "data/service_timeseries_200_verified"
SELECTION = EXPERIMENTS / "data/splits/selected_services_200.csv"
STEP_MS = 60_000
EXPECTED_ROWS = 18_720
FIELDS = (
    "replica_count", "node_count", "cpu_sum", "cpu_mean", "cpu_p95",
    "cpu_max", "memory_sum", "memory_mean", "memory_p95", "memory_max",
)


def main() -> None:
    selected = pd.read_csv(SELECTION)
    service_ids = selected.service_id.astype(str).tolist()
    assert len(service_ids) == len(set(service_ids)) == 200

    saved = {}
    for service_id in service_ids:
        frame = pd.read_parquet(
            VERIFIED / f"{service_id}.parquet",
            columns=["timestamp", "was_missing", *FIELDS],
        )
        assert len(frame) == EXPECTED_ROWS
        assert np.array_equal(
            frame.timestamp.to_numpy(), np.arange(EXPECTED_ROWS) * STEP_MS
        ), service_id
        saved[service_id] = frame

    partitions = sorted(UPSTREAM.glob("hour_id=*/part.parquet"))
    assert len(partitions) == 312, len(partitions)
    source_rows = 0
    checked_rows = 0
    for hour, path in enumerate(partitions, 1):
        frame = pq.read_table(
            path,
            columns=["timestamp", "msname", *FIELDS],
            filters=[("msname", "in", service_ids)],
        ).to_pandas()
        source_rows += len(frame)
        for service_id, observed in frame.groupby("msname", sort=False):
            idx = observed.timestamp.to_numpy(dtype=np.int64) // STEP_MS
            assert np.array_equal(idx * STEP_MS, observed.timestamp.to_numpy())
            assert len(np.unique(idx)) == len(idx), (path, service_id)
            current = saved[str(service_id)].iloc[idx]
            assert not current.was_missing.to_numpy().any(), (path, service_id)
            for field in FIELDS:
                actual = current[field].to_numpy()
                expected = observed[field].to_numpy()
                assert np.array_equal(actual, expected, equal_nan=True), (
                    path, service_id, field
                )
            checked_rows += len(observed)
        if hour % 24 == 0:
            print(f"verified {hour}/312 upstream hours", flush=True)

    assert source_rows == checked_rows == 3_742_687, (source_rows, checked_rows)
    focused_ids = selected.loc[selected.is_focused_20.astype(bool), "service_id"]
    assert len(focused_ids) == 20
    focused_demand_differences: dict[str, int] = {}
    focused_any_resource_differences: dict[str, int] = {}
    for service_id in focused_ids:
        focused = pd.read_parquet(
            EXPERIMENTS / "data/service_timeseries" / f"{service_id}.parquet",
            columns=["timestamp", "was_missing", *FIELDS],
        )
        verified = saved[str(service_id)]
        for field in ("timestamp", "was_missing"):
            assert np.array_equal(
                focused[field].to_numpy(), verified[field].to_numpy()
            ), (service_id, field)
        different_rows = np.zeros(EXPECTED_ROWS, dtype=bool)
        for field in FIELDS:
            different = ~np.isclose(
                focused[field].to_numpy(), verified[field].to_numpy(),
                atol=0, rtol=0, equal_nan=True,
            )
            assert not (different & ~focused.was_missing.to_numpy()).any(), (
                service_id, field, "upstream-observed row differs"
            )
            assert not different[14_400:].any(), (
                service_id, field, "held-out test row differs"
            )
            different_rows |= different
            if field == "cpu_sum" and different.any():
                focused_demand_differences[str(service_id)] = int(different.sum())
                assert not different[:11_520].any(), (
                    service_id, "training demand differs"
                )
        if different_rows.any():
            focused_any_resource_differences[str(service_id)] = int(
                different_rows.sum()
            )
    assert focused_demand_differences == {"MS_25320": 8}
    assert focused_any_resource_differences == {"MS_25320": 14}
    print(
        "PASS: 312 upstream partitions, 200 fixed services, "
        f"{checked_rows:,} observed rows; all 10 values and observed flags match; "
        "focused observed/test rows match; 8 focused calibration demand "
        "fills and 14 pre-test resource fills differ for MS_25320"
    )


if __name__ == "__main__":
    main()
