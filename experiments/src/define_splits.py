from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


TIME_SERIES_DIR = Path("experiments/data/service_timeseries")
OUTPUT_DIR = Path("experiments/data/splits")
SPLIT_JSON_PATH = OUTPUT_DIR / "split_definition.json"
SPLIT_SUMMARY_PATH = OUTPUT_DIR / "split_summary.csv"

TIMESTAMP_STEP = 60_000
MINUTES_PER_DAY = 24 * 60
DAY_MS = MINUTES_PER_DAY * TIMESTAMP_STEP
TOTAL_DAYS = 13
EXPECTED_TOTAL_ROWS = TOTAL_DAYS * MINUTES_PER_DAY

SPLITS = [
    {
        "name": "train",
        "day_start": 0,
        "day_end_exclusive": 8,
        "purpose": "Forecaster training history",
    },
    {
        "name": "calibration",
        "day_start": 8,
        "day_end_exclusive": 10,
        "purpose": "Conformal residual calibration and tuning/warm-up",
    },
    {
        "name": "test",
        "day_start": 10,
        "day_end_exclusive": 13,
        "purpose": "Held-out autoscaling replay evaluation",
    },
]


def split_bounds(split: dict[str, object]) -> dict[str, int]:
    start = int(split["day_start"]) * DAY_MS
    end_exclusive = int(split["day_end_exclusive"]) * DAY_MS
    return {
        "timestamp_start": start,
        "timestamp_end_exclusive": end_exclusive,
        "timestamp_end_inclusive": end_exclusive - TIMESTAMP_STEP,
        "expected_rows": (
            int(split["day_end_exclusive"]) - int(split["day_start"])
        )
        * MINUTES_PER_DAY,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_defs = []
    for split in SPLITS:
        bounds = split_bounds(split)
        split_defs.append({**split, **bounds})

    files = sorted(TIME_SERIES_DIR.glob("MS_*.parquet"))
    if not files:
        raise RuntimeError(f"No service time-series files found in {TIME_SERIES_DIR}")

    summary_rows = []
    for path in files:
        df = pd.read_parquet(path, columns=["timestamp", "msname", "was_missing"])
        service = str(df["msname"].iloc[0])
        total_rows = len(df)
        timestamp_min = int(df["timestamp"].min())
        timestamp_max = int(df["timestamp"].max())
        non_step_intervals = int(df["timestamp"].diff().dropna().ne(TIMESTAMP_STEP).sum())

        for split in split_defs:
            mask = (
                (df["timestamp"] >= split["timestamp_start"])
                & (df["timestamp"] < split["timestamp_end_exclusive"])
            )
            part = df.loc[mask]
            summary_rows.append(
                {
                    "msname": service,
                    "split": split["name"],
                    "timestamp_start": split["timestamp_start"],
                    "timestamp_end_inclusive": split["timestamp_end_inclusive"],
                    "expected_rows": split["expected_rows"],
                    "observed_rows": len(part),
                    "was_missing_count": int(part["was_missing"].sum()),
                    "total_service_rows": total_rows,
                    "service_timestamp_min": timestamp_min,
                    "service_timestamp_max": timestamp_max,
                    "service_non_60000_intervals": non_step_intervals,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SPLIT_SUMMARY_PATH, index=False)

    metadata = {
        "timestamp_unit": "trace-relative milliseconds",
        "timestamp_step": TIMESTAMP_STEP,
        "minutes_per_day": MINUTES_PER_DAY,
        "total_days": TOTAL_DAYS,
        "expected_total_rows_per_service": EXPECTED_TOTAL_ROWS,
        "split_policy": "Chronological day-boundary split: train days 0-7, calibration days 8-9, test days 10-12.",
        "splits": split_defs,
        "service_count": len(files),
        "services": [p.stem for p in files],
        "summary_csv": str(SPLIT_SUMMARY_PATH),
    }
    SPLIT_JSON_PATH.write_text(json.dumps(metadata, indent=2) + "\n")

    print("service_files", len(files))
    print("expected_total_rows_per_service", EXPECTED_TOTAL_ROWS)
    print("timestamp_step", TIMESTAMP_STEP)
    print("\nSPLIT_DEFINITION")
    for split in split_defs:
        print(
            split["name"],
            "day_start",
            split["day_start"],
            "day_end_exclusive",
            split["day_end_exclusive"],
            "timestamp_start",
            split["timestamp_start"],
            "timestamp_end_inclusive",
            split["timestamp_end_inclusive"],
            "expected_rows",
            split["expected_rows"],
        )

    print("\nSPLIT_COUNTS_BY_SPLIT")
    grouped = summary.groupby("split", sort=False).agg(
        services=("msname", "nunique"),
        min_rows=("observed_rows", "min"),
        max_rows=("observed_rows", "max"),
        total_rows=("observed_rows", "sum"),
        total_was_missing=("was_missing_count", "sum"),
    )
    print(grouped.to_csv().strip())

    print("\nSERVICES_WITH_MISSING_BY_SPLIT")
    missing = summary[summary["was_missing_count"] > 0][
        ["msname", "split", "was_missing_count"]
    ]
    if missing.empty:
        print("none")
    else:
        print(missing.to_csv(index=False).strip())

    print("\noutputs")
    print(SPLIT_JSON_PATH)
    print(SPLIT_SUMMARY_PATH)


if __name__ == "__main__":
    main()
