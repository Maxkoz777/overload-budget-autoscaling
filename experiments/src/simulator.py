from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metrics import CostConfig, evaluate_capacity
from policies import (
    observed_capacity,
    oracle_capacity,
    predictive_moving_average_capacity,
    predictive_persistence_capacity,
    reactive_threshold_capacity,
    static_quantile_capacity,
)


TIME_SERIES_DIR = Path("experiments/data/service_timeseries")
SPLIT_DEFINITION_PATH = Path("experiments/data/splits/split_definition.json")


def load_split_definition(path: Path = SPLIT_DEFINITION_PATH) -> dict:
    return json.loads(path.read_text())


def split_frame(df: pd.DataFrame, split: dict) -> pd.DataFrame:
    return df[
        (df["timestamp"] >= split["timestamp_start"])
        & (df["timestamp"] < split["timestamp_end_exclusive"])
    ].copy()


def run_service_replay(
    service_path: Path,
    split_definition: dict,
    mu: float = 1.0,
    cost_config: CostConfig | None = None,
) -> list[dict]:
    if cost_config is None:
        cost_config = CostConfig()

    df = pd.read_parquet(service_path).sort_values("timestamp")
    service = str(df["msname"].iloc[0])
    split_map = {split["name"]: split for split in split_definition["splits"]}
    train = split_frame(df, split_map["train"])
    calibration = split_frame(df, split_map["calibration"])
    test = split_frame(df, split_map["test"])
    history = pd.concat([train, calibration], ignore_index=True)

    policy_outputs = {
        "observed_capacity": observed_capacity(test),
        "oracle_demand_capacity": oracle_capacity(test, mu=mu),
        "static_p95_train_calibration": static_quantile_capacity(
            history,
            test,
            mu=mu,
            quantile=0.95,
        ),
        "reactive_threshold_u70_cooldown3": reactive_threshold_capacity(
            history,
            test,
            mu=mu,
            target_utilization=0.70,
            cooldown=3,
        ),
        "predictive_persistence_no_margin": predictive_persistence_capacity(
            history,
            test,
            mu=mu,
        ),
        "predictive_ma60_no_margin": predictive_moving_average_capacity(
            history,
            test,
            mu=mu,
            window=60,
        ),
    }

    demand = test["cpu_sum"].to_numpy(dtype=float)
    results = []
    for policy_name, capacity in policy_outputs.items():
        results.append(
            evaluate_capacity(
                service=service,
                policy=policy_name,
                timestamps=test["timestamp"],
                demand=demand,
                capacity=capacity,
                mu=mu,
                cost_config=cost_config,
            )
        )
    return results


def run_all_services(
    time_series_dir: Path = TIME_SERIES_DIR,
    split_definition_path: Path = SPLIT_DEFINITION_PATH,
    mu: float = 1.0,
    cost_config: CostConfig | None = None,
) -> pd.DataFrame:
    split_definition = load_split_definition(split_definition_path)
    rows = []
    for path in sorted(time_series_dir.glob("MS_*.parquet")):
        rows.extend(
            run_service_replay(
                service_path=path,
                split_definition=split_definition,
                mu=mu,
                cost_config=cost_config,
            )
        )
    return pd.DataFrame(rows)
