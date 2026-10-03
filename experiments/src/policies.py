from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting import moving_average_forecast, persistence_forecast


def observed_capacity(test: pd.DataFrame) -> np.ndarray:
    return test["replica_count"].to_numpy(dtype=int)


def oracle_capacity(test: pd.DataFrame, mu: float = 1.0) -> np.ndarray:
    demand = test["cpu_sum"].to_numpy(dtype=float)
    return np.maximum(np.ceil(demand / mu).astype(int), 1)


def static_quantile_capacity(
    history: pd.DataFrame,
    test: pd.DataFrame,
    mu: float = 1.0,
    quantile: float = 0.95,
) -> np.ndarray:
    demand_quantile = float(history["cpu_sum"].quantile(quantile))
    capacity = max(int(np.ceil(demand_quantile / mu)), 1)
    return np.full(len(test), capacity, dtype=int)


def reactive_threshold_capacity(
    history: pd.DataFrame,
    test: pd.DataFrame,
    mu: float = 1.0,
    target_utilization: float = 0.70,
    cooldown: int = 3,
) -> np.ndarray:
    if history.empty:
        raise ValueError("reactive_threshold_capacity requires non-empty history")

    previous_demand = float(history["cpu_sum"].iloc[-1])
    current_capacity = int(history["replica_count"].iloc[-1])
    last_scale_step = -cooldown
    capacities = []

    for step, demand in enumerate(test["cpu_sum"].to_numpy(dtype=float)):
        desired = max(int(np.ceil(previous_demand / (mu * target_utilization))), 1)
        if step - last_scale_step >= cooldown and desired != current_capacity:
            current_capacity = desired
            last_scale_step = step
        capacities.append(current_capacity)
        previous_demand = float(demand)

    return np.asarray(capacities, dtype=int)


def predictive_persistence_capacity(
    history: pd.DataFrame,
    test: pd.DataFrame,
    mu: float = 1.0,
) -> np.ndarray:
    forecast = persistence_forecast(history["cpu_sum"], test["cpu_sum"])
    return np.maximum(np.ceil(forecast / mu).astype(int), 1)


def predictive_moving_average_capacity(
    history: pd.DataFrame,
    test: pd.DataFrame,
    mu: float = 1.0,
    window: int = 60,
) -> np.ndarray:
    forecast = moving_average_forecast(history["cpu_sum"], test["cpu_sum"], window=window)
    return np.maximum(np.ceil(forecast / mu).astype(int), 1)
