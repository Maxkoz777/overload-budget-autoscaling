from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    c_res: float = 1.0
    c_act: float = 0.05
    c_vio: float = 10.0
    c_inf: float = 0.0
    c_train: float = 0.0


def consecutive_run_lengths(flags: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for flag in flags.astype(bool):
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def evaluate_capacity(
    service: str,
    policy: str,
    timestamps: pd.Series,
    demand: np.ndarray,
    capacity: np.ndarray,
    mu: float = 1.0,
    cost_config: CostConfig | None = None,
) -> dict[str, float | int | str]:
    if cost_config is None:
        cost_config = CostConfig()

    capacity = np.maximum(np.ceil(capacity).astype(int), 1)
    demand = demand.astype(float)
    overload = demand > (mu * capacity)
    actions = np.abs(np.diff(capacity, prepend=capacity[0]))
    runs = consecutive_run_lengths(overload)

    c_res = cost_config.c_res * float(capacity.sum())
    c_act = cost_config.c_act * float(actions.sum())
    c_vio = cost_config.c_vio * float(overload.sum())
    c_inf = cost_config.c_inf * float(len(demand))
    c_train = cost_config.c_train
    total_cost = c_res + c_act + c_vio + c_inf + c_train

    utilization = demand / np.maximum(mu * capacity, 1e-12)
    run_p99 = float(np.quantile(runs, 0.99)) if runs else 0.0

    return {
        "msname": service,
        "policy": policy,
        "n_intervals": int(len(demand)),
        "timestamp_start": int(timestamps.iloc[0]),
        "timestamp_end": int(timestamps.iloc[-1]),
        "mu": float(mu),
        "capacity_mean": float(capacity.mean()),
        "capacity_min": int(capacity.min()),
        "capacity_max": int(capacity.max()),
        "demand_mean": float(demand.mean()),
        "demand_max": float(demand.max()),
        "overload_count": int(overload.sum()),
        "overload_fraction": float(overload.mean()),
        "scaling_actions": int(np.count_nonzero(np.diff(capacity))),
        "scaling_churn": int(actions.sum()),
        "max_overload_run": int(max(runs) if runs else 0),
        "p99_overload_run": run_p99,
        "mean_utilization": float(utilization.mean()),
        "p95_utilization": float(np.quantile(utilization, 0.95)),
        "c_res": c_res,
        "c_act": c_act,
        "c_vio": c_vio,
        "c_inf": c_inf,
        "c_train": c_train,
        "c_total": total_cost,
    }
