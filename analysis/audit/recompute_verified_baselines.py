#!/usr/bin/env python3
"""Replay non-learned persistence baselines on the verified 200-service corpus.

Historical large-scale outputs are read only for an explicit reconciliation CSV;
they are never used as policy inputs or copied into verified results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
EXP = PAPER.parent / "experiments"
sys.path.insert(0, str(PAPER / "audit"))
sys.path.insert(0, str(EXP / "src"))

import recompute_margin_comparison as margin  # noqa: E402
from exp_core import simulate_policy  # noqa: E402

DATA = EXP / "data" / "service_timeseries_200_verified"
SELECTION = EXP / "data" / "splits" / "selected_services_200.csv"
SPLIT = EXP / "data" / "splits" / "split_definition.json"
OUT = PAPER / "audit" / "verified_results" / "baselines"
DELTAS = (0.10, 0.05, 0.03, 0.01)
W = 240


def result(capacity: np.ndarray, test: pd.DataFrame, denominator: float) -> dict[str, float | bool]:
    demand = test.cpu_sum.to_numpy(float)
    overload = demand > capacity
    return {
        "overload_fraction": float(overload.mean()),
        "relative_cost": margin.total_cost(capacity, demand) / denominator,
        "mean_capacity": float(capacity.mean()),
    }


def fixed_margin(history: pd.DataFrame, test: pd.DataFrame, delta: float) -> np.ndarray:
    hist = history.cpu_sum.to_numpy(float)
    q = float(np.quantile(np.maximum(np.diff(hist), 0), 1 - delta))
    forecast = np.r_[hist[-1], test.cpu_sum.to_numpy(float)[:-1]]
    return np.maximum(np.ceil(forecast + q), 1).astype(int)


def percentile_margin(history: pd.DataFrame, test: pd.DataFrame, delta: float) -> np.ndarray:
    hist = history.cpu_sum.to_numpy(float)
    demand = test.cpu_sum.to_numpy(float)
    residuals = np.maximum(np.diff(hist), 0).tolist()
    prev = float(hist[-1])
    overload_history: list[bool] = []
    soft_history: list[bool] = []
    capacity: list[int] = []
    for actual in demand:
        window = np.asarray(residuals[-W:], dtype=float)
        # Equivalent to np.quantile(window, 1-delta, method="linear").
        position = (len(window) - 1) * (1 - delta)
        low, high = int(np.floor(position)), int(np.ceil(position))
        selected = np.partition(window, (low, high))
        q = float(selected[low] + (position - low) * (selected[high] - selected[low]))
        nominal = max(int(np.ceil(prev + q)), 1)
        trigger = len(overload_history) >= 2 and (
            all(overload_history[-2:]) or all(soft_history[-2:])
        )
        applied = nominal + int(trigger)
        capacity.append(applied)
        overload_history.append(bool(actual > applied))
        soft_history.append(bool(actual > 0.70 * applied))
        residuals.append(max(float(actual - prev), 0.0))
        prev = float(actual)
    return np.asarray(capacity, dtype=int)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(SELECTION)
    split = json.loads(SPLIT.read_text())
    margin.DATA_ROOT = DATA
    existing = pd.read_csv(PAPER / "audit" / "verified_results" / "comparative" / "margin_comparison_per_service.csv")
    existing = existing[existing.subset.eq("all200")]
    rows: list[dict] = []
    denominators: dict[str, float] = {}
    for index, item in enumerate(selection.itertuples(), 1):
        history, test = margin.load_service(item.service_id, split)
        denominator = margin.observed_cost(test)
        denominators[item.service_id] = denominator
        base = {"service_id": item.service_id, "stratum": item.stratum, "focused": bool(item.is_focused_20)}
        demand = test.cpu_sum.to_numpy(float)
        previous = np.r_[history.cpu_sum.iloc[-1], demand[:-1]]
        pure = result(np.maximum(np.ceil(previous), 1).astype(int), test, denominator)
        guard = simulate_policy(history, test, delta=0.05, W=W, use_margin=False, use_guardrail=True)
        rows.append({**base, "policy": "B2_pure_predictive", "delta": 0.05, **pure})
        rows.append({**base, "policy": "B8_guard_only", "delta": 0.05,
                     "overload_fraction": guard["overload_fraction"],
                     "relative_cost": guard["c_total"] / denominator,
                     "mean_capacity": float(guard["capacities"].mean())})
        for delta in DELTAS:
            rows.append({**base, "policy": "B3_fixed_margin", "delta": delta,
                         **result(fixed_margin(history, test, delta), test, denominator)})
            rows.append({**base, "policy": "B9_linear_percentile_guard", "delta": delta,
                         **result(percentile_margin(history, test, delta), test, denominator)})
            if delta not in (0.05, 0.01):
                for name, has_guard in (("B6_conformal_guard", True), ("B7_conformal_only", False)):
                    replay = margin.simulate(history, test, delta=delta, margin_family="conformal", guard=has_guard)
                    rows.append({**base, "policy": name, "delta": delta,
                                 "overload_fraction": replay["overload_fraction"],
                                 "relative_cost": replay["total_cost"] / denominator,
                                 "mean_capacity": replay["mean_capacity"]})
        if index % 25 == 0:
            print(f"verified baselines {index}/200", flush=True)
    for item in existing.itertuples():
        name = f"B{'6' if item.guard else '7'}_conformal_{'guard' if item.guard else 'only'}" if item.margin_family == "conformal" else f"B4_gaussian_{'guard' if item.guard else 'only'}"
        rows.append({"service_id": item.service_id, "stratum": item.stratum,
                     "focused": bool(item.focused), "policy": name, "delta": float(item.delta),
                     "overload_fraction": float(item.overload_fraction),
                     "relative_cost": float(item.relative_cost), "mean_capacity": float(item.mean_capacity)})
    frame = pd.DataFrame(rows)
    frame["within_delta"] = frame.overload_fraction <= frame.delta
    if frame.duplicated(["service_id", "policy", "delta"]).any():
        raise AssertionError("duplicate verified policy rows")
    frame.to_csv(OUT / "persistence_baselines_per_service.csv", index=False)
    summary_rows = []
    for subset in ("all200", "focused20", "new180"):
        part = frame if subset == "all200" else frame[frame.focused.eq(subset == "focused20")]
        for (policy, delta), group in part.groupby(["policy", "delta"]):
            summary_rows.append({"subset": subset, "policy": policy, "delta": delta,
                                 "services": len(group), "median_overload": group.overload_fraction.median(),
                                 "mean_overload": group.overload_fraction.mean(),
                                 "median_relative_cost": group.relative_cost.median(),
                                 "mean_relative_cost": group.relative_cost.mean(),
                                 "service_compliance": group.within_delta.mean()})
    pd.DataFrame(summary_rows).to_csv(OUT / "persistence_baselines_summary.csv", index=False)
    old = pd.read_csv(EXP / "results" / "large_scale" / "analysis" / "ls_baselines_per_service_for_stats_200.csv")
    mapping = {"B2: pure predictive": "B2_pure_predictive", "B4: Gaussian margin": "B4_gaussian_only",
               "B6: conformal + guardrail": "B6_conformal_guard",
               "B7: conformal margin only": "B7_conformal_only",
               "B8: guardrail only": "B8_guard_only",
               "B9: empirical percentile + guardrail": "B9_linear_percentile_guard"}
    old = old[old.policy.isin(mapping)].copy()
    old["policy"] = old.policy.map(mapping)
    reconciliation = frame.merge(old[["service_id", "policy", "delta", "overload_fraction", "c_total"]],
                                 on=["service_id", "policy", "delta"], how="left", suffixes=("_verified", "_historical"))
    reconciliation["overload_difference"] = reconciliation.overload_fraction_verified - reconciliation.overload_fraction_historical
    reconciliation["verified_c_total"] = reconciliation.relative_cost * reconciliation.service_id.map(denominators)
    reconciliation["total_cost_difference"] = reconciliation.verified_c_total - reconciliation.c_total
    reconciliation.to_csv(OUT / "historical_reconciliation.csv", index=False)
    (OUT / "methodology.json").write_text(json.dumps({
        "input_series": str(DATA.relative_to(EXP)), "services": 200, "window": W,
        "historical_outputs_role": "read-only reconciliation, never simulation input",
        "delta_grid": DELTAS, "percentile": "NumPy linear quantile without W+1 rank correction",
        "cost": {"resource": 1.0, "churn": 0.05, "overload": 10.0},
    }, indent=2) + "\n")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
