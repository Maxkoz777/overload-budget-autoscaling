#!/usr/bin/env python3
"""Matched persistence margin/rank replay on upstream-verified service series.

The inverse-CDF arm differs from conformal *only* in the W versus W+1
order-statistic rank. The NumPy-linear and Gaussian arms are separate method
comparators. All four share forecasts, observed-cost denominator and guard.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm


PAPER = Path(__file__).resolve().parents[1]
EXP = PAPER.parent / "experiments"
DATA = EXP / "data" / "service_timeseries_200_verified"
SPLIT = EXP / "data" / "splits" / "split_definition.json"
SELECTION = EXP / "data" / "splits" / "selected_services_200.csv"
OUT = PAPER / "audit" / "verified_results" / "rank_ablation"
SAVED = PAPER / "audit" / "verified_results" / "comparative" / "margin_comparison_per_service.csv"
B9 = PAPER / "audit" / "verified_results" / "baselines" / "persistence_baselines_per_service.csv"
W = 240
DELTAS = (0.05, 0.01)
METHODS = ("conformal", "empirical_inverse_cdf", "numpy_linear_percentile", "gaussian")


def cost(capacity: np.ndarray, demand: np.ndarray) -> float:
    return float(capacity.sum() + 0.05 * np.abs(np.diff(capacity)).sum() + 10 * (demand > capacity).sum())


def guarded(nominal: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """Exact h=2, rho=.7, gamma=1 controller; soft includes hard overload."""
    capacity = nominal.copy()
    soft = np.zeros(len(demand), dtype=bool)
    for t in range(len(demand)):
        if t >= 2 and soft[t - 1] and soft[t - 2]:
            capacity[t] += 1
        soft[t] = demand[t] > 0.7 * capacity[t]
    return capacity


def service_rows(item: object, split: dict) -> list[dict]:
    frame = pd.read_parquet(DATA / f"{item.service_id}.parquet").sort_values("timestamp")
    test_spec = split["test"]
    history = frame[frame.timestamp < test_spec["timestamp_start"]]
    test = frame[
        (frame.timestamp >= test_spec["timestamp_start"])
        & (frame.timestamp < test_spec["timestamp_end_exclusive"])
    ]
    hy = history.cpu_sum.to_numpy(float)
    demand = test.cpu_sum.to_numpy(float)
    assert len(hy) == 14_400 and len(demand) == 4_320
    forecast = np.r_[hy[-1], demand[:-1]]
    residual = np.diff(np.r_[hy, demand])
    windows = np.lib.stride_tricks.sliding_window_view(residual, W)[len(hy) - 1 - W : len(hy) - 1 - W + len(demand)]
    assert np.array_equal(windows[0], np.diff(hy)[-W:])
    assert windows[1, -1] == demand[0] - hy[-1]
    ordered = np.sort(np.maximum(windows, 0), axis=1)
    observed = np.maximum(np.ceil(test.replica_count.to_numpy(float)), 1)
    denominator = cost(observed, demand)
    rows = []
    for delta in DELTAS:
        conformal_rank = min(W, math.ceil((W + 1) * (1 - delta)))
        empirical_rank = math.ceil(W * (1 - delta))
        position = (W - 1) * (1 - delta)
        lower, upper = math.floor(position), math.ceil(position)
        margins = {
            "conformal": ordered[:, conformal_rank - 1],
            "empirical_inverse_cdf": ordered[:, empirical_rank - 1],
            "numpy_linear_percentile": ordered[:, lower] + (position - lower) * (ordered[:, upper] - ordered[:, lower]),
            "gaussian": np.maximum(0, windows.mean(axis=1) + norm.ppf(1 - delta) * windows.std(axis=1, ddof=0)),
        }
        for method, margin in margins.items():
            nominal = np.maximum(np.ceil(forecast + margin), 1)
            for guard in (False, True):
                capacity = guarded(nominal, demand) if guard else nominal
                overload = float((demand > capacity).mean())
                rows.append({
                    "service_id": item.service_id,
                    "stratum": item.stratum,
                    "focused": bool(item.is_focused_20),
                    "delta": delta,
                    "method": method,
                    "guard": guard,
                    "overload_fraction": overload,
                    "relative_cost": cost(capacity, demand) / denominator,
                    "within_delta": overload <= delta,
                })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = {part["name"]: part for part in json.loads(SPLIT.read_text())["splits"]}
    selection = pd.read_csv(SELECTION)
    assert len(selection) == 200 and selection.service_id.is_unique
    rows = []
    for index, item in enumerate(selection.itertuples(index=False), 1):
        rows.extend(service_rows(item, split))
        if index % 25 == 0:
            print(f"verified rank replay {index}/200", flush=True)
    per = pd.DataFrame(rows)
    assert len(per) == 3_200 and not per.duplicated(["service_id", "delta", "method", "guard"]).any()

    # A second implementation must reproduce all saved conformal/Gaussian cells.
    saved = pd.read_csv(SAVED)
    saved = saved[saved.subset.eq("all200")]
    matched = per[per.method.isin(("conformal", "gaussian"))].merge(
        saved, left_on=["service_id", "delta", "method", "guard"],
        right_on=["service_id", "delta", "margin_family", "guard"], validate="one_to_one",
    )
    assert len(matched) == 1_600
    assert np.allclose(matched.overload_fraction_x, matched.overload_fraction_y, atol=1e-12, rtol=0)
    assert np.allclose(matched.relative_cost_x, matched.relative_cost_y, atol=1e-12, rtol=0)

    old_b9 = pd.read_csv(B9)
    old_b9 = old_b9[old_b9.policy.eq("B9_linear_percentile_guard") & old_b9.delta.isin(DELTAS)]
    linear = per[per.method.eq("numpy_linear_percentile") & per.guard].merge(
        old_b9, on=["service_id", "delta"], validate="one_to_one", suffixes=("_new", "_b9"),
    )
    assert len(linear) == 400
    assert np.allclose(linear.overload_fraction_new, linear.overload_fraction_b9, atol=1e-12, rtol=0)
    assert np.allclose(linear.relative_cost_new, linear.relative_cost_b9, atol=1e-12, rtol=0)

    per.to_csv(OUT / "rank_ablation_per_service.csv", index=False)
    panels = []
    for subset, part in (("all200", per), ("focused20", per[per.focused])):
        for (delta, method, guard), group in part.groupby(["delta", "method", "guard"]):
            panels.append({"subset": subset, "delta": delta, "method": method, "guard": bool(guard),
                           "services": len(group), "compliant_services": int(group.within_delta.sum()),
                           "compliance": float(group.within_delta.mean()),
                           "median_overload": float(group.overload_fraction.median()),
                           "median_relative_cost": float(group.relative_cost.median()),
                           "mean_relative_cost": float(group.relative_cost.mean())})
    summary = pd.DataFrame(panels)
    summary.to_csv(OUT / "rank_ablation_summary.csv", index=False)

    # Paired descriptives prevent identical medians on floor-bound services
    # from being mistaken for identical service-level economic outcomes.
    pairs = []
    for subset, part in (("all200", per), ("focused20", per[per.focused])):
        for delta in DELTAS:
            for guard in (False, True):
                cell = part[part.delta.eq(delta) & part.guard.eq(guard)]
                wide = cell.pivot(index="service_id", columns="method", values=["overload_fraction", "relative_cost", "within_delta"])
                for other in METHODS[1:]:
                    diff_ol = wide["overload_fraction"]["conformal"] - wide["overload_fraction"][other]
                    diff_cost = wide["relative_cost"]["conformal"] - wide["relative_cost"][other]
                    c_ok = wide["within_delta"]["conformal"].astype(bool)
                    o_ok = wide["within_delta"][other].astype(bool)
                    wins = int((c_ok & ~o_ok).sum())
                    losses = int((~c_ok & o_ok).sum())
                    pairs.append({"subset": subset, "delta": delta, "guard": guard, "comparator": other,
                                  "services": len(wide), "mean_overload_difference": float(diff_ol.mean()),
                                  "mean_relative_cost_difference": float(diff_cost.mean()),
                                  "conformal_only_compliant": wins,
                                  "comparator_only_compliant": losses,
                                  "exact_paired_compliance_p": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0})
    pd.DataFrame(pairs).to_csv(OUT / "rank_ablation_paired.csv", index=False)

    meta = {"source_series": str(DATA.relative_to(EXP)), "saved_historical_forecasts_used": False,
            "forecaster": "one-step persistence", "window": W, "deltas": list(DELTAS),
            "methods": {
                "conformal": "positive residual order statistic ceil((W+1)*(1-delta))",
                "empirical_inverse_cdf": "same positive residual order statistics; only rank changes to ceil(W*(1-delta))",
                "numpy_linear_percentile": "same positive residuals; linear interpolation at (W-1)*(1-delta)",
                "gaussian": "positive part of signed residual mean + z_(1-delta)*std_ddof0",
            }, "guard": "h=2, rho=0.70, gamma=1; identical rule for all arms",
            "matched_saved_conformal_gaussian_rows": len(matched), "matched_saved_b9_rows": len(linear),
            "rank_at_delta_005": {"conformal": 229, "empirical_inverse_cdf": 228, "numpy_linear_one_based_position": 228.05},
            "rank_at_delta_001": {"conformal": 239, "empirical_inverse_cdf": 238, "numpy_linear_one_based_position": 237.61},
            "scope": "retrospective fixed-suite rank ablation, not an IID or population coverage certificate"}
    (OUT / "methodology.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
