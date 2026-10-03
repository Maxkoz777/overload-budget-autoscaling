#!/usr/bin/env python3
"""Reproduce the Phase-8 focused-series imputation sensitivity.

The historical focused-20 preprocessing used an adjacent maximum for missing
resource values. This script compares it with the rebuilt past-only series,
repeats the calibration-only window selection on both versions, and reruns the
affected service's causal one-step ARIMA using the verified input. This script
does not retrain XGBoost or LSTM; the refit of those models on the past-only
history is ``audit/refit_focused_causal_fill.py``.

Run ``python3 audit/phase8_focused_fill_sensitivity.py``. The script is
read-only and prints a JSON summary; pipe/capture it if a record is wanted.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
EXPERIMENTS = PAPER.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS / "src"))
sys.path.insert(0, str(PAPER / "audit"))

from exp_core import simulate_policy  # noqa: E402
import recompute_arima_causal as arima  # noqa: E402


WINDOWS = (30, 60, 120, 240, 480, 960, 1440, 2880)
CAL_START = 691_200_000
TEST_START = 864_000_000
AFFECTED = "MS_25320"


def frames(directory: Path, service_ids: list[str]) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(directory / f"{name}.parquet").sort_values("timestamp")
        for name in service_ids
    }


def selected_window(data: dict[str, pd.DataFrame]) -> tuple[list[dict], int]:
    rows = []
    for window in WINDOWS:
        overloads = []
        for frame in data.values():
            history = frame[frame.timestamp < CAL_START]
            calibration = frame[
                frame.timestamp.ge(CAL_START) & frame.timestamp.lt(TEST_START)
            ]
            outcome = simulate_policy(
                history, calibration, delta=0.05, W=window,
                use_guardrail=False,
            )
            overloads.append(float(outcome["overload_fraction"]))
        median = float(np.median(overloads))
        rows.append({
            "W": window,
            "median_calibration_overload": median,
            "abs_deviation_from_delta": abs(median - 0.05),
        })
    selected = min(rows, key=lambda row: row["abs_deviation_from_delta"])["W"]
    return rows, selected


def main() -> None:
    selected = pd.read_csv(EXPERIMENTS / "data/splits/selected_services_200.csv")
    ids = selected.loc[selected.is_focused_20.astype(bool), "service_id"].tolist()
    assert len(ids) == 20
    old = frames(EXPERIMENTS / "data/service_timeseries", ids)
    verified = frames(EXPERIMENTS / "data/service_timeseries_200_verified", ids)

    demand_differences: dict[str, int] = {}
    for name in ids:
        a, b = old[name], verified[name]
        assert np.array_equal(a.timestamp.to_numpy(), b.timestamp.to_numpy())
        assert np.array_equal(a.was_missing.to_numpy(), b.was_missing.to_numpy())
        different = ~np.isclose(a.cpu_sum, b.cpu_sum, atol=0, rtol=0)
        assert not (different & ~a.was_missing).any(), name
        assert not (different & a.timestamp.ge(TEST_START)).any(), name
        if different.any():
            demand_differences[name] = int(different.sum())
            assert not (different & a.timestamp.lt(CAL_START)).any(), name
    assert demand_differences == {AFFECTED: 8}, demand_differences

    old_windows, old_choice = selected_window(old)
    verified_windows, verified_choice = selected_window(verified)
    archived = pd.read_csv(PAPER / "audit/calibration_window_selection.csv")
    assert np.array_equal(archived.W.to_numpy(), np.array(WINDOWS))
    assert np.allclose(
        archived.median_calibration_overload,
        [row["median_calibration_overload"] for row in old_windows],
        atol=1e-12, rtol=0,
    )
    assert old_choice == verified_choice == 240

    log = pd.read_csv(arima.LOG_PATH)
    row = log[log.msname.eq(AFFECTED)].iloc[0]
    split = json.loads(arima.SPLIT_PATH.read_text())
    arima.DATA = EXPERIMENTS / "data/service_timeseries_200_verified"
    corrected = arima.run_service((
        AFFECTED, int(row.best_k), ast.literal_eval(str(row.arima_order)), split,
    ))
    old_policy = pd.read_csv(arima.OUT / "arima_causal_policy_per_service.csv")
    old_policy = old_policy[
        old_policy.service_id.eq(AFFECTED)
        & old_policy.warm_start.eq("causal_arima_days_8_9")
        & old_policy.evaluation.eq("full_test")
    ]
    new_policy = pd.DataFrame(corrected["policy_rows"])
    new_policy = new_policy[
        new_policy.warm_start.eq("causal_arima_days_8_9")
        & new_policy.evaluation.eq("full_test")
    ]
    paired = old_policy.merge(
        new_policy,
        on=["warm_start", "evaluation", "delta"],
        suffixes=("_old", "_verified"),
        validate="one_to_one",
    )
    assert len(paired) == 3
    assert (paired.within_delta_old == paired.within_delta_verified).all()
    assert np.allclose(
        paired.overload_fraction_old.to_numpy(),
        paired.overload_fraction_verified.to_numpy(),
        atol=1e-12, rtol=0,
    )
    max_cost_change = float(np.max(np.abs(
        paired.relative_cost_vs_observed_old
        - paired.relative_cost_vs_observed_verified
    )))
    assert max_cost_change < 1e-4

    print(json.dumps({
        "upstream_reference": "available processed/final, not raw Alibaba archive",
        "focused_cpu_difference_rows": demand_differences,
        "difference_period": "days 8-9 calibration; no observed or held-out difference",
        "historical_window_choice": old_choice,
        "verified_window_choice": verified_choice,
        "historical_window_medians": old_windows,
        "verified_window_medians": verified_windows,
        "affected_arima_full_test": [
            {
                "delta": float(item.delta),
                "overload_fraction_old": float(item.overload_fraction_old),
                "overload_fraction_verified": float(item.overload_fraction_verified),
                "relative_cost_old": float(item.relative_cost_vs_observed_old),
                "relative_cost_verified": float(item.relative_cost_vs_observed_verified),
                "compliant": bool(item.within_delta_verified),
            }
            for item in paired.sort_values("delta").itertuples(index=False)
        ],
        "max_arima_relative_cost_change": max_cost_change,
        "xgboost_lstm_retrained_by_this_script": False,
        "xgboost_lstm_refit_record": "audit/verified_results/focused_causal_refit",
    }, indent=2))


if __name__ == "__main__":
    main()
