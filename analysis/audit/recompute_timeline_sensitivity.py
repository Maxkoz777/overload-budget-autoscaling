#!/usr/bin/env python3
"""Audit the forecasting/calibration timeline and warm-start sensitivity.

The historical experiment code uses all of days 0--9 as pre-test fitting
history and seeds every rolling score buffer with persistence residuals.  This
script makes that implementation choice explicit and measures whether the
headline B6 results change after the first W=240 held-out intervals, once the
buffer contains only residuals produced by the evaluated forecaster.

Run from any directory with::

    python3 audit/recompute_timeline_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
EXP = RESEARCH / "experiments"
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from exp_core import simulate_policy  # noqa: E402


OUT = PAPER / "audit" / "timeline_results"
SPLIT_PATH = EXP / "data" / "splits" / "split_definition.json"
FOCUSED_DATA = EXP / "data" / "service_timeseries"
LARGE_DATA = EXP / "data" / "service_timeseries_200"
FOCUSED_RESULTS = EXP / "results"
LARGE_FORECASTS = EXP / "results" / "large_scale" / "forecasts"
W = 240
DELTA = 0.05
MODELS = ("persistence", "xgb", "lstm", "arima")
FOCUSED_FORECASTS = {
    model: pd.read_parquet(FOCUSED_RESULTS / f"{model}_forecasts.parquet")
    for model in MODELS
    if model != "persistence"
}


def split_frames(path: Path, split: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(path).sort_values("timestamp")
    specs = {item["name"]: item for item in split["splits"]}

    def take(name: str) -> pd.DataFrame:
        spec = specs[name]
        return frame[
            frame.timestamp.ge(spec["timestamp_start"])
            & frame.timestamp.lt(spec["timestamp_end_exclusive"])
        ].copy()

    return take("train"), take("calibration"), take("test")


def aligned_forecast(tier: str, model: str, service: str, test: pd.DataFrame) -> np.ndarray | None:
    if model == "persistence":
        return None
    if tier == "focused20":
        frame = FOCUSED_FORECASTS[model]
        frame = frame[frame.msname.eq(service)].sort_values("timestamp")
    else:
        frame = pd.read_parquet(LARGE_FORECASTS / model / f"{service}.parquet").sort_values("timestamp")
    expected = test.timestamp.to_numpy(np.int64)
    actual = frame.timestamp.to_numpy(np.int64)
    if len(frame) != len(test) or not np.array_equal(actual, expected):
        raise AssertionError(f"unaligned forecast: {tier}/{model}/{service}")
    values = frame.forecast.to_numpy(float)
    if not np.isfinite(values).all():
        raise AssertionError(f"non-finite forecast: {tier}/{model}/{service}")
    return values


def cost(demand: np.ndarray, capacity: np.ndarray, replicas: np.ndarray) -> tuple[float, float]:
    def total(cap: np.ndarray) -> float:
        overload = demand > cap
        actions = np.abs(np.diff(cap, prepend=cap[0]))
        return float(cap.sum() + 0.05 * actions.sum() + 10.0 * overload.sum())

    policy = total(capacity.astype(float))
    observed = total(np.maximum(np.ceil(replicas), 1).astype(float))
    return policy, policy / observed


def one_row(
    tier: str,
    model: str,
    service: str,
    history: pd.DataFrame,
    test: pd.DataFrame,
    replay: dict,
    start: int,
) -> dict[str, object]:
    demand = test.cpu_sum.to_numpy(float)[start:]
    capacity = replay["capacities"][start:]
    replicas = test.replica_count.to_numpy(float)[start:]
    total, relative = cost(demand, capacity, replicas)
    overload = demand > capacity
    return {
        "tier": tier,
        "model": model,
        "forecast_protocol": (
            "one_step_causal"
            if model in {"persistence", "xgb", "lstm"}
            else "legacy_daily_block_stress"
        ),
        "service_id": service,
        "evaluation": "full_test" if start == 0 else "after_W_test_intervals",
        "excluded_intervals": start,
        "n_intervals": len(demand),
        "overload_fraction": float(overload.mean()),
        "within_delta": bool(overload.mean() <= DELTA),
        "c_total": total,
        "relative_cost_vs_observed": relative,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_PATH.read_text())
    rows: list[dict[str, object]] = []

    tiers = (("all200", LARGE_DATA),) if VERIFIED_MODE else (
        ("focused20", FOCUSED_DATA),
        ("all200", LARGE_DATA),
    )
    for tier, data_dir in tiers:
        for path in sorted(data_dir.glob("MS_*.parquet")):
            service = path.stem
            train, calibration, test = split_frames(path, split)
            history = pd.concat([train, calibration], ignore_index=True)
            if len(train) != 11_520 or len(calibration) != 2_880 or len(test) != 4_320:
                raise AssertionError(f"unexpected split lengths for {service}")
            for model in (("persistence",) if VERIFIED_MODE else MODELS):
                forecast = aligned_forecast(tier, model, service, test)
                replay = simulate_policy(
                    history,
                    test,
                    forecast_array=forecast,
                    delta=DELTA,
                    W=W,
                    use_margin=True,
                    use_guardrail=True,
                )
                rows.append(one_row(tier, model, service, history, test, replay, 0))
                rows.append(one_row(tier, model, service, history, test, replay, W))

    per_service = pd.DataFrame(rows)
    summary = (
        per_service.groupby(
            ["tier", "model", "forecast_protocol", "evaluation", "excluded_intervals"],
            as_index=False,
        )
        .agg(
            services=("service_id", "nunique"),
            median_relative_cost=("relative_cost_vs_observed", "median"),
            median_overload_fraction=("overload_fraction", "median"),
            service_compliance=("within_delta", "mean"),
        )
    )
    roles = pd.DataFrame(
        [
            {"period": "days_0_7", "rows": 11_520, "implemented_roles": "forecaster model/order/hyperparameter selection; included in final fitting; persistence-residual history"},
            {"period": "days_8_9", "rows": 2_880, "implemented_roles": "forecaster model/order/hyperparameter selection and final fitting; reactive/window tuning; persistence-residual warm start"},
            {"period": "days_10_12", "rows": 4_320, "implemented_roles": "held-out policy replay only"},
        ]
    )
    methodology = {
        "input_series": "service_timeseries_200_verified" if VERIFIED_MODE else "historical_series",
        "forecaster_scope": "persistence_only" if VERIFIED_MODE else "historical_all_four",
        "delta": DELTA,
        "window": W,
        "test_intervals": 4_320,
        "post_warm_start_intervals": 4_320 - W,
        "pre_test_description": "days 0-9 are pre-test fitting/tuning history; days 8-9 are not an independent conformal split for the learned forecasters",
        "historical_warm_start": "all forecasters receive the last W positive one-step persistence residuals from pre-test history",
        "sensitivity": "simulate the complete causal replay, then score only held-out intervals W..T-1; the policy state is not reset",
        "model_specific_pre_test_scores": "unavailable for XGBoost/LSTM in the archived artefacts; in-sample residuals were deliberately not substituted",
        "arima_status": "legacy saved ARIMA forecasts use daily blocks and are labelled stress-only pending the separate causal replay",
        "initial_conditions": {
            "minimum_capacity": 1,
            "guardrail": "disabled until h=2 held-out outcomes have been observed",
            "actuation_cost": "the first held-out capacity is the cost-accounting reference; no unobserved pre-test transition is charged",
        },
    }

    per_service.to_csv(OUT / "warmstart_sensitivity_per_service.csv", index=False)
    summary.to_csv(OUT / "warmstart_sensitivity_summary.csv", index=False)
    roles.to_csv(OUT / "timeline_role_audit.csv", index=False)
    (OUT / "timeline_methodology.json").write_text(json.dumps(methodology, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"\nWrote {len(per_service)} service/evaluation rows to {OUT}")


if __name__ == "__main__":
    VERIFIED_MODE = "--verified" in sys.argv
    if VERIFIED_MODE:
        OUT = PAPER / "audit" / "verified_results" / "timeline"
        LARGE_DATA = EXP / "data" / "service_timeseries_200_verified"
    main()
