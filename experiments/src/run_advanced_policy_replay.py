"""
run_advanced_policy_replay.py
==============================
Run the risk-aware conformal policy using ARIMA, XGBoost, and LSTM forecasts
alongside the simple baselines, producing a unified cost-risk comparison table.

Prerequisites:
    python3 experiments/src/train_arima.py    # produces arima_forecasts.parquet
    python3 experiments/src/train_xgb.py      # produces xgb_forecasts.parquet
    python3 experiments/src/train_lstm.py     # produces lstm_forecasts.parquet

Each forecast file must contain columns: msname, timestamp, forecast.

Usage:
    python3 experiments/src/run_advanced_policy_replay.py

Output:
    experiments/results/advanced_policy_replay_per_service.csv
    experiments/results/advanced_policy_replay_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import CostConfig, evaluate_capacity
from risk_policy import RiskPolicyConfig, simulate_risk_policy


TIME_SERIES_DIR = Path("experiments/data/service_timeseries")
SPLIT_DEFINITION_PATH = Path("experiments/data/splits/split_definition.json")
RESULTS_DIR = Path("experiments/results")
PER_SERVICE_PATH = RESULTS_DIR / "advanced_policy_replay_per_service.csv"
SUMMARY_PATH = RESULTS_DIR / "advanced_policy_replay_summary.csv"

FORECAST_FILES = {
    "arima": RESULTS_DIR / "arima_forecasts.parquet",
    "xgb":   RESULTS_DIR / "xgb_forecasts.parquet",
    "lstm":  RESULTS_DIR / "lstm_forecasts.parquet",
}

MU = 1.0
COST_CONFIG = CostConfig(c_res=1.0, c_act=0.05, c_vio=10.0)
DELTA_VALUES = [0.10, 0.05, 0.01]
WINDOW = 240
ALPHA = 1.0
RHO = 0.70
GUARDRAIL_H = 2
GUARDRAIL_GAMMA = 1


def split_frame(df: pd.DataFrame, split: dict) -> pd.DataFrame:
    return df[
        (df["timestamp"] >= split["timestamp_start"])
        & (df["timestamp"] < split["timestamp_end_exclusive"])
    ].copy()


def load_forecasts_for_service(service: str, model_name: str, forecast_df: pd.DataFrame) -> np.ndarray | None:
    """Return aligned forecast array for service test split, or None if unavailable."""
    sub = forecast_df[forecast_df["msname"] == service].sort_values("timestamp")
    if sub.empty:
        return None
    return sub["forecast"].to_numpy(dtype=float)


def conformal_margin_from_residuals(residuals: list[float], config: RiskPolicyConfig) -> float:
    """Compute conformal quantile from a list of positive residuals."""
    if not residuals or not config.use_margin:
        return 0.0
    window = residuals[-config.window:]
    w = len(window)
    level = min(w, int(np.ceil((w + 1) * (1.0 - config.delta))))
    sorted_r = np.sort(np.asarray(window, dtype=float))
    return float(config.alpha * sorted_r[level - 1])


def simulate_policy_with_external_forecast(
    history: pd.DataFrame,
    test: pd.DataFrame,
    external_forecasts: np.ndarray,
    config: RiskPolicyConfig,
) -> np.ndarray:
    """
    Like simulate_risk_policy but uses a pre-computed forecast array
    instead of the built-in persistence forecaster.

    The external_forecasts array must be aligned with the test split rows.
    Positive residuals are recomputed from history using the forecast's
    one-step persistence as a stand-in for the calibration phase,
    then updated causally during replay using the external forecast.
    """
    # Initialise residuals from history (using persistence as warm-up)
    hist_y = history["cpu_sum"].to_numpy(dtype=float)
    residuals = list(np.maximum(hist_y[1:] - hist_y[:-1], 0.0))

    overload_history: list[bool] = []
    soft_history: list[bool] = []
    capacities = []

    for i, (_, row) in enumerate(test.iterrows()):
        demand = float(row["cpu_sum"])
        forecast = max(float(external_forecasts[i]), 0.0)
        margin = conformal_margin_from_residuals(residuals, config)
        nominal_capacity = max(int(np.ceil((forecast + margin) / config.mu)), 1)

        trigger = False
        if config.use_guardrail and len(overload_history) >= config.guardrail_h:
            recent_overload = overload_history[-config.guardrail_h:]
            recent_soft = soft_history[-config.guardrail_h:]
            trigger = all(recent_overload) or all(recent_soft)

        correction = config.guardrail_gamma if trigger else 0
        capacity = nominal_capacity + correction

        overload = demand > config.mu * capacity
        soft = demand > config.rho * config.mu * capacity
        positive_residual = max(demand - forecast, 0.0)

        residuals.append(positive_residual)
        overload_history.append(bool(overload))
        soft_history.append(bool(soft))
        capacities.append(capacity)

    return np.asarray(capacities, dtype=int)


def run_service(
    path: Path,
    split_definition: dict,
    forecast_cache: dict[str, pd.DataFrame],
) -> list[dict]:
    df = pd.read_parquet(path).sort_values("timestamp")
    service = str(df["msname"].iloc[0])
    split_map = {s["name"]: s for s in split_definition["splits"]}

    train = split_frame(df, split_map["train"])
    calibration = split_frame(df, split_map["calibration"])
    test = split_frame(df, split_map["test"])
    history = pd.concat([train, calibration], ignore_index=True)

    demand = test["cpu_sum"].to_numpy(dtype=float)
    results = []

    for model_name, forecast_df in forecast_cache.items():
        if forecast_df is None:
            continue
        fc_array = load_forecasts_for_service(service, model_name, forecast_df)
        if fc_array is None or len(fc_array) != len(test):
            continue

        for delta in DELTA_VALUES:
            # Conformal margin only
            config_margin = RiskPolicyConfig(
                delta=delta, window=WINDOW, alpha=ALPHA, mu=MU,
                rho=RHO, guardrail_h=GUARDRAIL_H, guardrail_gamma=GUARDRAIL_GAMMA,
                use_margin=True, use_guardrail=False,
            )
            cap_margin = simulate_policy_with_external_forecast(
                history, test, fc_array, config_margin
            )
            results.append(evaluate_capacity(
                service=service,
                policy=f"{model_name}_conformal_margin_only_d{delta}",
                timestamps=test["timestamp"],
                demand=demand, capacity=cap_margin,
                mu=MU, cost_config=COST_CONFIG,
            ))

            # Full policy (margin + guardrail)
            config_full = RiskPolicyConfig(
                delta=delta, window=WINDOW, alpha=ALPHA, mu=MU,
                rho=RHO, guardrail_h=GUARDRAIL_H, guardrail_gamma=GUARDRAIL_GAMMA,
                use_margin=True, use_guardrail=True,
            )
            cap_full = simulate_policy_with_external_forecast(
                history, test, fc_array, config_full
            )
            results.append(evaluate_capacity(
                service=service,
                policy=f"{model_name}_conformal_guardrail_d{delta}",
                timestamps=test["timestamp"],
                demand=demand, capacity=cap_full,
                mu=MU, cost_config=COST_CONFIG,
            ))

    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grp = results.groupby("policy", sort=False)
    return grp.agg(
        services=("msname", "nunique"),
        median_relative_cost=("relative_cost_vs_observed", "median"),
        p10_relative_cost=("relative_cost_vs_observed", lambda s: s.quantile(0.10)),
        p90_relative_cost=("relative_cost_vs_observed", lambda s: s.quantile(0.90)),
        median_overload_fraction=("overload_fraction", "median"),
        p10_overload_fraction=("overload_fraction", lambda s: s.quantile(0.10)),
        p90_overload_fraction=("overload_fraction", lambda s: s.quantile(0.90)),
        median_scaling_churn=("scaling_churn", "median"),
        median_max_overload_run=("max_overload_run", "median"),
    ).reset_index()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_definition = json.loads(SPLIT_DEFINITION_PATH.read_text())

    forecast_cache: dict[str, pd.DataFrame | None] = {}
    for name, fpath in FORECAST_FILES.items():
        if fpath.exists():
            forecast_cache[name] = pd.read_parquet(fpath)
            print(f"  Loaded {name} forecasts: {len(forecast_cache[name])} rows")
        else:
            forecast_cache[name] = None
            print(f"  Missing {name} forecasts — skipped")

    available = [k for k, v in forecast_cache.items() if v is not None]
    if not available:
        print("No forecast files found. Run training scripts first.")
        return

    service_files = sorted(TIME_SERIES_DIR.glob("MS_*.parquet"))
    all_rows = []

    for path in service_files:
        service = path.stem
        rows = run_service(path, split_definition, forecast_cache)
        all_rows.extend(rows)

    results = pd.DataFrame(all_rows)

    # Relative cost w.r.t. observed capacity from the baseline replay, if present
    baseline_path = RESULTS_DIR / "baseline_replay_per_service.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        obs_cost = (
            baseline[baseline["policy"] == "observed_capacity"]
            .set_index("msname")["c_total"]
            .rename("observed_c_total")
        )
        results = results.join(obs_cost, on="msname")
        results["relative_cost_vs_observed"] = results["c_total"] / results["observed_c_total"]
    else:
        results["relative_cost_vs_observed"] = float("nan")

    summary = summarize(results)
    results.to_csv(PER_SERVICE_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"\nper_service_rows: {len(results)}")
    print(f"policies: {results['policy'].nunique()}")
    print(f"services: {results['msname'].nunique()}")
    print("\nSUMMARY")
    print(summary.to_csv(index=False).strip())
    print(f"\nOutputs:\n  {PER_SERVICE_PATH}\n  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
