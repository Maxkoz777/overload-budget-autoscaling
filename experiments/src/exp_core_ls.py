"""
exp_core_ls.py
==============
Large-scale (200-service) adapters for replay experiments.

This module intentionally leaves the focused-tier exp_core.py behavior intact.
It imports shared simulation primitives and adds loaders for the 200-service
selection, time-series directory, and sharded forecast outputs.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from exp_core import (
    DEFAULT_COST,
    SPLITS_PATH,
    _runlengths,
    load_service_data,
    simulate_policy,
)


SELECTION = Path("experiments/data/splits/selected_services_200.csv")
TIMESERIES_200 = Path("experiments/data/service_timeseries_200")
FORECASTS_200 = Path("experiments/results/large_scale/forecasts")
RESULTS_LS = Path("experiments/results/large_scale/analysis")
FIGS_LS = Path("experiments/figures/large_scale")
REPORTS_LS = Path("experiments/reports/large_scale")

FORECAST_MODELS = ("arima", "xgb", "lstm")
SIM_COST = {
    "c_res": DEFAULT_COST["c_res"],
    "c_act": DEFAULT_COST["c_act"],
    "c_vio": DEFAULT_COST["c_vio"],
    "c_inf_per_step": DEFAULT_COST["c_inf"],
    "c_train_total": DEFAULT_COST["c_train"],
}


def _ensure_large_scale_dirs() -> None:
    """Create writable large-scale output directories."""
    RESULTS_LS.mkdir(parents=True, exist_ok=True)
    FIGS_LS.mkdir(parents=True, exist_ok=True)
    REPORTS_LS.mkdir(parents=True, exist_ok=True)


def load_selection() -> pd.DataFrame:
    """
    Load the 200-service selection table.

    Returns columns including service_id, stratum, and is_focused_20. The
    boolean flag is normalised to bool because CSV round-trips may preserve it
    as object/string on some pandas versions.
    """
    df = pd.read_csv(SELECTION)
    required = {"service_id", "stratum", "is_focused_20"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{SELECTION} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["service_id"] = df["service_id"].astype(str)
    if df["is_focused_20"].dtype != bool:
        df["is_focused_20"] = df["is_focused_20"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
    return df


GROUP_MAP_200: dict[str, str] = dict(
    zip(load_selection()["service_id"], load_selection()["stratum"])
)


def load_split_def() -> dict:
    """Load the common train/calibration/test split definition."""
    return json.loads(SPLITS_PATH.read_text())


def load_all_services_200(
    split_def: dict,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """
    Load all selected 200 services as (service_id, history_df, test_df).

    history_df is train+calibration; test_df is the replay horizon. The service
    order follows selected_services_200.csv to keep downstream joins stable.
    """
    out: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for service in load_selection()["service_id"]:
        path = TIMESERIES_200 / f"{service}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing large-scale time series: {path}")
        history, test = load_service_data(path, split_def)
        out.append((service, history, test))
    return out


def _persistence_forecast(history: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """One-step persistence fallback aligned to test rows."""
    hist_y = history["cpu_sum"].to_numpy(dtype=float)
    test_y = test["cpu_sum"].to_numpy(dtype=float)
    if len(test_y) == 0:
        return np.asarray([], dtype=float)
    prev = hist_y[-1] if len(hist_y) else 0.0
    return np.concatenate([[prev], test_y[:-1]]).astype(float)


def load_forecast_ls(model: str, service: str) -> np.ndarray:
    """
    Load one sharded forecast and align it to the service test window.

    Forecast timestamps are merged onto the test timestamps from
    service_timeseries_200. Missing or NaN forecasts are filled first with the
    last valid forecast and then, for leading gaps, with causal persistence.
    A warning is emitted whenever such filling is needed.
    """
    if model not in FORECAST_MODELS:
        raise ValueError(f"model must be one of {FORECAST_MODELS}, got {model!r}")

    split_def = load_split_def()
    ts_path = TIMESERIES_200 / f"{service}.parquet"
    fc_path = FORECASTS_200 / model / f"{service}.parquet"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing large-scale time series: {ts_path}")
    if not fc_path.exists():
        raise FileNotFoundError(f"Missing forecast shard: {fc_path}")

    history, test = load_service_data(ts_path, split_def)
    forecast = pd.read_parquet(fc_path)
    required = {"timestamp", "forecast"}
    missing = required - set(forecast.columns)
    if missing:
        raise ValueError(f"{fc_path} missing required columns: {sorted(missing)}")

    forecast = forecast.sort_values("timestamp").copy()
    duplicated = int(forecast["timestamp"].duplicated().sum())
    if duplicated:
        warnings.warn(
            f"{model}/{service}: {duplicated} duplicate forecast timestamps; "
            "keeping the last value per timestamp",
            RuntimeWarning,
            stacklevel=2,
        )
        forecast = forecast.drop_duplicates("timestamp", keep="last")

    test_ts = test[["timestamp"]].reset_index(drop=True)
    merged = test_ts.merge(
        forecast[["timestamp", "forecast"]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    arr = merged["forecast"].astype(float)
    missing_mask = arr.isna()
    if bool(missing_mask.any()):
        n_missing = int(missing_mask.sum())
        warnings.warn(
            f"{model}/{service}: {n_missing} missing/NaN forecasts after "
            "timestamp alignment; filling with last valid forecast/persistence",
            RuntimeWarning,
            stacklevel=2,
        )
        persistence = pd.Series(_persistence_forecast(history, test), index=arr.index)
        arr = arr.ffill()
        arr = arr.where(~arr.isna(), persistence)

    out = arr.to_numpy(dtype=float)
    if len(out) != len(test):
        raise ValueError(
            f"{model}/{service}: aligned forecast length {len(out)} != test length {len(test)}"
        )
    return out


def concat_forecasts(model: str) -> pd.DataFrame:
    """
    Concatenate all forecast shards for a model into one DataFrame.

    This is optional convenience for downstream scripts. The returned frame is
    also written to RESULTS_LS/{model}_forecasts_200.parquet.
    """
    if model not in FORECAST_MODELS:
        raise ValueError(f"model must be one of {FORECAST_MODELS}, got {model!r}")
    _ensure_large_scale_dirs()
    parts = []
    for service in load_selection()["service_id"]:
        path = FORECASTS_200 / model / f"{service}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing forecast shard: {path}")
        parts.append(pd.read_parquet(path))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(RESULTS_LS / f"{model}_forecasts_200.parquet", index=False)
    return out


def _smoke_services(selection: pd.DataFrame) -> list[str]:
    """Pick three services from distinct strata for a quick replay check."""
    strata = sorted(selection["stratum"].dropna().unique())[:3]
    return [
        str(selection[selection["stratum"] == stratum].sort_values("service_id")["service_id"].iloc[0])
        for stratum in strata
    ]


def smoke_test() -> pd.DataFrame:
    """
    Run a small replay smoke test for three services.

    Policies: persistence plus ARIMA/XGB/LSTM forecasts, all with delta=0.05,
    conformal margin, and guardrail enabled.
    """
    _ensure_large_scale_dirs()
    split_def = load_split_def()
    selection = load_selection()
    rows = []

    for service in _smoke_services(selection):
        stratum = GROUP_MAP_200[service]
        history, test = load_service_data(TIMESERIES_200 / f"{service}.parquet", split_def)
        forecast_cases: list[tuple[str, np.ndarray | None]] = [("persistence", None)]
        forecast_cases.extend((model, load_forecast_ls(model, service)) for model in FORECAST_MODELS)

        for model, forecast_array in forecast_cases:
            result = simulate_policy(
                history,
                test,
                forecast_array=forecast_array,
                delta=0.05,
                use_margin=True,
                use_guardrail=True,
                **SIM_COST,
            )
            rows.append(
                {
                    "service_id": service,
                    "stratum": stratum,
                    "model": model,
                    "overload_fraction": result["overload_fraction"],
                    "c_total": result["c_total"],
                    "margin_mean": result["margin_mean"],
                    "guardrail_rate": result["guardrail_rate"],
                    "max_overload_run": result["max_overload_run"],
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    smoke = smoke_test()
    print("large_scale_services", len(load_selection()))
    print("smoke_services", ",".join(smoke["service_id"].drop_duplicates()))
    print(smoke.to_csv(index=False).strip())
