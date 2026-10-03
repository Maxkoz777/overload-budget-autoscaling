from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting import (
    AutoregressiveRidgeForecaster,
    moving_average_forecast,
    persistence_forecast,
    seasonal_naive_forecast,
)


TIME_SERIES_DIR = Path("experiments/data/service_timeseries")
SPLIT_DEFINITION_PATH = Path("experiments/data/splits/split_definition.json")
RESULTS_DIR = Path("experiments/results")
PER_SERVICE_PATH = RESULTS_DIR / "forecasting_metrics_per_service.csv"
SUMMARY_PATH = RESULTS_DIR / "forecasting_metrics_summary.csv"


def split_frame(df: pd.DataFrame, split: dict) -> pd.DataFrame:
    return df[
        (df["timestamp"] >= split["timestamp_start"])
        & (df["timestamp"] < split["timestamp_end_exclusive"])
    ].copy()


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    under = np.maximum(y_true - y_pred, 0.0)
    return {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mape_safe": float((abs_errors / np.maximum(y_true, 1e-9)).mean()),
        "bias": float(errors.mean()),
        "underprediction_mae": float(under.mean()),
        "underprediction_p95": float(np.quantile(under, 0.95)),
        "underprediction_rate": float((y_pred < y_true).mean()),
        "residual_p95": float(np.quantile(y_true - y_pred, 0.95)),
    }


def evaluate_service(path: Path, split_definition: dict) -> list[dict[str, float | int | str]]:
    df = pd.read_parquet(path).sort_values("timestamp")
    service = str(df["msname"].iloc[0])
    split_map = {split["name"]: split for split in split_definition["splits"]}
    train = split_frame(df, split_map["train"])
    calibration = split_frame(df, split_map["calibration"])
    test = split_frame(df, split_map["test"])

    train_y = train["cpu_sum"]
    calibration_y = calibration["cpu_sum"]
    test_y = test["cpu_sum"]
    pre_test_history = pd.concat([train_y, calibration_y], ignore_index=True)

    models = []

    start = time.perf_counter()
    pred = persistence_forecast(pre_test_history, test_y)
    elapsed = time.perf_counter() - start
    models.append(("persistence", pred, 0.0, elapsed))

    start = time.perf_counter()
    pred = moving_average_forecast(pre_test_history, test_y, window=60)
    elapsed = time.perf_counter() - start
    models.append(("moving_average_60", pred, 0.0, elapsed))

    start = time.perf_counter()
    pred = seasonal_naive_forecast(pre_test_history, test_y, seasonal_lag=1440)
    elapsed = time.perf_counter() - start
    models.append(("seasonal_naive_1d", pred, 0.0, elapsed))

    start_train = time.perf_counter()
    ar = AutoregressiveRidgeForecaster(alpha=1.0)
    ar.fit(train_y)
    train_elapsed = time.perf_counter() - start_train
    start_pred = time.perf_counter()
    pred = ar.predict_causal(pre_test_history, test_y)
    pred_elapsed = time.perf_counter() - start_pred
    models.append(("autoregressive_ridge_lags", pred, train_elapsed, pred_elapsed))

    rows = []
    y_true = test_y.to_numpy(dtype=float)
    for model_name, y_pred, train_seconds, inference_seconds_total in models:
        metrics = error_metrics(y_true, y_pred)
        rows.append(
            {
                "msname": service,
                "model": model_name,
                "train_rows": len(train),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "train_seconds": train_seconds,
                "inference_seconds_total": inference_seconds_total,
                "inference_seconds_per_step": inference_seconds_total / len(test),
                **metrics,
            }
        )
    return rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby("model", sort=False)
    return grouped.agg(
        services=("msname", "nunique"),
        median_mae=("mae", "median"),
        p10_mae=("mae", lambda s: s.quantile(0.10)),
        p90_mae=("mae", lambda s: s.quantile(0.90)),
        median_rmse=("rmse", "median"),
        median_bias=("bias", "median"),
        median_underprediction_mae=("underprediction_mae", "median"),
        median_underprediction_p95=("underprediction_p95", "median"),
        median_underprediction_rate=("underprediction_rate", "median"),
        median_train_seconds=("train_seconds", "median"),
        median_inference_seconds_per_step=("inference_seconds_per_step", "median"),
    ).reset_index()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_definition = json.loads(SPLIT_DEFINITION_PATH.read_text())
    rows = []
    for path in sorted(TIME_SERIES_DIR.glob("MS_*.parquet")):
        rows.extend(evaluate_service(path, split_definition))

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(PER_SERVICE_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("per_service_rows", len(results))
    print("services", results["msname"].nunique())
    print("models", results["model"].nunique())
    print("target", "cpu_sum")
    print("evaluation_split", "test")
    print("\nSUMMARY")
    print(summary.to_csv(index=False).strip())
    print("\noutputs")
    print(PER_SERVICE_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
