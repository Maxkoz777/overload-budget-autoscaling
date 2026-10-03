"""
evaluate_advanced_forecasting.py
=================================
Load pre-trained ARIMA / XGBoost / LSTM forecasts and compare them against
the simple baselines.

This script assumes the three training scripts have already been run:
    python3 experiments/src/train_arima.py
    python3 experiments/src/train_xgb.py
    python3 experiments/src/train_lstm.py

It reads the saved forecast Parquet files, joins them with ground truth from
the test split, computes a unified error metrics table, and writes a combined
report.

Usage:
    python3 experiments/src/evaluate_advanced_forecasting.py

Output:
    experiments/results/advanced_forecasting_comparison.csv
    experiments/reports/forecasting_advanced_report.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TIME_SERIES_DIR = Path("experiments/data/service_timeseries")
SPLIT_DEFINITION_PATH = Path("experiments/data/splits/split_definition.json")
RESULTS_DIR = Path("experiments/results")
REPORTS_DIR = Path("experiments/reports")
OUTPUT_CSV = RESULTS_DIR / "advanced_forecasting_comparison.csv"
OUTPUT_REPORT = REPORTS_DIR / "forecasting_advanced_report.md"

FORECAST_FILES = {
    "arima": RESULTS_DIR / "arima_forecasts.parquet",
    "xgb":   RESULTS_DIR / "xgb_forecasts.parquet",
    "lstm":  RESULTS_DIR / "lstm_forecasts.parquet",
}

TRAINING_LOGS = {
    "arima": RESULTS_DIR / "arima_training_log.csv",
    "xgb":   RESULTS_DIR / "xgb_training_log.csv",
    "lstm":  RESULTS_DIR / "lstm_training_log.csv",
}


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray, model: str, service: str) -> dict:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    under = np.maximum(y_true - y_pred, 0.0)
    return {
        "msname": service,
        "model": model,
        "n": len(y_true),
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape_safe": float((abs_errors / np.maximum(y_true, 1e-9)).mean()),
        "bias": float(errors.mean()),
        "underprediction_mae": float(under.mean()),
        "underprediction_p95": float(np.quantile(under, 0.95)),
        "underprediction_rate": float((y_pred < y_true).mean()),
        "residual_p95": float(np.quantile(y_true - y_pred, 0.95)),
    }


def load_ground_truth(split_definition: dict) -> pd.DataFrame:
    split_map = {s["name"]: s for s in split_definition["splits"]}
    test_split = split_map["test"]
    rows = []
    for path in sorted(TIME_SERIES_DIR.glob("MS_*.parquet")):
        df = pd.read_parquet(path, columns=["timestamp", "msname", "cpu_sum"])
        test = df[
            (df["timestamp"] >= test_split["timestamp_start"])
            & (df["timestamp"] < test_split["timestamp_end_exclusive"])
        ]
        rows.append(test)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    split_definition = json.loads(SPLIT_DEFINITION_PATH.read_text())
    gt = load_ground_truth(split_definition)
    gt = gt.rename(columns={"cpu_sum": "y_true"})

    all_metrics = []
    available_models = []

    for model_name, fpath in FORECAST_FILES.items():
        if not fpath.exists():
            print(f"  Skipping {model_name}: forecast file not found ({fpath})")
            continue
        available_models.append(model_name)
        forecasts = pd.read_parquet(fpath)
        merged = gt.merge(forecasts[["msname", "timestamp", "forecast"]], on=["msname", "timestamp"], how="inner")

        for service, grp in merged.groupby("msname"):
            y_true = grp["y_true"].to_numpy(dtype=float)
            y_pred = grp["forecast"].to_numpy(dtype=float)
            all_metrics.append(error_metrics(y_true, y_pred, model_name, str(service)))

    if not all_metrics:
        print("No forecast files found. Run the training scripts first.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Per-service metrics saved: {OUTPUT_CSV}")

    # Summary across services
    summary = (
        metrics_df.groupby("model")
        .agg(
            services=("msname", "nunique"),
            median_mae=("mae", "median"),
            p10_mae=("mae", lambda s: s.quantile(0.10)),
            p90_mae=("mae", lambda s: s.quantile(0.90)),
            median_rmse=("rmse", "median"),
            median_bias=("bias", "median"),
            median_underprediction_mae=("underprediction_mae", "median"),
            median_underprediction_p95=("underprediction_p95", "median"),
            median_underprediction_rate=("underprediction_rate", "median"),
        )
        .reset_index()
    )

    # Merge training overhead if available
    overhead_rows = []
    for model_name, log_path in TRAINING_LOGS.items():
        if log_path.exists() and model_name in available_models:
            log_df = pd.read_csv(log_path)
            if "train_seconds" in log_df.columns:
                overhead_rows.append({
                    "model": model_name,
                    "median_train_seconds": float(log_df["train_seconds"].dropna().median()),
                    "median_inference_seconds_per_step": float(
                        log_df["inference_seconds_per_step"].dropna().median()
                    ) if "inference_seconds_per_step" in log_df.columns else float("nan"),
                })
    if overhead_rows:
        overhead_df = pd.DataFrame(overhead_rows)
        summary = summary.merge(overhead_df, on="model", how="left")

    print("\nSUMMARY")
    print(summary.to_string(index=False))

    _write_report(summary, metrics_df, available_models)
    print(f"\nReport saved: {OUTPUT_REPORT}")


def _write_report(summary: pd.DataFrame, metrics_df: pd.DataFrame, models: list[str]) -> None:
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

    lines = [
        "# Advanced Forecasting Models Report",
        "",
        f"Generated: {now}",
        "",
        "## Scope",
        "",
        "This report evaluates ARIMA, XGBoost, and LSTM forecasters trained in Phase 7 (extended)",
        "and compares their one-step-ahead forecast accuracy on the held-out test split.",
        "",
        "Input forecast files:",
    ]
    for m in models:
        lines.append(f"- `experiments/results/{m}_forecasts.parquet`")
    lines += [
        "",
        "## Models Evaluated",
        "",
        "| Model | Description |",
        "|---|---|",
        "| `arima` | ARIMA with daily Fourier terms, K selected by 2-fold CV, auto_arima |",
        "| `xgb` | XGBoost with lag/rolling/EMA/calendar features, 200-trial Optuna search |",
        "| `lstm` | LSTM (CPU or GPU mode), Optuna search over lookback/units/layers |",
        "",
        "## Summary Metrics",
        "",
    ]

    cols = list(summary.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)
    for _, row in summary.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Per-Service MAE Distribution",
        "",
        "| Model | Services | Min MAE | Median MAE | Max MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in models:
        sub = metrics_df[metrics_df["model"] == m]["mae"]
        if len(sub):
            lines.append(
                f"| `{m}` | {len(sub)} | {sub.min():.4f} | {sub.median():.4f} | {sub.max():.4f} |"
            )

    lines += [
        "",
        "## Underprediction Characteristics",
        "",
        "Underprediction drives overload risk in the autoscaling replay.",
        "",
        "| Model | Median underpred MAE | Median underpred p95 | Median underpred rate |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        if "median_underprediction_mae" in summary.columns:
            lines.append(
                f"| `{row['model']}` | "
                f"{row.get('median_underprediction_mae', float('nan')):.4f} | "
                f"{row.get('median_underprediction_p95', float('nan')):.4f} | "
                f"{row.get('median_underprediction_rate', float('nan')):.4f} |"
            )

    lines += [
        "",
        "## Limitations",
        "",
        "1. ARIMA uses `pmdarima.auto_arima`; full parameter search (`stepwise=False`) may",
        "   be slow for services with complex autocorrelation structures.",
        "2. XGBoost Optuna search uses 200 trials; fewer trials reduce search time at the",
        "   cost of potentially suboptimal hyperparameters.",
        "3. LSTM uses CPU mode by default; GPU mode requires CUDA and produces significantly",
        "   better results for bursty services (see thesis Section 4.5.5).",
        "4. The predictor-agnosticism claim requires that all three models be available before",
        "   Phase 9 core experiments begin.",
        "",
        "## Next Phase",
        "",
        "Proceed to Phase 9 (core experiments) once all three model forecast files are available.",
        "Phase 9 should integrate these forecasts into the risk-aware policy replay via",
        "`run_advanced_policy_replay.py`.",
    ]

    with open(OUTPUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
