"""
train_arima.py
==============
Train per-service ARIMA models with daily Fourier terms.

Method:
  - auto_arima from pmdarima (start_p=0, start_q=0, max_p=7, max_q=7, max_d=3)
  - Daily seasonality via Fourier terms (period=1440 min), K harmonics in {1..7}
  - ARIMA order pre-selected without exogenous terms (stepwise=True); best K
    then selected by 3-fold time-series CV (test_size=2880) with that order
    fixed, optimising MAE
  - Final model trained on full history (stepwise=False, approximation=False)
  - Causal forecast on the test split in daily chunks, calling model.update()
    with the observed values after each chunk

Usage (original 20-service mode):
    python3 experiments/src/train_arima.py

Usage (large-scale 200-service mode):
    python3 experiments/src/train_arima.py \
        --services-csv experiments/data/splits/selected_services_200.csv \
        --timeseries-dir experiments/data/service_timeseries_200/ \
        --output-dir experiments/results

Output (original mode):
    experiments/models/arima/<msname>_arima.pkl    -- fitted model
    experiments/models/arima/<msname>_meta.json    -- best K and metadata
    experiments/results/arima_forecasts.parquet    -- test-split forecasts per service
    experiments/results/arima_training_log.csv     -- per-service training summary
    experiments/reports/arima_training_report.md   -- human-readable training report

Output (large-scale mode):
    experiments/results/large_scale/arima_training_log_200.csv
    experiments/results/large_scale/forecasts/arima/<service>.parquet
    experiments/results/large_scale/arima_registry.csv

Dependencies:
    pip install pmdarima statsmodels numpy pandas pyarrow
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


TIME_SERIES_DIR    = Path("experiments/data/service_timeseries")
SPLIT_DEFINITION_PATH = Path("experiments/data/splits/split_definition.json")
MODELS_DIR         = Path("experiments/models/arima")
RESULTS_DIR        = Path("experiments/results")
REPORTS_DIR        = Path("experiments/reports")
FORECASTS_PATH     = RESULTS_DIR / "arima_forecasts.parquet"
TRAINING_LOG_PATH  = RESULTS_DIR / "arima_training_log.csv"
REPORT_PATH        = REPORTS_DIR / "arima_training_report.md"

TARGET_COL      = "cpu_sum"
FOURIER_K_VALUES = [1, 2, 3, 4, 5, 6, 7]
SEASONAL_PERIOD = 1440                     # minutes per day
CV_TEST_SIZE    = 2880                     # 2 days per TimeSeriesSplit fold
N_CV_FOLDS      = 3
ARIMA_MAX_P     = 7
ARIMA_MAX_Q     = 7
ARIMA_MAX_D     = 3
ARIMA_MAX_ITER  = 50
EVAL_UPDATE_EVERY = 1440


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-service ARIMA forecasters.")
    p.add_argument(
        "--services-csv", default=None,
        help="CSV with service_id column. If given, runs in large-scale mode.",
    )
    p.add_argument(
        "--timeseries-dir", default=str(TIME_SERIES_DIR),
        help="Directory containing per-service parquets.",
    )
    p.add_argument(
        "--output-dir", default=str(RESULTS_DIR),
        help="Base output directory.",
    )
    p.add_argument(
        "--save-models", action="store_true", default=False,
        help="Save fitted model pickles (default: off in large-scale mode).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N services (dry-run).",
    )
    return p.parse_args()


SEP = "─" * 72

def _ts() -> str:
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

def _fmt_sec(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m{sec:.0f}s"

def _log_stage(idx: int, total: int, name: str, elapsed: float, extra: str = "") -> None:
    tag = f"Stage {idx}/{total}"
    extra_str = f"  {extra}" if extra else ""
    print(f"  {tag:12s} · {name:44s} done  ({_fmt_sec(elapsed)}){extra_str}", flush=True)

def _log_service_header(i: int, total: int, service: str) -> None:
    print(f"\n{SEP}", flush=True)
    print(f"[{i}/{total}]  {service}  ({_ts()})", flush=True)

def _log_service_footer(result: dict, elapsed_total: float, eta_sec: float | None) -> None:
    mae  = result.get("mae", float("nan"))
    rmse = result.get("rmse", float("nan"))
    ur   = result.get("underprediction_rate", float("nan"))
    print(
        f"  MAE={mae:.4f}  RMSE={rmse:.4f}  underpred_rate={ur:.2%}",
        flush=True,
    )
    eta_str = f"  |  ETA {_fmt_sec(eta_sec)}" if eta_sec is not None else ""
    print(f"  Total: {_fmt_sec(elapsed_total)}{eta_str}", flush=True)


def make_fourier_matrix(timestamps_minute: np.ndarray, K: int, period: int) -> np.ndarray:
    t    = timestamps_minute.astype(float)
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)


def minute_index(timestamps_ms: pd.Series) -> np.ndarray:
    return (timestamps_ms // 60_000).to_numpy(dtype=int)


def time_series_cv_splits(n: int, n_splits: int, test_size: int):
    min_train = n - n_splits * test_size
    if min_train <= 0:
        raise ValueError("Not enough data for the requested CV folds / test size.")
    for fold in range(n_splits):
        val_end   = n - (n_splits - fold - 1) * test_size
        val_start = val_end - test_size
        train_idx = np.arange(0, val_start)
        val_idx   = np.arange(val_start, val_end)
        yield train_idx, val_idx


def select_best_k(
    y_train: np.ndarray,
    ts_train: np.ndarray,
    k_values: list[int],
    log_indent: str = "    ",
) -> tuple[int, tuple]:
    from pmdarima import auto_arima, ARIMA as PmdARIMA  # type: ignore

    n = len(y_train)

    first_train_idx = next(iter(time_series_cv_splits(n, N_CV_FOLDS, CV_TEST_SIZE)))[0]
    t0_order = time.perf_counter()
    print(f"{log_indent}Pre-selecting ARIMA order (no exogenous, stepwise=True) ...", flush=True)
    order_model = auto_arima(
        y_train[first_train_idx],
        start_p=0, start_q=0,
        max_p=ARIMA_MAX_P, max_q=ARIMA_MAX_Q, max_d=ARIMA_MAX_D,
        stepwise=True, approximation=False,
        maxiter=ARIMA_MAX_ITER,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
    )
    base_order = order_model.order
    print(
        f"{log_indent}Base order: {base_order}  ({_fmt_sec(time.perf_counter() - t0_order)})",
        flush=True,
    )

    best_k   = k_values[0]
    best_mae = float("inf")

    for K in k_values:
        fold_maes  = []
        fold_times = []
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(
                time_series_cv_splits(n, N_CV_FOLDS, CV_TEST_SIZE)
            ):
                t0_fold      = time.perf_counter()
                X_fold_train = make_fourier_matrix(ts_train[train_idx], K, SEASONAL_PERIOD)
                X_fold_val   = make_fourier_matrix(ts_train[val_idx],   K, SEASONAL_PERIOD)
                model = PmdARIMA(order=base_order, suppress_warnings=True)
                model.fit(y_train[train_idx], X=X_fold_train)
                preds = model.predict(n_periods=len(val_idx), X=X_fold_val)
                preds = np.maximum(preds, 0.0)
                mae   = float(np.mean(np.abs(preds - y_train[val_idx])))
                fold_maes.append(mae)
                fold_times.append(time.perf_counter() - t0_fold)
                print(
                    f"{log_indent}K={K}  fold {fold_idx+1}/{N_CV_FOLDS}  "
                    f"MAE={mae:.4f}  ({_fmt_sec(fold_times[-1])})",
                    flush=True,
                )
        except Exception as e:
            print(f"{log_indent}K={K}  ERROR: {e}", flush=True)
            continue

        mean_mae = float(np.mean(fold_maes)) if fold_maes else float("inf")
        marker   = "  <- best" if mean_mae < best_mae else ""
        print(f"{log_indent}K={K}  mean_MAE={mean_mae:.4f}{marker}", flush=True)
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_k   = K

    return best_k, base_order


def predict_test(
    model,
    y_test:    np.ndarray,
    ts_test:   np.ndarray,
    K:         int,
    update_every: int = EVAL_UPDATE_EVERY,
) -> np.ndarray:
    n         = len(y_test)
    forecasts = []

    for chunk_start in range(0, n, update_every):
        chunk_end = min(chunk_start + update_every, n)
        ts_chunk  = ts_test[chunk_start:chunk_end]
        X_chunk   = make_fourier_matrix(ts_chunk, K, SEASONAL_PERIOD)

        preds = model.predict(n_periods=chunk_end - chunk_start, exogenous=X_chunk)
        forecasts.extend(np.maximum(preds, 0.0).tolist())

        if chunk_end < n:
            try:
                model.update(y_test[chunk_start:chunk_end], exogenous=X_chunk)
            except Exception:
                pass

    return np.maximum(np.array(forecasts, dtype=float), 0.0)


def load_registry(registry_path: Path) -> dict[str, dict]:
    if not registry_path.exists():
        return {}
    try:
        df = pd.read_csv(registry_path)
        return {str(row["service_id"]): row.to_dict() for _, row in df.iterrows()}
    except Exception:
        return {}


def append_registry(registry_path: Path, row: dict) -> None:
    """Append or update a registry row."""
    existing = load_registry(registry_path)
    existing[str(row["service_id"])] = row
    pd.DataFrame(list(existing.values())).to_csv(registry_path, index=False)


def train_service(
    path: Path,
    split_definition: dict,
    service_idx: int,
    total: int,
    save_models: bool = True,
    models_dir: Path | None = None,
) -> dict:
    from pmdarima import auto_arima  # type: ignore

    df      = pd.read_parquet(path).sort_values("timestamp")
    service = str(df["msname"].iloc[0])
    split_map = {s["name"]: s for s in split_definition["splits"]}

    _log_service_header(service_idx, total, service)

    def mask(split_name):
        s = split_map[split_name]
        return (df["timestamp"] >= s["timestamp_start"]) & (df["timestamp"] < s["timestamp_end_exclusive"])

    # Stage 1: load and split data
    t0 = time.perf_counter()
    train_df   = df[mask("train")]
    cal_df     = df[mask("calibration")]
    test_df    = df[mask("test")]
    history_df = pd.concat([train_df, cal_df], ignore_index=True)

    y_history  = history_df[TARGET_COL].to_numpy(dtype=float)
    y_test     = test_df[TARGET_COL].to_numpy(dtype=float)
    ts_history = minute_index(history_df["timestamp"])
    ts_test    = minute_index(test_df["timestamp"])
    _log_stage(1, 4, "Load data & split",
               time.perf_counter() - t0,
               f"history={len(y_history):,}  test={len(y_test):,}")

    # Stage 2: K-selection
    t0_select = time.perf_counter()
    print(
        f"  Stage 2/4  · K-selection  "
        f"(K∈{FOURIER_K_VALUES}, {N_CV_FOLDS}-fold CV, fixed-order method) ...",
        flush=True,
    )
    best_k, base_order = select_best_k(y_history, ts_history, FOURIER_K_VALUES)
    t_select = time.perf_counter() - t0_select
    _log_stage(2, 4, "K-selection CV", t_select, f"best_K={best_k}  base_order={base_order}")

    # Stage 3: Final model fit
    t0_train = time.perf_counter()
    print(
        f"  Stage 3/4  · Final model fit (K={best_k}, stepwise=False) ...",
        flush=True,
    )
    X_history = make_fourier_matrix(ts_history, best_k, SEASONAL_PERIOD)
    final_model = auto_arima(
        y_history,
        exogenous=X_history,
        start_p=0, start_q=0,
        max_p=ARIMA_MAX_P, max_q=ARIMA_MAX_Q, max_d=ARIMA_MAX_D,
        stepwise=False, approximation=False,
        maxiter=ARIMA_MAX_ITER,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
    )
    t_train = time.perf_counter() - t0_train
    _log_stage(3, 4, "Final model fit (thorough)",
               t_train,
               f"order={final_model.order}  K={best_k}")

    # Stage 4: test prediction
    t0_pred = time.perf_counter()
    n_chunks = int(np.ceil(len(y_test) / EVAL_UPDATE_EVERY))
    print(
        f"  Stage 4/4  · Prediction with daily updates  "
        f"({n_chunks} chunks × {EVAL_UPDATE_EVERY} steps) ...",
        flush=True,
    )
    forecasts = predict_test(final_model, y_test, ts_test, best_k)
    t_pred = time.perf_counter() - t0_pred
    _log_stage(4, 4, "Prediction (daily model updates)",
               t_pred,
               f"steps={len(y_test):,}  chunks={n_chunks}")

    errors         = forecasts - y_test
    mae            = float(np.mean(np.abs(errors)))
    rmse           = float(np.sqrt(np.mean(errors ** 2)))
    under          = np.maximum(y_test - forecasts, 0.0)
    underpred_rate = float(np.mean(forecasts < y_test))
    residual_p95   = float(np.quantile(y_test - forecasts, 0.95))

    if save_models and models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{service}_arima.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        with open(models_dir / f"{service}_meta.json", "w") as f:
            json.dump({
                "service":       service,
                "best_k":        best_k,
                "arima_order":   list(final_model.order),
                "model_path":    str(model_path),
                "history_rows":  len(y_history),
                "test_rows":     len(y_test),
            }, f, indent=2)

    return {
        "msname":                     service,
        "model":                      "arima",
        "best_k":                     best_k,
        "arima_order":                str(final_model.order),
        "arima_p":                    final_model.order[0],
        "arima_d":                    final_model.order[1],
        "arima_q":                    final_model.order[2],
        "train_rows":                 len(y_history),
        "test_rows":                  len(y_test),
        "mae":                        mae,
        "rmse":                       rmse,
        "underprediction_mae":        float(np.mean(under)),
        "underprediction_p95":        float(np.quantile(under, 0.95)),
        "underprediction_rate":       underpred_rate,
        "residual_p95":               residual_p95,
        "k_selection_seconds":        t_select,
        "train_seconds":              t_train,
        "inference_seconds_total":    t_pred,
        "inference_seconds_per_step": t_pred / len(y_test),
        "forecasts":                  forecasts,
        "timestamps":                 test_df["timestamp"].to_numpy(),
    }


def _write_report(log_rows: list[dict], t_wall: float) -> None:
    import datetime

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df  = pd.DataFrame([r for r in log_rows if "error" not in r])
    err_rows = [r for r in log_rows if "error" in r]

    lines = [
        "# ARIMA Training Report",
        "",
        f"Generated: {now}",
        f"Total wall time: {_fmt_sec(t_wall)}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| `FOURIER_K_VALUES` | {FOURIER_K_VALUES} |",
        f"| `SEASONAL_PERIOD` | {SEASONAL_PERIOD} min |",
        f"| `N_CV_FOLDS` | {N_CV_FOLDS} |",
        f"| `CV_TEST_SIZE` | {CV_TEST_SIZE} steps |",
        f"| `ARIMA_MAX_P` | {ARIMA_MAX_P} |",
        f"| `ARIMA_MAX_Q` | {ARIMA_MAX_Q} |",
        f"| `ARIMA_MAX_D` | {ARIMA_MAX_D} |",
        f"| `EVAL_UPDATE_EVERY` | {EVAL_UPDATE_EVERY} steps (daily) |",
        "",
    ]

    if not df.empty:
        lines += [
            "## Aggregate Metrics",
            "",
            "| Metric | Min | Median | Max |",
            "|---|---:|---:|---:|",
        ]
        for col, label in [
            ("mae",                  "MAE"),
            ("rmse",                 "RMSE"),
            ("underprediction_rate", "Underpred rate"),
            ("underprediction_p95",  "Underpred p95"),
            ("k_selection_seconds",  "K-selection time (s)"),
            ("train_seconds",        "Final fit time (s)"),
        ]:
            if col in df.columns:
                s = df[col].dropna()
                lines.append(
                    f"| {label} | {s.min():.4f} | {s.median():.4f} | {s.max():.4f} |"
                )

        if "best_k" in df.columns:
            k_counts = df["best_k"].value_counts().sort_index()
            lines += ["", "## Best K Distribution", ""]
            lines.append("| K | Count |")
            lines.append("|---:|---:|")
            for k, cnt in k_counts.items():
                lines.append(f"| {k} | {cnt} |")

        if "arima_order" in df.columns:
            order_counts = df["arima_order"].value_counts().head(10)
            lines += ["", "## Most Common ARIMA Orders (top 10)", ""]
            lines.append("| Order | Count |")
            lines.append("|---|---:|")
            for order, cnt in order_counts.items():
                lines.append(f"| `{order}` | {cnt} |")

        lines += [
            "",
            "## Per-Service Results",
            "",
            "| Service | MAE | RMSE | best_K | ARIMA order | underpred_rate | K-sel(s) | fit(s) |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
        for _, row in df.iterrows():
            lines.append(
                f"| `{row['msname']}` "
                f"| {row.get('mae', float('nan')):.4f} "
                f"| {row.get('rmse', float('nan')):.4f} "
                f"| {row.get('best_k', '?')} "
                f"| `{row.get('arima_order', '?')}` "
                f"| {row.get('underprediction_rate', float('nan')):.2%} "
                f"| {row.get('k_selection_seconds', float('nan')):.1f} "
                f"| {row.get('train_seconds', float('nan')):.1f} |"
            )

    if err_rows:
        lines += ["", "## Errors", ""]
        for r in err_rows:
            lines.append(f"- `{r['msname']}`: {r.get('error', 'unknown')}")

    lines += [""]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {REPORT_PATH}")


def main() -> None:
    args = parse_args()
    large_scale_mode = args.services_csv is not None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_definition = json.loads(SPLIT_DEFINITION_PATH.read_text())

    if large_scale_mode:
        ts_dir = Path(args.timeseries_dir)
        out_dir = Path(args.output_dir)
        large_out = out_dir / "large_scale"
        large_out.mkdir(parents=True, exist_ok=True)
        forecasts_dir = large_out / "forecasts" / "arima"
        forecasts_dir.mkdir(parents=True, exist_ok=True)

        training_log_path = large_out / "arima_training_log_200.csv"
        registry_path     = large_out / "arima_registry.csv"

        svc_df = pd.read_csv(args.services_csv)
        service_ids = svc_df["service_id"].tolist()
        if args.limit is not None:
            service_ids = service_ids[:args.limit]

        # Load existing registry and skip 'done' services
        registry = load_registry(registry_path)
        done_ids = {k for k, v in registry.items() if v.get("status") == "done"}
        pending  = [s for s in service_ids if s not in done_ids]

        total = len(service_ids)
        print(f"\n{'='*72}", flush=True)
        print(f"  ARIMA Large-Scale Training  —  {total} services  ({_ts()})", flush=True)
        print(f"  Already done: {len(done_ids)}, Pending: {len(pending)}", flush=True)
        print(f"  Config: K∈{FOURIER_K_VALUES} · {N_CV_FOLDS}-fold CV", flush=True)
        print(f"{'='*72}\n", flush=True)

        existing_log: list[dict] = []
        if training_log_path.exists():
            try:
                existing_log = pd.read_csv(training_log_path).to_dict("records")
            except Exception:
                pass

        log_rows = existing_log.copy()
        service_times = []
        t_wall_start = time.perf_counter()

        for idx, service_id in enumerate(pending, start=1):
            path = ts_dir / f"{service_id}.parquet"
            if not path.exists():
                print(f"  SKIP {service_id}: parquet not found at {path}", flush=True)
                reg_row = {
                    "service_id": service_id, "status": "failed",
                    "started_at": _ts(), "finished_at": _ts(),
                    "runtime_s": 0, "mae": float("nan"),
                    "error": "parquet not found",
                }
                append_registry(registry_path, reg_row)
                continue

            started_at = _ts()
            t_svc = time.perf_counter()

            append_registry(registry_path, {
                "service_id": service_id, "status": "running",
                "started_at": started_at, "finished_at": "",
                "runtime_s": 0, "mae": float("nan"), "error": "",
            })

            try:
                global_idx = service_ids.index(service_id) + 1
                result     = train_service(
                    path, split_definition, global_idx, total,
                    save_models=args.save_models,
                    models_dir=MODELS_DIR if args.save_models else None,
                )
                forecasts  = result.pop("forecasts")
                timestamps = result.pop("timestamps")

                fc_df = pd.DataFrame({
                    "msname":    service_id,
                    "model":     "arima",
                    "timestamp": timestamps.astype(int),
                    "forecast":  forecasts,
                })
                fc_path = forecasts_dir / f"{service_id}.parquet"
                fc_df.to_parquet(fc_path, index=False)

                log_rows.append(result)
                pd.DataFrame(log_rows).to_csv(training_log_path, index=False)

                elapsed_svc = time.perf_counter() - t_svc
                service_times.append(elapsed_svc)
                remaining = len(pending) - idx
                eta = (sum(service_times) / len(service_times)) * remaining if remaining > 0 else None

                finished_at = _ts()
                append_registry(registry_path, {
                    "service_id": service_id, "status": "done",
                    "started_at": started_at, "finished_at": finished_at,
                    "runtime_s": round(elapsed_svc, 1),
                    "mae": round(result.get("mae", float("nan")), 6),
                    "error": "",
                })
                _log_service_footer(result, elapsed_svc, eta)

            except Exception as e:
                import traceback
                elapsed_svc = time.perf_counter() - t_svc
                print(f"  ERROR ({_fmt_sec(elapsed_svc)}): {e}", flush=True)
                traceback.print_exc()
                append_registry(registry_path, {
                    "service_id": service_id, "status": "failed",
                    "started_at": started_at, "finished_at": _ts(),
                    "runtime_s": round(elapsed_svc, 1),
                    "mae": float("nan"),
                    "error": str(e)[:200],
                })

        t_wall = time.perf_counter() - t_wall_start
        ok  = [r for r in log_rows if "error" not in r]
        print(f"\n{'='*72}", flush=True)
        print(f"  Done: {len(ok)}/{total} services succeeded", flush=True)
        print(f"  Training log: {training_log_path}", flush=True)
        print(f"  Registry:     {registry_path}", flush=True)
        print(f"  Wall time:    {_fmt_sec(t_wall)}", flush=True)
        print(f"{'='*72}\n", flush=True)

    else:
        service_files = sorted(TIME_SERIES_DIR.glob("MS_*.parquet"))
        if args.limit is not None:
            service_files = service_files[:args.limit]
        total = len(service_files)

        print(f"\n{'='*72}", flush=True)
        print(f"  ARIMA Training  —  {total} services  ({_ts()})", flush=True)
        print(f"  Config: K∈{FOURIER_K_VALUES} · {N_CV_FOLDS}-fold CV · "
              f"max_p={ARIMA_MAX_P} max_q={ARIMA_MAX_Q} max_d={ARIMA_MAX_D}", flush=True)
        print(f"{'='*72}\n", flush=True)

        log_rows      = []
        forecast_rows = []
        t_wall_start  = time.perf_counter()
        service_times = []

        for idx, path in enumerate(service_files, start=1):
            service = path.stem
            t_svc   = time.perf_counter()
            try:
                result     = train_service(
                    path, split_definition, idx, total,
                    save_models=True, models_dir=MODELS_DIR,
                )
                forecasts  = result.pop("forecasts")
                timestamps = result.pop("timestamps")
                log_rows.append(result)
                for ts, fc in zip(timestamps, forecasts):
                    forecast_rows.append({"msname": service, "model": "arima",
                                          "timestamp": int(ts), "forecast": fc})
                elapsed_svc = time.perf_counter() - t_svc
                service_times.append(elapsed_svc)
                remaining   = total - idx
                eta         = (sum(service_times) / len(service_times)) * remaining if remaining > 0 else None
                _log_service_footer(result, elapsed_svc, eta)
            except Exception as e:
                import traceback
                elapsed_svc = time.perf_counter() - t_svc
                print(f"  ERROR ({_fmt_sec(elapsed_svc)}): {e}", flush=True)
                traceback.print_exc()
                log_rows.append({"msname": service, "model": "arima", "error": str(e)})

        t_wall = time.perf_counter() - t_wall_start

        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(TRAINING_LOG_PATH, index=False)

        if forecast_rows:
            forecast_df = pd.DataFrame(forecast_rows)
            forecast_df.to_parquet(FORECASTS_PATH, index=False)

        ok  = [r for r in log_rows if "error" not in r]
        err = [r for r in log_rows if "error" in r]
        print(f"\n{'='*72}", flush=True)
        print(f"  Done: {len(ok)}/{total} services succeeded, {len(err)} failed", flush=True)
        if ok:
            maes = [r["mae"] for r in ok]
            print(f"  MAE  min={min(maes):.4f}  median={float(np.median(maes)):.4f}  max={max(maes):.4f}", flush=True)
        print(f"  Wall time: {_fmt_sec(t_wall)}", flush=True)
        print(f"{'='*72}\n", flush=True)

        _write_report(log_rows, t_wall)


if __name__ == "__main__":
    main()
