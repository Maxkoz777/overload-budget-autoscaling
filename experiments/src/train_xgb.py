"""
train_xgb.py
============
Train per-service XGBoost forecasters.

Method:
  - Features: 14 lags, rolling stats (5,10 min windows), EMAs (3,5,10,15 min),
    rate-of-change, lag differences, minute-of-day Fourier calendar features
  - Hyperparameter search via Optuna (200 trials default, 50 for large-scale)
  - Expanding-window CV with CV_N_SPLITS=5 configured folds of test_size=2880;
    folds whose training part would be empty are skipped (``ts_cv_splits``).
    With the focused 14,400-minute pre-test history (14,386 rows after lag
    filtering) this yields four feasible folds; see ``effective_cv_folds``.
    Early stopping after 150 rounds.
  - Final model retrained on full history, walk-forward causal test prediction.
    The online feature vector for target minute t is the training row of t:
    lags/rolling/EMA use observations up to t-1 and the calendar features use
    t (``next_step_features``).

Usage (original 20-service mode):
    python3 experiments/src/train_xgb.py

Usage (large-scale 200-service mode):
    python3 experiments/src/train_xgb.py \
        --services-csv experiments/data/splits/selected_services_200.csv \
        --timeseries-dir experiments/data/service_timeseries_200/ \
        --output-dir experiments/results \
        --optuna-trials 50

Output (original mode):
    experiments/models/xgb/<msname>_xgb.json      -- XGBoost booster
    experiments/models/xgb/<msname>_meta.json      -- best params + metadata
    experiments/results/xgb_forecasts.parquet      -- test-split forecasts
    experiments/results/xgb_training_log.csv       -- per-service training summary
    experiments/reports/xgb_training_report.md     -- human-readable training report

Output (large-scale mode):
    experiments/results/large_scale/xgb_training_log_200.csv
    experiments/results/large_scale/forecasts/xgb/<service>.parquet
    experiments/results/large_scale/xgb_registry.csv

Dependencies:
    pip install xgboost optuna numpy pandas pyarrow scikit-learn
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


TIME_SERIES_DIR = Path("experiments/data/service_timeseries")
SPLIT_DEFINITION_PATH = Path("experiments/data/splits/split_definition.json")
MODELS_DIR = Path("experiments/models/xgb")
RESULTS_DIR = Path("experiments/results")
REPORTS_DIR = Path("experiments/reports")
FORECASTS_PATH = RESULTS_DIR / "xgb_forecasts.parquet"
TRAINING_LOG_PATH = RESULTS_DIR / "xgb_training_log.csv"
REPORT_PATH = REPORTS_DIR / "xgb_training_report.md"

TARGET_COL = "cpu_sum"
N_LAGS = 14
ROLLING_WINDOWS = [5, 10]
EMA_SPANS = [3, 5, 10, 15]
CV_N_SPLITS = 5
CV_TEST_SIZE = 2880
OPTUNA_N_TRIALS = 200          # default for original 20-service mode
OPTUNA_N_TRIALS_LARGE = 50     # default for large-scale mode
EARLY_STOPPING_ROUNDS = 150
SEASONAL_PERIOD = 1440
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-service XGBoost forecasters.")
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
        help="Save fitted XGBoost model files (default: off in large-scale mode).",
    )
    p.add_argument(
        "--optuna-trials", type=int, default=None,
        help=f"Number of Optuna trials. Defaults to {OPTUNA_N_TRIALS} (original) "
             f"or {OPTUNA_N_TRIALS_LARGE} (large-scale).",
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
    print(f"  {tag:12s} · {name:40s} done  ({_fmt_sec(elapsed)}){extra_str}", flush=True)

def _log_service_header(i: int, total: int, service: str) -> None:
    print(f"\n{SEP}", flush=True)
    print(f"[{i}/{total}]  {service}  ({_ts()})", flush=True)

def _log_service_footer(result: dict, elapsed_total: float, eta_sec: float | None) -> None:
    mae  = result.get("mae", float("nan"))
    rmse = result.get("rmse", float("nan"))
    ur   = result.get("underprediction_rate", float("nan"))
    cv   = result.get("best_mae_cv", float("nan"))
    print(
        f"  MAE={mae:.4f}  RMSE={rmse:.4f}  "
        f"underpred_rate={ur:.2%}  cv_MAE={cv:.4f}",
        flush=True,
    )
    eta_str = f"  |  ETA {_fmt_sec(eta_sec)}" if eta_sec is not None else ""
    print(f"  Total: {_fmt_sec(elapsed_total)}{eta_str}", flush=True)

class _OptunaProgressCallback:
    def __init__(self, n_trials: int, report_every: int = 25):
        self.n_trials = n_trials
        self.report_every = report_every
        self._start = time.perf_counter()

    def __call__(self, study, trial):
        n = trial.number + 1
        if n % self.report_every == 0 or n == self.n_trials:
            elapsed = time.perf_counter() - self._start
            rate = n / elapsed if elapsed > 0 else 0
            remaining = (self.n_trials - n) / rate if rate > 0 else 0
            best = study.best_value if study.best_trial else float("nan")
            print(
                f"    Optuna trial {n:>3}/{self.n_trials}  "
                f"best_MAE={best:.4f}  "
                f"elapsed={_fmt_sec(elapsed)}  ETA~{_fmt_sec(remaining)}",
                flush=True,
            )


def build_features(series: np.ndarray, timestamps_ms: np.ndarray) -> pd.DataFrame:
    s = pd.Series(series, dtype=float)
    minute_of_day = (timestamps_ms // 60_000) % SEASONAL_PERIOD
    minute_of_day = minute_of_day.astype(float)

    df = pd.DataFrame()

    for lag in range(1, N_LAGS + 1):
        df[f"lag_{lag}"] = s.shift(lag)

    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = s.shift(1).rolling(w).std()
        df[f"roll_max_{w}"]  = s.shift(1).rolling(w).max()
        df[f"roll_min_{w}"]  = s.shift(1).rolling(w).min()

    for span in EMA_SPANS:
        df[f"ema_{span}"] = s.shift(1).ewm(span=span, adjust=False).mean()

    df["roc"] = (s.shift(1) - s.shift(2)) / (s.shift(2).abs() + 1e-9)
    df["lag_diff_1_2"] = s.shift(1) - s.shift(2)
    df["lag_diff_1_5"] = s.shift(1) - s.shift(5)
    df["sin_minute"] = np.sin(2 * np.pi * minute_of_day / SEASONAL_PERIOD)
    df["cos_minute"] = np.cos(2 * np.pi * minute_of_day / SEASONAL_PERIOD)

    return df


def prepare_xy(
    series: np.ndarray,
    timestamps_ms: np.ndarray,
    drop_na: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    feat_df = build_features(series, timestamps_ms)
    X = feat_df.to_numpy(dtype=float)
    y = series.astype(float)
    if drop_na:
        valid = ~np.any(np.isnan(X), axis=1)
        return X[valid], y[valid]
    return X, y


def ts_cv_splits(n: int, n_splits: int, test_size: int):
    for fold in range(n_splits):
        val_end = n - (n_splits - fold - 1) * test_size
        val_start = val_end - test_size
        if val_start <= 0:
            continue
        yield np.arange(0, val_start), np.arange(val_start, val_end)


def effective_cv_folds(n: int, n_splits: int = CV_N_SPLITS, test_size: int = CV_TEST_SIZE) -> list[dict]:
    """Describe the folds actually produced by ``ts_cv_splits`` for ``n`` rows.

    Returns one dict per feasible fold with 0-based row boundaries
    (training rows ``[0, train_end)``, validation rows ``[val_start, val_end)``).
    """
    folds = []
    for train_idx, val_idx in ts_cv_splits(n, n_splits, test_size):
        folds.append({
            "train_start": int(train_idx[0]),
            "train_end": int(train_idx[-1]) + 1,
            "train_rows": int(len(train_idx)),
            "val_start": int(val_idx[0]),
            "val_end": int(val_idx[-1]) + 1,
            "val_rows": int(len(val_idx)),
        })
    return folds


def _xgb_objective(trial, X_train_full, y_train_full):
    import xgboost as xgb  # type: ignore

    params = {
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "eta":              trial.suggest_float("eta", 1e-3, 0.5, log=True),
        "max_depth":        trial.suggest_int("max_depth", 2, 12),
        "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 20.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.001, 100.0, log=True),
        "seed":             RANDOM_SEED,
    }

    n = len(X_train_full)
    fold_maes = []
    for train_idx, val_idx in ts_cv_splits(n, CV_N_SPLITS, CV_TEST_SIZE):
        dtrain = xgb.DMatrix(X_train_full[train_idx], label=y_train_full[train_idx])
        dval   = xgb.DMatrix(X_train_full[val_idx],   label=y_train_full[val_idx])
        model  = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        preds = np.maximum(model.predict(dval), 0.0)
        fold_maes.append(float(np.mean(np.abs(preds - y_train_full[val_idx]))))

    return float(np.mean(fold_maes)) if fold_maes else float("inf")


def train_final_model(X: np.ndarray, y: np.ndarray, best_params: dict):
    import xgboost as xgb  # type: ignore

    params = {
        "objective":   "reg:squarederror",
        "eval_metric": "mae",
        "seed":        RANDOM_SEED,
        **best_params,
    }
    dtrain  = xgb.DMatrix(X, label=y)
    best_n  = best_params.pop("best_n_estimators", 500)
    n_rounds = max(best_n, 200)
    model   = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
    return model


def next_step_features(
    history_series: np.ndarray,
    history_ts_ms: np.ndarray,
    target_ts_ms: int,
) -> np.ndarray:
    """Feature vector for the not-yet-observed target minute ``target_ts_ms``.

    The target minute is appended with an unknown demand (NaN) and its own
    timestamp, and the last row of ``build_features`` is returned.  Because
    every demand feature in ``build_features`` is shifted by at least one
    step, this row uses observations up to and including the last history
    value, never the target demand, and the calendar features of the target
    minute -- exactly the training row whose label is the target demand.
    """
    series = np.append(np.asarray(history_series, dtype=float), np.nan)
    ts = np.append(np.asarray(history_ts_ms, dtype=np.int64), np.int64(target_ts_ms))
    return build_features(series, ts).iloc[-1].to_numpy(dtype=float)


def walk_forward_predict(
    model,
    history_series: np.ndarray,
    history_ts_ms: np.ndarray,
    test_series: np.ndarray,
    test_ts_ms: np.ndarray,
    predict_fn=None,
) -> np.ndarray:
    """Causal one-step walk-forward forecasts for every test minute.

    For test minute ``t`` the features are built by ``next_step_features``
    from the history ending at ``t-1`` and the timestamp of ``t``; the
    observed demand of ``t`` is appended to the history only after the
    forecast has been issued.  ``predict_fn`` (row -> float) replaces the
    XGBoost booster in behavioural tests.
    """
    if predict_fn is None:
        import xgboost as xgb  # type: ignore

        def predict_fn(row: np.ndarray) -> float:
            return float(model.predict(xgb.DMatrix(row.reshape(1, -1)))[0])

    all_series = list(np.asarray(history_series, dtype=float))
    all_ts     = list(np.asarray(history_ts_ms, dtype=np.int64))
    forecasts  = []

    for demand, ts in zip(test_series, test_ts_ms):
        row = next_step_features(
            np.array(all_series, dtype=float), np.array(all_ts, dtype=np.int64), int(ts)
        )
        if np.any(np.isnan(row)):
            forecasts.append(float(all_series[-1]))
        else:
            forecasts.append(max(predict_fn(row), 0.0))
        all_series.append(float(demand))
        all_ts.append(int(ts))

    return np.array(forecasts, dtype=float)


def load_registry(registry_path: Path) -> dict[str, dict]:
    if not registry_path.exists():
        return {}
    try:
        df = pd.read_csv(registry_path)
        return {str(row["service_id"]): row.to_dict() for _, row in df.iterrows()}
    except Exception:
        return {}


def append_registry(registry_path: Path, row: dict) -> None:
    existing = load_registry(registry_path)
    existing[str(row["service_id"])] = row
    pd.DataFrame(list(existing.values())).to_csv(registry_path, index=False)


def train_service(
    path: Path,
    split_definition: dict,
    service_idx: int,
    total: int,
    n_optuna_trials: int = OPTUNA_N_TRIALS,
    use_tpe: bool = False,
    save_models: bool = True,
    models_dir: Path | None = None,
) -> dict:
    import optuna  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = pd.read_parquet(path).sort_values("timestamp")
    service = str(df["msname"].iloc[0])
    split_map = {s["name"]: s for s in split_definition["splits"]}

    _log_service_header(service_idx, total, service)

    def mask(name):
        s = split_map[name]
        return (df["timestamp"] >= s["timestamp_start"]) & (df["timestamp"] < s["timestamp_end_exclusive"])

    # Stage 1: load data and build features
    t0 = time.perf_counter()
    history_df = pd.concat([df[mask("train")], df[mask("calibration")]], ignore_index=True)
    test_df    = df[mask("test")]

    y_history     = history_df[TARGET_COL].to_numpy(dtype=float)
    ts_history_ms = history_df["timestamp"].to_numpy(dtype=np.int64)
    y_test        = test_df[TARGET_COL].to_numpy(dtype=float)
    ts_test_ms    = test_df["timestamp"].to_numpy(dtype=np.int64)

    X_history, y_history_clean = prepare_xy(y_history, ts_history_ms, drop_na=True)
    n_drop = len(y_history) - len(y_history_clean)
    _log_stage(1, 4, "Load & feature engineering",
               time.perf_counter() - t0,
               f"history={len(y_history):,}  test={len(y_test):,}  features={X_history.shape[1]}  dropped={n_drop}")

    # Stage 2: Optuna search
    t0_search = time.perf_counter()
    cv_folds = effective_cv_folds(len(y_history_clean))
    print(f"  Stage 2/4  · Optuna search ({n_optuna_trials} trials, "
          f"{len(cv_folds)} feasible of {CV_N_SPLITS} configured expanding-window folds) ...", flush=True)

    sampler_cls = optuna.samplers.TPESampler if use_tpe else optuna.samplers.RandomSampler
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler_cls(seed=RANDOM_SEED),
    )
    callback = _OptunaProgressCallback(n_optuna_trials, report_every=max(10, n_optuna_trials // 10))
    study.optimize(
        lambda trial: _xgb_objective(trial, X_history, y_history_clean),
        n_trials=n_optuna_trials,
        show_progress_bar=False,
        callbacks=[callback],
    )
    t_search = time.perf_counter() - t0_search
    best_params = study.best_params.copy()
    _log_stage(2, 4, f"Optuna search ({n_optuna_trials} trials)",
               t_search,
               f"best_cv_MAE={study.best_value:.4f}  "
               f"eta={best_params.get('eta', 0):.4f}  "
               f"depth={best_params.get('max_depth', '?')}")

    # Stage 3: Final model
    best_params["best_n_estimators"] = 500
    t0_train = time.perf_counter()
    final_model = train_final_model(X_history, y_history_clean, best_params.copy())
    t_train = time.perf_counter() - t0_train
    _log_stage(3, 4, "Final model training",
               t_train,
               f"n_rounds={final_model.num_boosted_rounds()}")

    # Stage 4: Walk-forward inference
    t0_pred = time.perf_counter()
    forecasts = walk_forward_predict(
        final_model, y_history, ts_history_ms, y_test, ts_test_ms
    )
    t_pred = time.perf_counter() - t0_pred
    _log_stage(4, 4, "Walk-forward inference",
               t_pred,
               f"steps={len(y_test):,}  per_step={t_pred/len(y_test)*1000:.2f}ms")

    errors = forecasts - y_test
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    under = np.maximum(y_test - forecasts, 0.0)

    if save_models and models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{service}_xgb.json"
        final_model.save_model(str(model_path))
        meta = {
            "service":       service,
            "best_params":   study.best_params,
            "best_mae_cv":   float(study.best_value),
            "model_path":    str(model_path),
            "history_rows":  len(y_history),
            "test_rows":     len(y_test),
            "n_features":    X_history.shape[1],
            "cv_configured_n_splits": CV_N_SPLITS,
            "cv_effective_n_splits":  len(cv_folds),
            "cv_folds":      cv_folds,
            "inference_feature_alignment": "target_minute (next_step_features)",
        }
        with open(models_dir / f"{service}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    result = {
        "msname":                    service,
        "model":                     "xgb",
        "best_mae_cv":               float(study.best_value),
        "train_rows":                len(y_history),
        "test_rows":                 len(y_test),
        "n_features":                X_history.shape[1],
        "cv_configured_n_splits":    CV_N_SPLITS,
        "cv_effective_n_splits":     len(cv_folds),
        "mae":                       mae,
        "rmse":                      rmse,
        "underprediction_mae":       float(np.mean(under)),
        "underprediction_p95":       float(np.quantile(under, 0.95)),
        "underprediction_rate":      float(np.mean(forecasts < y_test)),
        "residual_p95":              float(np.quantile(y_test - forecasts, 0.95)),
        "best_eta":                  float(study.best_params.get("eta", float("nan"))),
        "best_max_depth":            int(study.best_params.get("max_depth", -1)),
        "best_subsample":            float(study.best_params.get("subsample", float("nan"))),
        "search_seconds":            t_search,
        "train_seconds":             t_train,
        "inference_seconds_total":   t_pred,
        "inference_seconds_per_step": t_pred / len(y_test),
        "forecasts":                 forecasts,
        "timestamps":                test_df["timestamp"].to_numpy(),
    }
    return result


def _write_report(log_rows: list[dict], t_wall: float, n_trials: int) -> None:
    import datetime

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([r for r in log_rows if "error" not in r])
    err_rows = [r for r in log_rows if "error" in r]

    lines = [
        "# XGBoost Training Report",
        "",
        f"Generated: {now}",
        f"Total wall time: {_fmt_sec(t_wall)}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| `N_LAGS` | {N_LAGS} |",
        f"| `ROLLING_WINDOWS` | {ROLLING_WINDOWS} |",
        f"| `EMA_SPANS` | {EMA_SPANS} |",
        f"| `CV_N_SPLITS` | {CV_N_SPLITS} |",
        f"| `CV_TEST_SIZE` | {CV_TEST_SIZE} |",
        f"| `OPTUNA_N_TRIALS` | {n_trials} |",
        f"| `EARLY_STOPPING_ROUNDS` | {EARLY_STOPPING_ROUNDS} |",
        f"| `RANDOM_SEED` | {RANDOM_SEED} |",
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
            ("best_mae_cv",          "CV MAE (Optuna best)"),
            ("search_seconds",       "Search time (s)"),
            ("train_seconds",        "Train time (s)"),
        ]:
            if col in df.columns:
                s = df[col].dropna()
                lines.append(
                    f"| {label} | {s.min():.4f} | {s.median():.4f} | {s.max():.4f} |"
                )

        lines += [
            "",
            "## Per-Service Results",
            "",
            "| Service | MAE | RMSE | CV MAE | eta | max_depth | subsample | underpred_rate | search(s) | train(s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in df.iterrows():
            lines.append(
                f"| `{row['msname']}` "
                f"| {row.get('mae', float('nan')):.4f} "
                f"| {row.get('rmse', float('nan')):.4f} "
                f"| {row.get('best_mae_cv', float('nan')):.4f} "
                f"| {row.get('best_eta', float('nan')):.4f} "
                f"| {int(row.get('best_max_depth', -1))} "
                f"| {row.get('best_subsample', float('nan')):.3f} "
                f"| {row.get('underprediction_rate', float('nan')):.2%} "
                f"| {row.get('search_seconds', float('nan')):.1f} "
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

    if args.optuna_trials is not None:
        n_trials = args.optuna_trials
    elif large_scale_mode:
        n_trials = OPTUNA_N_TRIALS_LARGE
    else:
        n_trials = OPTUNA_N_TRIALS

    use_tpe = large_scale_mode  # TPE in large-scale, Random in original

    if large_scale_mode:
        ts_dir = Path(args.timeseries_dir)
        out_dir = Path(args.output_dir)
        large_out = out_dir / "large_scale"
        large_out.mkdir(parents=True, exist_ok=True)
        forecasts_dir = large_out / "forecasts" / "xgb"
        forecasts_dir.mkdir(parents=True, exist_ok=True)

        training_log_path = large_out / "xgb_training_log_200.csv"
        registry_path     = large_out / "xgb_registry.csv"

        svc_df = pd.read_csv(args.services_csv)
        service_ids = svc_df["service_id"].tolist()
        if args.limit is not None:
            service_ids = service_ids[:args.limit]

        registry = load_registry(registry_path)
        done_ids = {k for k, v in registry.items() if v.get("status") == "done"}
        pending  = [s for s in service_ids if s not in done_ids]

        total = len(service_ids)
        print(f"\n{'='*72}", flush=True)
        print(f"  XGBoost Large-Scale Training  —  {total} services  ({_ts()})", flush=True)
        print(f"  Already done: {len(done_ids)}, Pending: {len(pending)}", flush=True)
        print(f"  Config: {n_trials} Optuna trials (TPE) · {CV_N_SPLITS} configured CV folds", flush=True)
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
                print(f"  SKIP {service_id}: parquet not found", flush=True)
                append_registry(registry_path, {
                    "service_id": service_id, "status": "failed",
                    "started_at": _ts(), "finished_at": _ts(),
                    "runtime_s": 0, "mae": float("nan"),
                    "error": "parquet not found",
                })
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
                    n_optuna_trials=n_trials,
                    use_tpe=use_tpe,
                    save_models=args.save_models,
                    models_dir=MODELS_DIR if args.save_models else None,
                )
                forecasts  = result.pop("forecasts")
                timestamps = result.pop("timestamps")

                fc_df = pd.DataFrame({
                    "msname":    service_id,
                    "model":     "xgb",
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

                append_registry(registry_path, {
                    "service_id": service_id, "status": "done",
                    "started_at": started_at, "finished_at": _ts(),
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
        ok = [r for r in log_rows if "error" not in r]
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
        print(f"  XGBoost Training  —  {total} services  ({_ts()})", flush=True)
        print(f"  Config: {n_trials} Optuna trials · {CV_N_SPLITS} configured CV folds · {N_LAGS} lags", flush=True)
        print(f"{'='*72}\n", flush=True)

        log_rows      = []
        forecast_rows = []
        t_wall_start  = time.perf_counter()
        service_times = []

        for idx, path in enumerate(service_files, start=1):
            service  = path.stem
            t_svc    = time.perf_counter()
            try:
                result     = train_service(
                    path, split_definition, idx, total,
                    n_optuna_trials=n_trials,
                    use_tpe=False,
                    save_models=True,
                    models_dir=MODELS_DIR,
                )
                forecasts  = result.pop("forecasts")
                timestamps = result.pop("timestamps")
                log_rows.append(result)
                for ts, fc in zip(timestamps, forecasts):
                    forecast_rows.append({"msname": service, "model": "xgb",
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
                log_rows.append({"msname": service, "model": "xgb", "error": str(e)})

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

        _write_report(log_rows, t_wall, n_trials)


if __name__ == "__main__":
    main()
