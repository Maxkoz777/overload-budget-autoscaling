"""
train_lstm.py
=============
Train per-service LSTM forecasters.

Usage (original 20-service mode):
    python3 experiments/src/train_lstm.py

Usage (large-scale 200-service mode):
    python3 experiments/src/train_lstm.py \
        --services-csv experiments/data/splits/selected_services_200.csv \
        --timeseries-dir experiments/data/service_timeseries_200/ \
        --output-dir experiments/results

Output (original mode):
    experiments/models/lstm/<msname>_lstm.pt
    experiments/models/lstm/<msname>_scaler.json
    experiments/results/lstm_forecasts.parquet
    experiments/results/lstm_training_log.csv
    experiments/reports/lstm_training_report.md

Output (large-scale mode):
    experiments/results/large_scale/lstm_training_log_200.csv
    experiments/results/large_scale/forecasts/lstm/<service>.parquet
    experiments/results/large_scale/lstm_registry.csv
    (no model checkpoints saved)

Dependencies:
    pip install torch optuna numpy pandas pyarrow
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-service LSTM forecasters.")
    p.add_argument(
        "--services-csv", default=None,
        help="CSV with service_id column. If given, runs in large-scale mode.",
    )
    p.add_argument(
        "--timeseries-dir", default="experiments/data/service_timeseries",
        help="Directory containing per-service parquets.",
    )
    p.add_argument(
        "--output-dir", default="experiments/results",
        help="Base output directory.",
    )
    p.add_argument(
        "--save-models", action="store_true", default=False,
        help="Save model checkpoints (default: off in large-scale mode).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N services (dry-run).",
    )
    return p.parse_args()


TIME_SERIES_DIR    = Path("experiments/data/service_timeseries")
SPLIT_DEFINITION_PATH = Path("experiments/data/splits/split_definition.json")
MODELS_DIR         = Path("experiments/models/lstm")
RESULTS_DIR        = Path("experiments/results")
REPORTS_DIR        = Path("experiments/reports")
FORECASTS_PATH     = RESULTS_DIR / "lstm_forecasts.parquet"
TRAINING_LOG_PATH  = RESULTS_DIR / "lstm_training_log.csv"
REPORT_PATH        = REPORTS_DIR / "lstm_training_report.md"

TARGET_COL      = "cpu_sum"
SEASONAL_PERIOD = 1440      # minutes per day
RANDOM_SEED     = 42

# Minimum free RAM (GB) required to use MPS safely.
# M1 shares RAM/VRAM; if the system has less than this free, fall back to CPU.
MPS_MIN_FREE_GB = 3.0

# Optional device override. ``None`` keeps the automatic CUDA > MPS > CPU
# selection; "cpu" forces the CPU mode and the CPU search space even when
# an accelerator is visible.
FORCE_DEVICE_MODE: str | None = None

# CPU-only mode — also used as automatic fallback when MPS memory is low.
CPU_LOOKBACKS   = [16, 32, 48]
CPU_UNITS       = [16, 32, 64]
CPU_LAYERS      = [1, 2]
CPU_N_TRIALS    = 15
CPU_MAX_EPOCHS  = 25
CPU_PATIENCE    = 6
CPU_BATCH_SIZES = [256]

# MPS mode (Apple Metal — M1/M2/M3 Mac)
MPS_LOOKBACKS   = [24, 48, 72, 96]
MPS_UNITS       = [32, 64, 128]
MPS_LAYERS      = [1, 2, 3]
MPS_N_TRIALS    = 30
MPS_MAX_EPOCHS  = 60
MPS_PATIENCE    = 8
MPS_BATCH_SIZES = [256, 512]

# CUDA GPU mode (full search space)
GPU_LOOKBACKS   = [24, 48, 72, 96, 120]
GPU_UNITS       = [64, 128, 256]
GPU_LAYERS      = [2, 3]
GPU_N_TRIALS    = 50
GPU_MAX_EPOCHS  = 100
GPU_PATIENCE    = 10
GPU_BATCH_SIZES = [512, 1024]

CV_TEST_SIZE = 2880


SEP = "─" * 72

def _ts() -> str:
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")

# Set by an MPS OOM event: all later services in the same process then run on
# CPU, so exhausted shared RAM does not make every remaining service fail.
_MPS_OOM_DETECTED: bool = False


def _mps_free_gb() -> float:
    """Return available system RAM in GB (proxy for MPS free memory on M1)."""
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return float("inf")  # psutil not installed — assume OK


def _mps_is_functional() -> bool:
    """Probe MPS with a tiny allocation to confirm the backend is usable.

    psutil.virtual_memory().available can show plenty of free RAM while the MPS
    allocator's private pool is exhausted (other processes hold the physical
    pages).  A direct test allocation is the only reliable indicator.
    """
    try:
        import torch  # type: ignore
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return False
        test = torch.zeros(1, device="mps")
        del test
        torch.mps.empty_cache()
        return True
    except Exception:
        return False


def _clear_device_cache(device) -> None:
    """Free cached allocations on MPS or CUDA to prevent fragmentation."""
    try:
        import torch  # type: ignore
        device_str = str(device)
        if device_str == "mps":
            torch.mps.empty_cache()
        elif device_str.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass


def _apply_cpu_config(
    seq_cache_X: dict, seq_cache_y: dict,
    scaled_history: "np.ndarray", cal_history: "np.ndarray",
):
    """(Re)build sequence caches for CPU lookbacks and return CPU config tuple."""
    import torch  # type: ignore
    cpu_device = torch.device("cpu")
    new_X: dict = {}
    new_y: dict = {}
    for lb in CPU_LOOKBACKS:
        if lb not in seq_cache_X:
            Xs, ys = make_sequences(scaled_history, cal_history, lb)
            new_X[lb] = Xs
            new_y[lb] = ys
        else:
            new_X[lb] = seq_cache_X[lb]
            new_y[lb] = seq_cache_y[lb]
    return (
        cpu_device, "cpu",
        CPU_LOOKBACKS, CPU_UNITS, CPU_LAYERS,
        CPU_N_TRIALS, CPU_MAX_EPOCHS, CPU_PATIENCE, CPU_BATCH_SIZES,
        new_X, new_y,
    )

def _fmt_sec(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{sec:.0f}s"
    h, mm = divmod(int(m), 60)
    return f"{h}h{mm}m{sec:.0f}s"

def _log_stage(idx: int, total: int, name: str, elapsed: float, extra: str = "") -> None:
    tag = f"Stage {idx}/{total}"
    extra_str = f"  {extra}" if extra else ""
    print(f"  {tag:12s} · {name:44s} done  ({_fmt_sec(elapsed)}){extra_str}", flush=True)

def _log_service_header(i: int, total: int, service: str, mode: str) -> None:
    print(f"\n{SEP}", flush=True)
    print(f"[{i}/{total}]  {service}  (mode={mode}, {_ts()})", flush=True)

def _log_service_footer(result: dict, elapsed_total: float, eta_sec: float | None) -> None:
    mae  = result.get("mae", float("nan"))
    rmse = result.get("rmse", float("nan"))
    ur   = result.get("underprediction_rate", float("nan"))
    cv   = result.get("best_mae_cv", float("nan"))
    lb   = result.get("lookback", "?")
    nu   = result.get("n_units", "?")
    nl   = result.get("n_layers", "?")
    print(
        f"  ✓ MAE={mae:.4f}  RMSE={rmse:.4f}  underpred_rate={ur:.2%}  "
        f"cv_MAE={cv:.4f}  lookback={lb}  units={nu}  layers={nl}",
        flush=True,
    )
    eta_str = f"  |  ETA ≈ {_fmt_sec(eta_sec)} remaining" if eta_sec is not None else ""
    print(f"  Total: {_fmt_sec(elapsed_total)}{eta_str}", flush=True)

class _OptunaProgressCallback:
    """Prints a compact summary line every N trials."""
    def __init__(self, n_trials: int, report_every: int = 5):
        self.n_trials     = n_trials
        self.report_every = report_every
        self._start       = time.perf_counter()

    def __call__(self, study, trial):
        n = trial.number + 1
        if n % self.report_every == 0 or n == self.n_trials:
            elapsed   = time.perf_counter() - self._start
            rate      = n / elapsed if elapsed > 0 else 0
            remaining = (self.n_trials - n) / rate if rate > 0 else 0
            best      = study.best_value if study.best_trial else float("nan")
            p = trial.params
            print(
                f"    Optuna trial {n:>3}/{self.n_trials}  "
                f"best_MAE={best:.4f}  "
                f"lb={p.get('lookback','?')} units={p.get('n_units','?')} "
                f"layers={p.get('n_layers','?')}  "
                f"elapsed={_fmt_sec(elapsed)}  ETA≈{_fmt_sec(remaining)}",
                flush=True,
            )


class MinMaxScaler:
    def __init__(self, feature_range: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.min_ = 0.0
        self.max_ = 1.0
        self.a, self.b = feature_range

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        self.min_ = float(np.min(x))
        self.max_ = float(np.max(x))
        if self.max_ == self.min_:
            self.max_ = self.min_ + 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.a + (x - self.min_) / (self.max_ - self.min_) * (self.b - self.a)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.a) / (self.b - self.a) * (self.max_ - self.min_) + self.min_

    def to_dict(self) -> dict:
        return {"min_": self.min_, "max_": self.max_, "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "MinMaxScaler":
        s = cls((d["a"], d["b"]))
        s.min_ = d["min_"]
        s.max_ = d["max_"]
        return s


def calendar_features(timestamps_ms: np.ndarray) -> np.ndarray:
    minute_of_day = (timestamps_ms // 60_000).astype(float) % SEASONAL_PERIOD
    sin_feat = np.sin(2 * np.pi * minute_of_day / SEASONAL_PERIOD)
    cos_feat = np.cos(2 * np.pi * minute_of_day / SEASONAL_PERIOD)
    return np.column_stack([sin_feat, cos_feat])


def _build_model(lookback: int, n_extra_features: int, n_units: int, n_layers: int, dropout: float):
    import torch.nn as nn  # type: ignore

    class LSTMForecaster(nn.Module):
        def __init__(self):
            super().__init__()
            input_size = 1 + n_extra_features
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=n_units,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(n_units, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out).squeeze(-1)

    return LSTMForecaster()


def make_sequences(
    scaled_y: np.ndarray,
    cal_feat: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where X.shape = (N, lookback, 1+n_cal)."""
    n_cal = cal_feat.shape[1]
    X, y  = [], []
    for i in range(lookback, len(scaled_y)):
        seq_y = scaled_y[i - lookback:i].reshape(-1, 1)
        seq_c = cal_feat[i - lookback:i]
        X.append(np.concatenate([seq_y, seq_c], axis=1))
        y.append(scaled_y[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _run_train(
    X_train, y_train, X_val, y_val,
    n_units, n_layers, dropout, lr, batch_size,
    max_epochs, patience, device, use_amp
):
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    torch.manual_seed(RANDOM_SEED)
    n_extra = X_train.shape[2] - 1
    model   = _build_model(X_train.shape[1], n_extra, n_units, n_layers, dropout).to(device)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion   = nn.L1Loss()
    scaler_amp  = torch.cuda.amp.GradScaler() if use_amp else None

    Xt = torch.tensor(X_train).to(device)
    yt = torch.tensor(y_train).to(device)
    Xv = torch.tensor(X_val).to(device)
    yv = torch.tensor(y_val).to(device)

    dataset = TensorDataset(Xt, yt)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    no_improve    = 0
    best_state    = None

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(xb), yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(Xv), yv).item())
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            no_improve    = 0
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def _lstm_objective(
    trial,
    X_seq, y_seq,
    n_splits, test_size,
    lookbacks, units, layers, batch_sizes,
    max_epochs, patience, device, use_amp,
):
    lookback   = trial.suggest_categorical("lookback",   lookbacks)
    n_units    = trial.suggest_categorical("n_units",    units)
    n_layers   = trial.suggest_categorical("n_layers",   layers)
    dropout    = trial.suggest_float("dropout", 0.1, 0.4)
    lr         = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", batch_sizes)

    X_full = X_seq.get(lookback)
    y_full = y_seq.get(lookback)
    if X_full is None:
        return float("inf")

    n_full    = len(y_full)
    fold_maes = []
    for fold in range(n_splits):
        val_end   = n_full - (n_splits - fold - 1) * test_size
        val_start = val_end - test_size
        if val_start <= 0:
            continue
        X_tr, y_tr = X_full[:val_start], y_full[:val_start]
        X_vl, y_vl = X_full[val_start:val_end], y_full[val_start:val_end]
        if len(X_tr) == 0 or len(X_vl) == 0:
            continue
        try:
            _, val_loss = _run_train(
                X_tr, y_tr, X_vl, y_vl,
                n_units, n_layers, dropout, lr, batch_size,
                max_epochs, patience, device, use_amp,
            )
            fold_maes.append(val_loss)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM on MPS/CUDA: release cache and report as bad trial
                _clear_device_cache(device)
                return float("inf")
            raise

    return float(np.mean(fold_maes)) if fold_maes else float("inf")


def _walk_forward_predict_lstm(
    model,
    scaler: MinMaxScaler,
    scaled_history: np.ndarray,
    cal_history: np.ndarray,
    y_test: np.ndarray,
    ts_test_ms: np.ndarray,
    lookback: int,
    device,
) -> np.ndarray:
    import torch  # type: ignore

    model.eval()
    all_scaled  = list(scaled_history)
    all_cal     = list(cal_history)
    ts_test_cal = calendar_features(ts_test_ms)
    forecasts   = []

    with torch.no_grad():
        for demand, cal_row in zip(y_test, ts_test_cal):
            if len(all_scaled) < lookback:
                forecasts.append(float(scaler.inverse_transform(np.array([all_scaled[-1]]))[0]))
                all_scaled.append(float(scaler.transform(np.array([demand]))[0]))
                all_cal.append(cal_row)
                continue

            seq_y = np.array(all_scaled[-lookback:], dtype=np.float32).reshape(-1, 1)
            seq_c = np.array(all_cal[-lookback:],    dtype=np.float32)
            seq   = np.concatenate([seq_y, seq_c], axis=1)
            x     = torch.tensor(seq[np.newaxis]).to(device)
            pred_scaled = float(model(x).item())
            pred = float(scaler.inverse_transform(np.array([pred_scaled]))[0])
            forecasts.append(max(pred, 0.0))

            all_scaled.append(float(scaler.transform(np.array([demand]))[0]))
            all_cal.append(cal_row)

    return np.array(forecasts, dtype=float)


def train_service(
    path: Path,
    split_definition: dict,
    service_idx: int,
    total: int,
    save_models: bool = True,
) -> dict:
    import torch  # type: ignore
    import optuna  # type: ignore

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    global _MPS_OOM_DETECTED

    # Flush any stale MPS allocations from the previous service
    _clear_device_cache(torch.device("mps") if (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ) else torch.device("cpu"))

    # Device selection: CUDA > MPS (if healthy) > CPU
    # Three-layer check for MPS:
    #   1. No previous OOM detected in this process (_MPS_OOM_DETECTED flag)
    #   2. Enough free RAM via psutil (coarse check)
    #   3. A tiny test allocation actually succeeds (_mps_is_functional probe)
    if FORCE_DEVICE_MODE is not None and FORCE_DEVICE_MODE != "cpu":
        raise ValueError(f"unsupported FORCE_DEVICE_MODE={FORCE_DEVICE_MODE!r}")
    if FORCE_DEVICE_MODE == "cpu":
        device = torch.device("cpu")
        mode   = "cpu"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        mode   = "cuda"
    elif (
        not _MPS_OOM_DETECTED
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and _mps_free_gb() >= MPS_MIN_FREE_GB
        and _mps_is_functional()
    ):
        device = torch.device("mps")
        mode   = "mps"
    else:
        if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            reason = "global OOM flag set" if _MPS_OOM_DETECTED else "probe failed / low RAM"
            print(f"  ⚠  MPS available but bypassed ({reason}) — using CPU", flush=True)
        device = torch.device("cpu")
        mode   = "cpu"

    use_amp = mode == "cuda"
    is_gpu  = mode in ("cuda", "mps")

    if mode == "cuda":
        lookbacks, units, layers = GPU_LOOKBACKS, GPU_UNITS, GPU_LAYERS
        n_trials, max_epochs, patience = GPU_N_TRIALS, GPU_MAX_EPOCHS, GPU_PATIENCE
        batch_sizes = GPU_BATCH_SIZES
    elif mode == "mps":
        lookbacks, units, layers = MPS_LOOKBACKS, MPS_UNITS, MPS_LAYERS
        n_trials, max_epochs, patience = MPS_N_TRIALS, MPS_MAX_EPOCHS, MPS_PATIENCE
        batch_sizes = MPS_BATCH_SIZES
    else:
        lookbacks, units, layers = CPU_LOOKBACKS, CPU_UNITS, CPU_LAYERS
        n_trials, max_epochs, patience = CPU_N_TRIALS, CPU_MAX_EPOCHS, CPU_PATIENCE
        batch_sizes = CPU_BATCH_SIZES

    df      = pd.read_parquet(path).sort_values("timestamp")
    service = str(df["msname"].iloc[0])
    split_map = {s["name"]: s for s in split_definition["splits"]}

    _log_service_header(service_idx, total, service, mode)

    def mask(name):
        s = split_map[name]
        return (df["timestamp"] >= s["timestamp_start"]) & (df["timestamp"] < s["timestamp_end_exclusive"])

    # Stage 1: load and split data
    t0 = time.perf_counter()
    history_df = pd.concat([df[mask("train")], df[mask("calibration")]], ignore_index=True)
    test_df    = df[mask("test")]

    y_history     = history_df[TARGET_COL].to_numpy(dtype=float)
    ts_history_ms = history_df["timestamp"].to_numpy(dtype=np.int64)
    y_test        = test_df[TARGET_COL].to_numpy(dtype=float)
    ts_test_ms    = test_df["timestamp"].to_numpy(dtype=np.int64)

    train_split  = split_map["train"]
    train_mask   = (history_df["timestamp"] >= train_split["timestamp_start"]) & \
                   (history_df["timestamp"] <  train_split["timestamp_end_exclusive"])
    y_train_only = history_df.loc[train_mask, TARGET_COL].to_numpy(dtype=float)
    scaler       = MinMaxScaler(feature_range=(-1.0, 1.0))
    scaler.fit(y_train_only)

    scaled_history = scaler.transform(y_history)
    cal_history    = calendar_features(ts_history_ms)
    _log_stage(1, 5, "Load data & fit scaler",
               time.perf_counter() - t0,
               f"history={len(y_history):,}  test={len(y_test):,}  "
               f"scale=[{scaler.min_:.3f}, {scaler.max_:.3f}]")

    # Stage 2: build sequence caches
    t0 = time.perf_counter()
    seq_cache_X: dict[int, np.ndarray] = {}
    seq_cache_y: dict[int, np.ndarray] = {}
    for lb in lookbacks:
        Xs, ys = make_sequences(scaled_history, cal_history, lb)
        seq_cache_X[lb] = Xs
        seq_cache_y[lb] = ys
    _log_stage(2, 5, "Build sequence caches",
               time.perf_counter() - t0,
               f"lookbacks={lookbacks}  "
               f"max_seqs={max(len(v) for v in seq_cache_y.values()):,}")

    # Stage 3: Optuna hyperparameter search
    t0_search = time.perf_counter()
    print(
        f"  Stage 3/5  · Optuna search  "
        f"({n_trials} trials, 2-fold CV, device={device}) ...",
        flush=True,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    callback = _OptunaProgressCallback(n_trials, report_every=max(1, n_trials // 10))
    study.optimize(
        lambda trial: _lstm_objective(
            trial, seq_cache_X, seq_cache_y,
            2, CV_TEST_SIZE, lookbacks, units, layers, batch_sizes,
            max_epochs, patience, device, use_amp,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[callback],
    )
    t_search = time.perf_counter() - t0_search

    # If every Optuna trial returned inf (MPS OOM), fall back to CPU and retry
    if study.best_value == float("inf") and mode == "mps":
        _MPS_OOM_DETECTED = True
        _clear_device_cache(device)
        print(
            f"  ⚠  All {n_trials} MPS trials failed (OOM) — switching to CPU and retrying",
            flush=True,
        )
        (device, mode,
         lookbacks, units, layers,
         n_trials, max_epochs, patience, batch_sizes,
         seq_cache_X, seq_cache_y) = _apply_cpu_config(
            seq_cache_X, seq_cache_y, scaled_history, cal_history
        )
        is_gpu = False
        use_amp = False

        cpu_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        t0_cpu = time.perf_counter()
        cpu_cb  = _OptunaProgressCallback(n_trials, report_every=max(1, n_trials // 10))
        cpu_study.optimize(
            lambda trial: _lstm_objective(
                trial, seq_cache_X, seq_cache_y,
                2, CV_TEST_SIZE, lookbacks, units, layers, batch_sizes,
                max_epochs, patience, device, use_amp,
            ),
            n_trials=n_trials,
            show_progress_bar=False,
            callbacks=[cpu_cb],
        )
        t_search += time.perf_counter() - t0_cpu
        study = cpu_study

    best = study.best_params
    _log_stage(3, 5, f"Optuna search ({n_trials} trials, device={device})",
               t_search,
               f"best_cv_MAE={study.best_value:.4f}  "
               f"lookback={best['lookback']}  units={best['n_units']}  "
               f"layers={best['n_layers']}")

    # Stage 4: final model training on full history
    t0_train = time.perf_counter()
    lb     = best["lookback"]
    X_full = seq_cache_X[lb]
    y_full = seq_cache_y[lb]
    print(
        f"  Stage 4/5  · Final model training  "
        f"(lookback={lb}, units={best['n_units']}, layers={best['n_layers']}) ...",
        flush=True,
    )
    try:
        final_model, _ = _run_train(
            X_full, y_full, X_full[-CV_TEST_SIZE:], y_full[-CV_TEST_SIZE:],
            best["n_units"], best["n_layers"], best["dropout"],
            best["lr"], best["batch_size"],
            max_epochs, patience, device, use_amp,
        )
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        # MPS OOM during final training — fall back to CPU
        _MPS_OOM_DETECTED = True
        _clear_device_cache(device)
        print(
            f"  ⚠  Stage 4 OOM on {device} — retrying final training on CPU",
            flush=True,
        )
        device  = torch.device("cpu")
        mode    = "cpu"
        is_gpu  = False
        use_amp = False
        # Ensure X_full / y_full are available for CPU lookback
        if best["lookback"] not in seq_cache_X:
            X_full, y_full = make_sequences(scaled_history, cal_history, best["lookback"])
        else:
            X_full = seq_cache_X[best["lookback"]]
            y_full = seq_cache_y[best["lookback"]]
        final_model, _ = _run_train(
            X_full, y_full, X_full[-CV_TEST_SIZE:], y_full[-CV_TEST_SIZE:],
            best["n_units"], best["n_layers"], best["dropout"],
            best["lr"], best["batch_size"],
            max_epochs, patience, device, use_amp,
        )

    t_train = time.perf_counter() - t0_train
    _log_stage(4, 5, "Final model training", t_train)

    # Stage 5: walk-forward inference
    t0_pred = time.perf_counter()
    forecasts = _walk_forward_predict_lstm(
        final_model, scaler,
        scaled_history, cal_history,
        y_test, ts_test_ms,
        lb, device,
    )
    t_pred = time.perf_counter() - t0_pred
    _log_stage(5, 5, "Walk-forward inference",
               t_pred,
               f"steps={len(y_test):,}  per_step={t_pred/len(y_test)*1000:.2f}ms")

    errors = forecasts - y_test
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    under  = np.maximum(y_test - forecasts, 0.0)

    # Model checkpoints are skipped in large-scale mode unless --save-models.
    model_path  = MODELS_DIR / f"{service}_lstm.pt"
    scaler_path = MODELS_DIR / f"{service}_scaler.json"
    if save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), str(model_path))
        with open(scaler_path, "w") as f:
            json.dump(scaler.to_dict(), f)
        meta = {
            "service":       service,
            "best_params":   best,
            "best_mae_cv":   float(study.best_value),
            "device":        str(device),
            "mode":          mode,
            "gpu_mode":      is_gpu,
            "model_path":    str(model_path),
            "scaler_path":   str(scaler_path),
            "history_rows":  len(y_history),
            "test_rows":     len(y_test),
        }
        with open(MODELS_DIR / f"{service}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return {
        "msname":                    service,
        "model":                     "lstm",
        "device_mode":               mode,
        "gpu_mode":                  is_gpu,
        "lookback":                  lb,
        "n_units":                   best["n_units"],
        "n_layers":                  best["n_layers"],
        "dropout":                   best["dropout"],
        "lr":                        best["lr"],
        "batch_size":                best["batch_size"],
        "best_mae_cv":               float(study.best_value),
        "train_rows":                len(y_history),
        "test_rows":                 len(y_test),
        "mae":                       mae,
        "rmse":                      rmse,
        "underprediction_mae":       float(np.mean(under)),
        "underprediction_p95":       float(np.quantile(under, 0.95)),
        "underprediction_rate":      float(np.mean(forecasts < y_test)),
        "residual_p95":              float(np.quantile(y_test - forecasts, 0.95)),
        "search_seconds":            t_search,
        "train_seconds":             t_train,
        "inference_seconds_total":   t_pred,
        "inference_seconds_per_step": t_pred / len(y_test),
        "forecasts":                 forecasts,
        "timestamps":                test_df["timestamp"].to_numpy(),
    }


def _write_report(log_rows: list[dict], t_wall: float) -> None:
    import datetime

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok_rows  = [r for r in log_rows if "error" not in r]
    err_rows = [r for r in log_rows if "error" in r]
    df = pd.DataFrame(ok_rows) if ok_rows else pd.DataFrame()

    mode_label = ok_rows[0].get("device_mode", "unknown") if ok_rows else "unknown"

    lines = [
        "# LSTM Training Report",
        "",
        f"Generated: {now}",
        f"Total wall time: {_fmt_sec(t_wall)}",
        f"Device mode: `{mode_label}`",
        "",
        "## Configuration",
        "",
        "| Mode | Parameter | Value |",
        "|---|---|---|",
    ]
    if mode_label == "cuda":
        cfg = [
            ("lookbacks", GPU_LOOKBACKS), ("units", GPU_UNITS), ("layers", GPU_LAYERS),
            ("n_trials", GPU_N_TRIALS), ("max_epochs", GPU_MAX_EPOCHS),
            ("patience", GPU_PATIENCE), ("batch_sizes", GPU_BATCH_SIZES),
        ]
    elif mode_label == "mps":
        cfg = [
            ("lookbacks", MPS_LOOKBACKS), ("units", MPS_UNITS), ("layers", MPS_LAYERS),
            ("n_trials", MPS_N_TRIALS), ("max_epochs", MPS_MAX_EPOCHS),
            ("patience", MPS_PATIENCE), ("batch_sizes", MPS_BATCH_SIZES),
        ]
    else:
        cfg = [
            ("lookbacks", CPU_LOOKBACKS), ("units", CPU_UNITS), ("layers", CPU_LAYERS),
            ("n_trials", CPU_N_TRIALS), ("max_epochs", CPU_MAX_EPOCHS),
            ("patience", CPU_PATIENCE), ("batch_sizes", CPU_BATCH_SIZES),
        ]
    for k, v in cfg:
        lines.append(f"| `{mode_label}` | `{k}` | {v} |")
    lines += [
        f"| all | `CV_TEST_SIZE` | {CV_TEST_SIZE} |",
        f"| all | `SEASONAL_PERIOD` | {SEASONAL_PERIOD} min |",
        f"| all | `RANDOM_SEED` | {RANDOM_SEED} |",
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
            ("best_mae_cv",          "CV MAE (best trial)"),
            ("search_seconds",       "Search time (s)"),
            ("train_seconds",        "Final train time (s)"),
        ]:
            if col in df.columns:
                s = df[col].dropna()
                lines.append(
                    f"| {label} | {s.min():.4f} | {s.median():.4f} | {s.max():.4f} |"
                )

        # Hyperparameter distribution tables
        for col, label in [
            ("lookback",  "Best Lookback Distribution"),
            ("n_units",   "Best Units Distribution"),
            ("n_layers",  "Best Layers Distribution"),
        ]:
            if col in df.columns:
                counts = df[col].value_counts().sort_index()
                lines += ["", f"## {label}", ""]
                lines.append(f"| {col} | Count |")
                lines.append("|---:|---:|")
                for val, cnt in counts.items():
                    lines.append(f"| {val} | {cnt} |")

        lines += [
            "",
            "## Per-Service Results",
            "",
            "| Service | MAE | RMSE | CV MAE | lookback | units | layers | dropout | lr | underpred_rate | search(s) | train(s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in df.iterrows():
            lines.append(
                f"| `{row['msname']}` "
                f"| {row.get('mae', float('nan')):.4f} "
                f"| {row.get('rmse', float('nan')):.4f} "
                f"| {row.get('best_mae_cv', float('nan')):.4f} "
                f"| {row.get('lookback', '?')} "
                f"| {row.get('n_units', '?')} "
                f"| {row.get('n_layers', '?')} "
                f"| {row.get('dropout', float('nan')):.3f} "
                f"| {row.get('lr', float('nan')):.2e} "
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


def _load_registry(registry_path: Path) -> set[str]:
    """Return set of service_ids already marked 'done'."""
    if not registry_path.exists():
        return set()
    df = pd.read_csv(registry_path)
    return set(df[df["status"] == "done"]["service_id"].tolist())


def _update_registry(registry_path: Path, service_id: str, status: str,
                     started_at: str, finished_at: str, runtime_s: float,
                     mae: float = float("nan"), error: str = "") -> None:
    """Append or update a row in the registry CSV (incremental write)."""
    row = pd.DataFrame([{
        "service_id": service_id, "status": status,
        "started_at": started_at, "finished_at": finished_at,
        "runtime_s": round(runtime_s, 1), "mae": mae, "error": error,
    }])
    header = not registry_path.exists()
    row.to_csv(registry_path, mode="a", header=header, index=False)


def main() -> None:
    import datetime

    args = parse_args()
    large_scale = args.services_csv is not None

    split_definition = json.loads(SPLIT_DEFINITION_PATH.read_text())

    if large_scale:
        svc_df        = pd.read_csv(args.services_csv)
        service_ids   = svc_df["service_id"].tolist()
        ts_dir        = Path(args.timeseries_dir)
        out_dir       = Path(args.output_dir) / "large_scale"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "forecasts" / "lstm").mkdir(parents=True, exist_ok=True)
        log_path      = out_dir / "lstm_training_log_200.csv"
        registry_path = out_dir / "lstm_registry.csv"
        save_models   = args.save_models   # False by default in large-scale
        done_ids      = _load_registry(registry_path)
        if args.limit:
            service_ids = service_ids[: args.limit]
        service_files = [ts_dir / f"{sid}.parquet" for sid in service_ids
                         if (ts_dir / f"{sid}.parquet").exists()]
    else:
        ts_dir        = TIME_SERIES_DIR
        service_files = sorted(ts_dir.glob("MS_*.parquet"))
        log_path      = FORECASTS_PATH.parent / "lstm_training_log.csv"
        registry_path = None
        done_ids      = set()
        save_models   = True   # original mode always saves models
        if args.limit:
            service_files = service_files[: args.limit]

    total = len(service_files)

    # Detect device for header
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            _mode = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _mode = "mps"
        else:
            _mode = "cpu"
        _n_trials = {"cuda": GPU_N_TRIALS, "mps": MPS_N_TRIALS, "cpu": CPU_N_TRIALS}[_mode]
        _max_ep   = {"cuda": GPU_MAX_EPOCHS, "mps": MPS_MAX_EPOCHS, "cpu": CPU_MAX_EPOCHS}[_mode]
    except ImportError:
        _mode, _n_trials, _max_ep = "cpu", CPU_N_TRIALS, CPU_MAX_EPOCHS

    mode_str = "large-scale (no checkpoints)" if large_scale else "original"
    print(f"\n{'═'*72}")
    print(f"  LSTM Training  —  {total} services  ({_ts()})  [{mode_str}]")
    print(f"  Device: {_mode}  |  Optuna trials: {_n_trials}  |  Max epochs: {_max_ep}")
    if large_scale:
        skip_n = len(done_ids)
        print(f"  Resume: {skip_n} already done, {total - skip_n} to run")
    print(f"{'═'*72}\n")

    log_rows      = []
    forecast_rows = []
    t_wall_start  = time.perf_counter()
    service_times = []

    for idx, path in enumerate(service_files, start=1):
        service   = path.stem
        started   = datetime.datetime.now().strftime("%H:%M:%S")

        # Skip already-done services (large-scale resume)
        if large_scale and service in done_ids:
            print(f"  [skip] {service} (already done)", flush=True)
            continue

        t_svc = time.perf_counter()
        try:
            result     = train_service(path, split_definition, idx, total,
                                       save_models=save_models)
            forecasts  = result.pop("forecasts")
            timestamps = result.pop("timestamps")
            elapsed_svc = time.perf_counter() - t_svc
            service_times.append(elapsed_svc)

            if large_scale:
                # Write per-service forecast parquet immediately
                fc_df = pd.DataFrame({
                    "msname": service, "model": "lstm",
                    "timestamp": timestamps.astype(int),
                    "forecast":  forecasts,
                })
                fc_path = Path(args.output_dir) / "large_scale" / "forecasts" / "lstm" / f"{service}.parquet"
                fc_df.to_parquet(fc_path, index=False)

                # Append metrics row to training log immediately
                metrics_row = {k: v for k, v in result.items()}
                metrics_row["msname"] = service
                header = not log_path.exists()
                pd.DataFrame([metrics_row]).to_csv(log_path, mode="a", header=header, index=False)

                finished = datetime.datetime.now().strftime("%H:%M:%S")
                _update_registry(registry_path, service, "done",
                                  started, finished, elapsed_svc,
                                  mae=result.get("mae", float("nan")))
                done_ids.add(service)
            else:
                log_rows.append(result)
                for ts, fc in zip(timestamps, forecasts):
                    forecast_rows.append({"msname": service, "model": "lstm",
                                          "timestamp": int(ts), "forecast": fc})

            remaining = total - idx
            eta       = (sum(service_times) / len(service_times)) * remaining if remaining > 0 else None
            _log_service_footer(result, elapsed_svc, eta)

        except Exception as e:
            import traceback
            elapsed_svc = time.perf_counter() - t_svc
            print(f"  ✗ ERROR ({_fmt_sec(elapsed_svc)}): {e}", flush=True)
            traceback.print_exc()
            err_row = {"msname": service, "model": "lstm", "error": str(e)}
            if large_scale:
                finished = datetime.datetime.now().strftime("%H:%M:%S")
                _update_registry(registry_path, service, "failed",
                                  started, finished, elapsed_svc, error=str(e))
                header = not log_path.exists()
                pd.DataFrame([err_row]).to_csv(log_path, mode="a", header=header, index=False)
            else:
                log_rows.append(err_row)
        finally:
            try:
                import torch  # type: ignore
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    t_wall = time.perf_counter() - t_wall_start

    if not large_scale:
        # Original mode: write combined outputs at the end
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(TRAINING_LOG_PATH, index=False)
        if forecast_rows:
            forecast_df = pd.DataFrame(forecast_rows)
            forecast_df.to_parquet(FORECASTS_PATH, index=False)
        _write_report(log_rows, t_wall)

    ok  = [r for r in (log_rows if not large_scale else
                        pd.read_csv(log_path).to_dict("records") if log_path.exists() else [])
           if "error" not in r]
    print(f"\n{'═'*72}")
    print(f"  Wall time: {_fmt_sec(t_wall)}")
    if large_scale:
        done_count   = sum(1 for _ in (Path(args.output_dir) / "large_scale" / "forecasts" / "lstm").glob("*.parquet"))
        print(f"  Forecast shards written: {done_count}")
        print(f"  Registry: {registry_path}")
        print(f"  Training log: {log_path}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
