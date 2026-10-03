from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_forecast(history: pd.Series, horizon: pd.Series) -> np.ndarray:
    """One-step persistence forecast for the horizon, seeded by the last history value."""
    if history.empty:
        raise ValueError("persistence_forecast requires non-empty history")
    values = horizon.to_numpy(dtype=float)
    forecast = np.empty(len(values), dtype=float)
    forecast[0] = float(history.iloc[-1])
    if len(values) > 1:
        forecast[1:] = values[:-1]
    return np.maximum(forecast, 0.0)


def moving_average_forecast(
    history: pd.Series,
    horizon: pd.Series,
    window: int = 60,
) -> np.ndarray:
    """Causal rolling mean forecast, seeded with pre-horizon history."""
    if history.empty:
        raise ValueError("moving_average_forecast requires non-empty history")
    observed = list(history.tail(window).astype(float))
    forecasts = []
    for value in horizon.astype(float):
        forecasts.append(float(np.mean(observed[-window:])))
        observed.append(float(value))
    return np.maximum(np.asarray(forecasts), 0.0)


def seasonal_naive_forecast(
    history: pd.Series,
    horizon: pd.Series,
    seasonal_lag: int = 1440,
) -> np.ndarray:
    """Causal seasonal-naive forecast using the value from one seasonal lag ago."""
    if history.empty:
        raise ValueError("seasonal_naive_forecast requires non-empty history")
    observed = list(history.astype(float))
    fallback = float(history.iloc[-1])
    forecasts = []
    for value in horizon.astype(float):
        if len(observed) >= seasonal_lag:
            forecasts.append(float(observed[-seasonal_lag]))
        else:
            forecasts.append(fallback)
        observed.append(float(value))
    return np.maximum(np.asarray(forecasts), 0.0)


class AutoregressiveRidgeForecaster:
    def __init__(self, lags: list[int] | None = None, alpha: float = 1.0) -> None:
        self.lags = lags or [1, 2, 3, 5, 10, 30, 60, 1440]
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, train: pd.Series) -> "AutoregressiveRidgeForecaster":
        values = train.to_numpy(dtype=float)
        max_lag = max(self.lags)
        if len(values) <= max_lag:
            raise ValueError("not enough training observations for selected lags")

        rows = []
        targets = []
        for idx in range(max_lag, len(values)):
            rows.append([1.0, *[values[idx - lag] for lag in self.lags]])
            targets.append(values[idx])

        x = np.asarray(rows, dtype=float)
        y = np.asarray(targets, dtype=float)
        penalty = self.alpha * np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        return self

    def predict_causal(self, history: pd.Series, horizon: pd.Series) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("forecaster must be fit before prediction")
        observed = list(history.astype(float))
        forecasts = []
        for value in horizon.astype(float):
            features = np.asarray([1.0, *[observed[-lag] for lag in self.lags]], dtype=float)
            forecasts.append(float(features @ self.coef_))
            observed.append(float(value))
        return np.maximum(np.asarray(forecasts), 0.0)
