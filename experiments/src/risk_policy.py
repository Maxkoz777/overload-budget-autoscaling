from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskPolicyConfig:
    delta: float
    window: int = 240
    alpha: float = 1.0
    mu: float = 1.0
    rho: float = 0.70
    guardrail_h: int = 2
    guardrail_gamma: int = 1
    use_margin: bool = True
    use_guardrail: bool = True


def persistence_positive_residuals(history: pd.Series) -> list[float]:
    values = history.to_numpy(dtype=float)
    if len(values) < 2:
        return []
    forecasts = values[:-1]
    actuals = values[1:]
    return np.maximum(actuals - forecasts, 0.0).astype(float).tolist()


def conformal_margin(residuals: list[float], config: RiskPolicyConfig) -> float:
    if not config.use_margin:
        return 0.0
    if not residuals:
        return 0.0
    window = residuals[-config.window :]
    w = len(window)
    level = min(w, int(np.ceil((w + 1) * (1.0 - config.delta))))
    sorted_residuals = np.sort(np.asarray(window, dtype=float))
    return float(config.alpha * sorted_residuals[level - 1])


def simulate_risk_policy(
    history: pd.DataFrame,
    test: pd.DataFrame,
    config: RiskPolicyConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    residuals = persistence_positive_residuals(history["cpu_sum"])
    previous_demand = float(history["cpu_sum"].iloc[-1])
    overload_history: list[bool] = []
    soft_history: list[bool] = []

    capacities = []
    diagnostics = []

    for _, row in test.iterrows():
        demand = float(row["cpu_sum"])
        forecast = max(previous_demand, 0.0)
        margin = conformal_margin(residuals, config)
        nominal_capacity = max(int(np.ceil((forecast + margin) / config.mu)), 1)

        trigger = False
        if config.use_guardrail and len(overload_history) >= config.guardrail_h:
            recent_overload = overload_history[-config.guardrail_h :]
            recent_soft = soft_history[-config.guardrail_h :]
            trigger = all(recent_overload) or all(recent_soft)

        correction = config.guardrail_gamma if trigger else 0
        capacity = nominal_capacity + correction

        overload = demand > config.mu * capacity
        soft = demand > config.rho * config.mu * capacity
        positive_residual = max(demand - forecast, 0.0)
        residuals.append(positive_residual)
        overload_history.append(bool(overload))
        soft_history.append(bool(soft))
        previous_demand = demand
        capacities.append(capacity)

        diagnostics.append(
            {
                "timestamp": int(row["timestamp"]),
                "forecast": forecast,
                "margin": margin,
                "nominal_capacity": nominal_capacity,
                "guardrail_trigger": bool(trigger),
                "guardrail_correction": correction,
                "capacity": capacity,
                "demand": demand,
                "positive_residual": positive_residual,
                "overload": bool(overload),
                "soft_utilization": bool(soft),
            }
        )

    return np.asarray(capacities, dtype=int), pd.DataFrame(diagnostics)
