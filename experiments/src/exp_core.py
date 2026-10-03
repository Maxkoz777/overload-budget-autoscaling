"""
exp_core.py
===========
Shared simulation primitives for supplemental experiments (EXP-2 through EXP-10).
All experiments import from here to ensure consistency.
"""

from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RESULTS_DIR   = Path("experiments/results")
TIMESERIES    = Path("experiments/data/service_timeseries")
SPLITS_PATH   = Path("experiments/data/splits/split_definition.json")

# Default cost coefficients
DEFAULT_COST = dict(c_res=1.0, c_act=0.05, c_vio=10.0, c_inf=0.0, c_train=0.0)


def load_service_data(path: Path, split_def: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (history_df, test_df) for one service parquet file."""
    df = pd.read_parquet(path).sort_values("timestamp")
    smap = {s["name"]: s for s in split_def["splits"]}

    def mask(name):
        s = smap[name]
        return (df["timestamp"] >= s["timestamp_start"]) & \
               (df["timestamp"] < s["timestamp_end_exclusive"])

    history = pd.concat([df[mask("train")], df[mask("calibration")]], ignore_index=True)
    test    = df[mask("test")]
    return history, test


def simulate_policy(
    history: pd.DataFrame,
    test:    pd.DataFrame,
    forecast_array: np.ndarray | None = None,   # None → persistence
    *,
    delta:           float = 0.05,
    W:               int   = 240,
    alpha:           float = 1.0,
    mu:              float = 1.0,
    rho:             float = 0.70,
    guardrail_h:     int   = 2,
    guardrail_gamma: int   = 1,
    use_margin:      bool  = True,
    use_guardrail:   bool  = True,
    c_res:           float = 1.0,
    c_act:           float = 0.05,
    c_vio:           float = 10.0,
    c_inf_per_step:  float = 0.0,
    c_train_total:   float = 0.0,
    demand_override: np.ndarray | None = None,   # for synthetic shift experiments
) -> dict:
    """
    Full policy simulation returning per-step capacities and aggregate metrics.

    Parameters
    ----------
    history, test : DataFrames with cpu_sum and timestamp columns
    forecast_array : pre-computed forecasts (same length as test).
                     If None, uses one-step-ahead persistence.
    demand_override : if not None, replaces test["cpu_sum"] (for synthetic shift).
    """
    hist_y = history["cpu_sum"].to_numpy(dtype=float)
    test_y = (demand_override
              if demand_override is not None
              else test["cpu_sum"].to_numpy(dtype=float))
    n = len(test_y)

    # Initialise residual buffer from history (persistence residuals)
    prev = hist_y[-1]
    residuals: list[float] = list(np.maximum(hist_y[1:] - hist_y[:-1], 0.0))

    overload_hist: list[bool] = []
    soft_hist:     list[bool] = []
    capacities: list[int] = []
    margins:    list[float] = []

    prev_cap = 1

    for i in range(n):
        demand = float(test_y[i])
        if forecast_array is not None:
            forecast = max(float(forecast_array[i]), 0.0)
        else:
            forecast = max(prev, 0.0)

        # Conformal margin
        if use_margin and residuals:
            window = residuals[-W:]
            w      = len(window)
            level  = min(w, int(np.ceil((w + 1) * (1.0 - delta))))
            margin = alpha * float(sorted(window)[level - 1])
        else:
            margin = 0.0
        margins.append(margin)

        nominal = max(int(np.ceil((forecast + margin) / mu)), 1)

        # Guard-rail
        trigger = False
        if use_guardrail and len(overload_hist) >= guardrail_h:
            trigger = (all(overload_hist[-guardrail_h:]) or
                       all(soft_hist[-guardrail_h:]))
        correction = guardrail_gamma if trigger else 0
        cap = nominal + correction
        capacities.append(cap)

        overload = demand > mu * cap
        soft     = demand > rho * mu * cap
        residuals.append(max(demand - forecast, 0.0))
        overload_hist.append(bool(overload))
        soft_hist.append(bool(soft))
        prev     = demand
        prev_cap = cap

    cap_arr = np.array(capacities, dtype=int)
    ol_arr  = test_y > mu * cap_arr

    # Cost components
    T = n
    c_res_total   = c_res  * float(np.sum(cap_arr))
    c_act_total   = c_act  * float(np.sum(np.abs(np.diff(np.concatenate([[capacities[0]], cap_arr])))))
    c_vio_total   = c_vio  * float(np.sum(ol_arr))
    c_inf_total   = c_inf_per_step * T
    c_train_total_ = c_train_total

    # Run-length of consecutive overload
    runs = _runlengths(ol_arr)

    return {
        "capacities":        cap_arr,
        "overload_arr":      ol_arr,
        "margins":           np.array(margins),
        "overload_fraction": float(np.mean(ol_arr)),
        "c_res":             c_res_total,
        "c_act":             c_act_total,
        "c_vio":             c_vio_total,
        "c_inf":             c_inf_total,
        "c_train":           c_train_total_,
        "c_total":           c_res_total + c_act_total + c_vio_total + c_inf_total + c_train_total_,
        "margin_mean":       float(np.mean(margins)),
        "margin_p95":        float(np.quantile(margins, 0.95)),
        "scaling_churn":     int(np.sum(np.abs(np.diff(cap_arr)))),
        "max_overload_run":  int(max(runs)) if runs else 0,
        "run_lengths":       runs,
        "guardrail_rate":    float(np.mean([
            all(overload_hist[max(0,i-guardrail_h):i]) or
            all(soft_hist[max(0,i-guardrail_h):i])
            for i in range(guardrail_h, n)
        ])) if n > guardrail_h else 0.0,
    }


def _runlengths(ol_arr: np.ndarray) -> list[int]:
    """Return list of lengths of consecutive True runs."""
    runs, cur = [], 0
    for v in ol_arr:
        if v:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


def load_all_services(split_def: dict) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Load all 20 services as (name, history_df, test_df)."""
    out = []
    for f in sorted(TIMESERIES.glob("MS_*.parquet")):
        h, t = load_service_data(f, split_def)
        out.append((f.stem, h, t))
    return out


def load_forecast(model: str) -> pd.DataFrame:
    """Load forecast parquet for arima / xgb / lstm."""
    return pd.read_parquet(RESULTS_DIR / f"{model}_forecasts.parquet")


GROUP_MAP = {
    "MS_31285": "G1_stable",   "MS_6298":  "G1_stable",
    "MS_63525": "G1_stable",   "MS_8458":  "G1_stable",
    "MS_70053": "G2_moderate", "MS_29860": "G2_moderate",
    "MS_42222": "G2_moderate", "MS_66711": "G2_moderate",
    "MS_491":   "G3_high_var", "MS_48534": "G3_high_var",
    "MS_19988": "G3_high_var", "MS_21035": "G3_high_var",
    "MS_2024":  "G4_bursty",   "MS_49699": "G4_bursty",
    "MS_14526": "G4_bursty",   "MS_12652": "G4_bursty",
    "MS_51028": "G5_v_bursty", "MS_5201":  "G5_v_bursty",
    "MS_345":   "G5_v_bursty", "MS_25320": "G5_v_bursty",
}
