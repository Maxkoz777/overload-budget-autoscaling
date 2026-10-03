#!/usr/bin/env python3
"""Recompute focused-tier ARIMA with a verified one-step causal protocol.

The archived ARIMA run mixed horizons from one minute to one day and passed
Fourier regressors through the obsolete ``exogenous=`` keyword.  This script
keeps the archived, pre-test-selected order and Fourier K fixed, fits with the
current ``X=`` API, checks the fitted exogenous dimension, and advances the
state after every observed minute without refitting coefficients.

Run from any directory with::

    python3 audit/recompute_arima_causal.py --workers 4

To redraw only the residual diagnostic from saved focused outputs::

    python3 audit/recompute_arima_causal.py --diagnostic-figure-only
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
EXP = RESEARCH / "experiments"
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from train_arima import make_fourier_matrix, minute_index  # noqa: E402


OUT = PAPER / "audit" / "arima_causal_results"
DATA = EXP / "data" / "service_timeseries"
SPLIT_PATH = EXP / "data" / "splits" / "split_definition.json"
LOG_PATH = EXP / "results" / "arima_training_log.csv"
W = 240
DELTAS = (0.10, 0.05, 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--diagnostic-figure-only",
        action="store_true",
        help="redraw only the focused residual diagnostic from saved causal results",
    )
    return parser.parse_args()


def split_frame(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    return frame[
        frame.timestamp.ge(spec["timestamp_start"])
        & frame.timestamp.lt(spec["timestamp_end_exclusive"])
    ].copy()


def fit_verified(y: np.ndarray, timestamps: pd.Series, order: tuple[int, int, int], k: int):
    from pmdarima import ARIMA

    X = make_fourier_matrix(minute_index(timestamps), k, 1440)
    # Explicitly omit a drift/intercept.  This makes the state update
    # identifiable when statsmodels clones a single-row Fourier design and is
    # held fixed for every service (it is not selected on test outcomes).
    model = ARIMA(order=order, with_intercept=False, suppress_warnings=True, maxiter=50).fit(y, X=X)
    k_exog = int(model.arima_res_.model.k_exog)
    if not bool(model.fit_with_exog_) or k_exog != 2 * k:
        raise AssertionError(f"Fourier regressors not fitted: fit_with_exog={model.fit_with_exog_}, k_exog={k_exog}, K={k}")
    names = list(model.arima_res_.param_names)
    exog_names = [name for name in names if name.startswith("x")]
    if len(exog_names) != 2 * k:
        raise AssertionError(f"expected {2*k} exogenous coefficients, found {exog_names}")
    return model, X, names


def causal_forecast(result, y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, object, int]:
    forecasts: list[float] = []
    updates = 0
    current = result
    for actual, row in zip(y, X, strict=True):
        row2 = row.reshape(1, -1)
        predicted = float(np.asarray(current.forecast(steps=1, exog=row2)).reshape(-1)[0])
        forecasts.append(max(predicted, 0.0))
        current = current.extend(endog=np.asarray([actual]), exog=row2)
        updates += 1
    return np.asarray(forecasts, dtype=float), current, updates


def acf_mean_1_5(values: np.ndarray) -> tuple[float, float]:
    if np.std(values) <= 1e-12:
        return 0.0, 0.0
    acfs = [float(pd.Series(values).autocorr(lag=lag)) for lag in range(1, 6)]
    return acfs[0], float(np.nanmean(acfs))


def simulate_policy(
    demand: np.ndarray,
    forecasts: np.ndarray,
    initial_scores: np.ndarray,
    delta: float,
) -> np.ndarray:
    scores = initial_scores.astype(float).tolist()
    overload_history: list[bool] = []
    soft_history: list[bool] = []
    capacities: list[int] = []
    for actual, forecast in zip(demand, forecasts, strict=True):
        window = np.asarray(scores[-W:], dtype=float)
        level = min(len(window), int(np.ceil((len(window) + 1) * (1.0 - delta))))
        margin = float(np.partition(window, level - 1)[level - 1])
        nominal = max(1, int(np.ceil(max(forecast, 0.0) + margin)))
        trigger = False
        if len(overload_history) >= 2:
            trigger = all(overload_history[-2:]) or all(soft_history[-2:])
        capacity = nominal + int(trigger)
        overload = bool(actual > capacity)
        soft = bool(actual > 0.70 * capacity)
        capacities.append(capacity)
        scores.append(max(float(actual - forecast), 0.0))
        overload_history.append(overload)
        soft_history.append(soft)
    return np.asarray(capacities, dtype=int)


def cost(demand: np.ndarray, capacity: np.ndarray, observed_capacity: np.ndarray) -> tuple[float, float]:
    def total(cap: np.ndarray) -> float:
        overload = demand > cap
        action = np.abs(np.diff(cap, prepend=cap[0]))
        return float(cap.sum() + 0.05 * action.sum() + 10.0 * overload.sum())

    value = total(capacity.astype(float))
    reference = total(np.maximum(np.ceil(observed_capacity), 1).astype(float))
    return value, value / reference


def run_service(task: tuple[str, int, tuple[int, int, int], dict]) -> dict[str, object]:
    service, k, order, split = task
    started = time.perf_counter()
    frame = pd.read_parquet(DATA / f"{service}.parquet").sort_values("timestamp")
    specs = {item["name"]: item for item in split["splits"]}
    train = split_frame(frame, specs["train"])
    calibration = split_frame(frame, specs["calibration"])
    test = split_frame(frame, specs["test"])
    history = pd.concat([train, calibration], ignore_index=True)

    cal_model, _, _ = fit_verified(train.cpu_sum.to_numpy(float), train.timestamp, order, k)
    X_cal = make_fourier_matrix(minute_index(calibration.timestamp), k, 1440)
    cal_forecast, cal_result, cal_updates = causal_forecast(
        cal_model.arima_res_, calibration.cpu_sum.to_numpy(float), X_cal
    )

    test_model, _, param_names = fit_verified(history.cpu_sum.to_numpy(float), history.timestamp, order, k)
    X_test = make_fourier_matrix(minute_index(test.timestamp), k, 1440)
    initial_prediction = float(np.asarray(test_model.arima_res_.forecast(1, exog=X_test[:1])).reshape(-1)[0])
    test_forecast, final_result, test_updates = causal_forecast(
        test_model.arima_res_, test.cpu_sum.to_numpy(float), X_test
    )
    counterfactual_next = float(np.asarray(final_result.forecast(1, exog=X_test[-1:])).reshape(-1)[0])

    actual = test.cpu_sum.to_numpy(float)
    signed = actual - test_forecast
    positive = np.maximum(signed, 0.0)
    acf1, acf15 = acf_mean_1_5(positive)
    diagnostics = {
        "service_id": service,
        "best_k": k,
        "order": str(order),
        "fit_with_exog": bool(test_model.fit_with_exog_),
        "k_exog": int(test_model.arima_res_.model.k_exog),
        "expected_k_exog": 2 * k,
        "exogenous_coefficient_names": ";".join(name for name in param_names if name.startswith("x")),
        "calibration_state_updates": cal_updates,
        "test_state_updates": test_updates,
        "update_method": "statsmodels.SARIMAXResults.extend; fixed coefficients",
        "initial_prediction": initial_prediction,
        "post_update_prediction_probe": counterfactual_next,
        "mae": float(np.mean(np.abs(signed))),
        "rmse": float(np.sqrt(np.mean(signed**2))),
        "underprediction_rate": float(np.mean(signed > 0)),
        "bias": float(np.mean(signed)),
        "normalised_bias": float(np.mean(signed) / max(np.mean(actual), 1e-12)),
        "positive_residual_acf1": acf1,
        "positive_residual_acf_mean_1_5": acf15,
        "fit_and_replay_seconds": time.perf_counter() - started,
    }

    cal_scores = np.maximum(calibration.cpu_sum.to_numpy(float) - cal_forecast, 0.0)[-W:]
    persistence_scores = np.maximum(history.cpu_sum.to_numpy(float)[1:] - history.cpu_sum.to_numpy(float)[:-1], 0.0)[-W:]
    policy_rows: list[dict[str, object]] = []
    for warm_start, scores in (("causal_arima_days_8_9", cal_scores), ("legacy_persistence", persistence_scores)):
        for delta in DELTAS:
            capacities = simulate_policy(actual, test_forecast, scores, delta)
            for evaluation, start in (("full_test", 0), ("after_W_test_intervals", W)):
                subtotal, relative = cost(
                    actual[start:], capacities[start:], test.replica_count.to_numpy(float)[start:]
                )
                overload = actual[start:] > capacities[start:]
                policy_rows.append(
                    {
                        "service_id": service,
                        "warm_start": warm_start,
                        "delta": delta,
                        "evaluation": evaluation,
                        "excluded_intervals": start,
                        "n_intervals": len(overload),
                        "overload_fraction": float(overload.mean()),
                        "within_delta": bool(overload.mean() <= delta),
                        "relative_cost_vs_observed": relative,
                        "c_total": subtotal,
                    }
                )

    forecast_rows = pd.DataFrame(
        {
            "msname": service,
            "model": "arima_causal_one_step",
            "timestamp": test.timestamp.to_numpy(np.int64),
            "forecast": test_forecast,
        }
    )
    calibration_rows = pd.DataFrame(
        {
            "msname": service,
            "model": "arima_causal_one_step",
            "timestamp": calibration.timestamp.to_numpy(np.int64),
            "forecast": cal_forecast,
        }
    )
    return {
        "diagnostics": diagnostics,
        "policy_rows": policy_rows,
        "forecasts": forecast_rows,
        "calibration_forecasts": calibration_rows,
    }


def save_exchangeability_figure(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    """Render the diagnostic from frozen focused outputs without refitting ARIMA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = PAPER / "figures"
    colors = {"persistence": "#1f77b4", "lstm": "#2ca02c", "xgb": "#ff7f0e", "arima": "#d62728"}
    old_diag = pd.read_csv(EXP / "results" / "c4_diag_by_model.csv")
    diag = old_diag[~old_diag.model.eq("arima")].copy()
    causal = summary[
        summary.warm_start.eq("causal_arima_days_8_9")
        & summary.evaluation.eq("full_test")
        & np.isclose(summary.delta, .05)
    ]
    assert len(causal) == 1
    row = {
        "model": "arima",
        "underpred_rate": diagnostics.underprediction_rate.median(),
        "norm_bias": diagnostics.normalised_bias.median(),
        "acf1": diagnostics.positive_residual_acf1.median(),
        "acf_mean_1_5": diagnostics.positive_residual_acf_mean_1_5.median(),
        "frac_within_delta": float(causal.service_compliance.iloc[0]),
    }
    diag = pd.concat([diag, pd.DataFrame([row])], ignore_index=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    annotations = {
        "persistence": ((5, -10), "left", "top"),
        "arima": ((5, 8), "left", "bottom"),
        "xgb": ((-5, 4), "right", "bottom"),
    }
    for _, item in diag.iterrows():
        x, y = item.acf_mean_1_5, 100 * item.frac_within_delta
        ax.scatter(x, y, s=70, color=colors[item.model])
        label = item.model.upper() if item.model != "persistence" else "Persistence"
        offset, horizontal, vertical = annotations.get(item.model, ((5, 4), "left", "bottom"))
        ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8, va=vertical, ha=horizontal)
    ax.set_xlabel("Median positive-residual autocorrelation (lags 1--5)")
    ax.set_ylabel("Services within $\\delta=0.05$ (%)")
    ax.set_ylim(top=101.5)
    ax.grid(alpha=.2); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"exp_c4_exchangeability.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_figures(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = PAPER / "figures"
    exp2 = pd.read_csv(EXP / "results" / "exp2_coverage_calibration.csv")
    causal = summary[
        summary.warm_start.eq("causal_arima_days_8_9")
        & summary.evaluation.eq("full_test")
    ][["delta", "median_overload_fraction", "service_compliance"]].copy()
    causal["model"] = "arima"
    causal = causal.rename(columns={"median_overload_fraction": "median_ol", "service_compliance": "frac_within_delta"})
    combined = exp2[~exp2.model.eq("arima")].copy()
    for col in ("p10_ol", "p90_ol"):
        causal[col] = np.nan
    combined = pd.concat([combined, causal[combined.columns]], ignore_index=True)

    colors = {"persistence": "#1f77b4", "lstm": "#2ca02c", "xgb": "#ff7f0e", "arima": "#d62728"}

    advanced = pd.read_csv(EXP / "results" / "advanced_policy_replay_summary.csv")
    persistence = pd.read_csv(EXP / "results" / "risk_policy_replay_summary.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for model in ("lstm", "xgb"):
        costs, overloads = [], []
        for delta in DELTAS:
            found = advanced[advanced.policy.eq(f"{model}_conformal_guardrail_d{delta}")].iloc[0]
            costs.append(float(found.median_relative_cost))
            overloads.append(100 * float(found.median_overload_fraction))
        ax.plot(overloads, costs, marker="o", color=colors[model], label=model.upper())
    part = summary[summary.warm_start.eq("causal_arima_days_8_9") & summary.evaluation.eq("full_test")]
    part = part.set_index("delta").loc[list(DELTAS)]
    ax.plot(100 * part.median_overload_fraction, part.median_relative_cost, marker="o", color=colors["arima"], label="ARIMA (one-step)")
    p_costs, p_overloads = [], []
    for delta in DELTAS:
        found = persistence[persistence.policy.eq("risk_conformal_guardrail") & np.isclose(persistence.delta, delta)].iloc[0]
        p_costs.append(float(found.median_relative_cost)); p_overloads.append(100 * float(found.median_overload_fraction))
    ax.plot(p_overloads, p_costs, marker="o", linestyle="--", color=colors["persistence"], label="Persistence")
    ax.set_xlabel("Median overload fraction (%)")
    ax.set_ylabel("Median relative cost")
    ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"fig1_cost_risk_frontier.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model in ("persistence", "lstm", "xgb", "arima"):
        part = combined[combined.model.eq(model)].sort_values("delta", ascending=False)
        ax.plot(part.delta, 100 * part.median_ol, marker="o", label=model.upper() if model != "persistence" else "Persistence", color=colors[model])
    ax.plot([0.10, 0.01], [10, 1], "k--", linewidth=1, label="declared budget")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel("Risk budget $\\delta$"); ax.set_ylabel("Median overload (%)")
    ax.legend(fontsize=8); ax.grid(alpha=.2, which="both"); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"exp2_coverage_calibration.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    save_exchangeability_figure(summary, diagnostics)


def main() -> None:
    args = parse_args()
    if args.diagnostic_figure_only:
        summary = pd.read_csv(OUT / "arima_causal_summary.csv")
        diagnostics = pd.read_csv(OUT / "arima_causal_diagnostics.csv")
        save_exchangeability_figure(summary, diagnostics)
        print("redrew residual diagnostic from saved causal results")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_PATH.read_text())
    log = pd.read_csv(LOG_PATH).sort_values("msname")
    if args.limit:
        log = log.head(args.limit)
    tasks = [
        (str(row.msname), int(row.best_k), ast.literal_eval(str(row.arima_order)), split)
        for row in log.itertuples(index=False)
    ]

    outputs: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_service, task): task[0] for task in tasks}
        for future in as_completed(futures):
            service = futures[future]
            result = future.result()
            outputs.append(result)
            print(f"completed {service} ({len(outputs)}/{len(tasks)})", flush=True)

    diagnostics = pd.DataFrame([item["diagnostics"] for item in outputs]).sort_values("service_id")
    policies = pd.DataFrame([row for item in outputs for row in item["policy_rows"]]).sort_values(["service_id", "warm_start", "delta", "evaluation"])
    forecasts = pd.concat([item["forecasts"] for item in outputs], ignore_index=True).sort_values(["msname", "timestamp"])
    calibration = pd.concat([item["calibration_forecasts"] for item in outputs], ignore_index=True).sort_values(["msname", "timestamp"])
    summary = (
        policies.groupby(["warm_start", "delta", "evaluation", "excluded_intervals"], as_index=False)
        .agg(
            services=("service_id", "nunique"),
            median_relative_cost=("relative_cost_vs_observed", "median"),
            median_overload_fraction=("overload_fraction", "median"),
            service_compliance=("within_delta", "mean"),
        )
    )

    import pmdarima, statsmodels, sklearn
    methodology = {
        "protocol": "one-step causal focused-20 replay",
        "selection": "reuse archived pre-test-selected Fourier K and ARIMA order; no test selection",
        "fit_api": "pmdarima.ARIMA.fit(y, X=X)",
        "state_update": "statsmodels.SARIMAXResults.extend(endog=[observed_y], exog=X_t); coefficients are not refitted on test",
        "intercept": "disabled explicitly for all services so one-row Fourier state extension is identified; this choice is fixed and not test-selected",
        "legacy_api_probe": "the independent pmdarima 2.1.1 check found k_exog=0 with legacy exogenous= and k_exog=1 with X=; the causal replay requires and verifies k_exog=2K per service",
        "calibration_scores": (
            "causal forward forecasts for days 8-9 from coefficients fitted on days 0-7; "
            "the reused archived order and Fourier K were selected within the full days 0-9 "
            "pre-test history, so days 8-9 are not an independent model-selection holdout"
        ),
        "test_fit": "separate final model fitted on all days 0-9",
        "versions": {
            "python": platform.python_version(),
            "pmdarima": pmdarima.__version__,
            "statsmodels": statsmodels.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "services": len(tasks),
        "expected_calibration_updates_per_service": 2_880,
        "expected_test_updates_per_service": 4_320,
    }

    diagnostics.to_csv(OUT / "arima_causal_diagnostics.csv", index=False)
    policies.to_csv(OUT / "arima_causal_policy_per_service.csv", index=False)
    summary.to_csv(OUT / "arima_causal_summary.csv", index=False)
    forecasts.to_parquet(OUT / "arima_causal_forecasts_focused20.parquet", index=False)
    calibration.to_parquet(OUT / "arima_causal_calibration_forecasts_focused20.parquet", index=False)
    (OUT / "arima_api_and_methodology.json").write_text(json.dumps(methodology, indent=2) + "\n")
    if len(tasks) == 20:
        save_figures(summary, diagnostics)
    print("\n", diagnostics[["service_id", "k_exog", "mae", "underprediction_rate", "positive_residual_acf_mean_1_5"]].to_string(index=False))
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
