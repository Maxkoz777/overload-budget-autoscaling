"""
run_ls_frontier.py
==================
Large-scale replay: cost-risk frontier and coverage calibration for
200 services across persistence, ARIMA, XGBoost, and LSTM forecasters.

Outputs
-------
experiments/results/large_scale/analysis/ls_frontier_per_service_200.csv
experiments/results/large_scale/analysis/ls_frontier_summary_200.csv
experiments/results/large_scale/analysis/ls_coverage_calibration_200.csv
experiments/results/large_scale/analysis/ls_frontier_by_stratum_200.csv
experiments/figures/large_scale/ls_fig_cost_risk_frontier_200.pdf/.png
experiments/figures/large_scale/ls_fig_coverage_calibration_200.pdf/.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("experiments/src")))

from exp_core_ls import (  # noqa: E402
    FIGS_LS,
    FORECAST_MODELS,
    GROUP_MAP_200,
    RESULTS_LS,
    SIM_COST,
    load_all_services_200,
    load_forecast_ls,
    load_selection,
    load_split_def,
    simulate_policy,
)


MPLCONFIG = FIGS_LS / ".mplconfig"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402


DELTAS = [0.10, 0.05, 0.03, 0.01]
MODELS = ["persistence", *FORECAST_MODELS]
W = 240
ALPHA = 1.0
RHO = 0.70
GUARDRAIL_H = 2
GUARDRAIL_GAMMA = 1
MU = 1.0

PER_SERVICE_PATH = RESULTS_LS / "ls_frontier_per_service_200.csv"
SUMMARY_PATH = RESULTS_LS / "ls_frontier_summary_200.csv"
COVERAGE_PATH = RESULTS_LS / "ls_coverage_calibration_200.csv"
BY_STRATUM_PATH = RESULTS_LS / "ls_frontier_by_stratum_200.csv"

PAL = {
    "lstm": "#2196F3",
    "xgb": "#4CAF50",
    "arima": "#FF9800",
    "persistence": "#9C27B0",
}
MARKERS = {"lstm": "o", "xgb": "s", "arima": "^", "persistence": "D"}
LABELS = {
    "lstm": "LSTM",
    "xgb": "XGBoost",
    "arima": "ARIMA",
    "persistence": "Persistence",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def observed_cost(test: pd.DataFrame) -> float:
    """Cost of the recorded replica_count over the test window."""
    if "replica_count" not in test.columns:
        raise ValueError("test frame is missing replica_count; cannot compute observed rel_cost")
    demand = test["cpu_sum"].to_numpy(dtype=float)
    cap = np.maximum(np.ceil(test["replica_count"].to_numpy(dtype=float)).astype(int), 1)
    overload = demand > (MU * cap)
    actions = np.abs(np.diff(cap, prepend=cap[0]))
    return (
        SIM_COST["c_res"] * float(cap.sum())
        + SIM_COST["c_act"] * float(actions.sum())
        + SIM_COST["c_vio"] * float(overload.sum())
        + SIM_COST["c_inf_per_step"] * float(len(demand))
        + SIM_COST["c_train_total"]
    )


def quantile_10(series: pd.Series) -> float:
    return float(series.quantile(0.10))


def quantile_90(series: pd.Series) -> float:
    return float(series.quantile(0.90))


def summarize(per_service: pd.DataFrame) -> pd.DataFrame:
    grouped = per_service.groupby(["model", "delta"], sort=False, dropna=False)
    return grouped.agg(
        services=("service_id", "nunique"),
        median_rel_cost=("rel_cost", "median"),
        p10_rel_cost=("rel_cost", quantile_10),
        p90_rel_cost=("rel_cost", quantile_90),
        median_overload_fraction=("overload_fraction", "median"),
        p10_overload_fraction=("overload_fraction", quantile_10),
        p90_overload_fraction=("overload_fraction", quantile_90),
        frac_within_delta=("within_delta", "mean"),
        median_margin_mean=("margin_mean", "median"),
        median_margin_p95=("margin_p95", "median"),
        median_guardrail_rate=("guardrail_rate", "median"),
        median_max_overload_run=("max_overload_run", "median"),
        median_scaling_churn=("scaling_churn", "median"),
    ).reset_index()


def summarize_by_stratum(per_service: pd.DataFrame) -> pd.DataFrame:
    grouped = per_service.groupby(["stratum", "model", "delta"], sort=True, dropna=False)
    return grouped.agg(
        services=("service_id", "nunique"),
        median_rel_cost=("rel_cost", "median"),
        p10_rel_cost=("rel_cost", quantile_10),
        p90_rel_cost=("rel_cost", quantile_90),
        median_overload_fraction=("overload_fraction", "median"),
        p10_overload_fraction=("overload_fraction", quantile_10),
        p90_overload_fraction=("overload_fraction", quantile_90),
        frac_within_delta=("within_delta", "mean"),
        median_margin_mean=("margin_mean", "median"),
        median_margin_p95=("margin_p95", "median"),
        median_guardrail_rate=("guardrail_rate", "median"),
        median_max_overload_run=("max_overload_run", "median"),
        median_scaling_churn=("scaling_churn", "median"),
    ).reset_index()


def save(fig: plt.Figure, name: str) -> None:
    FIGS_LS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_LS / f"{name}.pdf")
    fig.savefig(FIGS_LS / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {FIGS_LS / (name + '.pdf')}")
    print(f"saved {FIGS_LS / (name + '.png')}")


def plot_cost_risk(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.6))

    for model in MODELS:
        sub = summary[summary["model"] == model].sort_values("delta", ascending=False)
        x = sub["median_overload_fraction"].to_numpy(dtype=float) * 100.0
        y = sub["median_rel_cost"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=PAL[model],
            marker=MARKERS[model],
            linewidth=1.8,
            markersize=6.5,
            label=LABELS[model],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['delta']:.2f}",
                xy=(row["median_overload_fraction"] * 100.0, row["median_rel_cost"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=PAL[model],
            )

    ax.set_xlabel("Median realised overload (%)")
    ax.set_ylabel("Median relative cost vs observed capacity")
    ax.set_title("Large-scale cost-risk frontier (200 services)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.legend(loc="best", ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "ls_fig_cost_risk_frontier_200")


def plot_coverage_calibration(coverage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))

    max_axis = max(float(coverage["delta"].max()), float(coverage["p90_overload_fraction"].max()))
    grid = np.linspace(0.0, max_axis * 1.08, 100)
    ax.plot(grid, grid, color="black", linestyle=":", linewidth=1.3, label="ideal: realised = δ")

    for model in MODELS:
        sub = coverage[coverage["model"] == model].sort_values("delta")
        x = sub["delta"].to_numpy(dtype=float)
        y = sub["median_overload_fraction"].to_numpy(dtype=float)
        ylo = sub["p10_overload_fraction"].to_numpy(dtype=float)
        yhi = sub["p90_overload_fraction"].to_numpy(dtype=float)
        ax.fill_between(x, ylo, yhi, color=PAL[model], alpha=0.12, linewidth=0)
        ax.plot(
            x,
            y,
            color=PAL[model],
            marker=MARKERS[model],
            linewidth=1.8,
            markersize=6.5,
            label=LABELS[model],
        )

    ax.set_xlabel("Nominal risk budget δ")
    ax.set_ylabel("Realised overload fraction")
    ax.set_title("Coverage calibration (median with p10-p90 band)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save(fig, "ls_fig_coverage_calibration_200")


def assert_outputs_clean(summary: pd.DataFrame, by_stratum: pd.DataFrame) -> None:
    numeric_summary = summary.select_dtypes(include=[np.number])
    numeric_stratum = by_stratum.select_dtypes(include=[np.number])
    if not np.isfinite(numeric_summary.to_numpy()).all():
        raise ValueError("summary contains NaN or infinite numeric values")
    if not np.isfinite(numeric_stratum.to_numpy()).all():
        raise ValueError("by-stratum summary contains NaN or infinite numeric values")
    for frame_name, frame in [("summary", summary), ("by_stratum", by_stratum)]:
        bad = frame[(frame["frac_within_delta"] < 0) | (frame["frac_within_delta"] > 1)]
        if not bad.empty:
            raise ValueError(f"{frame_name} has frac_within_delta outside [0, 1]")


def main() -> None:
    RESULTS_LS.mkdir(parents=True, exist_ok=True)
    FIGS_LS.mkdir(parents=True, exist_ok=True)

    split_def = load_split_def()
    services = load_all_services_200(split_def)
    selection = load_selection()
    expected_services = set(selection["service_id"].astype(str))
    if {service for service, _, _ in services} != expected_services:
        raise ValueError("loaded services do not match selected_services_200.csv")

    rows = []
    total = len(services)
    for idx, (service, history, test) in enumerate(services, start=1):
        denom = observed_cost(test)
        if denom <= 0:
            raise ValueError(f"{service}: observed cost denominator is non-positive")
        stratum = GROUP_MAP_200[service]

        forecast_cache: dict[str, np.ndarray | None] = {"persistence": None}
        for model in FORECAST_MODELS:
            forecast_cache[model] = load_forecast_ls(model, service)

        for model in MODELS:
            forecast_array = forecast_cache[model]
            for delta in DELTAS:
                result = simulate_policy(
                    history,
                    test,
                    forecast_array=forecast_array,
                    delta=delta,
                    W=W,
                    alpha=ALPHA,
                    mu=MU,
                    rho=RHO,
                    guardrail_h=GUARDRAIL_H,
                    guardrail_gamma=GUARDRAIL_GAMMA,
                    use_margin=True,
                    use_guardrail=True,
                    **SIM_COST,
                )
                rows.append(
                    {
                        "service_id": service,
                        "stratum": stratum,
                        "model": model,
                        "delta": delta,
                        "overload_fraction": result["overload_fraction"],
                        "within_delta": bool(result["overload_fraction"] <= delta),
                        "rel_cost": float(result["c_total"] / denom),
                        "observed_c_total": denom,
                        "c_total": result["c_total"],
                        "c_res": result["c_res"],
                        "c_act": result["c_act"],
                        "c_vio": result["c_vio"],
                        "c_inf": result["c_inf"],
                        "c_train": result["c_train"],
                        "margin_mean": result["margin_mean"],
                        "margin_p95": result["margin_p95"],
                        "guardrail_rate": result["guardrail_rate"],
                        "max_overload_run": result["max_overload_run"],
                        "scaling_churn": result["scaling_churn"],
                        "window": W,
                        "alpha": ALPHA,
                        "rho": RHO,
                        "guardrail_h": GUARDRAIL_H,
                        "guardrail_gamma": GUARDRAIL_GAMMA,
                    }
                )

        if idx % 25 == 0 or idx == total:
            print(f"processed {idx}/{total} services")

    per_service = pd.DataFrame(rows)
    summary = summarize(per_service)
    coverage = summary[
        [
            "model",
            "delta",
            "median_overload_fraction",
            "p10_overload_fraction",
            "p90_overload_fraction",
            "frac_within_delta",
        ]
    ].copy()
    by_stratum = summarize_by_stratum(per_service)

    assert_outputs_clean(summary, by_stratum)

    per_service.to_csv(PER_SERVICE_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    coverage.to_csv(COVERAGE_PATH, index=False)
    by_stratum.to_csv(BY_STRATUM_PATH, index=False)

    plot_cost_risk(summary)
    plot_coverage_calibration(coverage)

    print("per_service_rows", len(per_service))
    print("services", per_service["service_id"].nunique())
    print("models", ",".join(MODELS))
    print("deltas", ",".join(str(d) for d in DELTAS))
    print("rel_cost_definition", "c_total(policy) / c_total(observed_capacity)")
    print("\nSUMMARY")
    print(summary.to_csv(index=False).strip())
    print("\noutputs")
    for path in [
        PER_SERVICE_PATH,
        SUMMARY_PATH,
        COVERAGE_PATH,
        BY_STRATUM_PATH,
        FIGS_LS / "ls_fig_cost_risk_frontier_200.pdf",
        FIGS_LS / "ls_fig_cost_risk_frontier_200.png",
        FIGS_LS / "ls_fig_coverage_calibration_200.pdf",
        FIGS_LS / "ls_fig_coverage_calibration_200.png",
    ]:
        print(path)


if __name__ == "__main__":
    main()
