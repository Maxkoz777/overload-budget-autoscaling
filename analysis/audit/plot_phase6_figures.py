#!/usr/bin/env python3
"""Regenerate Phase-6 focused figures from matched, saved per-service replays.

The archived ARIMA rows in c5_frontier_density.csv and
exp2_coverage_calibration.csv are never used. Corrected ARIMA is restricted to
the full-test, causal one-step, model-specific warm-start output.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER.parent / "experiments" / "results"
ARIMA = PAPER / "audit" / "arima_causal_results"
REACTIVE = PAPER / "audit" / "verified_results" / "comparative"
OUT = PAPER / "audit" / "phase6_results"
FIGURES = PAPER / "figures" / "verified"
DELTAS = (0.01, 0.05, 0.10)
MODELS = ("persistence", "arima", "lstm", "xgb")


def one_row(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= np.isclose(frame[column], value) if isinstance(value, float) else frame[column].eq(value)
    selected = frame.loc[mask]
    assert len(selected) == 1, (conditions, len(selected))
    return selected.iloc[0]


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    risk = pd.read_csv(RESULTS / "risk_policy_replay_per_service.csv")
    advanced = pd.read_csv(RESULTS / "advanced_policy_replay_per_service.csv")
    arima = pd.read_csv(ARIMA / "arima_causal_policy_per_service.csv")
    risk_summary = pd.read_csv(RESULTS / "risk_policy_replay_summary.csv")
    advanced_summary = pd.read_csv(RESULTS / "advanced_policy_replay_summary.csv")
    arima_summary = pd.read_csv(ARIMA / "arima_causal_summary.csv")
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for delta in DELTAS:
            if model == "persistence":
                subset = risk.loc[
                    risk.policy.eq("risk_conformal_guardrail") & np.isclose(risk.delta, delta)
                ]
                cost_column = "relative_cost_vs_observed"
                overload_column = "overload_fraction"
                expected = one_row(risk_summary, policy="risk_conformal_guardrail", delta=delta)
                expected_overload = expected.median_overload_fraction
                expected_cost = expected.median_relative_cost
            elif model == "arima":
                subset = arima.loc[
                    arima.warm_start.eq("causal_arima_days_8_9")
                    & arima.evaluation.eq("full_test")
                    & np.isclose(arima.delta, delta)
                ]
                cost_column = "relative_cost_vs_observed"
                overload_column = "overload_fraction"
                expected = one_row(
                    arima_summary,
                    warm_start="causal_arima_days_8_9",
                    evaluation="full_test",
                    delta=delta,
                )
                expected_overload = expected.median_overload_fraction
                expected_cost = expected.median_relative_cost
            else:
                policy = f"{model}_conformal_guardrail_d{delta}"
                subset = advanced.loc[advanced.policy.eq(policy)]
                cost_column = "relative_cost_vs_observed"
                overload_column = "overload_fraction"
                expected = one_row(advanced_summary, policy=policy)
                expected_overload = expected.median_overload_fraction
                expected_cost = expected.median_relative_cost
            service_column = "service_id" if model == "arima" else "msname"
            assert len(subset) == 20 and subset[service_column].nunique() == 20
            assert (subset.n_intervals == 4320).all()
            ol = subset[overload_column].to_numpy(float)
            costs = subset[cost_column].to_numpy(float)
            median_ol = float(np.median(ol))
            median_cost = float(np.median(costs))
            compliance = float(np.mean(ol <= delta))
            np.testing.assert_allclose([median_ol, median_cost], [expected_overload, expected_cost], atol=1e-10)
            if model == "arima":
                np.testing.assert_allclose(compliance, expected.service_compliance, atol=1e-10)
            rows.append({
                "model": model,
                "protocol": "causal_arima_one_step" if model == "arima" else "focused_one_step",
                "delta": delta,
                "services": 20,
                "median_overload": median_ol,
                "p10_overload": float(np.quantile(ol, 0.1)),
                "p90_overload": float(np.quantile(ol, 0.9)),
                "median_relative_cost": median_cost,
                "service_compliance": compliance,
            })
    window = pd.read_csv(RESULTS / "exp3_window_sensitivity.csv").sort_values("W")
    np.testing.assert_allclose(window.abs_dev_median, abs(window.median_ol - 0.05), atol=1e-12)
    assert (window.p10_ol <= window.median_ol).all() and (window.median_ol <= window.p90_ol).all()
    assert int(window.loc[window.abs_dev_median.idxmin(), "W"]) == 240
    return pd.DataFrame(rows), window


def table_text(data: pd.DataFrame) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(data.columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(data.to_dict(orient="records"))
    return stream.getvalue()


def draw(data: pd.DataFrame, window: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "persistence": "#265dab", "arima": "#c43c39",
        "lstm": "#23834b", "xgb": "#e5931d",
    }
    labels = {
        "persistence": "Persistence", "arima": "ARIMA (causal 1-step)",
        "lstm": "LSTM", "xgb": "XGBoost",
    }
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.3, 4.1), constrained_layout=True)
    for model in MODELS:
        part = data.loc[data.model.eq(model)].sort_values("median_overload")
        ax.plot(
            100 * part.median_overload, part.median_relative_cost,
            marker="o", linewidth=1.7, markersize=5,
            linestyle="--" if model == "persistence" else "-",
            color=colors[model], label=labels[model],
        )
    tuned = one_row(pd.read_csv(REACTIVE / "reactive_selected_summary.csv"), subset="focused20")
    ax.scatter(
        [100 * tuned.median_overload], [tuned.median_relative_cost],
        marker="P", s=100, color="#6b3f91", edgecolors="white", linewidths=0.7, zorder=6,
        label="Reactive (pre-test selected)",
    )
    ax.annotate("Reactive", xy=(100 * tuned.median_overload, tuned.median_relative_cost),
                xytext=(4.7, 0.220), fontsize=10, color="#6b3f91",
                arrowprops={"arrowstyle": "->", "color": "#6b3f91"})
    ref = one_row(pd.read_csv(RESULTS / "baseline_replay_summary.csv"), policy="oracle_demand_capacity")
    ax.axhline(ref.median_relative_cost, color="#505050", linestyle=":", linewidth=1.3,
               label="Clairvoyant zero-OL reference")
    ax.set(xlabel="Median held-out overload (%)", ylabel="Median relative replay cost", xlim=(0, 8.2))
    ax.set_ylim(0.188, 0.302)
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.94)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_focused_frontier_phase6.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.1, 3.8), constrained_layout=True)
    ax.plot([0, 10], [0, 10], color="#303030", linestyle="--", linewidth=1.1,
            label="Declared budget")
    for model in MODELS:
        part = data.loc[data.model.eq(model)].sort_values("delta")
        ax.plot(100 * part.delta, 100 * part.median_overload, marker="o",
                linewidth=1.7, markersize=5, color=colors[model], label=labels[model])
    ax.set(xlabel="Declared risk budget (%)", ylabel="Median realised overload (%)",
           xlim=(0, 10.5), ylim=(0, 11))
    ax.set_xticks([1, 5, 10]); ax.grid(alpha=0.22, linestyle="--")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.94)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_focused_calibration_phase6.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.7, 5.7), constrained_layout=True)
    xx = window.W.to_numpy(float)
    yy = window.median_ol.to_numpy(float)
    ax1.semilogx(xx, yy, marker="o", color="#7545a1", linewidth=1.7,
                 label="Median realised overload")
    ax1.fill_between(xx, window.p10_ol.to_numpy(float), window.p90_ol.to_numpy(float),
                     color="#b887cf", alpha=0.32, label="Service p10-p90")
    ax1.axhline(0.05, linestyle="--", color="#303030", linewidth=1.1, label="Budget 0.05")
    ax1.set(ylabel="Realised overload fraction", title="Trace-level window sensitivity")
    ax1.grid(alpha=0.2, linestyle="--"); ax1.legend(fontsize=7.3)
    ax2.semilogx(xx, window.abs_dev_median.to_numpy(float), marker="o",
                 color="#7545a1", linewidth=1.7)
    ax2.scatter([240], [float(window.loc[window.W.eq(240), "abs_dev_median"].iloc[0])],
                s=60, facecolors="none", edgecolors="#b43333", linewidths=1.5,
                zorder=5, label="Pre-test choice W=240")
    ax2.set(xlabel="Calibration window W (minutes)",
            ylabel="|median overload - 0.05|", title="Trace-level budget deviation")
    ax2.grid(alpha=0.2, linestyle="--"); ax2.legend(fontsize=7.3)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_window_sensitivity_phase6.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data, window = prepare()
    content = table_text(data)
    target = OUT / "focused_figure_data.csv"
    names = ("fig_focused_frontier_phase6", "fig_focused_calibration_phase6", "fig_window_sensitivity_phase6")
    if args.write:
        OUT.mkdir(exist_ok=True)
        target.write_text(content, encoding="utf-8")
        draw(data, window)
        print(f"wrote {target} and {len(names)} verified figures")
        return 0
    assert target.exists() and target.read_text(encoding="utf-8") == content
    assert all((FIGURES / f"{name}.{ext}").exists() for name in names for ext in ("pdf", "png"))
    print("PASS: focused figure data regenerate from matched one-step replays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
