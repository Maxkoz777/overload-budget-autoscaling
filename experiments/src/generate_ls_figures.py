"""
generate_ls_figures.py
======================
Publication figures (PDF+PNG) and LaTeX table fragments for the 200-service
large-scale tier.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("experiments/src")))

from exp_core_ls import FIGS_LS, RESULTS_LS, REPORTS_LS  # noqa: E402


MPLCONFIG = FIGS_LS / ".mplconfig"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIG))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402


TABLES_PATH = REPORTS_LS / "ls_tables.tex"

MODELS = ["persistence", "arima", "xgb", "lstm"]
MODEL_LABEL = {
    "persistence": "Persistence",
    "arima": "ARIMA",
    "xgb": "XGBoost",
    "lstm": "LSTM",
}
MODEL_COLOR = {
    "persistence": "#9C27B0",
    "arima": "#FF9800",
    "xgb": "#4CAF50",
    "lstm": "#2196F3",
}
MODEL_MARKER = {
    "persistence": "D",
    "arima": "^",
    "xgb": "s",
    "lstm": "o",
}
STRATA = ["G1", "G2", "G3", "G4", "G5"]
DELTAS = [0.10, 0.05, 0.03, 0.01]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    }
)


def load_csv(name: str) -> pd.DataFrame:
    path = RESULTS_LS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def save(fig: plt.Figure, name: str) -> None:
    FIGS_LS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_LS / f"{name}.pdf")
    fig.savefig(FIGS_LS / f"{name}.png", dpi=220)
    plt.close(fig)
    print(f"saved {FIGS_LS / (name + '.pdf')}")
    print(f"saved {FIGS_LS / (name + '.png')}")


def pct_axis(ax, axis: str = "y") -> None:
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
    if axis == "x":
        ax.xaxis.set_major_formatter(fmt)
    else:
        ax.yaxis.set_major_formatter(fmt)


def fig_cost_risk_frontier(frontier: pd.DataFrame, hpa: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for model in MODELS:
        sub = frontier[frontier["model"].eq(model)].sort_values("delta", ascending=False)
        ax.plot(
            sub["median_overload_fraction"] * 100.0,
            sub["median_rel_cost"],
            color=MODEL_COLOR[model],
            marker=MODEL_MARKER[model],
            linewidth=1.9,
            markersize=6.5,
            label=MODEL_LABEL[model],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['delta']:.2f}",
                (row["median_overload_fraction"] * 100.0, row["median_rel_cost"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=MODEL_COLOR[model],
            )

    best_hpa = hpa.sort_values("median_rel_cost").iloc[0]
    ax.scatter(
        best_hpa["median_overload_fraction"] * 100.0,
        best_hpa["median_rel_cost"],
        color="#F44336",
        marker="P",
        s=95,
        label=f"HPA u={best_hpa['hpa_threshold']:.1f}",
        zorder=5,
    )
    ax.set_xlabel("Median realised overload (%)")
    ax.set_ylabel("Median relative cost vs observed capacity")
    ax.set_title("Large-scale cost-risk frontier (n=200)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save(fig, "ls_fig_cost_risk_frontier_200")


def fig_coverage_calibration(coverage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    max_axis = max(float(coverage["delta"].max()), float(coverage["p90_overload_fraction"].max()))
    grid = np.linspace(0, max_axis * 1.08, 100)
    ax.plot(grid, grid, color="black", linestyle=":", linewidth=1.2, label="realised = δ")

    for model in MODELS:
        sub = coverage[coverage["model"].eq(model)].sort_values("delta")
        ax.fill_between(
            sub["delta"],
            sub["p10_overload_fraction"],
            sub["p90_overload_fraction"],
            color=MODEL_COLOR[model],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            sub["delta"],
            sub["median_overload_fraction"],
            color=MODEL_COLOR[model],
            marker=MODEL_MARKER[model],
            linewidth=1.9,
            markersize=6.5,
            label=MODEL_LABEL[model],
        )

    ax.set_xlabel("Nominal risk budget δ")
    ax.set_ylabel("Realised overload fraction")
    ax.set_title("Coverage calibration (median with p10-p90 band)")
    pct_axis(ax, "x")
    pct_axis(ax, "y")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save(fig, "ls_fig_coverage_calibration_200")


def annotate_heatmap(ax, data: np.ndarray, fmt: str, threshold: float) -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center", color=color, fontsize=8)


def fig_stratum_heatmap(by_stratum: pd.DataFrame) -> None:
    sub = by_stratum[by_stratum["delta"].eq(0.05)].copy()
    cost = sub.pivot(index="stratum", columns="model", values="median_rel_cost").reindex(index=STRATA, columns=MODELS)
    overload = sub.pivot(index="stratum", columns="model", values="median_overload_fraction").reindex(index=STRATA, columns=MODELS) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    im0 = axes[0].imshow(cost.to_numpy(), cmap="YlGnBu", aspect="auto")
    axes[0].set_title("Median relative cost")
    axes[0].set_xticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS], rotation=25, ha="right")
    axes[0].set_yticks(range(len(STRATA)), STRATA)
    annotate_heatmap(axes[0], cost.to_numpy(), ".3f", float(np.nanmedian(cost.to_numpy())))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

    im1 = axes[1].imshow(overload.to_numpy(), cmap="YlOrRd", aspect="auto")
    axes[1].set_title("Median overload (%)")
    axes[1].set_xticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS], rotation=25, ha="right")
    axes[1].set_yticks(range(len(STRATA)), STRATA)
    annotate_heatmap(axes[1], overload.to_numpy(), ".2f", max(0.01, float(np.nanmedian(overload.to_numpy()))))
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    fig.suptitle("Large-scale stratum summary at δ=0.05", y=1.03, fontsize=12)
    save(fig, "ls_fig_stratum_heatmap_200")


def fig_baselines(baselines: pd.DataFrame, hpa: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    h = hpa.sort_values("hpa_threshold")
    ax.plot(
        h["median_overload_fraction"] * 100.0,
        h["median_rel_cost"],
        color="#F44336",
        marker="P",
        linewidth=1.7,
        label="B1 tuned HPA grid",
    )
    for _, row in h.iterrows():
        ax.annotate(
            f"u={row['hpa_threshold']:.1f}",
            (row["median_overload_fraction"] * 100.0, row["median_rel_cost"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.4,
            color="#F44336",
        )

    colors = {
        "B2": "#9E9E9E",
        "B3": "#795548",
        "B4": "#E91E63",
        "B6": "#4CAF50",
        "B7": "#2196F3",
        "B8": "#607D8B",
        "B9": "#FF9800",
    }
    markers = {"B2": "x", "B3": "v", "B4": "^", "B6": "o", "B7": "s", "B8": "D", "B9": "*"}
    for _, row in baselines[~baselines["policy_family"].eq("B1")].iterrows():
        fam = row["policy_family"]
        label = row["policy"].replace(": ", " ")
        ax.scatter(
            row["median_overload_fraction"] * 100.0,
            row["median_rel_cost"],
            color=colors.get(fam, "#000000"),
            marker=markers.get(fam, "o"),
            s=100 if fam == "B9" else 72,
            label=label,
            zorder=4,
        )
    ax.axvline(5.0, color="black", linestyle=":", linewidth=1.2, label="δ=0.05")
    ax.set_xlabel("Median realised overload (%)")
    ax.set_ylabel("Median relative cost vs observed capacity")
    ax.set_title("Baseline comparison at δ=0.05")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save(fig, "ls_fig_baselines_200")


def fig_percentile_vs_conformal(percentile: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    styles = [
        ("B6: conformal + guardrail", "#4CAF50", "o"),
        ("B9: empirical percentile + guardrail", "#FF9800", "^"),
        ("B7: conformal margin only", "#2196F3", "s"),
    ]
    for policy, color, marker in styles:
        sub = percentile[percentile["policy"].eq(policy)].sort_values("delta")
        ax.plot(
            sub["delta"],
            sub["frac_within_delta"],
            color=color,
            marker=marker,
            linewidth=1.9,
            markersize=6.5,
            label=policy,
        )
    ax.set_xlabel("Nominal risk budget δ")
    ax.set_ylabel("Fraction of services within δ")
    ax.set_title("Conformal correction vs empirical percentile")
    pct_axis(ax, "x")
    pct_axis(ax, "y")
    ax.set_ylim(0.88, 1.01)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "ls_fig_percentile_vs_conformal_200")


def fig_actuation_delay(delay: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    colors = {"B6: conformal + guardrail": "#4CAF50", "B1: tuned HPA": "#F44336"}
    markers = {"B6: conformal + guardrail": "o", "B1: tuned HPA": "P"}
    for policy in ["B6: conformal + guardrail", "B1: tuned HPA"]:
        sub = delay[delay["policy"].eq(policy)].sort_values("tau")
        ax.plot(
            sub["tau"],
            sub["median_overload_fraction"] * 100.0,
            color=colors[policy],
            marker=markers[policy],
            linewidth=1.9,
            markersize=7,
            label=policy,
        )
        ax.fill_between(
            sub["tau"],
            sub["p10_overload_fraction"] * 100.0,
            sub["p90_overload_fraction"] * 100.0,
            color=colors[policy],
            alpha=0.12,
            linewidth=0,
        )
    ax.axhline(5.0, color="black", linestyle=":", linewidth=1.2, label="δ=0.05")
    ax.set_xlabel("Actuation delay τ (steps)")
    ax.set_ylabel("Realised overload (%)")
    ax.set_title("Actuation-delay sensitivity")
    ax.xaxis.set_major_locator(mticker.FixedLocator([0, 1, 2, 5]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "ls_fig_actuation_delay_200")


def fig_coverage_fraction(frontier: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for model in MODELS:
        sub = frontier[frontier["model"].eq(model)].sort_values("delta")
        ax.plot(
            sub["delta"],
            sub["frac_within_delta"],
            color=MODEL_COLOR[model],
            marker=MODEL_MARKER[model],
            linewidth=1.9,
            markersize=6.5,
            label=MODEL_LABEL[model],
        )
    ax.set_xlabel("Nominal risk budget δ")
    ax.set_ylabel("Fraction of services within δ")
    ax.set_title("Large-scale coverage fraction by forecaster")
    pct_axis(ax, "x")
    pct_axis(ax, "y")
    ax.set_ylim(0.88, 1.01)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    save(fig, "ls_fig_coverage_fraction_200")


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def tabular(df: pd.DataFrame, caption: str, label: str, floatfmt: str = ".3f") -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        df.to_latex(index=False, escape=True, float_format=lambda x: format(x, floatfmt)),
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def write_tables(
    baselines: pd.DataFrame,
    hpa: pd.DataFrame,
    frontier: pd.DataFrame,
    by_stratum: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    REPORTS_LS.mkdir(parents=True, exist_ok=True)

    t1 = baselines.copy()
    t1 = t1[
        ["policy", "median_rel_cost", "median_overload_fraction", "frac_within_delta", "run_length_p99"]
    ].rename(
        columns={
            "policy": "Policy",
            "median_rel_cost": "Rel. cost",
            "median_overload_fraction": "Overload",
            "frac_within_delta": "Within δ",
            "run_length_p99": "p99 run",
        }
    )

    f2 = frontier[
        ["model", "delta", "median_rel_cost", "median_overload_fraction", "frac_within_delta"]
    ].copy()
    f2["Model"] = f2["model"].map(MODEL_LABEL)
    f2 = f2.rename(
        columns={
            "delta": "δ",
            "median_rel_cost": "Rel. cost",
            "median_overload_fraction": "Overload",
            "frac_within_delta": "Within δ",
        }
    )[["Model", "δ", "Rel. cost", "Overload", "Within δ"]]
    h2 = hpa[["policy", "hpa_threshold", "median_rel_cost", "median_overload_fraction", "frac_within_delta"]].copy()
    h2["Model"] = "HPA u=" + h2["hpa_threshold"].map(lambda x: f"{x:.1f}")
    h2["δ"] = 0.05
    h2 = h2.rename(
        columns={
            "median_rel_cost": "Rel. cost",
            "median_overload_fraction": "Overload",
            "frac_within_delta": "Within δ",
        }
    )[["Model", "δ", "Rel. cost", "Overload", "Within δ"]]
    t2 = pd.concat([f2, h2], ignore_index=True)

    t3 = by_stratum.pivot(index="stratum", columns="delta", values="frac_within_delta").reindex(STRATA)
    t3 = t3[[0.10, 0.05, 0.03, 0.01]].reset_index()
    t3.columns = ["Stratum", "δ=0.10", "δ=0.05", "δ=0.03", "δ=0.01"]

    t4 = tests[["comparison", "p_value", "alpha_bonferroni", "reject_H0"]].copy()
    t4 = t4.rename(
        columns={
            "comparison": "Comparison",
            "p_value": "p-value",
            "alpha_bonferroni": "Bonf. α",
            "reject_H0": "Reject",
        }
    )

    content = [
        "% Auto-generated by experiments/src/generate_ls_figures.py",
        tabular(t1, "T-LS-1: Large-scale policy summary at $\\delta=0.05$.", "tab:ls-policy-summary"),
        tabular(t2, "T-LS-2: Forecaster $\\delta$ sweep with tuned HPA references.", "tab:ls-delta-sweep"),
        tabular(t3, "T-LS-3: B6 fraction of services within $\\delta$ by stratum.", "tab:ls-strata"),
        tabular(t4, "T-LS-4: Paired Wilcoxon tests with Bonferroni correction, n=200.", "tab:ls-wilcoxon", floatfmt=".3g"),
    ]
    TABLES_PATH.write_text("\n".join(content))
    print(f"saved {TABLES_PATH}")


def assert_artifacts(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing/empty artifacts: {missing}")


def main() -> None:
    FIGS_LS.mkdir(parents=True, exist_ok=True)
    REPORTS_LS.mkdir(parents=True, exist_ok=True)

    frontier = load_csv("ls_frontier_summary_200.csv")
    coverage = load_csv("ls_coverage_calibration_200.csv")
    by_stratum_frontier = load_csv("ls_frontier_by_stratum_200.csv")
    baselines = load_csv("ls_baselines_summary_200.csv")
    hpa = load_csv("ls_hpa_grid_200.csv")
    percentile = load_csv("ls_percentile_vs_conformal_200.csv")
    delay = load_csv("ls_actuation_delay_200.csv")
    strata_b6 = load_csv("ls_by_stratum_summary_200.csv")
    tests = load_csv("ls_statistical_tests_200.csv")

    fig_cost_risk_frontier(frontier, hpa)
    fig_coverage_calibration(coverage)
    fig_stratum_heatmap(by_stratum_frontier)
    fig_baselines(baselines, hpa)
    fig_percentile_vs_conformal(percentile)
    fig_actuation_delay(delay)
    fig_coverage_fraction(frontier)
    write_tables(baselines, hpa, frontier, strata_b6, tests)

    expected = []
    for name in [
        "ls_fig_cost_risk_frontier_200",
        "ls_fig_coverage_calibration_200",
        "ls_fig_stratum_heatmap_200",
        "ls_fig_baselines_200",
        "ls_fig_percentile_vs_conformal_200",
        "ls_fig_actuation_delay_200",
        "ls_fig_coverage_fraction_200",
    ]:
        expected.extend([FIGS_LS / f"{name}.pdf", FIGS_LS / f"{name}.png"])
    expected.append(TABLES_PATH)
    assert_artifacts(expected)

    print("\noutputs")
    for path in expected:
        print(path)


if __name__ == "__main__":
    main()
