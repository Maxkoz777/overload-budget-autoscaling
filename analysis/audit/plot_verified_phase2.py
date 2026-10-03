#!/usr/bin/env python3
"""Render only verified persistence/comparator figures for the manuscript."""

from __future__ import annotations

import os
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "figures" / "verified"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PAPER / "tmp" / "mplconfig"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = PAPER / "audit" / "verified_results" / "baselines"
COMP = PAPER / "audit" / "verified_results" / "comparative"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = pd.read_csv(BASE / "persistence_baselines_summary.csv")
    test = pd.read_csv(BASE / "persistence_baselines_per_service.csv")
    reactive = pd.read_csv(COMP / "reactive_selected_summary.csv")

    palette = {"B6_conformal_guard": "#2e7d32", "B4_gaussian_guard": "#ad1457",
               "B9_linear_percentile_guard": "#ef6c00"}
    labels = {"B6_conformal_guard": "Conformal + guard", "B4_gaussian_guard": "Gaussian + guard",
              "B9_linear_percentile_guard": "Linear percentile + guard"}
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    short_labels = {"B6_conformal_guard": "Conformal", "B4_gaussian_guard": "Gaussian",
                    "B9_linear_percentile_guard": "Percentile"}
    names = list(palette)
    for delta, offset, colour, marker in ((0.05, -0.10, "#1565c0", "o"), (0.01, 0.10, "#ef6c00", "s")):
        part = summary[summary.subset.eq("all200") & summary.delta.eq(delta)]
        values = [float(part[part.policy.eq(name)].service_compliance.iloc[0]) for name in names]
        ax.scatter(np.arange(3) + offset, 100 * np.asarray(values), color=colour,
                   marker=marker, s=60, zorder=3)
        for pos, value in enumerate(values):
            ax.text(pos + offset, 100 * value + (0.9 if delta == 0.05 else -1.6),
                    f"{value:.0%}" if value == 1 else f"{value:.1%}",
                    ha="center", fontsize=7)
    ax.set_xticks(np.arange(3), [short_labels[name] for name in names])
    ax.set_ylim(85, 104)
    ax.set_ylabel("Services within budget (%)")
    ax.grid(axis="y", alpha=0.2)
    save(fig, "fig_verified_coverage_200")

    frame = test[test.policy.eq("B6_conformal_guard") & test.delta.eq(0.05)]
    strata = frame.groupby("stratum", sort=True).agg(
        services=("service_id", "size"), compliance=("within_delta", "mean"),
        cost=("relative_cost", "median"), mean_overload=("overload_fraction", "mean")
    )
    fig, ax = plt.subplots(figsize=(3.6, 2.6), constrained_layout=True)
    x = np.arange(len(strata))
    ax.bar(x, strata.cost, color="#1565c0")
    ax.set_ylabel("Median relative cost")
    ax.set_ylim(0, 0.56)
    ax.set_xticks(x, strata.index)
    ax.set_xlabel("Burstiness stratum (full-period)")
    ax.grid(axis="y", alpha=0.2)
    save(fig, "fig_verified_stratum_200")

    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    for name in ("B6_conformal_guard", "B9_linear_percentile_guard"):
        part = summary[summary.subset.eq("all200") & summary.policy.eq(name)].sort_values("delta")
        ax.plot(100 * part.delta, 100 * part.service_compliance, marker="o", linewidth=1.8,
                label=labels[name], color=palette[name])
    ax.set_xticks([1, 3, 5, 10])
    ax.set_xlabel("Declared risk budget δ (%)")
    ax.set_ylabel("Services within temporal budget (%)")
    ax.set_ylim(90, 101)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, "fig_verified_percentile_vs_conformal_200")
    print(strata.to_string())
    print(reactive[reactive.subset.eq("all200")].to_string(index=False))


if __name__ == "__main__":
    main()
