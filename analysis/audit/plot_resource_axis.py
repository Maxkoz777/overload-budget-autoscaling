#!/usr/bin/env python3
"""Figures for the resource-axis extension (read-only on result CSVs).

1. fig_resource_overload_envelope: mean held-out overload versus mean
   normalised resource cost for fleet-uniform configurations, the descriptive
   test-selected envelope evaluated exactly at every tested configuration's
   resource value (envelope_exact.csv; the protocol's 30-cap samples in
   envelope.csv are not drawn as a step line), and the pre-test-selected points.
2. fig_guardrail_characterisation_v2: cosmetic re-draw of the frozen guard
   figure from the saved verified CSVs, with the observability floor of the
   recovery metric marked.  The original figure files are left unchanged.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
EXT = PAPER / "audit/verified_results/resource_axis_study"
GUARD = PAPER / "audit/verified_results/guardrail"
FIG = PAPER / "figures/verified"
STYLE = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
FAM = {"conformal": ("#1f5fa8", "o", "Conformal margin family"),
       "gaussian": ("#d0781c", "s", "Gaussian margin family"),
       "reactive": ("#b43333", "^", "Reactive threshold grid"),
       "pure_predictive": ("#555555", "X", "Pure prediction")}


def envelope_figure() -> None:
    s = pd.read_csv(EXT / "summary.csv")
    # exact envelope: steps only at tested configurations' resource values
    e = pd.read_csv(EXT / "envelope_exact.csv")
    plt.rcParams.update(STYLE)
    cohorts = [("all200", "All 200"), ("Q3Q4", "Q3–Q4 (upper half, pre-test)"), ("Q4", "Q4 (upper quarter, pre-test)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75), constrained_layout=True)
    for ax, (cohort, title) in zip(axes, cohorts):
        fu = s[(s.cohort == cohort) & (s.point_type == "fleet_uniform") & (s.evaluation_budget == 0.01)]
        for fam, (color, marker, label) in FAM.items():
            part = fu[fu.family == fam]
            ax.scatter(part.mean_norm_resource, 100 * part.mean_overload, s=13, marker=marker, color=color,
                       alpha=0.35, linewidths=0, label=label)
            env = e[(e.cohort == cohort) & (e.evaluation_budget == 0.01) & (e.family == fam) & e.available]
            if fam != "pure_predictive" and len(env):
                # solid up to the family's largest tested resource; dotted beyond it,
                # where the line is flat only because the grid ends
                top = part.mean_norm_resource.max()
                inside, beyond = env[env.resource_cap <= top], env[env.resource_cap >= top]
                ax.step(inside.resource_cap, 100 * inside.best_mean_overload, where="post", color=color, linewidth=1.1)
                if len(beyond) > 1:
                    ax.step(beyond.resource_cap, 100 * beyond.best_mean_overload, where="post", color=color,
                            linewidth=0.9, linestyle=":")
        for budget, face in ((0.01, "filled"), (0.05, "none")):
            sp = s[(s.cohort == cohort) & (s.point_type == "pretest_selected") & (s.evaluation_budget == budget)]
            for _, r in sp.iterrows():
                if r.config_id.endswith("_g0"):
                    continue  # show the offset variants and reactive only, to keep the panel legible
                color, marker, _ = FAM[r.family]
                ax.scatter(r.mean_norm_resource, 100 * r.mean_overload, s=46, marker=marker,
                           facecolors=color if face == "filled" else "white", edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Mean normalised resource cost")
        ax.grid(alpha=0.2, linestyle="--")
    axes[0].set_ylabel("Mean held-out overload (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], marker="o", color="white", markerfacecolor="#777777", markeredgecolor="black", markersize=6),
                Line2D([], [], marker="o", color="white", markerfacecolor="white", markeredgecolor="black", markersize=6)]
    labels += ["Pre-test selected, δ=1%", "Pre-test selected, δ=5%"]
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_resource_overload_envelope.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def guard_figure_v2() -> None:
    activation = pd.read_csv(GUARD / "activation_by_scale.csv")
    recovery = pd.read_csv(GUARD / "recovery_summary.csv")
    plt.rcParams.update(STYLE)
    colors = {"Q1": "#d9eaf7", "Q2": "#8ecae6", "Q3": "#219ebc", "Q4": "#023047"}
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), constrained_layout=True)
    scale = activation[(activation.period == "test") & (activation.tier == "all200")]
    axes[0].bar(scale.scale_quartile, 100 * scale.median_trigger_rate, color=[colors[q] for q in scale.scale_quartile])
    axes[0].set_ylabel("Median trigger rate (%)")
    axes[0].set_xlabel("Pre-test demand-scale quartile")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Default offset activation (held-out)", fontsize=9)
    variants = ["margin_only", "fixed_guard", "always_plus1", "proportional_5pct"]
    labels = ["Margin only", "Fixed +1", "Always +1", "Proportional 5%"]
    part = recovery[(recovery["shift"] == 1.0) & recovery.variant.isin(variants)].set_index("variant").loc[variants]
    x = np.arange(len(part))
    axes[1].errorbar(x, part.median_observable_end,
                     yerr=[part.median_observable_end - part.p10_observable_end,
                           part.p90_observable_end - part.median_observable_end],
                     fmt="o", capsize=3, color="#7a1f5c")
    axes[1].axhline(59, color="#505050", linestyle=":", linewidth=1.0)
    axes[1].annotate("earliest observable end = 59 steps\n(60-sample window, zero-indexed)", xy=(2.5, 59.2),
                     xytext=(2.45, 100), ha="center", va="center", fontsize=6.5, color="#404040",
                     arrowprops={"arrowstyle": "->", "color": "#707070", "linewidth": 0.7})
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("Observable end after shift (steps)")
    axes[1].set_title("Persistent +100% shift (focused 20)", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_guardrail_characterisation_v2.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    envelope_figure()
    guard_figure_v2()
    print("figures written")
