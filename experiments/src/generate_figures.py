"""
generate_figures.py
====================
Generate the publication figures for the paper
"From Forecasts to Guarantees: Risk-Aware Predictive Autoscaling
with Adaptive Safety Margins"

Figures produced
----------------
  fig1_cost_risk_frontier.pdf / .png
  fig2_forecasting_accuracy_by_group.pdf / .png
  fig3_policy_comparison_all.pdf / .png
  fig4_ablation_guardrail.pdf / .png
  fig5_delta_sweep.pdf / .png
  fig6_group_cost_heatmap.pdf / .png
  fig7_underpred_vs_cost.pdf / .png

Usage
-----
    python3 experiments/src/generate_figures.py

Outputs
-------
    experiments/figures/fig*.pdf   (LaTeX-ready vector)
    experiments/figures/fig*.png   (preview, 200 DPI)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

OUT_DIR = Path("experiments/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.9,
    "figure.dpi":         150,
    "savefig.dpi":        200,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
})

# Colour-blind-friendly palette
PAL = {
    "lstm":        "#2196F3",
    "xgb":         "#4CAF50",
    "arima":       "#FF9800",
    "persistence": "#9C27B0",
    "oracle":      "#000000",
    "reactive":    "#F44336",
    "observed":    "#607D8B",
}
MARKERS = {"lstm": "o", "xgb": "s", "arima": "^",
           "persistence": "D", "oracle": "*", "reactive": "P"}
LABELS  = {
    "lstm":        "LSTM",
    "xgb":         "XGBoost",
    "arima":       "ARIMA",
    "persistence": "Persistence",
    "oracle":      "Oracle",
    "reactive":    "Reactive threshold",
    "observed":    "Observed capacity",
}

RESULTS = Path("experiments/results")

adv     = pd.read_csv(RESULTS / "advanced_policy_replay_summary.csv")
risk    = pd.read_csv(RESULTS / "risk_policy_replay_summary.csv")
base    = pd.read_csv(RESULTS / "baseline_replay_summary.csv")
fc      = pd.read_csv(RESULTS / "advanced_forecasting_comparison.csv")
arima_l = pd.read_csv(RESULTS / "arima_training_log.csv")
xgb_l   = pd.read_csv(RESULTS / "xgb_training_log.csv")
lstm_l  = pd.read_csv(RESULTS / "lstm_training_log.csv")
per_svc = pd.read_csv(RESULTS / "advanced_policy_replay_per_service.csv")

GROUP_MAP = {
    "MS_31285": "G1\nStable",   "MS_6298":  "G1\nStable",
    "MS_63525": "G1\nStable",   "MS_8458":  "G1\nStable",
    "MS_70053": "G2\nModerate", "MS_29860": "G2\nModerate",
    "MS_42222": "G2\nModerate", "MS_66711": "G2\nModerate",
    "MS_491":   "G3\nHighVar",  "MS_48534": "G3\nHighVar",
    "MS_19988": "G3\nHighVar",  "MS_21035": "G3\nHighVar",
    "MS_2024":  "G4\nBursty",   "MS_49699": "G4\nBursty",
    "MS_14526": "G4\nBursty",   "MS_12652": "G4\nBursty",
    "MS_51028": "G5\nV.Bursty", "MS_5201":  "G5\nV.Bursty",
    "MS_345":   "G5\nV.Bursty", "MS_25320": "G5\nV.Bursty",
}
GROUP_ORDER = ["G1\nStable", "G2\nModerate", "G3\nHighVar",
               "G4\nBursty", "G5\nV.Bursty"]
fc["group"] = fc["msname"].map(GROUP_MAP)


def save(fig, name: str):
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")


def fig1_cost_risk_frontier():
    """Main contribution figure: overload fraction vs relative cost,
    all models and delta values, compared against key baselines."""

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    deltas = [0.10, 0.05, 0.01]
    models = ["lstm", "xgb", "arima"]

    for model in models:
        costs, overloads = [], []
        for d in deltas:
            row = adv[adv["policy"] == f"{model}_conformal_guardrail_d{d}"]
            if row.empty:
                continue
            costs.append(float(row["median_relative_cost"]))
            overloads.append(float(row["median_overload_fraction"]) * 100)
        ax.plot(overloads, costs,
                color=PAL[model], marker=MARKERS[model],
                linewidth=1.8, markersize=7, label=f"{LABELS[model]} + conformal")
        # Annotate delta values on LSTM line only (to avoid clutter)
        if model == "lstm":
            for i, d in enumerate(deltas):
                ax.annotate(f"δ={d}", xy=(overloads[i], costs[i]),
                            xytext=(4, -10), textcoords="offset points",
                            fontsize=7.5, color=PAL[model])

    # Persistence + conformal
    pers_costs, pers_over = [], []
    for d in deltas:
        row = risk[(risk["policy"] == "risk_conformal_guardrail") & (risk["delta"] == d)]
        if not row.empty:
            pers_costs.append(float(row["median_relative_cost"]))
            pers_over.append(float(row["median_overload_fraction"]) * 100)
    ax.plot(pers_over, pers_costs,
            color=PAL["persistence"], marker=MARKERS["persistence"],
            linewidth=1.8, markersize=7, linestyle="--",
            label=f"{LABELS['persistence']} + conformal")

    # Oracle
    oracle_cost = float(base[base["policy"] == "oracle_demand_capacity"]["median_relative_cost"])
    ax.axhline(oracle_cost, color=PAL["oracle"], linestyle=":", linewidth=1.5,
               label=f"Oracle lower bound ({oracle_cost:.3f})")

    # Reactive threshold baseline
    react_cost = float(base[base["policy"] == "reactive_threshold_u70_cooldown3"]["median_relative_cost"])
    react_over = float(base[base["policy"] == "reactive_threshold_u70_cooldown3"]["median_overload_fraction"]) * 100
    ax.scatter([react_over], [react_cost],
               color=PAL["reactive"], marker=MARKERS["reactive"],
               s=100, zorder=5, label=f"{LABELS['reactive']} ({react_cost:.3f})")
    ax.annotate("Reactive\nthreshold", xy=(react_over, react_cost),
                xytext=(3, 5), textcoords="offset points",
                fontsize=7.5, color=PAL["reactive"])

    ax.set_xlabel("Median Overload Fraction (%)")
    ax.set_ylabel("Median Relative Cost\n(fraction of observed capacity cost)")
    ax.set_title("")
    ax.legend(loc="upper right", framealpha=0.9)

    ax.annotate("← Better SLA", xy=(0.02, 0.15), xytext=(0.02, 0.15),
                fontsize=8, color="grey", style="italic")

    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=0.15, top=0.40)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    fig.tight_layout()
    save(fig, "fig1_cost_risk_frontier")


def fig2_forecasting_accuracy_by_group():
    """Grouped bar chart: median MAE per (model, volatility group)."""

    grp = (fc.groupby(["group", "model"])["mae"]
             .median()
             .reset_index()
             .rename(columns={"mae": "median_mae"}))
    grp["group"] = pd.Categorical(grp["group"], categories=GROUP_ORDER, ordered=True)
    grp = grp.sort_values("group")

    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    x     = np.arange(len(GROUP_ORDER))
    width = 0.24
    offsets = {"arima": -width, "lstm": 0, "xgb": width}

    for model, offset in offsets.items():
        vals = []
        for g in GROUP_ORDER:
            row = grp[(grp["group"] == g) & (grp["model"] == model)]
            vals.append(float(row["median_mae"]) if not row.empty else 0)
        bars = ax.bar(x + offset, vals,
                      width=width * 0.92,
                      color=PAL[model], alpha=0.85,
                      label=LABELS[model])
        for bar, v in zip(bars, vals):
            if v > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=6.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("\n", " ") for g in GROUP_ORDER])
    ax.set_ylabel("Median MAE (cpu_sum units)")
    ax.set_title("Figure 2: Forecasting Accuracy by Volatility Group\n(Median MAE — lower is better)")
    ax.legend()
    ax.set_yscale("symlog", linthresh=10)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save(fig, "fig2_forecasting_accuracy_by_group")


def fig3_policy_comparison():
    """Horizontal bar chart comparing all policies at δ=0.05."""

    policies = {
        "Oracle lower bound":            (float(base[base["policy"]=="oracle_demand_capacity"]["median_relative_cost"]),
                                           float(base[base["policy"]=="oracle_demand_capacity"]["median_overload_fraction"])),
        "LSTM + conformal (δ=0.05)":     (float(adv[adv["policy"]=="lstm_conformal_guardrail_d0.05"]["median_relative_cost"]),
                                           float(adv[adv["policy"]=="lstm_conformal_guardrail_d0.05"]["median_overload_fraction"])),
        "XGBoost + conformal (δ=0.05)":  (float(adv[adv["policy"]=="xgb_conformal_guardrail_d0.05"]["median_relative_cost"]),
                                           float(adv[adv["policy"]=="xgb_conformal_guardrail_d0.05"]["median_overload_fraction"])),
        "Persistence + conformal (δ=0.05)": (float(risk[(risk["policy"]=="risk_conformal_guardrail")&(risk["delta"]==0.05)]["median_relative_cost"]),
                                              float(risk[(risk["policy"]=="risk_conformal_guardrail")&(risk["delta"]==0.05)]["median_overload_fraction"])),
        "ARIMA + conformal (δ=0.05)":    (float(adv[adv["policy"]=="arima_conformal_guardrail_d0.05"]["median_relative_cost"]),
                                           float(adv[adv["policy"]=="arima_conformal_guardrail_d0.05"]["median_overload_fraction"])),
        "Reactive threshold (u70)":       (float(base[base["policy"]=="reactive_threshold_u70_cooldown3"]["median_relative_cost"]),
                                           float(base[base["policy"]=="reactive_threshold_u70_cooldown3"]["median_overload_fraction"])),
        "Static p95 (train+cal)":         (float(base[base["policy"]=="static_p95_train_calibration"]["median_relative_cost"]),
                                           float(base[base["policy"]=="static_p95_train_calibration"]["median_overload_fraction"])),
        "Observed capacity":              (1.000, 0.000),
    }

    labels  = list(policies.keys())
    costs   = [v[0]   for v in policies.values()]
    over    = [v[1]*100 for v in policies.values()]

    # Sort by cost ascending
    order   = np.argsort(costs)
    labels  = [labels[i] for i in order]
    costs   = [costs[i]  for i in order]
    over    = [over[i]   for i in order]

    colours = []
    for lbl in labels:
        if "LSTM"        in lbl: colours.append(PAL["lstm"])
        elif "XGBoost"   in lbl: colours.append(PAL["xgb"])
        elif "ARIMA"     in lbl: colours.append(PAL["arima"])
        elif "Persistence" in lbl: colours.append(PAL["persistence"])
        elif "Oracle"    in lbl: colours.append(PAL["oracle"])
        elif "Reactive"  in lbl: colours.append(PAL["reactive"])
        else:                     colours.append(PAL["observed"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    y = np.arange(len(labels))

    # Left: cost
    bars = ax1.barh(y, costs, color=colours, alpha=0.85, edgecolor="white")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8.5)
    ax1.set_xlabel("Median Relative Cost (fraction of observed)")
    ax1.set_title("Resource Cost")
    ax1.axvline(costs[0], color=PAL["oracle"], linestyle=":", linewidth=1.2)
    for bar, v in zip(bars, costs):
        ax1.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=7.5)

    # Right: overload
    bars2 = ax2.barh(y, over, color=colours, alpha=0.85, edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(labels))
    ax2.set_xlabel("Median Overload Fraction (%)")
    ax2.set_title("SLA Violation Rate")
    for bar, v in zip(bars2, over):
        ax2.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{v:.1f}%", va="center", fontsize=7.5)

    fig.suptitle("Figure 3: Policy Comparison at δ=0.05 (all policies)\n"
                 "Sorted by median relative cost; lower is better for both metrics",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig3_policy_comparison_all")


def fig4_ablation_guardrail():
    """Paired bars showing the effect of adding the guard-rail correction."""

    models = ["arima", "xgb", "lstm"]
    delta  = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    x      = np.arange(len(models))
    width  = 0.35

    for ax_idx, (metric, ylabel, title, scale) in enumerate([
        ("median_relative_cost",   "Median Relative Cost",       "Resource Cost",       None),
        ("median_overload_fraction", "Median Overload Fraction", "SLA Violation Rate",  None),
    ]):
        ax = axes[ax_idx]
        vals_margin = []
        vals_full   = []
        for m in models:
            r1 = adv[adv["policy"] == f"{m}_conformal_margin_only_d{delta}"]
            r2 = adv[adv["policy"] == f"{m}_conformal_guardrail_d{delta}"]
            vals_margin.append(float(r1[metric]) if not r1.empty else 0)
            vals_full.append(  float(r2[metric]) if not r2.empty else 0)

        if metric == "median_overload_fraction":
            vals_margin = [v*100 for v in vals_margin]
            vals_full   = [v*100 for v in vals_full]
            ylabel += " (%)"

        b1 = ax.bar(x - width/2, vals_margin,
                    width=width*0.92, color=[PAL[m] for m in models],
                    alpha=0.55, label="Margin only", hatch="//", edgecolor="white")
        b2 = ax.bar(x + width/2, vals_full,
                    width=width*0.92, color=[PAL[m] for m in models],
                    alpha=0.90, label="Margin + guardrail", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in models])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Annotate reduction
        if metric == "median_overload_fraction":
            for i, (vm, vf) in enumerate(zip(vals_margin, vals_full)):
                if vm > 0:
                    pct = (vm - vf) / vm * 100
                    ax.annotate(f"−{pct:.0f}%",
                                xy=(x[i] + width/2, vf),
                                xytext=(0, 5), textcoords="offset points",
                                ha="center", fontsize=8, color="green",
                                fontweight="bold")

        if ax_idx == 0:
            ax.legend(loc="upper right")

    fig.suptitle(f"Figure 4: Ablation Study — Guard-Rail Contribution (δ={delta})\n"
                 "Hatched: margin-only;  Solid: full policy (margin + guardrail)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig4_ablation_guardrail")


def fig5_delta_sweep():
    """Two-panel figure: cost and overload as a function of δ for all models."""

    deltas = [0.10, 0.05, 0.01]
    models = ["arima", "xgb", "lstm"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    for model in models:
        costs, overs = [], []
        for d in deltas:
            row = adv[adv["policy"] == f"{model}_conformal_guardrail_d{d}"]
            costs.append(float(row["median_relative_cost"]) if not row.empty else np.nan)
            overs.append(float(row["median_overload_fraction"])*100 if not row.empty else np.nan)
        ax1.plot(deltas, costs, color=PAL[model], marker=MARKERS[model],
                 linewidth=1.8, markersize=8, label=LABELS[model])
        ax2.plot(deltas, overs, color=PAL[model], marker=MARKERS[model],
                 linewidth=1.8, markersize=8, label=LABELS[model])

    # Persistence
    p_costs, p_overs = [], []
    for d in deltas:
        row = risk[(risk["policy"] == "risk_conformal_guardrail") & (risk["delta"] == d)]
        p_costs.append(float(row["median_relative_cost"]) if not row.empty else np.nan)
        p_overs.append(float(row["median_overload_fraction"])*100 if not row.empty else np.nan)
    ax1.plot(deltas, p_costs, color=PAL["persistence"], marker=MARKERS["persistence"],
             linewidth=1.8, markersize=8, linestyle="--", label="Persistence")
    ax2.plot(deltas, p_overs, color=PAL["persistence"], marker=MARKERS["persistence"],
             linewidth=1.8, markersize=8, linestyle="--", label="Persistence")

    # Reactive reference line
    react_cost = float(base[base["policy"]=="reactive_threshold_u70_cooldown3"]["median_relative_cost"])
    ax1.axhline(react_cost, color=PAL["reactive"], linestyle=":", linewidth=1.5,
                label=f"Reactive threshold ({react_cost:.3f})")

    oracle_cost = float(base[base["policy"]=="oracle_demand_capacity"]["median_relative_cost"])
    ax1.axhline(oracle_cost, color=PAL["oracle"], linestyle=":", linewidth=1.2,
                label=f"Oracle ({oracle_cost:.3f})")

    for ax, ylabel, title in [
        (ax1, "Median Relative Cost", "Resource Cost vs δ"),
        (ax2, "Median Overload (%)", "Overload Fraction vs δ"),
    ]:
        ax.set_xlabel("Risk budget δ\n(tighter SLA →)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(deltas)
        ax.set_xticklabels([str(d) for d in deltas])
        ax.invert_xaxis()   # δ=0.01 on right = tightest SLA
        ax.legend(fontsize=8)

    fig.suptitle("Figure 5: Cost and Overload vs Risk Budget δ (guardrail variant)\n"
                 "Left x-axis = most permissive (δ=0.10), right = most conservative (δ=0.01)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "fig5_delta_sweep")


def fig6_group_heatmap():
    """Heatmap: relative cost at δ=0.05 (guardrail) for each (model, group)."""

    per_svc["group"] = per_svc["msname"].map(GROUP_MAP)
    key_policies = {
        "ARIMA":    "arima_conformal_guardrail_d0.05",
        "XGBoost":  "xgb_conformal_guardrail_d0.05",
        "LSTM":     "lstm_conformal_guardrail_d0.05",
    }
    rows = []
    for label, policy in key_policies.items():
        sub = per_svc[per_svc["policy"] == policy].copy()
        sub["group"] = pd.Categorical(sub["group"], categories=GROUP_ORDER, ordered=True)
        grp = sub.groupby("group")["relative_cost_vs_observed"].median().reset_index()
        grp["model"] = label
        rows.append(grp)

    hdf = pd.concat(rows)
    pivot = hdf.pivot(index="model", columns="group", values="relative_cost_vs_observed")
    pivot = pivot[GROUP_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5),
                              gridspec_kw={"width_ratios": [1.8, 1.0]})

    # Left: heatmap
    ax = axes[0]
    annot = pivot.applymap(lambda x: f"{x:.2f}")
    sns.heatmap(pivot, ax=ax, annot=annot, fmt="", cmap="RdYlGn_r",
                linewidths=0.5, linecolor="white",
                vmin=0.05, vmax=0.60,
                cbar_kws={"label": "Relative cost\n(fraction of observed)"})
    ax.set_title("Median Relative Cost by Model and Volatility Group\n(δ=0.05, guardrail)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([g.replace("\n", " ") for g in GROUP_ORDER], rotation=20, ha="right")

    # Right: overload heatmap
    ax2 = axes[1]
    rows2 = []
    for label, policy in key_policies.items():
        sub = per_svc[per_svc["policy"] == policy].copy()
        sub["group"] = pd.Categorical(sub["group"], categories=GROUP_ORDER, ordered=True)
        grp = sub.groupby("group")["overload_fraction"].median().reset_index()
        grp["model"] = label
        rows2.append(grp)
    hdf2 = pd.concat(rows2)
    piv2 = hdf2.pivot(index="model", columns="group", values="overload_fraction")
    piv2 = piv2[GROUP_ORDER] * 100
    annot2 = piv2.applymap(lambda x: f"{x:.1f}%")
    sns.heatmap(piv2, ax=ax2, annot=annot2, fmt="", cmap="Reds",
                linewidths=0.5, linecolor="white",
                vmin=0, vmax=30,
                cbar_kws={"label": "Overload fraction (%)"})
    ax2.set_title("Overload\nFraction")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_xticklabels([g.replace("\n", " ") for g in GROUP_ORDER], rotation=20, ha="right")

    fig.suptitle("Figure 6: Per-Volatility-Group Policy Performance (δ=0.05, guardrail variant)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_group_cost_heatmap")


def fig7_underpred_vs_cost():
    """Scatter: median underprediction rate vs policy cost at δ=0.05.
    Illustrates the practical boundary of predictor-agnosticism."""

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    model_underpred = {
        m: float(fc[fc["model"] == m]["underprediction_rate"].median())
        for m in ["arima", "xgb", "lstm"]
    }
    # Persistence underpred approximated from risk policy residuals
    model_underpred["persistence"] = 0.361  # from baseline replay

    for model in ["arima", "xgb", "lstm", "persistence"]:
        if model == "persistence":
            row = risk[(risk["policy"]=="risk_conformal_guardrail") & (risk["delta"]==0.05)]
            cost = float(row["median_relative_cost"]) if not row.empty else np.nan
        else:
            row = adv[adv["policy"] == f"{model}_conformal_guardrail_d0.05"]
            cost = float(row["median_relative_cost"]) if not row.empty else np.nan

        ax.scatter([model_underpred[model] * 100], [cost],
                   color=PAL[model], marker=MARKERS[model],
                   s=120, zorder=5, label=LABELS[model])
        ax.annotate(LABELS[model],
                    xy=(model_underpred[model]*100, cost),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=8.5, color=PAL[model])

    # Oracle reference
    oracle_cost = float(base[base["policy"]=="oracle_demand_capacity"]["median_relative_cost"])
    ax.axhline(oracle_cost, color=PAL["oracle"], linestyle=":", linewidth=1.2,
               label=f"Oracle ({oracle_cost:.3f})")

    # Reactive reference
    react_cost = float(base[base["policy"]=="reactive_threshold_u70_cooldown3"]["median_relative_cost"])
    ax.axhline(react_cost, color=PAL["reactive"], linestyle=":", linewidth=1.2,
               label=f"Reactive threshold ({react_cost:.3f})")

    ax.set_xlabel("Median Underprediction Rate (%)\n(higher → forecaster misses demand spikes more often)")
    ax.set_ylabel("Policy Cost\n(median relative cost at δ=0.05)")
    ax.set_title("")
    ax.legend(loc="lower right")

    # Shade "agnosticism zone"
    ax.axvspan(0, 60, alpha=0.06, color="green", label="_nolegend_")
    ax.text(5, 0.175, "Predictor-agnostic\nzone (≤55% underpred.)",
            fontsize=7.5, color="green", style="italic")
    ax.axvspan(60, 80, alpha=0.06, color="red", label="_nolegend_")
    ax.text(62, 0.175, "Policy\ndegrades",
            fontsize=7.5, color="red", style="italic")

    ax.set_xlim(0, 85)
    ax.set_ylim(0.16, 0.33)
    fig.tight_layout()
    save(fig, "fig7_underpred_vs_cost")


if __name__ == "__main__":
    print(f"\nGenerating figures → {OUT_DIR}/\n")
    fig1_cost_risk_frontier()
    fig2_forecasting_accuracy_by_group()
    fig3_policy_comparison()
    fig4_ablation_guardrail()
    fig5_delta_sweep()
    fig6_group_heatmap()
    fig7_underpred_vs_cost()
    print(f"\nDone. {len(list(OUT_DIR.glob('*.pdf')))} PDF and {len(list(OUT_DIR.glob('*.png')))} PNG files written to {OUT_DIR}/")
