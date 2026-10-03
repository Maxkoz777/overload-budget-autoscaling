#!/usr/bin/env python3
"""Figure: service/unit compliance under closed-loop actuation delay on three traces (R4, R5)."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "verified_results" / "revision_2026-10-02"
C = HERE / "verified_results_canonical_2026-10-03" / "revision_2026-10-02"
if C.exists():  # canonical capacity units (revision_canonical_2026_10_03.py), as reported in the article
    R = C
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "figures" / "verified" / "fig_delay_three_traces"
ali = pd.read_csv(R / "r5_delay_alibaba_summary.csv").query("cohort == 'all200'").assign(dataset="alibaba200")
ex = pd.read_csv(R / "r4_delay_external_summary.csv").query("stratum == 'all'")
g7 = pd.read_csv(R / "r7_pac_grid_summary.csv").query("phase == 'test' and eta == 0.05 and score in ['tau0', 'horizon']")
sel = pd.read_csv(R / "r7_window_selection.csv")
pac = g7.merge(sel[["dataset", "budget", "selected_W"]], left_on=["dataset", "budget", "W"], right_on=["dataset", "budget", "selected_W"])
pac = pac.assign(policy="PAC-h", alpha=1.0)[["dataset", "budget", "policy", "alpha", "tau", "n", "compliant", "mean_relative_cost"]]
s = pd.concat([ali, ex, pac], ignore_index=True)
s["pct"] = 100 * s.compliant / s.n
# B6-h at tau = 0 is B6
b0 = s[(s.policy == "B6") & (s.alpha == 1.0) & (s.tau == 0)].assign(policy="B6-h")
s = pd.concat([s, b0], ignore_index=True)
styles = [("PAC-h", 1.0, r"PAC-h (proposed): PAC rank, horizon-aligned, pre-test $W$, no offset", "#1b7837", "-", "P"),
          ("B6-h", 1.0, "Conformal + offset, horizon-aligned (B6-h)", "#b2182b", "-", "o"),
          ("B6", 1.0, "Conformal + offset, one-step (B6)", "#ef8a62", "--", "s"),
          ("B6", 2.0, r"B6 with $\alpha=2$", "#999999", ":", "^"),
          ("reactive_cal_ext_strict", 1.0, r"Reactive, extended grid, calibrated at $\delta/4$", "#2166ac", "-", "D"),
          ("reactive_cal_orig", 1.0, r"Reactive, pre-test-selected at $\delta$ (B1)", "#67a9cf", "--", "v")]
names = {"alibaba200": "Alibaba (200 services)", "huawei2023": "Huawei (64 functions)", "azure2019": "Azure (2,839 applications)"}
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
for r, budget in enumerate((0.01, 0.05)):
    for c, ds in enumerate(names):
        ax = axes[r, c]
        for pol, a, lab, col, ls, mk in styles:
            t = s[(s.dataset == ds) & (s.budget == budget) & (s.policy == pol) & (s.alpha == a)].sort_values("tau")
            ax.plot(t.tau, t.pct, color=col, ls=ls, marker=mk, ms=3.5, lw=1.4, label=lab)
        ax.set_xticks([0, 1, 2, 5]); ax.grid(alpha=0.25, ls="--")
        ax.set_ylim(30 if ds != "alibaba200" else 70, 101)
        if r == 0:
            ax.set_title(names[ds], fontsize=8.5)
        if c == 0:
            ax.set_ylabel(rf"Within budget (%), $\delta={int(budget*100)}\%$")
        if r == 1:
            ax.set_xlabel(r"Actuation delay $\tau$ (min)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=6.8, frameon=False)
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=220)
print("saved", out)
