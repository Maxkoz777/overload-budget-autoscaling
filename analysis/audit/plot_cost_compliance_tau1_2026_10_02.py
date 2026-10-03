#!/usr/bin/env python3
"""Figure: compliance against mean relative replay cost at tau = 1 (data of Table 1; R4, R5, R7)."""
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
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "figures" / "verified" / "fig_cost_compliance_tau1"
TAU = 1
ali = pd.read_csv(R / "r5_delay_alibaba_summary.csv").query("cohort == 'all200'").assign(dataset="alibaba200")
ex = pd.read_csv(R / "r4_delay_external_summary.csv").query("stratum == 'all'")
g7 = pd.read_csv(R / "r7_pac_grid_summary.csv").query("phase == 'test' and eta == 0.05 and score in ['horizon', 'one-step']")
sel = pd.read_csv(R / "r7_window_selection.csv")
pac = g7.merge(sel[["dataset", "budget", "selected_W"]], left_on=["dataset", "budget", "W"], right_on=["dataset", "budget", "selected_W"])
pac = pac.assign(policy=pac.score.map({"horizon": "PAC-h", "one-step": "PAC-1"}), alpha=1.0)
cols = ["dataset", "budget", "policy", "alpha", "tau", "n", "compliant", "mean_relative_cost"]
s = pd.concat([ali[cols], ex[cols], pac[cols]], ignore_index=True).query("tau == @TAU")
s["pct"] = 100 * s.compliant / s.n
styles = [("PAC-h", 1.0, r"PAC-h (proposed)", "#1b7837", "P", 70),
          ("PAC-1", 1.0, "PAC rank, one-step scores", "#a6dba0", "X", 40),
          ("B6-h", 1.0, "B6-h (conformal + offset, horizon-aligned)", "#b2182b", "o", 34),
          ("B6", 1.0, "B6 (one-step)", "#ef8a62", "s", 34),
          ("B6", 2.0, r"B6, $\alpha=2$", "#777777", "^", 38),
          ("reactive_cal_ext_strict", 1.0, r"Reactive, extended grid, cal. $\delta/4$", "#2166ac", "D", 32),
          ("reactive_cal_orig", 1.0, r"Reactive B1, cal. $\delta$", "#67a9cf", "v", 38)]
names = {"alibaba200": "Alibaba (200 services)", "huawei2023": "Huawei (64 functions)", "azure2019": "Azure (2,839 applications)"}
xlab = {"alibaba200": "(vs. observed)", "huawei2023": "(vs. clairvoyant)", "azure2019": "(vs. clairvoyant)"}
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
for r, budget in enumerate((0.01, 0.05)):
    for c, ds in enumerate(names):
        ax = axes[r, c]
        for pol, a, lab, col, mk, sz in styles:
            t = s[(s.dataset == ds) & (s.budget == budget) & (s.policy == pol) & (s.alpha == a)]
            assert len(t) == 1, (ds, budget, pol, a, len(t))
            ax.scatter(t.mean_relative_cost, t.pct, color=col, marker=mk, s=sz, label=lab, zorder=3,
                       edgecolor="black" if pol == "PAC-h" else "none", linewidth=0.5)
        ax.grid(alpha=0.25, ls="--")
        if r == 0:
            ax.set_title(names[ds], fontsize=8.5)
        if c == 0:
            ax.set_ylabel(rf"Within budget (%), $\delta={int(budget*100)}\%$")
        ax.set_xlabel(f"Mean relative cost {xlab[ds]}")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=6.8, frameon=False)
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=220)
print(s.sort_values(["dataset", "budget", "policy"]).to_string(index=False))
print("saved", out)
