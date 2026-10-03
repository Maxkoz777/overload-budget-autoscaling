#!/usr/bin/env python3
"""Figure and LaTeX table rows for the external replication (read-only on result CSVs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "audit/verified_results/external_replication_2026-09-30"
FIG = PAPER / "figures/verified"
FAM = {"conformal": ("#1f5fa8", "o", "Conformal margin family"),
       "gaussian": ("#d0781c", "s", "Gaussian margin family"),
       "reactive": ("#b43333", "^", "Reactive threshold grid"),
       "pure_predictive": ("#555555", "X", "Pure prediction")}
LABEL = {"huawei2023": "Huawei Cloud 2023 (64 functions)", "azure2019": "Azure Functions 2019 (2,839 apps)"}


def figure() -> None:
    fu = pd.read_csv(OUT / "fleet_uniform_primary.csv")
    env = pd.read_csv(OUT / "envelope_exact_primary.csv")
    s = pd.read_csv(OUT / "summary_policies.csv")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), constrained_layout=True)
    for ax, fam in zip(axes, ("huawei2023", "azure2019")):
        f = fu[(fu.family_trace == fam) & (fu.cohort == "all")]
        e = env[(env.family_trace == fam) & (env.cohort == "all")]
        for k, (color, marker, label) in FAM.items():
            part = f[f.family == k]
            ax.scatter(part.mean_norm_resource, 100 * part.mean_overload, s=13, marker=marker, color=color,
                       alpha=0.35, linewidths=0, label=label)
            if k == "pure_predictive":
                continue
            top = part.mean_norm_resource.max()
            line = e[(e.family == k) & e.available]
            inside, beyond = line[line.resource_cap <= top], line[line.resource_cap >= top]
            ax.step(inside.resource_cap, 100 * inside.best_mean_overload, where="post", color=color, linewidth=1.1)
            if len(beyond) > 1:
                ax.step(beyond.resource_cap, 100 * beyond.best_mean_overload, where="post", color=color, linewidth=0.9, linestyle=":")
        for budget, face in ((0.01, "filled"), (0.05, "none")):
            for pol, k in (("B6", "conformal"), ("B1", "reactive")):
                r = s[(s.family == fam) & (s.mu == "primary") & (s.budget == budget) & (s.policy == pol)].iloc[0]
                color, marker, _ = FAM[k]
                ax.scatter(r.mean_norm_resource, 100 * r.mean_overload, s=48, marker=marker,
                           facecolors=color if face == "filled" else "white", edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_yscale("log")
        ax.set_title(LABEL[fam], fontsize=8.5)
        ax.set_xlabel("Mean resource cost relative to clairvoyant")
        ax.grid(alpha=0.2, linestyle="--")
    axes[0].set_ylabel("Mean held-out overload (%)")
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color="white", markerfacecolor="#777777", markeredgecolor="black", markersize=6),
          Line2D([], [], marker="o", color="white", markerfacecolor="white", markeredgecolor="black", markersize=6)]
    l += ["B6 / pre-test B1, δ=1%", "B6 / pre-test B1, δ=5%"]
    fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_external_replication.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def tables() -> None:
    s = pd.read_csv(OUT / "summary_policies.csv")
    T = OUT / "tables"; T.mkdir(exist_ok=True)
    fams = (("huawei2023", "primary", "Huawei 2023"), ("azure2019", "primary", "Azure Fn.\\ 2019"))
    lines = []
    for fam, mu, name in fams:
        for d in (0.01, 0.05):
            g = s[(s.family == fam) & (s.mu == mu) & (s.budget == d)].set_index("policy")
            n = int(g.loc["B6", "n"])
            cells = " & ".join(f"{int(g.loc[p, 'compliant'])}" for p in ("B6", "invcdf_G", "B4G", "conf1440", "B1"))
            lines.append(f"{name} & {int(100*d)}\\% & {n} & {cells} & {100*g.loc['B6','mean_overload']:.2f} / {100*g.loc['B1','mean_overload']:.2f} & "
                         f"{g.loc['B6','mean_relative_cost']:.2f} / {g.loc['B1','mean_relative_cost']:.2f} \\\\")
        lines.append("\\midrule")
    (T / "replication_main_rows.tex").write_text("\n".join(lines[:-1]) + "\n")
    # sensitivity rows (supplement)
    lines = []
    for fam, mu, name in (("huawei2023", "p90", "Huawei, $\\mu$ from p90"), ("azure2019", "K5", "Azure, $K=5$"), ("azure2019", "K20", "Azure, $K=20$")):
        for d in (0.01, 0.05):
            g = s[(s.family == fam) & (s.mu == mu) & (s.budget == d)].set_index("policy")
            cells = " & ".join(f"{int(g.loc[p, 'compliant'])}" for p in ("B6", "invcdf_G", "B4G", "conf1440", "B1"))
            lines.append(f"{name} & {int(100*d)}\\% & {int(g.loc['B6','n'])} & {cells} & {g.loc['B6','mean_relative_cost']:.2f} / {g.loc['B1','mean_relative_cost']:.2f} \\\\")
    (T / "replication_sensitivity_rows.tex").write_text("\n".join(lines) + "\n")
    # strata rows (supplement)
    st = pd.read_csv(OUT / "summary_strata.csv")
    lines = []
    names = {"lower_half": "Huawei lower half", "upper_half": "Huawei upper half", "http": "Azure http", "timer": "Azure timer",
             "queue": "Azure queue", "other": "Azure other"}
    for fam, strata in (("huawei2023", ("lower_half", "upper_half")), ("azure2019", ("http", "timer", "queue", "other"))):
        for stt in strata:
            for d in (0.01,):
                g = st[(st.family == fam) & (st.mu == "primary") & (st.budget == d) & (st.stratum == stt)].set_index("policy")
                cells = " & ".join(f"{int(g.loc[p, 'compliant'])}" for p in ("B6", "invcdf_G", "B4G", "B1"))
                lines.append(f"{names[stt]} & {int(g.loc['B6','n'])} & {cells} & {g.loc['B6','mean_relative_cost']:.2f} / {g.loc['B1','mean_relative_cost']:.2f} \\\\")
    (T / "replication_strata_rows.tex").write_text("\n".join(lines) + "\n")
    for f in sorted(T.glob("replication_*")):
        print(f.name); print(f.read_text())


if __name__ == "__main__":
    figure()
    tables()
