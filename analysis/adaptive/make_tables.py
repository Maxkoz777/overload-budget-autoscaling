"""LaTeX tables of Additional file 1, Sect. S12, from out_v1/*_summary.csv (reporting only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive import policies as pol

R = "out_v1"
s = pd.read_csv(f"{R}/test_summary.csv")
pp = pd.read_csv(f"{R}/confirm_pooled_summary.csv"); pp["trace"] = "periods"
cp = pd.read_csv(f"{R}/confirm_summary.csv")
ALL = pd.concat([s, pp])
TR = ["alibaba200", "huawei2023", "azure2019", "periods"]
TRN = {"alibaba200": "Alibaba (200)", "huawei2023": "Huawei (64)", "azure2019": "Azure (2,839)", "periods": "Huawei periods (195)"}
NAME = {"PAC-h (no bound)": "PAC-h, no bound", "PAC-h": "PAC-h", "ACI": "ACI", "ACI-G": "ACI-G", "ACI-G (R1)": "ACI-G, rule R1", "BACC-r": "BACC-r"}


def fmt_int(x):
    return f"{int(x):,}"


def cell(r, ref=False):
    if r is None:
        return "n/a"
    if ref:
        return fmt_int(r.within)
    mark = r"\textsuperscript{a}" if r.trace == "periods" and r.n < 195 else ""
    return f"{fmt_int(r.within)}{mark} / {r.paired_median_ratio:.2f} / {r.ratio_of_means:.2f}"


def get(tr, b, tau, p):
    x = ALL[(ALL.trace == tr) & (ALL.budget == b) & (ALL.tau == tau) & (ALL.policy == p)]
    return None if x.empty else x.iloc[0]


lines = []
for b in (0.01, 0.05):
    lines.append(r"\begin{table*}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}")
    lines.append(rf"\caption{{Adaptive comparators at $\delta={int(b*100)}\%$: units within budget / paired median cost ratio / ratio of mean costs, both relative to PAC-h under the common resource bound.}}")
    lines.append(rf"\label{{tab:adaptive_grid_{int(b*100)}}}")
    lines.append(r"\begin{tabular}{@{}llcccc@{}}\toprule")
    lines.append(r"$\tau$ & Policy & " + " & ".join(TRN[t] for t in TR) + r" \\ \midrule")
    for tau in (0, 1, 5):
        pols = ["PAC-h (no bound)", "PAC-h", "ACI", "ACI-G", "BACC-r"]
        if tau == 5:
            pols.insert(4, "ACI-G (R1)")
        for i, p in enumerate(pols):
            ref = p.startswith("PAC-h")
            cells = [cell(get(t, b, tau, p), ref) for t in TR]
            lines.append((f"{tau}" if i == 0 else "") + f" & {NAME[p]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace" if tau != 5 else r"\bottomrule")
    lines.append(r"\end{tabular}\par\vspace{3pt}")
    feas = [fmt_int(get(t, b, 1, "PAC-h").feasible_under_bound) for t in TR]
    lines.append(r"\begin{minipage}{\textwidth}\footnotesize\raggedright PAC-h rows give units within budget only. Ratios are over units where both the policy and bounded PAC-h comply. "
                 r"n/a: the primary step-size rule gives $\alpha^\star\le0$ for this cell. Units whose budget is attainable at constant capacity $M_u$: "
                 + ", ".join(f"{TRN[t]} {f}" for t, f in zip(TR, feas)) + r". \textsuperscript{a}Periods A and B (124 unit-runs); for period C the primary rule gives $\alpha^\star\le0$.\end{minipage}")
    lines.append(r"\end{table*}")
    lines.append("")

# BACC-r sensitivities
VN = [("BACC-r", "primary"), ("BACC-r V1", "$c=0.1\\mu$"), ("BACC-r V2", "$c=10\\mu$"), ("BACC-r V3", "no change limit"),
      ("BACC-r V4", "HPA-style limit"), ("BACC-r V5", "$A_k$ over missed minutes"), ("BACC-r V6", "no trigger"), ("BACC-r V9", "multiplicative replan")]
lines.append(r"\begin{table*}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}")
lines.append(r"\caption{BACC-r sensitivity variants at $\tau=1$: units within budget; for Azure also paired median cost ratio / ratio of mean costs relative to bounded PAC-h.}")
lines.append(r"\label{tab:bacc_variants}")
lines.append(r"\begin{tabular}{@{}lcccccccc@{}}\toprule")
lines.append(r" & \multicolumn{4}{c}{$\delta=1\%$} & \multicolumn{4}{c}{$\delta=5\%$} \\ \cmidrule(lr){2-5}\cmidrule(l){6-9}")
lines.append(r"Variant & Alibaba & Huawei & Azure & Azure cost & Alibaba & Huawei & Azure & Azure cost \\ \midrule")
for p, lab in VN:
    row = [lab]
    for b in (0.01, 0.05):
        for t in ("alibaba200", "huawei2023", "azure2019"):
            row.append(fmt_int(get(t, b, 1, p).within))
        r = get("azure2019", b, 1, p)
        row.append(f"{r.paired_median_ratio:.2f} / {r.ratio_of_means:.2f}")
    lines.append(" & ".join(row) + r" \\")
pac = ["bounded PAC-h"] + [fmt_int(get(t, b, 1, "PAC-h").within) if k < 3 else "1 / 1" for b in (0.01, 0.05) for k, t in enumerate(("alibaba200", "huawei2023", "azure2019", None))]
lines.append(r"\midrule " + " & ".join(pac) + r" \\ \bottomrule")
lines.append(r"\end{tabular}\end{table*}")
lines.append("")

# ACI diagnostics at tau = 1
lines.append(r"\begin{table*}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}")
lines.append(r"\caption{ACI and ACI-G diagnostics at $\tau=1$. max: largest formal miscoverage of a unit; $>\delta$: units whose formal miscoverage exceeds $\delta$; infinite: share of unit-minutes with an infinite set (capacity $M_u$); exc.: units over budget although their formal miscoverage is within $\delta$, because of overload in exceptional minutes.}")
lines.append(r"\label{tab:aci_diag}")
lines.append(r"\begin{tabular}{@{}llcccccccc@{}}\toprule")
lines.append(r" & & \multicolumn{4}{c}{ACI} & \multicolumn{4}{c}{ACI-G} \\ \cmidrule(lr){3-6}\cmidrule(l){7-10}")
lines.append(r"$\delta$ & Trace & max & $>\delta$ & infinite & exc. & max & $>\delta$ & infinite & exc. \\ \midrule")
for b in (0.01, 0.05):
    for i, t in enumerate(TR):
        row = [f"{int(b*100)}\\%" if i == 0 else "", TRN[t]]
        for p in ("ACI", "ACI-G"):
            r = get(t, b, 1, p)
            row += [f"{r.max_formal_err:.4f}", fmt_int(r.units_formal_err_over), f"{100*r.infinite_set_share:.1f}\\%", fmt_int(r.units_over_budget_with_formal_err_within)]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\addlinespace" if b == 0.01 else r"\bottomrule")
lines.append(r"\end{tabular}\end{table*}")
lines.append("")

# Huawei periods, tau = 1
lines.append(r"\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}")
lines.append(r"\caption{Huawei periods (post hoc baseline analysis), $\tau=1$: unit-runs within budget per period (days 0--18, 147--165, 168--184).}")
lines.append(r"\label{tab:adaptive_periods}")
lines.append(r"\begin{tabular}{@{}lcccccc@{}}\toprule")
lines.append(r" & \multicolumn{3}{c}{$\delta=1\%$} & \multicolumn{3}{c}{$\delta=5\%$} \\ \cmidrule(lr){2-4}\cmidrule(l){5-7}")
lines.append(r"Policy & A (57) & B (67) & C (71) & A & B & C \\ \midrule")
for p in ("PAC-h (no bound)", "PAC-h", "ACI", "ACI-G", "BACC-r"):
    row = [NAME[p]]
    for b in (0.01, 0.05):
        for per in ("A_days000_018", "B_days147_165", "C_days168_184"):
            x = cp[(cp.trace == per) & (cp.budget == b) & (cp.tau == 1) & (cp.policy == p)].iloc[0]
            row.append(fmt_int(x.within))
    lines.append(" & ".join(row) + r" \\")
lines.append(r"\bottomrule\end{tabular}\end{table}")
lines.append("")

# gamma
cal = pd.concat([pd.read_csv(f"{R}/cal_traces.csv"), pd.read_csv(f"{R}/cal_periods.csv")])
T = {"alibaba200": 4320, "huawei2023": 33120, "azure2019": 5760}
lines.append(r"\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}")
lines.append(r"\caption{Step size of ACI-G chosen on calibration days 8--9 (primary rule) and target level $\alpha^\star$ on the test period by delay; R1 gives the step size used where $\alpha^\star\le0$.}")
lines.append(r"\label{tab:aci_gamma}")
lines.append(r"\begin{tabular}{@{}llcccc@{}}\toprule")
lines.append(r"Trace & $\delta$ & $\gamma$ & $\alpha^\star$, $\tau=0$ & $\alpha^\star$, $\tau=1$ & $\alpha^\star$, $\tau=5$ (R1 $\gamma$) \\ \midrule")
for t in ("alibaba200", "huawei2023", "azure2019"):
    for b in (0.01, 0.05):
        c = cal[(cal.unit == t) & (cal.budget == b)].sort_values("order")
        g0 = float(c.iloc[0].gamma)
        a = [pol.alpha_star(b, g0, tau, T[t]) for tau in (0, 1, 5)]
        r1 = next(float(g) for g in c.gamma if pol.alpha_star(b, float(g), 5, T[t]) > 0)
        a5 = f"{a[2]:.4f}" if a[2] > 0 else f"n/a ({r1:g}: {pol.alpha_star(b, r1, 5, T[t]):.4f})"
        lines.append(f"{TRN[t] if b == 0.01 else ''} & {int(b*100)}\\% & {g0:g} & {a[0]:.4f} & {a[1]:.4f} & {a5} \\\\")
lines.append(r"\bottomrule\end{tabular}\end{table}")
open(f"{R}/tables_S12.tex", "w").write("\n".join(lines) + "\n")
print("written", len(lines))
