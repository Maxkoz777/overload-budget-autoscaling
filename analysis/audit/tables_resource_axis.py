#!/usr/bin/env python3
"""Generate LaTeX table bodies for the resource-axis extension from saved CSVs only."""
from pathlib import Path
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
EXT = PAPER / "audit/verified_results/resource_axis_study"
OUT = EXT / "tables"
OUT.mkdir(exist_ok=True)
per = pd.read_csv(EXT / "per_service.csv"); per = per[per.phase.eq("test")]
sel = pd.read_csv(EXT / "pretest_selected_per_service.csv")
summ = pd.read_csv(EXT / "summary.csv")
dec = pd.read_csv(EXT / "cost_decomposition.csv")
LAB = {"all200": "All 200", "Q3Q4": "Q3--Q4", "Q4": "Q4", "focused20": "Focused 20"}


def masks(frame):
    return {"all200": frame.index == frame.index, "Q3Q4": frame.pretest_size_quartile.isin(["Q3", "Q4"]),
            "Q4": frame.pretest_size_quartile.eq("Q4"), "focused20": frame.focused.astype(bool)}


# Table: scale strata (main paper)
lines = []
for d in (0.01, 0.05):
    b6 = per[per.config_id.eq(f"conformal_d{d:g}_a1_g1")]
    r = sel[sel.selected_point.eq("reactive_selected") & sel.evaluation_budget.eq(d)]
    mb, mr = masks(b6), masks(r)
    for c in ("all200", "Q3Q4", "Q4"):
        x, y = b6[mb[c]], r[mr[c]]
        lines.append(f"{int(100*d)}\\% & {LAB[c]} & {len(x)} & {100*x.floor_fraction.mean():.1f}\\% & {(x.floor_fraction == 1).sum()} & "
                     f"{(x.overload_fraction <= d).sum()} / {(y.overload_fraction <= d).sum()} & "
                     f"{100*x.overload_fraction.mean():.3f} / {100*y.overload_fraction.mean():.3f} & "
                     f"{x.relative_cost.mean():.4f} / {y.relative_cost.mean():.4f} \\\\")
    if d == 0.01:
        lines.append("\\midrule")
(OUT / "scale_strata_rows.tex").write_text("\n".join(lines) + "\n")

# Table: cost decomposition (supplement)
order = ["B2_pure_predictive", "B7_conformal_margin", "B6_conformal_plus_offset", "B4_gaussian_nominal",
         "B4G_gaussian_nominal_plus_offset", "B1_reactive_pretest_selected"]
names = {"B2_pure_predictive": "B2 pure prediction", "B7_conformal_margin": "B7 conformal margin",
         "B6_conformal_plus_offset": "B6 margin $+$ offset", "B4_gaussian_nominal": "B4 Gaussian (nominal $z$)",
         "B4G_gaussian_nominal_plus_offset": "B4-G Gaussian $+$ offset", "B1_reactive_pretest_selected": "B1 reactive (pre-test)"}
lines = []
for c, d in (("focused20", 0.05), ("all200", 0.01), ("Q4", 0.01)):
    part = dec[dec.cohort.eq(c) & dec.evaluation_budget.eq(d)].set_index("policy").loc[order]
    lines.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{LAB[c]}, $\\delta={d:g}$}}}} \\\\")
    for pol, rrow in part.iterrows():
        lines.append(f"{names[pol]} & {rrow.mean_resource_share:.4f} & {rrow.mean_actuation_share:.4f} & {rrow.mean_violation_share:.4f} & "
                     f"{rrow.mean_relative_cost:.4f} & {100*rrow.mean_overload:.2f}\\% & {int(rrow.compliant)}/{int(rrow.services)} \\\\")
    lines.append("\\midrule")
lines = lines[:-1]
(OUT / "cost_decomposition_rows.tex").write_text("\n".join(lines) + "\n")

# Table: pre-test-selected points (supplement)
lab = {"conformal_selected_g0": "Conformal ($\\alpha$ sel.)", "conformal_selected_g1": "Conformal $+$ offset ($\\alpha$ sel.)",
       "gaussian_selected_g0": "Gaussian ($z$ sel.)", "gaussian_selected_g1": "Gaussian $+$ offset ($z$ sel.)",
       "reactive_selected": "Reactive (B1 rule)"}
lines = []
for c in ("all200", "Q4"):
    for d in (0.01, 0.05):
        part = summ[summ.cohort.eq(c) & summ.evaluation_budget.eq(d) & summ.point_type.eq("pretest_selected")].set_index("config_id").loc[list(lab)]
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{LAB[c]}, $\\delta={d:g}$}}}} \\\\")
        for cid, rrow in part.iterrows():
            lines.append(f"{lab[cid]} & {int(rrow.compliant)}/{int(rrow.services)} & {100*rrow.mean_overload:.3f}\\% & {rrow.mean_norm_resource:.4f} & "
                         f"{rrow.pooled_norm_resource:.4f} & {int(rrow.calibration_infeasible_services)} \\\\")
        lines.append("\\midrule")
lines = lines[:-1]
(OUT / "pretest_selected_rows.tex").write_text("\n".join(lines) + "\n")
print((OUT / "scale_strata_rows.tex").read_text()); print((OUT / "cost_decomposition_rows.tex").read_text()); print((OUT / "pretest_selected_rows.tex").read_text())
