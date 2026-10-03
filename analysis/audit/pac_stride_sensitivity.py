#!/usr/bin/env python3
"""R9: stride-(tau+1) PAC rank and a cost-aware window rule (sensitivities).

Protocol: verified_results/delay_pac_study/protocol.json, key "R9_round3_addendum".
Usage:  python3 audit/pac_stride_sensitivity.py
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delay_and_reactive_grid as rev  # noqa: E402
import pac_window_grid as r7  # noqa: E402
import confirmatory_replay as r8  # noqa: E402

OUT = rev.OUT
ETA = 0.05
BUDGETS = (0.01, 0.05)
STRIDE_W = (960, 1440, 2880)
TAUS = (0, 1, 5)


def stride_margin(D, W, h, rank, start):
    y, t0, T = D["y"], D["t0"], D["T"]
    pos = np.maximum(y[:, h:] - y[:, :-h], 0.0)
    weff = W // h
    out = np.empty((y.shape[0], T - start))
    offs = h * np.arange(weff)
    for col, i in enumerate(range(start, T)):
        g = t0 + i
        idx = (g - h - 1) - offs
        out[:, col] = np.partition(pos[:, idx], rank - 1, axis=1)[:, rank - 1]
    return out


def stride_rows(D, tag, maxtau=5):
    rows = []
    forecast = D["y"][:, D["t0"] - maxtau - 1: D["t0"] + D["T"] - 1]
    for tau in (1, 5):
        h = tau + 1
        for W in STRIDE_W:
            weff = W // h
            for d in BUDGETS:
                if not r7.admissible(weff, d, ETA):
                    continue
                r = r7.pac_rank(weff, d, ETA)
                m = stride_margin(D, W, h, r, -maxtau)
                nom = np.maximum(np.ceil((forecast + m) / D["mu"][:, None]), 1.0)
                frac, rel = r7.evaluate(D, nom, tau, maxtau)
                for k, u in enumerate(D["uid"]):
                    rows.append((tag, W, weff, d, r, tau, "stride", u, frac[k], rel[k]))
        print(f"  stride {tag} tau={tau} done", flush=True)
    return rows


def overlap_rows(D, tag, maxtau=5):
    """Overlapping-score PAC-h at every admissible W (needed for the confirmatory periods)."""
    rows = []
    forecast = D["y"][:, D["t0"] - maxtau - 1: D["t0"] + D["T"] - 1]
    for W in r7.WINDOWS:
        combos = [(d, r7.pac_rank(W, d, ETA)) for d in BUDGETS if r7.admissible(W, d, ETA)]
        if not combos:
            continue
        for tau in TAUS:
            h = tau + 1
            ranks = tuple(sorted({c[1] for c in combos}))
            m = r7.margins(D, W, h, ranks, start=-maxtau)
            for d, r in combos:
                nom = np.maximum(np.ceil((forecast + m[r]) / D["mu"][:, None]), 1.0)
                frac, rel = r7.evaluate(D, nom, tau, maxtau)
                for k, u in enumerate(D["uid"]):
                    rows.append((tag, W, W, d, r, tau, "overlap", u, frac[k], rel[k]))
        print(f"  overlap {tag} W={W} done", flush=True)
    return rows


def cal_rows(Dc, tag):
    rows = []
    for W in r7.WINDOWS:
        for d in BUDGETS:
            if not r7.admissible(W, d, ETA) or Dc["t0"] - 6 - W < 0:
                continue
            frac, rel = r8.pac_rows(Dc, W, d, (0,), horizon=False)[0]
            rows.append({"dataset": tag, "W": W, "budget": d, "cal_compliant": int((frac <= d).sum()),
                         "n": len(frac), "cal_mean_cost": float(rel.mean())})
    return rows


COLS = ["dataset", "W", "W_eff", "budget", "rank", "tau", "variant", "unit_id", "overload_fraction", "relative_cost"]


def main():
    rows, cal = [], []
    for ds in ("alibaba200", "huawei2023", "azure2019"):
        D = r7.load(ds, "test")
        rows += stride_rows(D, ds)
    for run in r8.RUNS:
        Dt, Dc = r8.phase(run, "test"), r8.phase(run, "cal")
        rows += stride_rows(Dt, run)
        rows += overlap_rows(Dt, run)
        cal += cal_rows(Dc, run)
    per = pd.DataFrame(rows, columns=COLS)
    per.to_csv(OUT / "r9_stride_and_confirm_grid.csv.gz", index=False)
    # regression: confirmatory overlap rows at the R8-selected W must reproduce R8
    r8per = pd.read_csv(OUT / "r8_confirmatory_per_unit.csv.gz")
    sel8 = pd.read_csv(OUT / "r8_window_selection.csv")
    a = r8per[r8per.policy == "PAC-h (pre-test W)"].merge(sel8[["run", "budget", "selected_W"]], on=["run", "budget"])
    b = per[per.variant == "overlap"].rename(columns={"dataset": "run"})
    m = a.merge(b, left_on=["run", "budget", "tau", "unit_id", "selected_W"], right_on=["run", "budget", "tau", "unit_id", "W"], suffixes=("_r8", ""))
    rev.check("R9 vs R8 rows", len(m), len(a))
    rev.check("R9 vs R8 max diff", float((m.overload_fraction - m.overload_fraction_r8).abs().max() + (m.relative_cost - m.relative_cost_r8).abs().max()), 0.0, 1e-9)

    # combine with R7 for the main traces
    g7 = pd.concat([pd.read_csv(OUT / f"r7_pac_grid_{ds}.csv.gz") for ds in ("alibaba200", "huawei2023", "azure2019")])
    main_overlap = g7[(g7.phase == "test") & (g7.eta == 0.05) & g7.score.isin(["tau0", "horizon"])]
    main_overlap = main_overlap.assign(W_eff=main_overlap.W, variant="overlap")[COLS]
    allrows = pd.concat([per, main_overlap], ignore_index=True)
    allrows["within"] = allrows.overload_fraction <= allrows.budget
    allrows["group"] = np.where(allrows.dataset.str.startswith(("A_", "B_", "C_")), "huawei_confirmatory", allrows.dataset)
    summ = allrows.groupby(["group", "W", "W_eff", "budget", "rank", "tau", "variant"]).agg(
        n=("unit_id", "size"), compliant=("within", "sum"), mean_relative_cost=("relative_cost", "mean"),
        mean_overload=("overload_fraction", "mean")).reset_index()
    summ.to_csv(OUT / "r9_stride_summary.csv", index=False)

    # cost-aware window rule
    c7 = g7[(g7.phase == "cal") & (g7.eta == 0.05) & (g7.tau == 0)]
    c7 = c7.assign(within=c7.overload_fraction <= c7.budget).groupby(["dataset", "W", "budget"]).agg(
        cal_compliant=("within", "sum"), n=("unit_id", "size"), cal_mean_cost=("relative_cost", "mean")).reset_index()
    calall = pd.concat([c7, pd.DataFrame(cal)], ignore_index=True)
    sel = []
    for (ds, d), grp in calall.groupby(["dataset", "budget"]):
        ok = grp[grp.cal_compliant >= 0.99 * grp.n]
        best = (ok if len(ok) else grp).sort_values(["cal_mean_cost"]).iloc[0]
        sel.append({"dataset": ds, "budget": d, "cost_rule_W": int(best.W)})
    sel = pd.DataFrame(sel)
    sel.to_csv(OUT / "r9_cost_rule_selection.csv", index=False)
    ov = allrows[allrows.variant == "overlap"].merge(sel, on=["dataset", "budget"])
    ov = ov[ov.W == ov.cost_rule_W]
    cr = ov.groupby(["group", "budget", "tau"]).agg(n=("unit_id", "size"), compliant=("within", "sum"),
                                                    mean_relative_cost=("relative_cost", "mean")).reset_index()
    cr.to_csv(OUT / "r9_cost_rule_summary.csv", index=False)
    (OUT / "methodology_r9.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}, indent=2) + "\n")
    print(sel.to_string(index=False)); print(cr.to_string(index=False))
    print(summ[summ.tau > 0].to_string(index=False))


if __name__ == "__main__":
    main()
