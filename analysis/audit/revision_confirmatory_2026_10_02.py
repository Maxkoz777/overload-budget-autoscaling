#!/usr/bin/env python3
"""R8: confirmatory replay on untouched Huawei 2023 runs (days 0-18, 147-165, 168-184).

Protocol (frozen before the run, with this file's SHA-256): verified_results/revision_2026-10-02/protocol.json,
key "R8_confirmatory". Proposed method: margin-only PAC rank (eta = 0.05) with horizon-aligned scores, window
chosen by the pre-test rule on calibration days 8-9. Comparators: B6, B6-h (W = 240, conformal rank, offset),
pre-test-selected reactive B1 (original grid, budget delta) and the strict reactive controller (extended grid,
calibration budget delta/4).
Usage:  python3 audit/revision_confirmatory_2026_10_02.py
"""
from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_2026_10_02 as rev  # noqa: E402
import revision_external_delay_2026_10_02 as xd  # noqa: E402
import revision_round2_2026_10_02 as r7  # noqa: E402
ext = xd.ext

M = 1440
RUNS = ("A_days000_018", "B_days147_165", "C_days168_184")
SRC = HERE.parent.parent / "experiments" / "data" / "external_traces" / "huawei2023_confirmatory"
OUT = rev.OUT
BUDGETS = (0.01, 0.05)
TAUS = (0, 1, 5)
ETA = 0.05


def phase(run: str, which: str) -> dict:
    z = np.load(SRC / run / "series.npz", allow_pickle=True)
    units = pd.read_csv(SRC / run / "units.csv")
    assert list(units.unit_id) == list(z["unit_id"])
    X = z["demand"].astype(np.float64); mu = units.mu_p50.to_numpy(float)
    t0, t1 = (8 * M, 10 * M) if which == "cal" else (10 * M, X.shape[1])
    dem = X[:, t0:t1]
    clair = np.maximum(np.ceil(dem / mu[:, None]), 1.0)
    denom = ext.C_RES * clair.sum(1) + ext.C_ACT * np.abs(np.diff(clair, axis=1)).sum(1) + ext.C_VIO * (dem > mu[:, None] * clair).sum(1)
    return dict(uid=units.unit_id.to_numpy(), y=X[:, :t1], t0=t0, T=t1 - t0, mu=mu, init=None, warm=True, denom=denom, X=X)


def pac_rows(D, W, delta, taus, horizon: bool):
    r = r7.pac_rank(W, delta, ETA)
    maxtau = max(taus)
    forecast = D["y"][:, D["t0"] - maxtau - 1: D["t0"] + D["T"] - 1]
    out = {}
    for tau in taus:
        h = tau + 1 if horizon else 1
        m = r7.margins(D, W, h, (r,), start=-maxtau)[r]
        nom = np.maximum(np.ceil((forecast + m) / D["mu"][:, None]), 1.0)
        out[tau] = r7.evaluate(D, nom, tau, maxtau)
    return out


def b6_rows(D, budget, horizon: bool, taus):
    y, t0, T, mu = D["y"], D["t0"], D["T"], D["mu"]
    dem = y[:, t0:t0 + T]
    maxtau = max(taus)
    forecast = y[:, t0 - maxtau - 1: t0 + T - 1]
    rank = ext.conf_rank(240, budget)
    out = {}
    for tau in taus:
        h = tau + 1 if horizon else 1
        m = xd.margins(y, t0, T, h, (rank,), start=-maxtau)[rank]
        nominal = np.maximum(np.ceil((forecast + m) / mu[:, None]), 1.0)[:, maxtau - tau:]
        frac, rel, _ = xd.costs(*xd.closed_loop_margin(nominal, dem, mu, tau, True), T, D["denom"])
        out[tau] = (frac, rel)
    return out


def reactive_rows(Dc, Dt, budget_sel, thresholds, taus):
    X, mu = Dt["X"], Dt["mu"]
    c0 = Dc["t0"]
    cap0c = np.maximum(1.0, np.ceil(X[:, c0 - 1] / (0.8 * mu)))
    grid = rev.reactive_vec(X[:, c0 - 1], cap0c, X[:, c0:c0 + Dc["T"]], mu, thresholds, (1, 3, 5, 10))
    grid["unit_id"] = Dc["uid"][grid.unit_idx]
    grid["relative_cost"] = grid.total_cost / Dc["denom"][grid.unit_idx]
    ch = xd.select(grid, budget_sel).set_index("unit_id").loc[Dt["uid"]]
    t0 = Dt["t0"]
    cap0 = np.maximum(1.0, np.ceil(X[:, t0 - 1] / (0.8 * mu)))
    out = {}
    for tau in taus:
        frac, rel, _ = xd.costs(*xd.closed_loop_reactive(X[:, t0 - 1], cap0, X[:, t0:t0 + Dt["T"]], mu,
                                                         ch.threshold.to_numpy(float), ch.cooldown.to_numpy(float), tau), Dt["T"], Dt["denom"])
        out[tau] = (frac, rel)
    return out, ch


def main() -> None:
    rows, selections = [], []
    for run in RUNS:
        Dc, Dt = phase(run, "cal"), phase(run, "test")
        for budget in BUDGETS:
            # pre-test window selection on calibration days (eta = 0.05, tau = 0, one-step scores)
            cand = []
            for W in r7.WINDOWS:
                if not r7.admissible(W, budget, ETA) or Dc["t0"] - 6 - W < 0:
                    continue
                frac, rel = pac_rows(Dc, W, budget, (0,), horizon=False)[0]
                cand.append((W, int((frac <= budget).sum()), float(rel.mean())))
            Wsel = sorted(cand, key=lambda c: (-c[1], c[2]))[0][0]
            selections.append({"run": run, "budget": budget, "selected_W": Wsel,
                               "candidates": ";".join(f"{W}:{k}/{c:.3f}" for W, k, c in cand)})
            policies = {
                "PAC-h (pre-test W)": pac_rows(Dt, Wsel, budget, TAUS, True),
                "PAC-h (W=1440)": pac_rows(Dt, 1440, budget, TAUS, True),
                "B6": b6_rows(Dt, budget, False, TAUS),
                "B6-h": b6_rows(Dt, budget, True, TAUS),
            }
            policies["Reactive B1"], _ = reactive_rows(Dc, Dt, budget, rev.ORIG_U, TAUS)
            policies["Reactive strict"], _ = reactive_rows(Dc, Dt, budget / 4, rev.ORIG_U + rev.NEW_U, TAUS)
            for name, res in policies.items():
                for tau, (frac, rel) in res.items():
                    for k, u in enumerate(Dt["uid"]):
                        rows.append({"run": run, "budget": budget, "policy": name, "tau": tau, "unit_id": u,
                                     "overload_fraction": float(frac[k]), "relative_cost": float(rel[k])})
            print(f"{run} delta={budget} W_sel={Wsel} done", flush=True)
    per = pd.DataFrame(rows)
    per["within"] = per.overload_fraction <= per.budget
    per["util"] = per.overload_fraction / per.budget
    per.to_csv(OUT / "r8_confirmatory_per_unit.csv.gz", index=False)
    pd.DataFrame(selections).to_csv(OUT / "r8_window_selection.csv", index=False)
    summ = []
    for cohort, part in [*[(r, per[per.run == r]) for r in RUNS], ("pooled", per)]:
        for k, grp in part.groupby(["budget", "policy", "tau"]):
            summ.append({"cohort": cohort, **dict(zip(["budget", "policy", "tau"], k)), "n": len(grp),
                         "compliant": int(grp.within.sum()), "mean_relative_cost": grp.relative_cost.mean(),
                         "mean_overload": grp.overload_fraction.mean(), "mean_util": grp.util.mean(), "median_util": grp.util.median()})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "r8_confirmatory_summary.csv", index=False)
    # pre-declared hypotheses
    hyp = []
    for budget in BUDGETS:
        for tau in TAUS:
            a = per[(per.policy == "PAC-h (pre-test W)") & (per.budget == budget) & (per.tau == tau)].set_index(["run", "unit_id"])
            b = per[(per.policy == "Reactive strict") & (per.budget == budget) & (per.tau == tau)].set_index(["run", "unit_id"]).loc[a.index]
            wins, losses = int((a.within & ~b.within).sum()), int((~a.within & b.within).sum())
            hyp.append({"budget": budget, "tau": tau, "n": len(a), "pac_compliant": int(a.within.sum()),
                        "strict_reactive_compliant": int(b.within.sum()), "pac_only": wins, "reactive_only": losses,
                        "exact_p_one_sided": binomtest(wins, wins + losses, alternative="greater").pvalue if wins + losses else 1.0,
                        "pac_fraction": float(a.within.mean()), "H2_pac_at_least_95pct": bool(a.within.mean() >= 0.95),
                        "cost_ratio_pac_over_reactive": float(a.relative_cost.mean() / b.relative_cost.mean())})
    pd.DataFrame(hyp).to_csv(OUT / "r8_confirmatory_hypotheses.csv", index=False)
    (OUT / "methodology_r8.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}, indent=2) + "\n")
    print(pd.DataFrame(selections).to_string(index=False))
    print(pd.DataFrame(hyp).to_string(index=False))
    print(summ[summ.cohort == "pooled"].to_string(index=False))


if __name__ == "__main__":
    main()
