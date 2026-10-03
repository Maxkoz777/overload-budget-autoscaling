#!/usr/bin/env python3
"""R5: Alibaba delayed actuation at delta in {0.01, 0.05} with the same policy set as R4.

Protocol: verified_results/delay_pac_study/protocol.json, key "R5_alibaba_delay_symmetric_addendum".
Usage:  python3 audit/alibaba_delay.py
"""
from __future__ import annotations

import json
import platform
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delay_and_reactive_grid as rev  # noqa: E402
g = rev.g

BUDGETS = (0.01, 0.05)
SERVICES: list[dict] = []
PARAMS: dict = {}


def _service(index: int) -> list[dict]:
    s = SERVICES[index]
    sid = s["service_id"]
    rows = []
    for budget in BUDGETS:
        g.DELTA = budget
        for tau in rev.TAUS:
            def add(policy, alpha, r):
                rows.append({"service_id": sid, "budget": budget, "policy": policy, "alpha": alpha, "tau": tau,
                             "overload_fraction": r["overload_fraction"], "relative_cost": r["relative_cost"]})
            for alpha in (1.0, 2.0):
                add("B6", alpha, g.simulate_conformal(s["history"], s["test"], correction="fixed_plus1", alpha=alpha, tau=tau))
            add("B7", 1.0, g.simulate_conformal(s["history"], s["test"], correction="none", alpha=1.0, tau=tau))
            if tau:
                add("B6-h", 1.0, rev.simulate_horizon_aligned(s["history"], s["test"], correction="fixed_plus1", tau=tau))
                add("B7-h", 1.0, rev.simulate_horizon_aligned(s["history"], s["test"], correction="none", tau=tau))
            for name, (u, c) in PARAMS[(sid, budget)].items():
                add(name, 1.0, g.simulate_reactive_closed_loop(s["history"], s["test"], u, c, tau))
    return rows


def main() -> None:
    global SERVICES, PARAMS
    g.DATA = rev.DATA
    _, SERVICES = g.load_services()
    grid = pd.read_csv(rev.OUT / "r2_alibaba_reactive_grid.csv.gz")
    grid["service_id"] = grid.service_id.astype(str)
    cal = grid[grid.phase == "calibration"]
    for budget in BUDGETS:
        for name, gg, b in (("reactive_cal_orig", cal[cal.threshold.isin(rev.ORIG_U)], budget),
                            ("reactive_cal_ext", cal, budget), ("reactive_cal_ext_strict", cal, budget / 4)):
            ch = rev.select(gg, "service_id", b).set_index("service_id")
            for s in SERVICES:
                PARAMS.setdefault((s["service_id"], budget), {})[name] = (float(ch.loc[s["service_id"], "threshold"]), int(ch.loc[s["service_id"], "cooldown"]))
    with Pool(2) as pool:
        parts = pool.map(_service, range(len(SERVICES)), chunksize=5)
    per = pd.DataFrame([r for p in parts for r in p])
    per["within"] = per.overload_fraction <= per.budget

    # regression checks
    r1 = pd.read_csv(rev.OUT / "r1_delay_strict_per_service.csv")
    r3 = pd.read_csv(rev.OUT / "r3_horizon_aligned_per_service.csv")
    for ref, pols in ((r1, {"B6": "B6", "B7": "B7", "reactive_cal1pct": "reactive_cal_orig"}), (r3, {"B6-h": "B6-h", "B7-h": "B7-h"})):
        for rp, mp in pols.items():
            a = ref[(ref.policy == rp) & (ref.alpha == 1.0) & (ref.tau > (0 if mp.endswith("-h") else -1))][["service_id", "tau", "overload_fraction", "relative_cost"]]
            b = per[(per.budget == 0.01) & (per.policy == mp) & (per.alpha == 1.0)]
            m = a.merge(b, on=["service_id", "tau"], suffixes=("_ref", ""), validate="one_to_one")
            rev.check(f"R5 vs R1/R3 {mp} rows", len(m), len(a))
            rev.check(f"R5 vs R1/R3 {mp} max diff", float((m.overload_fraction - m.overload_fraction_ref).abs().max() + (m.relative_cost - m.relative_cost_ref).abs().max()), 0.0, 1e-12)
    s8 = pd.read_csv(rev.VER / "guardrail" / "closed_loop_delay_per_service.csv")
    for alpha in (1.0, 2.0):
        a = s8[(s8.policy == "conformal_fixed_guard") & (s8.alpha == alpha)][["service_id", "tau", "overload_fraction", "relative_cost"]]
        b = per[(per.budget == 0.05) & (per.policy == "B6") & (per.alpha == alpha)]
        m = a.merge(b, on=["service_id", "tau"], suffixes=("_ref", ""), validate="one_to_one")
        rev.check(f"R5 vs S8 B6 alpha={alpha} rows", len(m), len(a))
        rev.check(f"R5 vs S8 B6 alpha={alpha} max diff", float((m.overload_fraction - m.overload_fraction_ref).abs().max() + (m.relative_cost - m.relative_cost_ref).abs().max()), 0.0, 1e-12)

    quart = pd.read_csv(rev.VER / "strict_budget" / "strict_budget_per_service.csv")[["service_id", "pretest_size_quartile"]].drop_duplicates()
    crit = pd.read_csv(rev.OUT / "subset_pretest_criterion.csv")
    per = per.merge(quart, on="service_id").merge(crit[["service_id", "nontrivial_pretest"]], on="service_id")
    rows = []
    for cohort, part in (("all200", per), ("Q4", per[per.pretest_size_quartile == "Q4"]), ("nontrivial95", per[per.nontrivial_pretest])):
        for k, grp in part.groupby(["budget", "policy", "alpha", "tau"], sort=True):
            rows.append({"cohort": cohort, **dict(zip(["budget", "policy", "alpha", "tau"], k)), "n": len(grp), "compliant": int(grp.within.sum()),
                         "mean_relative_cost": float(grp.relative_cost.mean()), "mean_overload": float(grp.overload_fraction.mean())})
    per.to_csv(rev.OUT / "r5_delay_alibaba_per_service.csv.gz", index=False)
    pd.DataFrame(rows).to_csv(rev.OUT / "r5_delay_alibaba_summary.csv", index=False)
    (rev.OUT / "methodology_r5.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}, indent=2) + "\n")
    print(pd.DataFrame(rows).query("cohort=='all200'").to_string(index=False))


if __name__ == "__main__":
    main()
