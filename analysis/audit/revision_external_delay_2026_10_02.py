#!/usr/bin/env python3
"""R4: closed-loop actuation delay on the Huawei 2023 and Azure Functions 2019 replication traces.

Protocol: verified_results/revision_2026-10-02/protocol.json, key "R4_external_delay_addendum".
Semantics follow the canonical external replay (external_replication_2026_09_30.py) exactly at
tau = 0, which is checked against the stored per-unit rows before anything is written.

Usage:  python3 audit/revision_external_delay_2026_10_02.py [huawei2023] [azure2019]
"""
from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import external_replication_2026_09_30 as ext  # noqa: E402

PAPER = HERE.parent
VER = PAPER / "audit" / "verified_results"
OUT = VER / "revision_2026-10-02"
TAUS = (0, 1, 2, 5)
BUDGETS = (0.01, 0.05)
W = 240
ORIG_U = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
NEW_U = (0.2, 0.3, 0.4)


def check(label, got, want, tol=0.0):
    ok = abs(float(got) - float(want)) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: got {got}, expected {want}", flush=True)
    if not ok:
        raise SystemExit(f"regression check failed: {label}")


def select(grid: pd.DataFrame, budget: float) -> pd.DataFrame:
    rows = []
    for uid, g in grid.groupby("unit_id", sort=True):
        feas = g[g.overload_fraction <= budget]
        if len(feas):
            r = feas.sort_values(["relative_cost", "overload_fraction", "threshold", "cooldown"], ascending=[True, True, False, True]).iloc[0]
        else:
            r = g.sort_values(["overload_fraction", "relative_cost", "threshold", "cooldown"], ascending=[True, True, False, True]).iloc[0]
        rows.append({"unit_id": uid, "threshold": float(r.threshold), "cooldown": int(r.cooldown), "calibration_feasible": bool(len(feas))})
    return pd.DataFrame(rows)


def costs(cap_sum, churn, over, T, denom):
    total = ext.C_RES * cap_sum + ext.C_ACT * churn + ext.C_VIO * over
    return over / T, total / denom, over


def margins(y: np.ndarray, t0: int, T: int, h: int, ranks: tuple[int, ...], start: int) -> dict[int, np.ndarray]:
    """Rank statistics of the last W h-step positive residuals observable before each decision.

    Decision index i (test step, may be negative for warm start) is made after observing y[t0+i-1];
    the window holds scores max(y[j]-y[j-h], 0) for j = t0+i-W .. t0+i-1."""
    pos = np.maximum(y[:, h:] - y[:, :-h], 0.0)        # pos[:, k] = score with j = k + h
    out = {r: np.empty((y.shape[0], T - start)) for r in ranks}
    kth = [r - 1 for r in ranks]
    for col, i in enumerate(range(start, T)):
        g = t0 + i
        win = pos[:, g - h - W: g - h]
        part = np.partition(win, kth, axis=1)
        for r in ranks:
            out[r][:, col] = part[:, r - 1]
    return out


def closed_loop_margin(nominal: np.ndarray, dem: np.ndarray, mu: np.ndarray, tau: int, guard: bool):
    """nominal[:, i + tau] is the decision rule's nominal capacity for test step i (index shifted by tau)."""
    n, T = dem.shape
    dec = np.zeros((n, T + tau))
    dec[:, :tau] = nominal[:, :tau]                     # warm start: decisions made before the test
    s1 = np.zeros(n, bool); s2 = np.zeros(n, bool)
    cap_sum = np.zeros(n); churn = np.zeros(n); over = np.zeros(n); prev = None
    for i in range(T):
        d = nominal[:, i + tau].copy()
        if guard and i >= 2:
            d = d + ext.GAMMA * (s1 & s2)
        dec[:, i + tau] = d
        cap = dec[:, i]                                 # decision made at step i - tau
        soft = dem[:, i] > ext.RHO * mu * cap
        over += dem[:, i] > mu * cap
        cap_sum += cap
        if prev is not None:
            churn += np.abs(cap - prev)
        prev = cap.copy()
        s2 = s1; s1 = soft
    return cap_sum, churn, over


def closed_loop_reactive(prev0, cap0, dem, mu, u, c, tau):
    n, T = dem.shape
    cur = cap0.astype(float).copy(); last = -c.astype(float); prev = prev0.astype(float).copy()
    umu = u * mu
    queue = [cap0.astype(float).copy() for _ in range(tau)]
    cap_sum = np.zeros(n); churn = np.zeros(n); over = np.zeros(n); before = None
    for s in range(T):
        desired = np.maximum(np.ceil(prev / umu), 1.0)
        change = ((s - last) >= c) & (desired != cur)
        cur = np.where(change, desired, cur)
        last = np.where(change, s, last)
        if tau:
            queue.append(cur.copy()); cap = queue.pop(0)
        else:
            cap = cur
        over += dem[:, s] > mu * cap
        cap_sum += cap
        if before is not None:
            churn += np.abs(cap - before)
        before = cap.copy()
        prev = dem[:, s].copy()
    return cap_sum, churn, over


def run(family: str) -> pd.DataFrame:
    print(f"R4 {family}", flush=True)
    X, mu, units, split, _ = ext.load_family(family, "primary")
    X = X.astype(np.float64); mu = mu.astype(float)
    t0, t1 = split["test"]
    y = X[:, :t1]; dem = X[:, t0:t1]; n, T = dem.shape
    clair = np.maximum(np.ceil(dem / mu[:, None]), 1.0)
    denom = ext.C_RES * clair.sum(1) + ext.C_ACT * np.abs(np.diff(clair, axis=1)).sum(1) + ext.C_VIO * (dem > mu[:, None] * clair).sum(1)
    uid = units.unit_id.to_numpy()
    ranks = tuple(sorted({ext.conf_rank(W, d) for d in BUDGETS}))
    rows = []

    def add(budget, policy, alpha, tau, res):
        frac, rel, cnt = costs(*res, T, denom)
        for k in range(n):
            rows.append((family, budget, policy, alpha, tau, uid[k], frac[k], rel[k], int(cnt[k])))

    for h in sorted({t + 1 for t in TAUS}):
        taus_here = [t for t in TAUS if t + 1 == h or (h == 1)]
        maxtau = max(TAUS)
        m = margins(y, t0, T, h, ranks, start=-maxtau)  # columns: decision index -maxtau .. T-1
        forecast = y[:, t0 - maxtau - 1: t1 - 1]          # y[t0+i-1] for i = -maxtau .. T-1
        for budget in BUDGETS:
            r = ext.conf_rank(W, budget)
            for alpha in ((1.0, 2.0) if h == 1 else (1.0,)):
                nominal_full = np.maximum(np.ceil((forecast + alpha * m[r]) / mu[:, None]), 1.0)
                for tau in TAUS:
                    if h == 1:
                        label = None  # one-step score: B6/B7 for every tau
                    elif tau + 1 != h:
                        continue
                    nominal = nominal_full[:, maxtau - tau:]     # starts at decision index -tau
                    for guard, base in ((True, "B6"), (False, "B7")):
                        if alpha == 2.0 and not guard:
                            continue
                        name = base if h == 1 else base + "-h"
                        add(budget, name, alpha, tau, closed_loop_margin(nominal, dem, mu, tau, guard))
        print(f"  h={h} done", flush=True)

    # reactive comparators
    grid = pd.read_csv(OUT / f"r2_{family}_reactive_grid.csv.gz")
    cal = grid[grid.phase == "calibration"]
    cap0 = np.maximum(1.0, np.ceil(X[:, t0 - 1] / (0.8 * mu)))
    for budget in BUDGETS:
        variants = (("reactive_cal_orig", cal[cal.threshold.isin(ORIG_U)], budget),
                    ("reactive_cal_ext", cal, budget),
                    ("reactive_cal_ext_strict", cal, budget / 4))
        for name, g, b in variants:
            ch = select(g, b).set_index("unit_id").loc[uid]
            u = ch.threshold.to_numpy(float); c = ch.cooldown.to_numpy(float)
            for tau in TAUS:
                add(budget, name, 1.0, tau, closed_loop_reactive(X[:, t0 - 1], cap0, dem, mu, u, c, tau))
            if name != "reactive_cal_ext_strict":
                test = grid[grid.phase == "test"].merge(ch.reset_index()[["unit_id", "threshold", "cooldown"]], on=["unit_id", "threshold", "cooldown"])
                mine = pd.DataFrame([r for r in rows if r[1] == budget and r[2] == name and r[4] == 0],
                                    columns=["f", "b", "p", "a", "t", "unit_id", "of", "rel", "cnt"])
                mrg = test.merge(mine, on="unit_id", validate="one_to_one")
                check(f"{family} d={budget} {name} tau=0 overload counts", int((mrg.cnt - mrg.overload_count).abs().max()), 0)
                check(f"{family} d={budget} {name} tau=0 rel. cost", float((mrg.rel - mrg.relative_cost).abs().max()), 0.0, 1e-9)

    per = pd.DataFrame(rows, columns=["dataset", "budget", "policy", "alpha", "tau", "unit_id", "overload_fraction", "relative_cost", "overload_count"])
    # regression of margin policies at tau = 0 against the stored external replay
    ref = pd.read_csv(VER / "external_replication_2026-09-30" / f"per_unit_{family}_primary.csv.gz")
    for budget in BUDGETS:
        for policy, alpha, cid in (("B6", 1.0, f"conformal_W240_d{budget:g}_a1_g1"), ("B6", 2.0, f"conformal_W240_d{budget:g}_a2_g1"),
                                   ("B7", 1.0, f"conformal_W240_d{budget:g}_a1_g0"), ("B6-h", 1.0, f"conformal_W240_d{budget:g}_a1_g1")):
            if policy == "B6-h":
                continue
            r = ref[(ref.phase == "test") & (ref.config_id == cid)][["unit_id", "overload_count", "relative_cost"]]
            mine = per[(per.budget == budget) & (per.policy == policy) & (per.alpha == alpha) & (per.tau == 0)]
            mrg = r.merge(mine, on="unit_id", suffixes=("_ref", ""), validate="one_to_one")
            check(f"{family} {cid} tau=0 overload counts", int((mrg.overload_count - mrg.overload_count_ref).abs().max()), 0)
            check(f"{family} {cid} tau=0 rel. cost", float((mrg.relative_cost - mrg.relative_cost_ref).abs().max()), 0.0, 1e-9)
    return per


def summarise(per: pd.DataFrame) -> pd.DataFrame:
    strata = []
    for fam in per.dataset.unique():
        u = pd.read_csv(ext.EXT / fam / "units.csv")
        if fam == "huawei2023":
            v = u.mean_requests_pre / u.mu_p50
            u["stratum"] = np.where(v >= v.median(), "upper_half", "lower_half")
        else:
            u["stratum"] = u.group
        strata.append(u[["unit_id", "stratum"]].assign(dataset=fam))
    per = per.merge(pd.concat(strata), on=["dataset", "unit_id"], validate="many_to_one")
    per["within"] = per.overload_fraction <= per.budget
    rows = []
    for cohort_key in ("all", "stratum"):
        keys = ["dataset", "budget", "policy", "alpha", "tau"] + ([] if cohort_key == "all" else ["stratum"])
        for k, g in per.groupby(keys, sort=True):
            d = dict(zip(keys, k))
            d.setdefault("stratum", "all")
            rows.append({**d, "n": len(g), "compliant": int(g.within.sum()), "mean_relative_cost": float(g.relative_cost.mean()),
                         "median_relative_cost": float(g.relative_cost.median()), "mean_overload": float(g.overload_fraction.mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fams = [a for a in sys.argv[1:] if a in ("huawei2023", "azure2019")] or ["huawei2023", "azure2019"]
    parts = []
    for fam in fams:
        per = run(fam)
        per.to_csv(OUT / f"r4_delay_{fam}_per_unit.csv.gz", index=False)
        parts.append(per)
    allper = pd.concat([pd.read_csv(OUT / f"r4_delay_{f}_per_unit.csv.gz") for f in ("huawei2023", "azure2019")
                        if (OUT / f"r4_delay_{f}_per_unit.csv.gz").exists()], ignore_index=True)
    summ = summarise(allper)
    summ.to_csv(OUT / "r4_delay_external_summary.csv", index=False)
    (OUT / "methodology_r4.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                                          "protocol": "protocol.json#R4_external_delay_addendum"}, indent=2) + "\n")
    s = summ[summ.stratum == "all"]
    print(s.to_string(index=False))
