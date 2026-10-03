#!/usr/bin/env python3
"""R6: training-conditional (PAC) residual rank, alone and with horizon-aligned scores, on three traces.

Protocol: verified_results/revision_2026-10-02/protocol.json, key "R6_pac_rank_addendum".
Usage:  python3 audit/revision_pac_rank_2026_10_02.py [alibaba200] [huawei2023] [azure2019]
"""
from __future__ import annotations

import json
import platform
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revision_2026_10_02 as rev  # noqa: E402
import revision_external_delay_2026_10_02 as xd  # noqa: E402
g, ext = rev.g, xd.ext

ETA = 0.05
TAUS = (0, 1, 5)
BUDGETS = (0.01, 0.05)
W_PRIMARY = {0.01: 480, 0.05: 240}
W_SECONDARY = 1440


def pac_rank(W: int, delta: float, eta: float = ETA) -> int | None:
    for l in range(1, W + 1):
        if binom.sf(l - 1, W, 1 - delta) <= eta:
            return l
    return None


def conf_rank(W: int, delta: float) -> int:
    return min(W, int(np.ceil((W + 1) * (1 - delta))))


# (name, W, rank-function, offset) per budget
def configs(budget: float) -> list[tuple[str, int, int, str]]:
    wp = W_PRIMARY[budget]
    return [("B7-PAC", wp, pac_rank(wp, budget), "none"),
            ("B6-PAC", wp, pac_rank(wp, budget), "fixed_plus1"),
            ("B7-PAC-1440", W_SECONDARY, pac_rank(W_SECONDARY, budget), "none"),
            ("B7-check", 240, conf_rank(240, budget), "none")]          # regression arm


# Alibaba
def simulate(history: pd.DataFrame, evaluation: pd.DataFrame, *, W: int, level: int, correction: str, tau: int) -> dict:
    h = tau + 1
    hist = history.cpu_sum.to_numpy(float)
    demand = evaluation.cpu_sum.to_numpy(float)
    y = np.r_[hist, demand]
    n = len(hist)
    residuals = np.maximum(hist[h:] - hist[:-h], 0.0).tolist()
    previous = float(hist[-1])
    pending = deque([max(int(np.ceil(float(history.replica_count.iloc[-1]))), 1)] * tau)
    over_hist, soft_hist, applied = [], [], []
    for i, actual in enumerate(demand):
        window = np.asarray(residuals[-W:], dtype=float)
        lv = min(len(window), level)
        margin = float(np.partition(window, lv - 1)[lv - 1])
        nominal = max(1, int(np.ceil(previous + margin)))
        trigger = False
        if correction != "none" and len(over_hist) >= g.H:
            trigger = all(over_hist[-g.H:]) or all(soft_hist[-g.H:])
        decision = nominal + g.correction_value(correction, nominal, trigger)
        if tau:
            pending.append(decision); capacity = int(pending.popleft())
        else:
            capacity = decision
        applied.append(capacity)
        over_hist.append(bool(actual > g.MU * capacity))
        soft_hist.append(bool(actual > 0.70 * g.MU * capacity))
        residuals.append(max(float(actual - y[n + i - h]), 0.0))
        previous = float(actual)
    cap = np.asarray(applied, dtype=int)
    over = demand > g.MU * cap
    return {"overload_fraction": float(over.mean()), "relative_cost": g.policy_cost(demand, cap) / g.observed_cost(evaluation)}


SERVICES: list[dict] = []


def _service(i: int) -> list[dict]:
    s = SERVICES[i]
    rows = []
    for budget in BUDGETS:
        for name, W, level, corr in configs(budget):
            for tau in TAUS:
                if name == "B7-check" and tau not in (0, 1):
                    continue
                r = simulate(s["history"], s["test"], W=W, level=level, correction=corr, tau=tau)
                rows.append({"dataset": "alibaba200", "unit_id": s["service_id"], "budget": budget, "policy": name, "W": W,
                             "rank": level, "tau": tau, **r})
    return rows


def run_alibaba() -> pd.DataFrame:
    global SERVICES
    g.DATA = rev.DATA
    _, SERVICES = g.load_services()
    with Pool(2) as pool:
        parts = pool.map(_service, range(len(SERVICES)), chunksize=4)
    per = pd.DataFrame([r for p in parts for r in p])
    ref = pd.read_csv(rev.OUT / "r5_delay_alibaba_per_service.csv.gz")
    for budget in BUDGETS:
        for tau, pol in ((0, "B7"), (1, "B7-h")):
            a = ref[(ref.budget == budget) & (ref.policy == pol) & (ref.tau == tau)][["service_id", "overload_fraction", "relative_cost"]]
            b = per[(per.budget == budget) & (per.policy == "B7-check") & (per.tau == tau)].rename(columns={"unit_id": "service_id"})
            m = a.merge(b, on="service_id", suffixes=("_ref", ""), validate="one_to_one")
            rev.check(f"alibaba d={budget} {pol} tau={tau} rows", len(m), 200)
            rev.check(f"alibaba d={budget} {pol} tau={tau} max diff",
                      float((m.overload_fraction - m.overload_fraction_ref).abs().max() + (m.relative_cost - m.relative_cost_ref).abs().max()), 0.0, 1e-12)
    return per


# External traces
def margins_w(y, t0, T, h, W, ranks, start):
    pos = np.maximum(y[:, h:] - y[:, :-h], 0.0)
    out = {r: np.empty((y.shape[0], T - start)) for r in ranks}
    kth = sorted({r - 1 for r in ranks})
    for col, i in enumerate(range(start, T)):
        gidx = t0 + i
        part = np.partition(pos[:, gidx - h - W: gidx - h], kth, axis=1)
        for r in ranks:
            out[r][:, col] = part[:, r - 1]
    return out


def run_external(family: str) -> pd.DataFrame:
    X, mu, units, split, _ = ext.load_family(family, "primary")
    X = X.astype(np.float64); mu = mu.astype(float)
    t0, t1 = split["test"]
    y = X[:, :t1]; dem = X[:, t0:t1]; n, T = dem.shape
    clair = np.maximum(np.ceil(dem / mu[:, None]), 1.0)
    denom = ext.C_RES * clair.sum(1) + ext.C_ACT * np.abs(np.diff(clair, axis=1)).sum(1) + ext.C_VIO * (dem > mu[:, None] * clair).sum(1)
    uid = units.unit_id.to_numpy()
    maxtau = max(TAUS)
    forecast = y[:, t0 - maxtau - 1: t1 - 1]
    rows = []
    # group configurations by window so that each (W, h) is partitioned once
    plan: dict[int, list[tuple[float, str, int, str]]] = {}
    for budget in BUDGETS:
        for name, W, level, corr in configs(budget):
            plan.setdefault(W, []).append((budget, name, level, corr))
    for W, items in sorted(plan.items()):
        ranks = tuple(sorted({lv for _, _, lv, _ in items}))
        for tau in TAUS:
            h = tau + 1
            m = margins_w(y, t0, T, h, W, ranks, start=-maxtau)
            for budget, name, level, corr in items:
                if name == "B7-check" and tau not in (0, 1):
                    continue
                nominal = np.maximum(np.ceil((forecast + m[level]) / mu[:, None]), 1.0)[:, maxtau - tau:]
                frac, rel, cnt = xd.costs(*xd.closed_loop_margin(nominal, dem, mu, tau, corr != "none"), T, denom)
                for k in range(n):
                    rows.append((family, uid[k], budget, name, W, level, tau, frac[k], rel[k], int(cnt[k])))
            print(f"  {family} W={W} tau={tau} done", flush=True)
    per = pd.DataFrame(rows, columns=["dataset", "unit_id", "budget", "policy", "W", "rank", "tau", "overload_fraction", "relative_cost", "overload_count"])
    ref = pd.read_csv(rev.OUT / f"r4_delay_{family}_per_unit.csv.gz")
    for budget in BUDGETS:
        for tau, pol in ((0, "B7"), (1, "B7-h")):
            a = ref[(ref.budget == budget) & (ref.policy == pol) & (ref.tau == tau)][["unit_id", "overload_count", "relative_cost"]]
            b = per[(per.budget == budget) & (per.policy == "B7-check") & (per.tau == tau)]
            mm = a.merge(b, on="unit_id", suffixes=("_ref", ""), validate="one_to_one")
            rev.check(f"{family} d={budget} {pol} tau={tau} rows", len(mm), n)
            rev.check(f"{family} d={budget} {pol} tau={tau} counts", int((mm.overload_count - mm.overload_count_ref).abs().max()), 0)
            rev.check(f"{family} d={budget} {pol} tau={tau} cost", float((mm.relative_cost - mm.relative_cost_ref).abs().max()), 0.0, 1e-9)
    return per


def summarise() -> pd.DataFrame:
    rows = []
    for f in rev.OUT.glob("r6_pac_*_per_unit.csv.gz"):
        per = pd.read_csv(f)
        per["within"] = per.overload_fraction <= per.budget
        for k, grp in per.groupby(["dataset", "budget", "policy", "W", "rank", "tau"], sort=True):
            rows.append({**dict(zip(["dataset", "budget", "policy", "W", "rank", "tau"], k)), "n": len(grp),
                         "compliant": int(grp.within.sum()), "mean_relative_cost": float(grp.relative_cost.mean()),
                         "median_relative_cost": float(grp.relative_cost.median()), "mean_overload": float(grp.overload_fraction.mean())})
        if per.dataset.iloc[0] == "alibaba200":
            crit = pd.read_csv(rev.OUT / "subset_pretest_criterion.csv")
            sub = per.merge(crit, left_on="unit_id", right_on="service_id")
            sub = sub[sub.nontrivial_pretest]
            for k, grp in sub.groupby(["dataset", "budget", "policy", "W", "rank", "tau"], sort=True):
                rows.append({**dict(zip(["dataset", "budget", "policy", "W", "rank", "tau"], k)), "dataset": "alibaba_nontrivial95",
                             "n": len(grp), "compliant": int(grp.within.sum()), "mean_relative_cost": float(grp.relative_cost.mean()),
                             "median_relative_cost": float(grp.relative_cost.median()), "mean_overload": float(grp.overload_fraction.mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    todo = [a for a in sys.argv[1:]] or ["alibaba200", "huawei2023", "azure2019"]
    for b in BUDGETS:
        print("delta", b, [(c[0], c[1], c[2]) for c in configs(b)])
    for ds in todo:
        print("R6", ds, flush=True)
        per = run_alibaba() if ds == "alibaba200" else run_external(ds)
        per.to_csv(rev.OUT / f"r6_pac_{ds}_per_unit.csv.gz", index=False)
    s = summarise()
    s.to_csv(rev.OUT / "r6_pac_summary.csv", index=False)
    (rev.OUT / "methodology_r6.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__,
                                                              "pandas": pd.__version__, "eta": ETA}, indent=2) + "\n")
    print(s[s.policy != "B7-check"].to_string(index=False))
