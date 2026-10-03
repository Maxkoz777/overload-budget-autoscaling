#!/usr/bin/env python3
"""Revision analyses of 2 October 2026 (protocol: verified_results/delay_pac_study/protocol.json).

R1  closed-loop delayed actuation at the strict budget delta = 0.01 (Alibaba, 200 services);
R2  reactive baseline with an extended threshold grid {0.2, 0.3, 0.4} on Alibaba, Huawei and Azure;
R3  horizon-aligned conformal scores under delay (protocol addendum written after R1, before R3).

Both reuse the canonical replay functions unchanged and write only to the new output folder;
canonical results are never overwritten. Each part first reproduces the published numbers
(regression checks) and stops if they do not match.

Usage:  python3 audit/delay_and_reactive_grid.py [r1] [r2] [r3]
"""
from __future__ import annotations

import json
import math
import platform
import sys
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import recompute_guardrail_actuation as g  # noqa: E402
import recompute_reactive_baseline as rb  # noqa: E402
import external_replication as ext  # noqa: E402

PAPER = HERE.parent
EXP = PAPER.parent / "experiments"
VER = PAPER / "audit" / "verified_results"
OUT = VER / "delay_pac_study"
DATA = EXP / "data" / "service_timeseries_200_verified"

DELTA_STRICT = 0.01
TAUS = (0, 1, 2, 5)
ALPHAS = (1.0, 1.25, 1.5, 2.0)
NEW_U = (0.2, 0.3, 0.4)
ORIG_U = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
COOLDOWNS = (1, 3, 5, 10)
SORT_FEAS = (["relative_cost", "overload_fraction", "threshold", "cooldown"], [True, True, False, True])
SORT_INFEAS = (["overload_fraction", "relative_cost", "threshold", "cooldown"], [True, True, False, True])


def select(grid: pd.DataFrame, key: str, budget: float) -> pd.DataFrame:
    """Canonical per-unit reactive selection rule (identical in all published scripts)."""
    rows = []
    for uid, grp in grid.groupby(key, sort=True):
        feas = grp[grp.overload_fraction <= budget]
        cols, asc = SORT_FEAS if len(feas) else SORT_INFEAS
        r = (feas if len(feas) else grp).sort_values(cols, ascending=asc).iloc[0]
        rows.append({key: uid, "threshold": float(r.threshold), "cooldown": int(r.cooldown),
                     "calibration_feasible": bool(len(feas))})
    return pd.DataFrame(rows)


def check(label: str, got, want, tol: float = 5e-5) -> None:
    ok = (got == want) if isinstance(want, (int, np.integer)) else abs(float(got) - float(want)) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: got {got}, expected {want}", flush=True)
    if not ok:
        raise SystemExit(f"regression check failed: {label}")


# ===================================================================== R1
SERVICES: list[dict] = []
PARAMS: dict[str, dict[str, tuple[float, int]]] = {}


def _r1_service(index: int) -> list[dict]:
    s = SERVICES[index]
    sid = s["service_id"]
    rows = []
    for tau in TAUS:
        for label, (u, c) in PARAMS[sid].items():
            r = g.simulate_reactive_closed_loop(s["history"], s["test"], u, c, tau)
            rows.append({"service_id": sid, "policy": label, "tau": tau, "alpha": 1.0,
                         "overload_fraction": r["overload_fraction"], "relative_cost": r["relative_cost"]})
        for alpha in ALPHAS:
            for label, corr in (("B6", "fixed_plus1"), ("B7", "none")):
                r = g.simulate_conformal(s["history"], s["test"], correction=corr, alpha=alpha, tau=tau)
                rows.append({"service_id": sid, "policy": label, "tau": tau, "alpha": alpha,
                             "overload_fraction": r["overload_fraction"], "relative_cost": r["relative_cost"]})
    return rows


def run_r1() -> None:
    global SERVICES, PARAMS
    print("R1: delayed actuation at delta = 0.01", flush=True)
    g.DATA = DATA
    g.DELTA = DELTA_STRICT
    _, SERVICES = g.load_services()
    cal = pd.read_csv(VER / "comparative" / "reactive_calibration_grid_per_service.csv")
    cal["service_id"] = cal.service_id.astype(str)
    sel = {b: select(cal, "service_id", b).set_index("service_id") for b in (0.01, 0.0025)}
    PARAMS = {s["service_id"]: {
        "reactive_cal1pct": (float(sel[0.01].loc[s["service_id"], "threshold"]), int(sel[0.01].loc[s["service_id"], "cooldown"])),
        "reactive_cal0.25pct": (float(sel[0.0025].loc[s["service_id"], "threshold"]), int(sel[0.0025].loc[s["service_id"], "cooldown"])),
    } for s in SERVICES}
    with Pool(2) as pool:
        parts = pool.map(_r1_service, range(len(SERVICES)), chunksize=5)
    per = pd.DataFrame([row for part in parts for row in part])
    quart = pd.read_csv(VER / "strict_budget" / "strict_budget_per_service.csv")[["service_id", "pretest_size_quartile"]].drop_duplicates()
    per = per.merge(quart, on="service_id", validate="many_to_one")
    per["within_budget"] = per.overload_fraction <= DELTA_STRICT

    rows = []
    for cohort, part in (("all200", per), ("Q4", per[per.pretest_size_quartile == "Q4"])):
        for (pol, tau, alpha), grp in part.groupby(["policy", "tau", "alpha"], sort=True):
            rows.append({"cohort": cohort, "policy": pol, "tau": tau, "alpha": alpha, "n": len(grp),
                         "compliant": int(grp.within_budget.sum()),
                         "mean_relative_cost": float(grp.relative_cost.mean()),
                         "median_relative_cost": float(grp.relative_cost.median()),
                         "median_overload": float(grp.overload_fraction.median()),
                         "mean_overload": float(grp.overload_fraction.mean())})
    summ = pd.DataFrame(rows)

    # regression against Table 2 (strict_budget_per_service.csv)
    sb = pd.read_csv(VER / "strict_budget" / "strict_budget_per_service.csv")
    base = summ[(summ.cohort == "all200") & (summ.tau == 0) & (summ.alpha == 1.0)].set_index("policy")
    ref = {"B6": sb[(sb.policy == "conformal") & (sb.W == 240) & sb.guard],
           "B7": sb[(sb.policy == "conformal") & (sb.W == 240) & ~sb.guard],
           "reactive_cal1pct": sb[(sb.policy == "reactive") & (sb.calibration_budget == 0.01)],
           "reactive_cal0.25pct": sb[(sb.policy == "reactive") & (sb.calibration_budget == 0.0025)]}
    for pol, r in ref.items():
        check(f"R1 tau=0 {pol} compliant", int(base.loc[pol, "compliant"]), int(r.within_budget.sum()))
        check(f"R1 tau=0 {pol} mean cost", base.loc[pol, "mean_relative_cost"], r.relative_cost.mean(), 1e-9)

    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / "r1_delay_strict_per_service.csv", index=False)
    summ.to_csv(OUT / "r1_delay_strict_summary.csv", index=False)
    show = summ[(summ.alpha == 1.0) | (summ.policy == "B6")]
    print(show.to_string(index=False), flush=True)


# ===================================================================== R2
def _alibaba_rows(index_sid: tuple[int, str]) -> list[dict]:
    _, sid = index_sid
    split = json.loads((EXP / "data" / "splits" / "split_definition.json").read_text())
    train, calib, test = rb.split_frames(sid, split)
    pre = pd.concat([train, calib], ignore_index=True)
    rows = []
    for phase, hist, ev in (("calibration", train, calib), ("test", pre, test)):
        for c in COOLDOWNS:
            for u in ORIG_U + NEW_U:
                cap = rb.reactive_capacity(hist, ev, threshold=u, cooldown=c)
                m = rb.metrics(ev, cap)
                rows.append({"service_id": sid, "phase": phase, "threshold": u, "cooldown": c,
                             "overload_fraction": m["overload_fraction"], "relative_cost": m["relative_cost"]})
    return rows


def reactive_vec(prev0: np.ndarray, cap0: np.ndarray, demand: np.ndarray, mu: np.ndarray,
                 thresholds: tuple, cooldowns: tuple) -> pd.DataFrame:
    """Time-stepped NumPy version of ext.reactive for all units x configurations at once.

    Semantics per unit and configuration are identical to ext.reactive + ext.cost_parts.
    """
    n_units, T = demand.shape
    cfg = [(u, c) for c in cooldowns for u in thresholds]
    U = np.array([u for u, _ in cfg])[None, :]
    C = np.array([c for _, c in cfg])[None, :]
    k = len(cfg)
    umu = U * mu[:, None]                       # u * mu, as in ext.reactive
    cur = np.repeat(cap0[:, None], k, axis=1).astype(float)
    last = np.repeat(-C, n_units, axis=0).astype(float)
    prev = np.repeat(prev0[:, None], k, axis=1).astype(float)
    cap_sum = np.zeros((n_units, k)); over = np.zeros((n_units, k)); churn = np.zeros((n_units, k))
    before = None
    mu_col = mu[:, None]
    for s in range(T):
        desired = np.maximum(np.ceil(prev / umu), 1.0)
        change = ((s - last) >= C) & (desired != cur)
        cur = np.where(change, desired, cur)
        last = np.where(change, s, last)
        d = demand[:, s][:, None]
        cap_sum += cur
        over += d > mu_col * cur
        if before is not None:
            churn += np.abs(cur - before)
        before = cur.copy()
        prev = np.repeat(d, k, axis=1)
    total = ext.C_RES * cap_sum + ext.C_ACT * churn + ext.C_VIO * over
    out = []
    for j, (u, c) in enumerate(cfg):
        out.append(pd.DataFrame({"unit_idx": np.arange(n_units), "threshold": u, "cooldown": c,
                                 "overload_count": over[:, j].astype(int), "overload_fraction": over[:, j] / T,
                                 "total_cost": total[:, j], "c_res": cap_sum[:, j]}))
    return pd.concat(out, ignore_index=True)


def external_grid(family: str) -> pd.DataFrame:
    X, mu, units, split, _ = ext.load_family(family, "primary")
    X = X.astype(np.float64)
    mu = mu.astype(float)
    frames = []
    for phase, (e0, e1) in (("calibration", split["cal"]), ("test", split["test"])):
        hist_last = X[:, e0 - 1]
        cap0 = np.maximum(1.0, np.ceil(hist_last / (0.8 * mu)))
        dem = X[:, e0:e1]
        clair = np.maximum(np.ceil(dem / mu[:, None]), 1.0)
        churn = np.abs(np.diff(clair, axis=1)).sum(axis=1)
        over = (dem > mu[:, None] * clair).sum(axis=1)
        denom_total = ext.C_RES * clair.sum(axis=1) + ext.C_ACT * churn + ext.C_VIO * over
        grid = reactive_vec(hist_last, cap0, dem, mu, ORIG_U + NEW_U, COOLDOWNS)
        grid["unit_id"] = units.unit_id.to_numpy()[grid.unit_idx]
        grid["phase"] = phase
        grid["denom_total"] = denom_total[grid.unit_idx]
        grid["relative_cost"] = grid.total_cost / grid.denom_total
        frames.append(grid.drop(columns="unit_idx"))
        print(f"  {family} {phase}: {len(grid)} rows", flush=True)
    return pd.concat(frames, ignore_index=True)


def summarise_choice(grid: pd.DataFrame, key: str, budget: float, thresholds: tuple) -> dict:
    sub = grid[grid.threshold.isin(thresholds)]
    choice = select(sub[sub.phase == "calibration"], key, budget)
    test = sub[sub.phase == "test"].merge(choice, on=[key, "threshold", "cooldown"], validate="one_to_one")
    assert len(test) == grid[key].nunique()
    return {"n": len(test), "compliant": int((test.overload_fraction <= budget).sum()),
            "mean_relative_cost": float(test.relative_cost.mean()),
            "median_relative_cost": float(test.relative_cost.median()),
            "calibration_infeasible": int((~test.calibration_feasible).sum()),
            "selected_added_threshold": int(test.threshold.isin(NEW_U).sum())}, test


def run_r2() -> None:
    print("R2: extended reactive threshold grid", flush=True)
    rb.DATA_ROOT = DATA
    OUT.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(EXP / "data" / "splits" / "selected_services_200.csv")
    with Pool(2) as pool:
        parts = pool.map(_alibaba_rows, list(enumerate(sel.service_id.astype(str))), chunksize=5)
    ali = pd.DataFrame([r for p in parts for r in p])

    # equivalence of the original-grid rows with the canonical grids
    for phase, name in (("calibration", "reactive_calibration_grid_per_service.csv"), ("test", "reactive_test_grid_per_service.csv")):
        ref = pd.read_csv(VER / "comparative" / name)
        ref["service_id"] = ref.service_id.astype(str)
        mine = ali[(ali.phase == phase) & ali.threshold.isin(ORIG_U)]
        mrg = ref.merge(mine, on=["service_id", "threshold", "cooldown"], suffixes=("_ref", ""), validate="one_to_one")
        check(f"Alibaba {phase} grid rows", len(mrg), len(ref))
        check(f"Alibaba {phase} max |overload diff|", float((mrg.overload_fraction - mrg.overload_fraction_ref).abs().max()), 0.0, 1e-12)
        check(f"Alibaba {phase} max |cost diff|", float((mrg.relative_cost - mrg.relative_cost_ref).abs().max()), 0.0, 1e-9)
    ali.to_csv(OUT / "r2_alibaba_reactive_grid.csv.gz", index=False)

    grids = {"alibaba200": (ali, "service_id")}
    for fam in ("huawei2023", "azure2019"):
        eg = external_grid(fam)
        ref = pd.read_csv(VER / "external_replication_study" / f"per_unit_{fam}_primary.csv.gz")
        ref = ref[ref.family == "reactive"]
        mrg = ref.merge(eg, on=["unit_id", "phase", "threshold", "cooldown"], suffixes=("_ref", ""), validate="one_to_one")
        check(f"{fam} grid rows", len(mrg), len(ref))
        check(f"{fam} max |overload count diff|", int((mrg.overload_count - mrg.overload_count_ref).abs().max()), 0)
        check(f"{fam} max |relative cost diff|", float((mrg.relative_cost - mrg.relative_cost_ref).abs().max()), 0.0, 1e-9)
        eg.to_csv(OUT / f"r2_{fam}_reactive_grid.csv.gz", index=False)
        grids[fam] = (eg, "unit_id")

    rows, chosen = [], []
    for fam, (grid, key) in grids.items():
        for budget in (0.01, 0.05):
            for gname, th in (("original", ORIG_U), ("extended", ORIG_U + NEW_U)):
                res, test = summarise_choice(grid, key, budget, th)
                rows.append({"dataset": fam, "budget": budget, "grid": gname, **res})
                chosen.append(test.assign(dataset=fam, budget=budget, grid=gname).rename(columns={key: "unit"}))
    summ = pd.DataFrame(rows)
    s = summ.set_index(["dataset", "budget", "grid"])
    check("Alibaba 1% original", int(s.loc[("alibaba200", 0.01, "original"), "compliant"]), 190)
    check("Alibaba 5% original", int(s.loc[("alibaba200", 0.05, "original"), "compliant"]), 198)
    check("Huawei 1% original", int(s.loc[("huawei2023", 0.01, "original"), "compliant"]), 47)
    check("Azure 1% original", int(s.loc[("azure2019", 0.01, "original"), "compliant"]), 1123)
    check("Azure 1% original infeasible", int(s.loc[("azure2019", 0.01, "original"), "calibration_infeasible"]), 1562)
    summ.to_csv(OUT / "r2_extended_grid_summary.csv", index=False)
    keep = ["dataset", "budget", "grid", "unit", "threshold", "cooldown", "calibration_feasible", "overload_fraction", "relative_cost"]
    pd.concat(chosen, ignore_index=True)[keep].to_csv(OUT / "r2_selected_per_unit.csv.gz", index=False)
    print(summ.to_string(index=False), flush=True)


# ===================================================================== R3
def simulate_horizon_aligned(history: pd.DataFrame, evaluation: pd.DataFrame, *, correction: str, tau: int) -> dict:
    """g.simulate_conformal with the (tau+1)-step persistence residual as the conformity score.

    For tau = 0 this is line-for-line the canonical replay (h = 1)."""
    h = tau + 1
    hist = history.cpu_sum.to_numpy(float)
    demand = evaluation.cpu_sum.to_numpy(float)
    y = np.r_[hist, demand]
    n = len(hist)
    residuals = np.maximum(hist[h:] - hist[:-h], 0.0).tolist()
    previous = float(hist[-1])
    initial_capacity = max(int(np.ceil(float(history.replica_count.iloc[-1]))), 1)
    pending = deque([initial_capacity] * tau)
    over_hist: list[bool] = []
    soft_hist: list[bool] = []
    applied: list[int] = []
    for i, actual in enumerate(demand):
        window = np.asarray(residuals[-g.W:], dtype=float)
        level = min(len(window), int(np.ceil((len(window) + 1) * (1.0 - g.DELTA))))
        margin = float(np.partition(window, level - 1)[level - 1])
        nominal = max(1, int(np.ceil(previous + margin)))
        trigger = False
        if correction != "none" and len(over_hist) >= g.H:
            trigger = all(over_hist[-g.H:]) or all(soft_hist[-g.H:])
        decision = nominal + g.correction_value(correction, nominal, trigger)
        if tau:
            pending.append(decision)
            capacity = int(pending.popleft())
        else:
            capacity = decision
        applied.append(capacity)
        over_hist.append(bool(actual > g.MU * capacity))
        soft_hist.append(bool(actual > 0.70 * g.MU * capacity))
        residuals.append(max(float(actual - y[n + i - h]), 0.0))
        previous = float(actual)
    cap = np.asarray(applied, dtype=int)
    overload = demand > g.MU * cap
    return {"overload_fraction": float(overload.mean()),
            "relative_cost": g.policy_cost(demand, cap) / g.observed_cost(evaluation)}


def _r3_service(index: int) -> list[dict]:
    s = SERVICES[index]
    rows = []
    for tau in TAUS:
        for label, corr in (("B6-h", "fixed_plus1"), ("B7-h", "none")):
            r = simulate_horizon_aligned(s["history"], s["test"], correction=corr, tau=tau)
            rows.append({"service_id": s["service_id"], "policy": label, "tau": tau, "alpha": 1.0, **r})
    return rows


def run_r3() -> None:
    global SERVICES
    print("R3: horizon-aligned scores under delay, delta = 0.01", flush=True)
    g.DATA = DATA
    g.DELTA = DELTA_STRICT
    _, SERVICES = g.load_services()
    with Pool(2) as pool:
        parts = pool.map(_r3_service, range(len(SERVICES)), chunksize=5)
    per = pd.DataFrame([row for part in parts for row in part])
    quart = pd.read_csv(VER / "strict_budget" / "strict_budget_per_service.csv")[["service_id", "pretest_size_quartile"]].drop_duplicates()
    per = per.merge(quart, on="service_id", validate="many_to_one")
    per["within_budget"] = per.overload_fraction <= DELTA_STRICT
    rows = []
    for cohort, part in (("all200", per), ("Q4", per[per.pretest_size_quartile == "Q4"])):
        for (pol, tau), grp in part.groupby(["policy", "tau"], sort=True):
            rows.append({"cohort": cohort, "policy": pol, "tau": tau, "n": len(grp), "compliant": int(grp.within_budget.sum()),
                         "mean_relative_cost": float(grp.relative_cost.mean()),
                         "median_relative_cost": float(grp.relative_cost.median()),
                         "mean_overload": float(grp.overload_fraction.mean())})
    summ = pd.DataFrame(rows)
    b = summ[(summ.cohort == "all200") & (summ.tau == 0)].set_index("policy")
    check("R3 tau=0 B6-h compliant", int(b.loc["B6-h", "compliant"]), 199)
    check("R3 tau=0 B6-h mean cost", b.loc["B6-h", "mean_relative_cost"], 0.4478102350448739, 1e-12)
    check("R3 tau=0 B7-h compliant", int(b.loc["B7-h", "compliant"]), 197)
    check("R3 tau=0 B7-h mean cost", b.loc["B7-h", "mean_relative_cost"], 0.44396808922044123, 1e-12)
    per.to_csv(OUT / "r3_horizon_aligned_per_service.csv", index=False)
    summ.to_csv(OUT / "r3_horizon_aligned_summary.csv", index=False)
    print(summ.to_string(index=False), flush=True)


if __name__ == "__main__":
    todo = set(sys.argv[1:]) or {"r1", "r2", "r3"}
    if "r2" in todo:
        run_r2()
    if "r1" in todo:
        run_r1()
    if "r3" in todo:
        run_r3()
    meta = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "parts_run": sorted(todo), "protocol": "verified_results/delay_pac_study/protocol.json"}
    (OUT / f"methodology_{'_'.join(sorted(todo))}.json").write_text(json.dumps(meta, indent=2) + "\n")
