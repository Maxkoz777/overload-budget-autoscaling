#!/usr/bin/env python3
"""External replication of the persistence-based comparison (protocol 2026-09-30).

One generic replay engine is used for every trace family:
    capacity_t = max(1, ceil((forecast_t + margin_t) / mu)),
with the shared offset (h=2, rho=0.70, gamma=1), overload d_t > mu * capacity_t,
and cost C = sum k + 0.05 sum |dk| + 10 * #overload.

Stages (each fits in a few minutes; outputs are written under
audit/verified_results/external_replication_study/):
    python3 audit/external_replication.py regress-alibaba
    python3 audit/external_replication.py run --family huawei2023 --mu primary
    python3 audit/external_replication.py run --family huawei2023 --mu p90
    python3 audit/external_replication.py run --family azure2019 --mu primary
    python3 audit/external_replication.py run --family azure2019 --mu K5
    python3 audit/external_replication.py run --family azure2019 --mu K20
    python3 audit/external_replication.py summarise
Numba is used when available; the pure-Python fallback gives identical results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

try:
    from numba import njit
except ImportError:  # pragma: no cover - identical semantics, slower
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]

PAPER = Path(__file__).resolve().parents[1]
EXP = PAPER.parent / "experiments"
EXT = EXP / "data" / "external_traces"
OUT = PAPER / "audit" / "verified_results" / "external_replication_study"
PROTOCOL = OUT / "protocol.json"
M = 1440
W, W_LONG = 240, 1440
RHO, GAMMA = 0.70, 1
C_RES, C_ACT, C_VIO = 1.0, 0.05, 10.0
BUDGETS = (0.01, 0.05)
ALPHAS = (1.0, 1.25, 1.5, 2.0)
Z_GRID = (1.28, 1.64, float(norm.ppf(0.95)), 1.96, float(norm.ppf(0.99)), 3.0, 4.0)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
COOLDOWNS = (1, 3, 5, 10)


# ---------------------------------------------------------------- kernels
@njit(cache=True)
def rolling_order_stats(pos, start, T, w, ranks):
    """Order statistics (1-based ranks) of pos[start+j : start+j+w] for j=0..T-1."""
    buf = np.sort(pos[start:start + w].copy())
    out = np.empty((T, len(ranks)))
    for j in range(T):
        if j > 0:
            old = pos[start + j - 1]
            new = pos[start + j + w - 1]
            i = np.searchsorted(buf, old)
            while buf[i] != old:
                i += 1
            buf[i:w - 1] = buf[i + 1:w].copy()
            k = np.searchsorted(buf[:w - 1], new)
            buf[k + 1:w] = buf[k:w - 1].copy()
            buf[k] = new
        for r in range(len(ranks)):
            out[j, r] = buf[ranks[r] - 1]
    return out


@njit(cache=True)
def guarded(nominal, demand, rho, mu, gamma):
    cap = nominal.copy()
    s1 = False
    s2 = False
    for t in range(len(demand)):
        if t >= 2 and s1 and s2:
            cap[t] += gamma
        s = demand[t] > rho * mu * cap[t]
        s2 = s1
        s1 = s
    return cap


@njit(cache=True)
def reactive(prev0, cap0, demand, u, mu, cooldown):
    n = len(demand)
    out = np.empty(n)
    cur = cap0
    last = -cooldown
    prev = prev0
    for s in range(n):
        desired = max(math.ceil(prev / (u * mu)), 1.0)
        if s - last >= cooldown and desired != cur:
            cur = desired
            last = s
        out[s] = cur
        prev = demand[s]
    return out


def cost_parts(cap: np.ndarray, demand: np.ndarray, mu: float) -> dict:
    over = demand > mu * cap
    churn = float(np.abs(np.diff(cap)).sum())
    return {"overload_count": int(over.sum()), "overload_fraction": float(over.mean()),
            "mean_capacity": float(cap.mean()), "floor_fraction": float((cap == 1).mean()),
            "c_res": C_RES * float(cap.sum()), "c_act": C_ACT * churn, "c_vio": C_VIO * float(over.sum()),
            "total_cost": C_RES * float(cap.sum()) + C_ACT * churn + C_VIO * float(over.sum())}


# ---------------------------------------------------------------- one unit
def margin_configs():
    for d in BUDGETS:
        for a in ALPHAS:
            for g in (False, True):
                yield dict(family="conformal", W=W, rank_delta=d, alpha=a, z=np.nan, guard=g,
                           config_id=f"conformal_W{W}_d{d:g}_a{a:g}_g{int(g)}")
        for g in (False, True):
            yield dict(family="inverse_cdf", W=W, rank_delta=d, alpha=1.0, z=np.nan, guard=g,
                       config_id=f"invcdf_W{W}_d{d:g}_g{int(g)}")
            yield dict(family="conformal", W=W_LONG, rank_delta=d, alpha=1.0, z=np.nan, guard=g,
                       config_id=f"conformal_W{W_LONG}_d{d:g}_a1_g{int(g)}")
    for z in Z_GRID:
        for g in (False, True):
            yield dict(family="gaussian", W=W, rank_delta=np.nan, alpha=np.nan, z=z, guard=g,
                       config_id=f"gaussian_W{W}_z{z:.6f}_g{int(g)}")
    for g in (False, True):
        yield dict(family="pure_predictive", W=W, rank_delta=np.nan, alpha=np.nan, z=np.nan, guard=g,
                   config_id=f"pure_predictive_g{int(g)}")


def conf_rank(w, d):
    return min(w, math.ceil((w + 1) * (1 - d)))


def inv_rank(w, d):
    return math.ceil(w * (1 - d))


def replay_margins(history, demand, mu, denom_cap):
    n, T = len(history), len(demand)
    forecast = np.r_[history[-1], demand[:-1]]
    residual = np.diff(np.r_[history, demand])
    pos = np.maximum(residual, 0.0)
    out = {}
    for w in (W, W_LONG):
        if n - 1 - w < 0:
            continue
        ranks = sorted({conf_rank(w, d) for d in BUDGETS} | {inv_rank(w, d) for d in BUDGETS})
        stats = rolling_order_stats(pos, n - 1 - w, T, w, np.array(ranks, dtype=np.int64))
        out[w] = {r: stats[:, i] for i, r in enumerate(ranks)}
    win = np.lib.stride_tricks.sliding_window_view(residual, W)[n - 1 - W: n - 1 - W + T]
    assert np.array_equal(win[0], np.diff(history)[-W:])
    mean, std = win.mean(axis=1), win.std(axis=1, ddof=0)
    denom = cost_parts(denom_cap, demand, mu)
    rows = []
    for cfg in margin_configs():
        if cfg["family"] == "conformal":
            margin = cfg["alpha"] * out[cfg["W"]][conf_rank(cfg["W"], cfg["rank_delta"])]
        elif cfg["family"] == "inverse_cdf":
            margin = out[W][inv_rank(W, cfg["rank_delta"])]
        elif cfg["family"] == "gaussian":
            margin = np.maximum(0.0, mean + cfg["z"] * std)
        else:
            margin = np.zeros(T)
        nominal = np.maximum(np.ceil((forecast + margin) / mu), 1.0)
        cap = guarded(nominal, demand, RHO, mu, GAMMA) if cfg["guard"] else nominal
        part = cost_parts(cap, demand, mu)
        part["guard_rate"] = float((cap != nominal).mean())
        rows.append({**cfg, **part, "denom_total": denom["total_cost"], "denom_res": denom["c_res"]})
    return rows


def replay_reactive(history, demand, mu, cap0, denom_cap):
    denom = cost_parts(denom_cap, demand, mu)
    rows = []
    for c in COOLDOWNS:
        for u in THRESHOLDS:
            cap = reactive(float(history[-1]), float(cap0), demand, u, mu, c)
            part = cost_parts(cap, demand, mu)
            rows.append({"family": "reactive", "threshold": u, "cooldown": c, "guard": False,
                         "config_id": f"reactive_u{u:g}_c{c}", **part,
                         "denom_total": denom["total_cost"], "denom_res": denom["c_res"]})
    return rows


def choose_reactive(cal: pd.DataFrame, budget: float) -> pd.DataFrame:
    rows = []
    for uid, g in cal.groupby("unit_id", sort=True):
        feas = g[g.overload_fraction <= budget]
        if len(feas):
            r = feas.sort_values(["relative_cost", "overload_fraction", "threshold", "cooldown"],
                                 ascending=[True, True, False, True]).iloc[0]
        else:
            r = g.sort_values(["overload_fraction", "relative_cost", "threshold", "cooldown"],
                              ascending=[True, True, False, True]).iloc[0]
        rows.append({"unit_id": uid, "config_id": r.config_id, "calibration_feasible": bool(len(feas))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- families
def load_family(family: str, mu_variant: str):
    if family == "huawei2023":
        z = np.load(EXT / "huawei2023" / "series.npz", allow_pickle=True)
        units = pd.read_csv(EXT / "huawei2023" / "units.csv")
        assert list(units.unit_id) == list(z["unit_id"])
        mu = units["mu_p50" if mu_variant == "primary" else "mu_p90"].to_numpy(float)
        split = dict(history_end=8 * M, cal=(8 * M, 10 * M), test=(10 * M, 33 * M))
        return z["demand"], mu, units, split, z["observed"]
    if family == "azure2019":
        z = np.load(EXT / "azure2019" / "series.npz", allow_pickle=True)
        units = pd.read_csv(EXT / "azure2019" / "units.csv")
        assert list(units.unit_id) == list(z["unit_id"])
        K = {"primary": 10, "K5": 5, "K20": 20}[mu_variant]
        mu = units[f"mu_K{K}"].to_numpy(float)
        split = dict(history_end=8 * M, cal=(8 * M, 10 * M), test=(10 * M, 14 * M))
        return z["demand"], mu, units, split, None
    raise ValueError(family)


def run_family(family: str, mu_variant: str) -> None:
    X, mu, units, split, observed = load_family(family, mu_variant)
    rows, rrows = [], []
    for i, uid in enumerate(units.unit_id):
        d = X[i].astype(np.float64)
        m = float(mu[i])
        c0, c1 = split["cal"]
        t0, t1 = split["test"]
        # calibration phase (reactive selection): history days 0-7, evaluation days 8-9
        hist_c, dem_c = d[:c0], d[c0:c1]
        cap0 = max(1.0, math.ceil(hist_c[-1] / (0.8 * m)))
        clair_c = np.maximum(np.ceil(dem_c / m), 1.0)
        for r in replay_reactive(hist_c, dem_c, m, cap0, clair_c):
            rrows.append({"unit_id": uid, "phase": "calibration", **r})
        # test phase: history = everything before the test
        hist_t, dem_t = d[:t0], d[t0:t1]
        clair_t = np.maximum(np.ceil(dem_t / m), 1.0)
        cap0 = max(1.0, math.ceil(hist_t[-1] / (0.8 * m)))
        for r in replay_reactive(hist_t, dem_t, m, cap0, clair_t):
            rrows.append({"unit_id": uid, "phase": "test", **r})
        for r in replay_margins(hist_t, dem_t, m, clair_t):
            rows.append({"unit_id": uid, "phase": "test", **r})
        if observed is not None:
            obs_res = float(np.maximum(observed[i, t0:t1], 1.0).sum())
            for r in rows[-sum(1 for _ in margin_configs()):]:
                r["observed_res"] = obs_res
            for r in rrows[-len(THRESHOLDS) * len(COOLDOWNS):]:
                r["observed_res"] = obs_res
        if (i + 1) % 250 == 0:
            print(f"{family}/{mu_variant}: {i + 1}/{len(units)}", flush=True)
    per = pd.concat([pd.DataFrame(rows), pd.DataFrame(rrows)], ignore_index=True)
    per["relative_cost"] = per.total_cost / per.denom_total
    per["norm_resource"] = per.c_res / per.denom_res
    if "observed_res" in per:
        per["resource_vs_observed"] = per.c_res / per.observed_res
    per["family_trace"], per["mu_variant"] = family, mu_variant
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / f"per_unit_{family}_{mu_variant}.csv.gz", index=False)
    print(f"wrote {len(per)} rows for {family}/{mu_variant}")


# ---------------------------------------------------------------- regression on Alibaba
def regress_alibaba() -> None:
    split = {s["name"]: s for s in json.loads((EXP / "data/splits/split_definition.json").read_text())["splits"]}
    sel = pd.read_csv(EXP / "data/splits/selected_services_200.csv")
    rank = pd.read_csv(PAPER / "audit/verified_results/rank_ablation/rank_ablation_per_service.csv")
    grid = pd.read_csv(PAPER / "audit/verified_results/comparative/reactive_test_grid_per_service.csv")
    calg = pd.read_csv(PAPER / "audit/verified_results/comparative/reactive_calibration_grid_per_service.csv")
    rows, rrows = [], []
    for sid in sel.service_id:
        f = pd.read_parquet(EXP / "data/service_timeseries_200_verified" / f"{sid}.parquet").sort_values("timestamp")
        ts = f.timestamp
        pre = f[ts < split["test"]["timestamp_start"]]
        test = f[(ts >= split["test"]["timestamp_start"]) & (ts < split["test"]["timestamp_end_exclusive"])]
        train = f[ts < split["train"]["timestamp_end_exclusive"]]
        cal = f[(ts >= split["calibration"]["timestamp_start"]) & (ts < split["calibration"]["timestamp_end_exclusive"])]
        d = test.cpu_sum.to_numpy(float)
        obs = np.maximum(np.ceil(test.replica_count.to_numpy(float)), 1.0)
        for r in replay_margins(pre.cpu_sum.to_numpy(float), d, 1.0, obs):
            rows.append({"service_id": sid, **r})
        for phase, hist, ev in (("test", pre, test), ("calibration", train, cal)):
            ob = np.maximum(np.ceil(ev.replica_count.to_numpy(float)), 1.0)
            for r in replay_reactive(hist.cpu_sum.to_numpy(float), ev.cpu_sum.to_numpy(float), 1.0,
                                     int(hist.replica_count.iloc[-1]), ob):
                rrows.append({"service_id": sid, "phase": phase, **r})
    per = pd.DataFrame(rows); per["relative_cost"] = per.total_cost / per.denom_total
    rr = pd.DataFrame(rrows); rr["relative_cost"] = rr.total_cost / rr.denom_total
    checks = {}
    mapping = {"conformal": ("conformal", 1.0), "empirical_inverse_cdf": ("inverse_cdf", 1.0)}
    for meth, (fam, a) in mapping.items():
        for dlt in BUDGETS:
            mine = per[(per.family == fam) & (per.W == W) & np.isclose(per.rank_delta, dlt) & ((per.alpha == a) | per.alpha.isna())]
            ref = rank[(rank.method == meth) & np.isclose(rank.delta, dlt)]
            m = mine.merge(ref, on=["service_id", "guard"], suffixes=("", "_ref"), validate="one_to_one")
            checks[f"{meth}_{dlt}"] = bool(len(m) == 400 and np.allclose(m.overload_fraction, m.overload_fraction_ref, rtol=0, atol=1e-12)
                                           and np.allclose(m.relative_cost, m.relative_cost_ref, rtol=0, atol=1e-12))
    for dlt in BUDGETS:
        z = float(norm.ppf(1 - dlt))
        mine = per[(per.family == "gaussian") & np.isclose(per.z, z, rtol=0, atol=1e-12)]
        ref = rank[(rank.method == "gaussian") & np.isclose(rank.delta, dlt)]
        m = mine.merge(ref, on=["service_id", "guard"], suffixes=("", "_ref"), validate="one_to_one")
        checks[f"gaussian_{dlt}"] = bool(len(m) == 400 and np.allclose(m.overload_fraction, m.overload_fraction_ref, rtol=0, atol=1e-12)
                                         and np.allclose(m.relative_cost, m.relative_cost_ref, rtol=0, atol=1e-12))
    for phase, ref in (("test", grid), ("calibration", calg)):
        m = rr[rr.phase == phase].merge(ref, on=["service_id", "threshold", "cooldown"], suffixes=("", "_ref"), validate="one_to_one")
        checks[f"reactive_{phase}"] = bool(len(m) == 6400 and np.allclose(m.overload_fraction, m.overload_fraction_ref, rtol=0, atol=1e-12)
                                           and np.allclose(m.relative_cost, m.relative_cost_ref, rtol=0, atol=1e-12))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "alibaba_regression.json").write_text(json.dumps({"checks": checks, "all_passed": all(checks.values())}, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    assert all(checks.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["regress-alibaba", "run", "summarise"])
    ap.add_argument("--family")
    ap.add_argument("--mu", default="primary")
    a = ap.parse_args()
    if a.stage == "regress-alibaba":
        regress_alibaba()
    elif a.stage == "run":
        run_family(a.family, a.mu)
    else:
        from external_replication_summary import summarise  # noqa: E402
        summarise()
