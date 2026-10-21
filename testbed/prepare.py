#!/usr/bin/env python3
"""Select the testbed functions by the rule fixed in PROTOCOL.md and write the run plan.

Inputs: the three confirmatory Huawei periods and testbed/capacity.json (the common per-pod
capacity measured on the cloud workers). Outputs in testbed/plan/:
  plan.json                 functions, parameters, and the per-function B1 and HPA settings
  <name>_history_dD.csv     for test day D, the HISTORY minutes before the warm-up
  <name>_run_dD.csv         the WARMUP minutes before day D followed by day D (replayed live)
  <name>_pilot_history.csv  the HISTORY minutes before calibration day 9, minute 0
  <name>_pilot.csv          the first two hours of calibration day 9 (engineering checks only)
  sizing.json               the peak concurrent replicas of every custom policy on days 3-9
Demand is in requests per minute, scaled so that one pod serves mu_phys requests per minute.

Usage (from the repository root):  python3 testbed/prepare.py
The trace folder defaults to experiments/data/external_traces; set TESTBED_DATA to use another one.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testbed"))
from policies import PACH, ReactiveB1  # noqa: E402

DATA = Path(os.environ.get("TESTBED_DATA", ROOT / "experiments" / "data" / "external_traces")) / "huawei2023_confirmatory"
OUT = ROOT / "testbed" / "plan"
RUNS = ("A_days000_018", "B_days147_165", "C_days168_184")
M = 1440
DELTA, ETA, TAU = 0.01, 0.05, 1
SELECTED_W = {"A_days000_018": 1440, "B_days147_165": 1440, "C_days168_184": 2880}
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)
COOLDOWNS = (1, 3, 5, 10)
C_RES, C_ACT, C_VIO = 1.0, 0.05, 10.0
SEED = 20261017
PER_STRATUM = 4
MAX_PEAK_REPLICAS = 10
MIN_MEDIAN_REPLICAS = 2
WARMUP = 30
HISTORY = 3000
DAYS = (10, 11)
PILOT_MINUTES = 120
SIZING_FROM_DAY = 3
POD_CPU_MILLICORES = 40
SIZING_MARGIN = 1.25


def b1_selection(cal: np.ndarray, prev0: float, mu: float) -> tuple[float, int]:
    rows = []
    for c in COOLDOWNS:
        for u in THRESHOLDS:
            ctl = ReactiveB1(u, c, mu, prev0)
            caps = np.array([ctl.initial()] + [ctl.decide(d) for d in cal[:-1]], float)
            over = int((cal > mu * caps).sum())
            total = C_RES * caps.sum() + C_ACT * np.abs(np.diff(caps)).sum() + C_VIO * over
            rows.append((u, c, over / len(cal), total))
    g = pd.DataFrame(rows, columns=["threshold", "cooldown", "overload_fraction", "total"])
    feas = g[g.overload_fraction <= DELTA]
    if len(feas):
        r = feas.sort_values(["total", "overload_fraction", "threshold", "cooldown"], ascending=[True, True, False, True]).iloc[0]
    else:
        r = g.sort_values(["overload_fraction", "total", "threshold", "cooldown"], ascending=[True, True, False, True]).iloc[0]
    return float(r.threshold), int(r.cooldown)


def candidates() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        z = np.load(DATA / run / "series.npz", allow_pickle=True)
        units = pd.read_csv(DATA / run / "units.csv")
        X = z["demand"].astype(np.float64)
        for k, u in units.iterrows():
            hist = X[k, :10 * M]
            mu = float(u.mu_p50)
            rep = np.ceil(hist / mu)
            rows.append({"run": run, "row": k, "unit_id": u.unit_id, "function": int(u.function), "mu": mu,
                         "peak_replicas": float(rep.max()), "median_replicas_cal": float(np.median(rep[8 * M:])),
                         "burstiness": float(hist.max() / max(hist.mean(), 1e-9))})
    c = pd.DataFrame(rows)
    c["eligible"] = (c.peak_replicas <= MAX_PEAK_REPLICAS) & (c.median_replicas_cal >= MIN_MEDIAN_REPLICAS)
    return c


def select(c: pd.DataFrame) -> pd.DataFrame:
    e = c[c.eligible].copy()
    e["stratum"] = pd.qcut(e.burstiness, 3, labels=["low", "mid", "high"])
    rng = np.random.RandomState(SEED)
    chosen, used = [], set()
    for stratum in ("low", "mid", "high"):
        pool = e[e.stratum == stratum].sort_values(["run", "unit_id"]).reset_index(drop=True)
        for i in rng.permutation(len(pool)):
            if pool.loc[i, "function"] in used:
                continue
            chosen.append(pool.loc[i]); used.add(pool.loc[i, "function"])
            if sum(1 for r in chosen if r.stratum == stratum) == PER_STRATUM:
                break
    return pd.DataFrame(chosen).reset_index(drop=True)


def sizing(series: dict[str, np.ndarray], functions: list[dict], mu: float) -> dict:
    start, end = SIZING_FROM_DAY * M, 10 * M
    out = {"days": f"{SIZING_FROM_DAY}-9", "mu_phys_per_minute": mu, "pod_cpu_millicores": POD_CPU_MILLICORES,
           "margin": SIZING_MARGIN, "policies": {}}
    for pol in ("pach", "conf", "b1"):
        total = np.zeros(end - start)
        for fn in functions:
            n = series[fn["name"]]
            hist = list(n[:start])
            if pol == "b1":
                ctl = ReactiveB1(fn["b1_threshold"], fn["b1_cooldown"], mu, hist[-1])
            else:
                ctl = PACH(fn["W"], DELTA, ETA, TAU, mu, hist, "pac" if pol == "pach" else "conformal")
            d = [ctl.initial()] + [ctl.decide(x) for x in n[start:end - 1]]
            total += np.array(d, float)
        peak = int(total.max())
        out["policies"][pol] = {"peak_pods": peak, "mean_pods": round(float(total.mean()), 2),
                                "required_millicores_without_daemonsets": int(math.ceil(SIZING_MARGIN * peak * POD_CPU_MILLICORES))}
    return out


def main() -> None:
    cap = json.loads((ROOT / "testbed" / "capacity.json").read_text())
    mu_phys = float(cap["mu_phys_per_minute"])
    util_at_capacity = float(cap["cpu_utilisation_at_capacity"])
    c = candidates()
    s = select(c)
    OUT.mkdir(exist_ok=True)
    c.to_csv(OUT / "candidates.csv", index=False)
    functions, series = [], {}
    for i, r in s.iterrows():
        z = np.load(DATA / r.run / "series.npz", allow_pickle=True)
        X = z["demand"].astype(np.float64)[int(r.row)]
        scale = mu_phys / r.mu
        n = np.rint(X[:(max(DAYS) + 1) * M] * scale)
        name = f"f{i + 1}"
        series[name] = n
        for D in DAYS:
            pd.DataFrame({"demand": n[D * M - WARMUP - HISTORY:D * M - WARMUP].astype(int)}).to_csv(OUT / f"{name}_history_d{D}.csv", index=False)
            pd.DataFrame({"demand": n[D * M - WARMUP:(D + 1) * M].astype(int)}).to_csv(OUT / f"{name}_run_d{D}.csv", index=False)
        pd.DataFrame({"demand": n[9 * M - HISTORY:9 * M].astype(int)}).to_csv(OUT / f"{name}_pilot_history.csv", index=False)
        pd.DataFrame({"demand": n[9 * M:9 * M + PILOT_MINUTES].astype(int)}).to_csv(OUT / f"{name}_pilot.csv", index=False)
        u, cd = b1_selection(n[8 * M:10 * M], n[8 * M - 1], mu_phys)
        functions.append({"name": name, "run": r.run, "unit_id": r.unit_id, "function": int(r.function),
                          "stratum": str(r.stratum), "burstiness": round(float(r.burstiness), 3),
                          "mu_trace": r.mu, "scale": scale, "W": SELECTED_W[r.run],
                          "b1_threshold": u, "b1_cooldown": cd,
                          "hpa_target_cpu_percent": int(round(100 * u * util_at_capacity)),
                          "initial_replicas_reactive": {str(D): int(max(1, math.ceil(n[D * M - WARMUP - 1] / (0.8 * mu_phys))))
                                                        for D in DAYS},
                          "initial_replicas_reactive_pilot": int(max(1, math.ceil(n[9 * M - 1] / (0.8 * mu_phys))))})
    plan = {"delta": DELTA, "eta": ETA, "tau": TAU, "mu_phys_per_minute": mu_phys, "seed": SEED,
            "warmup_minutes": WARMUP, "minutes": WARMUP + M, "pod_cpu_millicores": POD_CPU_MILLICORES,
            "capacity": {k: v for k, v in cap.items() if k != "measurements"}, "functions": functions}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    size = sizing(series, functions, mu_phys)
    (OUT / "sizing.json").write_text(json.dumps(size, indent=2) + "\n")
    print(pd.DataFrame(functions)[["name", "run", "unit_id", "stratum", "W", "b1_threshold", "b1_cooldown",
                                   "hpa_target_cpu_percent"]].to_string(index=False))
    print(json.dumps(size["policies"], indent=1))


if __name__ == "__main__":
    main()
