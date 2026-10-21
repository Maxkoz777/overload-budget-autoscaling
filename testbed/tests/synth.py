#!/usr/bin/env python3
"""Synthetic testbed data for the tests in run_tests.py (no cluster, no trace data).

  synth.py plan ROOT                       writes ROOT/testbed/plan for three synthetic functions
  synth.py run ROOT OUT MODE [MINUTES]     writes the raw files of one run as the cluster would
  synth.py capacity PHASE_DIR OUT MODE     writes load.csv, cpu.csv and meta.json of a capacity phase
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MU = 420.0
T0 = 1_800_000_000.0
DAY = 10
WARM = 10
HISTORY = 600
LOAD_COLS = ["second", "target", "planned", "dispatched", "ok", "shed", "timeout", "failed", "lag_ms_mean", "lag_ms_max",
             "ok_le5", "ok_le10", "ok_le25", "ok_le50", "ok_le100", "ok_le250", "ok_le500", "ok_le1000", "ok_le2000",
             "ok_gt2000"]
POD_COLS = ["namespace", "pod", "uid", "function", "replicaset", "node", "created", "scheduled", "started", "ready",
            "endpoint_first", "endpoint_last", "deleted", "gone"]
DEC_COLS = ["minute", "function", "demand", "replicas", "planned_at", "decided_at", "called_at", "scaled_at", "attempts",
            "status", "acknowledged", "error"]


def make_plan(root: Path) -> None:
    pdir = root / "testbed" / "plan"
    pdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    functions = []
    for i, (level, burst) in enumerate([(2.0, 0.1), (0.6, 0.4), (1.5, 0.8)]):
        name = f"f{i + 1}"
        n = HISTORY + 200
        base = level * MU * (1 + 0.3 * np.sin(np.arange(n) / 40))
        x = np.maximum(0, np.rint(base * np.exp(burst * rng.standard_normal(n)))).astype(int)
        pd.DataFrame({"demand": x[:HISTORY]}).to_csv(pdir / f"{name}_history_d{DAY}.csv", index=False)
        pd.DataFrame({"demand": x[HISTORY:]}).to_csv(pdir / f"{name}_run_d{DAY}.csv", index=False)
        functions.append({"name": name, "W": 500, "b1_threshold": 0.8, "b1_cooldown": 1, "hpa_target_cpu_percent": 50,
                          "initial_replicas_reactive": {str(DAY): 2}})
    plan = {"delta": 0.01, "eta": 0.05, "tau": 1, "mu_phys_per_minute": MU, "seed": 1, "warmup_minutes": WARM,
            "minutes": 200, "pod_cpu_millicores": 40, "functions": functions}
    (pdir / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")


def make_run(root: Path, out: Path, mode: str, minutes: int) -> None:
    sys.path.insert(0, str(root / "testbed"))
    from policies import PACH, ReactiveB1
    pdir = root / "testbed" / "plan"
    plan = json.loads((pdir / "plan.json").read_text())
    rng = np.random.default_rng(1)
    out.mkdir(parents=True, exist_ok=True)
    load_rows, count_rows, pod_rows, served_rows = [], [], [], []
    decs = {p: [] for p in ("pach", "conf", "b1")}
    for fn in plan["functions"]:
        name = fn["name"]
        h = pd.read_csv(pdir / f"{name}_history_d{DAY}.csv").demand.astype(float).tolist()
        r = pd.read_csv(pdir / f"{name}_run_d{DAY}.csv").demand.to_numpy(float)[:minutes]
        for pol in ("pach", "conf", "b1", "hpa"):
            if pol in ("pach", "conf"):
                c = PACH(fn["W"], plan["delta"], plan["eta"], plan["tau"], MU, h, "pac" if pol == "pach" else "conformal")
            else:
                c = ReactiveB1(fn["b1_threshold"], fn["b1_cooldown"], MU, h[-1])
            d = [c.initial()] + [c.decide(x) for x in r[:-1]]
            pods: list[dict] = []

            def scale(k: int, t: float) -> None:
                live = [p for p in pods if p["deleted"] is None]
                if k > len(live):
                    for _ in range(k - len(live)):
                        ready = t + rng.uniform(18, 20)
                        pods.append({"pod": f"{name}-{pol}-{len(pods)}", "created": float(np.floor(t + 0.05)),
                                     "scheduled": float(np.floor(t + 0.05)), "started": t + 2, "ready": float(np.floor(ready)),
                                     "endpoint_first": ready, "deleted": None, "endpoint_last": None, "gone": None})
                elif k < len(live):
                    live.sort(key=lambda p: (p["endpoint_first"] <= t, -p["created"]))
                    for p in live[:len(live) - k]:
                        p["deleted"] = float(np.floor(t)) + 5.0
                        p["gone"] = t + 0.4
                        p["endpoint_last"] = t + 0.1 if p["endpoint_first"] <= t else None

            scale(d[0], T0 - 50)
            for p in pods:
                p["ready"], p["endpoint_first"] = T0 - 30, T0 - 29.5
            if pol != "hpa":
                decs[pol].append([-1, name, "", d[0], T0 - 50, T0 - 50, T0 - 50, T0 - 49.98, 1, "ok", d[0], ""])
            for m in range(minutes):
                if m >= 1:
                    ts = T0 + m * 60 + 0.6
                    scale(d[m], ts)
                    if pol != "hpa" and not (mode == "nodecisions" and pol == "pach"):
                        decs[pol].append([m - 1, name, r[m - 1], d[m], ts - 0.1, ts - 0.05, ts, ts + 0.02, 1, "ok", d[m], ""])
                per_sec = rng.multinomial(int(r[m]), [1 / 60] * 60)
                for s in range(60):
                    t = T0 + m * 60 + s + 0.2
                    ready = sum(1 for p in pods if p["endpoint_first"] <= t and (p["endpoint_last"] is None or p["endpoint_last"] >= t))
                    existing = sum(1 for p in pods if p["created"] <= t and (p["gone"] is None or p["gone"] > t))
                    gap = (mode == "gap_one_function" and pol == "pach" and name == "f1" and 30 <= m < 33)
                    missing = mode == "missing_series" and pol == "hpa" and name == "f2"
                    partial = mode == "partial" and m >= minutes // 2
                    if not (gap or missing or partial):
                        shown = ready + (5 if mode == "ready_above_bound" and pol == "b1" and m == 40 else 0)
                        count_rows.append([t, pol, name, d[min(m, minutes - 1)], existing, 0, shown])
                    if partial:
                        continue
                    a = int(per_sec[s])
                    bad = max(0, a - int(ready * MU / 60.0))
                    lag = 1500.0 if (mode == "lag" and pol == "b1") else 0.4
                    disp = a - (1 if (mode == "dropped" and pol == "conf" and m == 50 and s == 0 and a > 0) else 0)
                    okc = disp - min(bad, disp)
                    load_rows.append([m * 60 + s, f"{pol}-{name}", a, disp, okc, min(bad, disp), 0, 0, 0.2, lag,
                                      0, 0, okc, 0, 0, 0, 0, 0, 0, 0])
            for p in pods:
                last = p["endpoint_last"] if p["endpoint_last"] else (T0 + minutes * 60 if p["deleted"] is None else None)
                pod_rows.append([pol, p["pod"], p["pod"] + "-uid", name, f"{name}-rs", "node", p["created"], p["scheduled"],
                                 p["started"], p["ready"], p["endpoint_first"], last, p["deleted"], p["gone"]])
                served_rows.append([f"{pol}-{name}", p["pod"], p["endpoint_first"] + 0.3, p["endpoint_first"] + 10, 5])
    pd.DataFrame(load_rows, columns=LOAD_COLS).to_csv(out / "load.csv", index=False)
    pd.DataFrame(count_rows, columns=["t", "namespace", "function", "spec_replicas", "pods_existing", "pods_terminating",
                                      "endpoints_ready"]).to_csv(out / "counts.csv", index=False)
    pd.DataFrame(pod_rows, columns=POD_COLS).to_csv(out / "pods.csv", index=False)
    pd.DataFrame(served_rows, columns=["target", "pod", "first_ok", "last_ok", "served"]).to_csv(out / "served.csv", index=False)
    for pol, rows in decs.items():
        pd.DataFrame(rows, columns=DEC_COLS).to_csv(out / f"decisions_{pol}.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"t0": T0, "interval_seconds": 60, "day": DAY, "seed": 1, "run": "synthetic",
                                               "minutes": minutes, "warmup_minutes": WARM, "assignment": {}}))


def make_capacity(phase: Path, out: Path, mode: str) -> None:
    cfg = json.loads((phase / "loadgen.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    n_sec = cfg["minutes"] * 60
    rows = []
    for t in cfg["targets"]:
        sched = pd.read_csv(phase / f"{t['function']}.csv").demand.to_numpy()
        for s in range(n_sec + 1):
            m = s // 60
            p = 0 if s == n_sec else int(sched[m] // 60 + (1 if s % 60 < sched[m] % 60 else 0))
            r = dict.fromkeys(LOAD_COLS, 0)
            r.update(second=s, target=t["name"], planned=p, dispatched=p, ok=p)
            slow = mode == "cohort_fail" and m >= cfg["minutes"] - 2 and t["function"] == "rest"
            r["ok_le2000" if slow else "ok_le100"] = p
            if mode == "lost" and s == 100 and p:
                r.update(dispatched=0, ok=0, ok_le100=0)
            if mode == "missing_second" and s == 200:
                continue
            rows.append(r)
    pd.DataFrame(rows, columns=LOAD_COLS).to_csv(out / "load.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"t0": T0}))
    namespaces = sorted({t["url"].split(".")[1] for t in cfg["targets"]})
    pods = json.loads((phase / "meta.json").read_text())["deployments"]
    cpu_ms = 7.0 if mode == "slowcpu" else 3.5
    rates = {fn: pd.read_csv(phase / f"{fn}.csv").demand.to_numpy() / 60.0 / pods[fn] for fn in cfg["schedules"]}
    cpu = [{"t": T0 + k * 15, "namespace": ns, "pod": f"p{k}", "function": fn, "window": "15s",
            "millicores": min(40.0, 0.4 + cpu_ms * rates[fn][k // 4])}
           for k in range(cfg["minutes"] * 4) for ns in namespaces for fn in cfg["schedules"]]
    pd.DataFrame(cpu).to_csv(out / "cpu.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan":
        make_plan(Path(sys.argv[2]))
    elif cmd == "run":
        make_run(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 90)
    elif cmd == "capacity":
        make_capacity(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
