#!/usr/bin/env python3
"""Measure the common per-pod capacity mu_phys and the CPU utilisation c on every worker.

The rule (fixed in PROTOCOL.md before the cloud calibration):
  phase 1  on every worker, one pod: 3 minutes at 2 requests/s, then 4, 6, ..., 30 requests/s for
           3 minutes each. r1_w is the highest level whose last two minutes have a bad-request
           fraction below 1% (bad: 503, timeout, error, or slower than 500 ms). r* = min_w r1_w.
  phase 2  on every worker, one pod: r*-2, r*-1, r*, r*+1 requests/s, three repeats each in a
           seeded random order, 3 minutes each. mu_w is the highest rate at which all three
           repeats pass; the candidate mu = min_w mu_w. c is the mean pod CPU at rate mu (last
           two minutes, all repeats and workers) divided by the pod's CPU request.
  phase 3  verification of the candidate mu, on all workers at the same time: 2, 4 and 8 pods at
           k*mu requests/s, 5 minutes each, then the peak cohort: P pods at P*mu requests/s for
           5 minutes, where P is the largest peak of concurrent pods of one policy in
           plan/sizing.json. Every step must pass on every worker. If a step fails, mu is
           lowered by 1 request/s (never below r*-2, where c is still measured in phase 2),
           prepare.py is rerun and phase 3 is repeated at the new mu. If r*-2 fails, the
           calibration has failed and the protocol must be amended before the freeze.
Work per request: before r* is determined, phase 1 measures the CPU time of one request on every
worker (mean pod CPU over the last two minutes of the 2 and 4 requests/s levels, divided by the
rate). If the mean over workers lies outside [3.0, 4.0] ms, ITER is rescaled to
round(ITER * 3.5 / measured, -2), recorded in pod_iter.json, and phase 1 is repeated (exit code 3).
This fixes the work per request in CPU time on the cluster, not in hash rounds.
Every level must be delivered exactly as scheduled (planned = schedule, dispatched = planned,
an outcome for every request) and every CPU estimate needs at least four samples; otherwise
the phase is incomplete and is repeated, never scored. The full curves are kept in capacity.json.

Usage (from the repository root):
  python3 testbed/capacity.py plan1
  python3 testbed/capacity.py analyze1 --results DIR
  python3 testbed/capacity.py plan2
  python3 testbed/capacity.py analyze2 --results DIR      writes the candidate capacity.json
  python3 testbed/prepare.py
  python3 testbed/capacity.py plan3
  python3 testbed/capacity.py analyze3 --results DIR      verifies or lowers mu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import POD_ITER, pod_iter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "testbed"
DIR = TB / "capacity"
CAPACITY = TB / "capacity.json"
WORKERS = ("w1", "w2", "w3", "w4")
POD_CPU_MILLICORES = 40
MAX_BAD = 0.01
MIN_CPU_SAMPLES = 4
SEED = 20261017
PHASE1 = [2] + list(range(4, 31, 2))
STEP1, STEP2, STEP3 = 3, 3, 5
LINEAR_PODS = (2, 4, 8)
CPU_MS_BAND = (3.0, 4.0)
CPU_MS_TARGET = 3.5
ITER_CHANGED = 3


class Incomplete(Exception):
    pass


def load_results(results: Path) -> pd.DataFrame:
    load = pd.read_csv(results / "load.csv")
    load["minute"] = load.second // 60
    load["bad"] = load.shed + load.timeout + load.failed + load[["ok_le1000", "ok_le2000", "ok_gt2000"]].sum(axis=1)
    load["outcomes"] = load.ok + load.shed + load.timeout + load.failed
    return load


def write_phase(name: str, targets: list[dict], schedules: dict[str, list[int]], meta: dict) -> None:
    d = DIR / name
    d.mkdir(parents=True, exist_ok=True)
    for fn, counts in schedules.items():
        pd.DataFrame({"demand": counts}).to_csv(d / f"{fn}.csv", index=False)
    minutes = max(len(v) for v in schedules.values())
    cfg = {"schedules": {fn: f"/plan/{fn}.csv" for fn in schedules}, "targets": targets, "seed": SEED,
           "minutes": minutes, "interval_seconds": 60}
    (d / "loadgen.json").write_text(json.dumps(cfg, indent=1) + "\n")
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{name}: {minutes} minutes, {len(targets)} targets")


def targets_for(schedules: dict) -> list[dict]:
    return [{"name": f"{w}-{fn}", "function": fn, "url": f"http://{fn}.cap-{w}.svc.cluster.local/work"}
            for w in WORKERS for fn in schedules]


def check_complete(load: pd.DataFrame, phase: str) -> None:
    """Every target must have every second, the scheduled count per minute, and an outcome per request."""
    d = DIR / phase
    cfg = json.loads((d / "loadgen.json").read_text())
    problems = []
    for t in cfg["targets"]:
        sched = pd.read_csv(d / f"{t['function']}.csv").demand.to_numpy()
        n_sec = cfg["minutes"] * 60
        all_rows = load[load.target == t["name"]]
        g = all_rows[all_rows.second < n_sec]
        if len(g) != n_sec or g.second.duplicated().any() or all_rows[all_rows.second >= n_sec].planned.sum():
            problems.append(f"{t['name']}: {len(g)} of {n_sec} seconds")
            continue
        m = g.groupby("minute")[["planned", "dispatched", "outcomes"]].sum().reindex(range(cfg["minutes"]), fill_value=-1)
        if not np.array_equal(m.planned.to_numpy(), sched):
            problems.append(f"{t['name']}: planned requests differ from the schedule")
        if not m.dispatched.equals(m.planned):
            problems.append(f"{t['name']}: dispatched differ from planned")
        if not m.outcomes.equals(m.dispatched):
            problems.append(f"{t['name']}: requests without outcome")
    if problems:
        raise Incomplete(f"{phase} incomplete, repeat it: " + "; ".join(problems[:10]))


def level_bad(load: pd.DataFrame, target: str, start: int, length: int) -> tuple[float, int]:
    g = load[(load.target == target) & (load.minute >= start + length - 2) & (load.minute < start + length)]
    n = int(g.dispatched.sum())
    return (float(g.bad.sum() / n) if n else float("nan")), n


def passes(frac: float) -> bool:
    return bool(frac < MAX_BAD)


def cpu_mean(cpu: pd.DataFrame, ns: str, fn: str, start: int, length: int) -> tuple[float, int]:
    g = cpu[(cpu.namespace == ns) & (cpu.function == fn) & (cpu.minute >= start + length - 2) & (cpu.minute < start + length)]
    return float(g.millicores.mean()) if len(g) else float("nan"), int(len(g))


def read_cpu(results: Path) -> pd.DataFrame:
    cpu = pd.read_csv(results / "cpu.csv")
    t0 = json.loads((results / "meta.json").read_text())["t0"]
    cpu["minute"] = ((cpu.t - t0) // 60).astype(int)
    return cpu


def plan1() -> None:
    counts = [r * 60 for r in PHASE1 for _ in range(STEP1)]
    write_phase("phase1", targets_for({"single": counts}), {"single": counts},
                {"levels": PHASE1, "minutes_per_level": STEP1, "deployments": {"single": 1}, "iter": pod_iter()})


def analyze1(results: Path) -> None:
    meta = json.loads((DIR / "phase1" / "meta.json").read_text())
    if meta["iter"] != pod_iter():
        raise SystemExit("phase 1 was planned with another ITER; rerun plan1")
    load = load_results(results)
    check_complete(load, "phase1")
    cpu = read_cpu(results)
    per_request = []
    for w in WORKERS:
        for i, r in enumerate(PHASE1):
            if r <= 4:
                c, k = cpu_mean(cpu, f"cap-{w}", "single", i * STEP1, STEP1)
                if k < MIN_CPU_SAMPLES:
                    raise Incomplete(f"phase1 incomplete, repeat it: {w} level {r} has {k} CPU samples")
                per_request.append(c / r)
    cpu_ms = float(np.mean(per_request))
    history = json.loads(POD_ITER.read_text())["history"] if POD_ITER.exists() else []
    history.append({"iter": meta["iter"], "cpu_ms_per_request": round(cpu_ms, 3)})
    if not CPU_MS_BAND[0] <= cpu_ms <= CPU_MS_BAND[1]:
        new = int(round(meta["iter"] * CPU_MS_TARGET / cpu_ms, -2))
        POD_ITER.write_text(json.dumps({"iter": new, "history": history}, indent=1) + "\n")
        print(f"one request costs {cpu_ms:.2f} ms of CPU with ITER = {meta['iter']}; ITER set to {new}; repeat phase 1")
        raise SystemExit(ITER_CHANGED)
    POD_ITER.write_text(json.dumps({"iter": meta["iter"], "history": history}, indent=1) + "\n")
    print(f"one request costs {cpu_ms:.2f} ms of CPU with ITER = {meta['iter']}")
    out = {}
    for w in WORKERS:
        rows = []
        for i, r in enumerate(PHASE1):
            frac, n = level_bad(load, f"{w}-single", i * STEP1, STEP1)
            rows.append({"rate": r, "bad_fraction": frac, "requests": n})
        ok = [x["rate"] for x in rows if passes(x["bad_fraction"])]
        out[w] = {"r1": max(ok) if ok else None, "levels": rows}
    if any(v["r1"] is None for v in out.values()):
        raise SystemExit("phase 1: a worker fails even the lowest level")
    r_star = min(v["r1"] for v in out.values())
    if r_star - 2 < 1:
        raise SystemExit(f"phase 1: r* = {r_star} is too small for the phase-2 grid")
    (DIR / "phase1_result.json").write_text(json.dumps({"r_star": r_star, "workers": out}, indent=1) + "\n")
    print(json.dumps({w: v["r1"] for w, v in out.items()}), "r* =", r_star)


def plan2() -> None:
    r_star = json.loads((DIR / "phase1_result.json").read_text())["r_star"]
    rates = [r_star - 2, r_star - 1, r_star, r_star + 1]
    order = [r for r in rates for _ in range(3)]
    np.random.RandomState(SEED).shuffle(order)
    single = [r * 60 for r in order for _ in range(STEP2)]
    write_phase("phase2", targets_for({"single": single}), {"single": single},
                {"r_star": r_star, "order": order, "minutes_per_level": STEP2, "deployments": {"single": 1}})


def cpu_at(phase2: dict, mu: int) -> float:
    vals = [x["cpu_millicores"] for v in phase2.values() for x in v["repeats"] if x["rate"] == mu]
    return float(np.mean(vals) / POD_CPU_MILLICORES)


def write_capacity(mu: int, r_star: int, phase2: dict, verification: list[dict], verified: bool, note: str) -> None:
    c = cpu_at(phase2, mu)
    out = {"mu_phys_per_minute": int(mu * 60), "rate_per_second": int(mu), "cpu_utilisation_at_capacity": round(c, 4),
           "pod_cpu_millicores": POD_CPU_MILLICORES, "iter": pod_iter(), "verified": verified, "note": note, "r_star": r_star,
           "measurements": {"phase1": json.loads((DIR / "phase1_result.json").read_text()), "phase2": phase2,
                            "phase3": verification}}
    CAPACITY.write_text(json.dumps(out, indent=1) + "\n")
    print(f"mu = {mu} requests/s, c = {c:.3f}, verified = {verified}" + (f" ({note})" if note else ""))


def analyze2(results: Path) -> None:
    meta = json.loads((DIR / "phase2" / "meta.json").read_text())
    r_star, order = meta["r_star"], meta["order"]
    load = load_results(results)
    check_complete(load, "phase2")
    cpu = read_cpu(results)
    workers = {}
    for w in WORKERS:
        rep = []
        for i, r in enumerate(order):
            frac, n = level_bad(load, f"{w}-single", i * STEP2, STEP2)
            c, k = cpu_mean(cpu, f"cap-{w}", "single", i * STEP2, STEP2)
            if k < MIN_CPU_SAMPLES:
                raise Incomplete(f"phase2 incomplete, repeat it: {w} repeat {i} has {k} CPU samples")
            rep.append({"rate": r, "bad_fraction": frac, "requests": n, "cpu_millicores": c, "cpu_samples": k})
        rep_df = pd.DataFrame(rep)
        passing = [int(r) for r, g in rep_df.groupby("rate") if all(passes(x) for x in g.bad_fraction)]
        workers[w] = {"mu_w": max(passing) if passing else None, "repeats": rep}
    if any(v["mu_w"] is None for v in workers.values()):
        raise SystemExit("phase 2: a worker fails every rate; the calibration has failed")
    mu = min(v["mu_w"] for v in workers.values())
    print(json.dumps({w: v["mu_w"] for w, v in workers.items()}))
    write_capacity(mu, r_star, workers, [], False, "candidate; run prepare.py and phase 3")


def plan3() -> None:
    cap = json.loads(CAPACITY.read_text())
    if cap["verified"]:
        raise SystemExit("capacity.json is already verified")
    mu = cap["rate_per_second"]
    sizing = json.loads((TB / "plan" / "sizing.json").read_text())
    if int(sizing["mu_phys_per_minute"]) != mu * 60:
        raise SystemExit("plan/sizing.json was computed for another mu; rerun prepare.py")
    peak = max(v["peak_pods"] for v in sizing["policies"].values())
    rest = peak - sum(LINEAR_PODS)
    if rest < 1:
        raise SystemExit(f"peak cohort {peak} is not larger than the linear steps")
    lin_minutes = len(LINEAR_PODS) * STEP3
    minutes = lin_minutes + STEP3
    schedules = {}
    for j, k in enumerate(LINEAR_PODS):
        s = [0] * minutes
        for m in range(STEP3):
            s[j * STEP3 + m] = k * mu * 60
            s[lin_minutes + m] = k * mu * 60
        schedules[f"pods{k}"] = s
    schedules["rest"] = [0] * lin_minutes + [rest * mu * 60] * STEP3
    write_phase("phase3", targets_for(schedules), schedules,
                {"mu": mu, "peak_pods": peak, "minutes_per_step": STEP3, "linear_pods": list(LINEAR_PODS),
                 "deployments": {**{f"pods{k}": k for k in LINEAR_PODS}, "rest": rest}})


def analyze3(results: Path) -> None:
    meta = json.loads((DIR / "phase3" / "meta.json").read_text())
    cap = json.loads(CAPACITY.read_text())
    mu, r_star = meta["mu"], cap["r_star"]
    if mu != cap["rate_per_second"]:
        raise SystemExit("phase 3 was planned for another mu")
    load = load_results(results)
    check_complete(load, "phase3")
    cpu = read_cpu(results)
    lin_minutes = len(meta["linear_pods"]) * STEP3
    steps, ok = [], True
    for w in WORKERS:
        for j, k in enumerate(meta["linear_pods"]):
            frac, n = level_bad(load, f"{w}-pods{k}", j * STEP3, STEP3)
            c, _ = cpu_mean(cpu, f"cap-{w}", f"pods{k}", j * STEP3, STEP3)
            steps.append({"worker": w, "step": f"{k} pods", "pods": k, "bad_fraction": frac, "requests": n,
                          "cpu_millicores_per_pod": c, "pass": passes(frac)})
        g = load[load.target.str.startswith(f"{w}-") & (load.minute >= lin_minutes + STEP3 - 2) & (load.minute < lin_minutes + STEP3)]
        n = int(g.dispatched.sum())
        frac = float(g.bad.sum() / n) if n else float("nan")
        steps.append({"worker": w, "step": "peak cohort", "pods": meta["peak_pods"], "bad_fraction": frac, "requests": n,
                      "pass": passes(frac)})
    ok = all(s["pass"] for s in steps)
    verification = cap["measurements"].get("phase3", []) + [{"mu": mu, "steps": steps, "pass": ok}]
    phase2 = cap["measurements"]["phase2"]
    if ok:
        write_capacity(mu, r_star, phase2, verification, True, "")
        return
    failed = [f"{s['worker']} {s['step']}" for s in steps if not s["pass"]]
    if mu - 1 < r_star - 2:
        earlier = sum(1 for v in cap["measurements"].get("phase3", []) if v["mu"] == mu and not v["pass"])
        if earlier == 0:
            write_capacity(mu, r_star, phase2, verification, False,
                           f"phase 3 failed at r*-2 ({', '.join(failed)}); one repeat at the same mu (amendment A1)")
            return
        write_capacity(mu, r_star, phase2, verification, False, "calibration failed at r*-2")
        raise SystemExit(f"phase 3 failed at the lowest admissible mu ({failed}); amend the protocol before the freeze")
    write_capacity(mu - 1, r_star, phase2, verification, False, f"phase 3 failed at {mu} ({', '.join(failed)}); rerun prepare.py and phase 3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan1", "analyze1", "plan2", "analyze2", "plan3", "analyze3"))
    ap.add_argument("--results")
    a = ap.parse_args()
    try:
        if a.cmd.startswith("plan"):
            {"plan1": plan1, "plan2": plan2, "plan3": plan3}[a.cmd]()
        else:
            {"analyze1": analyze1, "analyze2": analyze2, "analyze3": analyze3}[a.cmd](Path(a.results))
    except Incomplete as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
