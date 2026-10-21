#!/usr/bin/env python3
"""Run PAC-h, its standard-rank ablation, or B1 for every function of one namespace.

After interval m ends and the load generator has dispatched all its requests, the controller
reads the requests each function received in interval m, asks the policy for the replica count,
and sets it with the Kubernetes scale API. Before interval 0 it sets the policy's initial count.
Every decision is logged with its planned and actual time, the number of attempts, the outcome
of the scale call, and the replica count the API acknowledged.

Usage: controller.py --plan plan.json --policy pach|conf|b1 --namespace NS --out decisions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from kube import Kube  # noqa: E402
from policies import PACH, ReactiveB1  # noqa: E402

ATTEMPTS = 5
COLS = ["minute", "function", "demand", "replicas", "planned_at", "decided_at", "called_at", "scaled_at", "attempts",
        "status", "acknowledged", "error"]


def demand_history(plan_dir: Path, name: str) -> list[float]:
    with open(plan_dir / f"{name}_history.csv") as f:
        return [float(r["demand"]) for r in csv.DictReader(f)]


def stats(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.load(r)


def scale(kube: Kube, ns: str, name: str, k: int) -> tuple[float, float, int, str, object, str]:
    err = ""
    for attempt in range(1, ATTEMPTS + 1):
        called = time.time()
        try:
            r = kube.scale(ns, name, k)
            return called, time.time(), attempt, "ok", r.get("spec", {}).get("replicas"), ""
        except OSError as e:
            err = str(e)
            time.sleep(0.5)
    return called, time.time(), ATTEMPTS, "failed", None, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--policy", choices=("pach", "conf", "b1"), required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--stats", default="http://127.0.0.1:9100/stats")
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan_path = Path(a.plan)
    plan = json.loads(plan_path.read_text())
    kube = Kube(a.proxy)
    mu = plan["mu_phys_per_minute"]
    ctls = {}
    for fn in plan["functions"]:
        hist = demand_history(plan_path.parent, fn["name"])
        if a.policy in ("pach", "conf"):
            ctls[fn["name"]] = PACH(fn["W"], plan["delta"], plan["eta"], plan["tau"], mu, hist,
                                    "pac" if a.policy == "pach" else "conformal")
        else:
            ctls[fn["name"]] = ReactiveB1(fn["b1_threshold"], fn["b1_cooldown"], mu, hist[-1])
    out = open(a.out, "w", newline="")
    log = csv.writer(out)
    log.writerow(COLS)
    for name, ctl in ctls.items():
        k = ctl.initial()
        now = time.time()
        log.writerow([-1, name, "", k, now, now, *scale(kube, a.namespace, name, k)])
    out.flush()
    while True:
        try:
            s = stats(a.stats)
            break
        except OSError:
            time.sleep(1)
    t0, interval = s["t0"], s["interval_seconds"]
    minutes = plan["minutes"]
    (Path(a.out).parent / "meta.json").write_text(json.dumps(
        {"t0": t0, "interval_seconds": interval, "day": plan.get("day"), "seed": plan.get("seed"),
         "run": plan.get("run"), "minutes": minutes, "warmup_minutes": plan["warmup_minutes"],
         "assignment": plan.get("assignment")}) + "\n")
    for m in range(minutes - 1):
        planned = t0 + (m + 1) * interval + 0.5
        time.sleep(max(0.0, planned - time.time()))
        while True:
            try:
                s = stats(a.stats)
                if all(len(s["arrivals"][f"{a.namespace}-{n}"]) > m for n in ctls):
                    break
            except (OSError, KeyError, ValueError):
                pass
            time.sleep(0.2)
        for name, ctl in ctls.items():
            d = s["arrivals"][f"{a.namespace}-{name}"][m]
            decided = time.time()
            k = ctl.decide(d)
            log.writerow([m, name, d, k, planned, decided, *scale(kube, a.namespace, name, k)])
        out.flush()
    out.close()


if __name__ == "__main__":
    main()
