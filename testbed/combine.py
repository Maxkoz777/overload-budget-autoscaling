#!/usr/bin/env python3
"""Combine the analysed runs into the results of PROTOCOL.md, Sect. 4.

Reads per_function.csv, summary.json and validity.json of every run folder and writes
combined.json and combined_per_function.csv next to the first folder's parent:
  validity of every run; H1 per policy (share of function-runs within the tolerance, with
  undetermined pairs counted as not within; ready pod-minutes ratio per run and policy);
  H2 per function-run; H3 with the function as unit; repeatability between the two runs of each day.

H3 uses interval arithmetic. For every function the overload minutes of a policy, averaged over
runs, lie in [lo, hi]; the difference pach - other lies in [pach_lo - other_hi, pach_hi - other_lo].
The one-sided sign test of pach < other is computed at the upper ends of these intervals (the
largest possible p-value, which decides) and at the lower ends (the smallest possible p-value);
Holm's correction is applied to each. A comparison is "supported" if the largest adjusted p is
below 0.05, "not supported" if the smallest is not, and "undetermined" otherwise.

Only a complete campaign is combined: four valid runs run1-run4 with the assignments of
render.py and one row per function and policy. With --descriptive, an incomplete set is combined
and every output is marked as descriptive (engineering checks only).

Usage (from the repository root):
  python3 testbed/combine.py testbed/results/run1 testbed/results/run2 testbed/results/run3 testbed/results/run4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

H1_SHARE = 0.90
H1_RATIO = (0.95, 1.05)
BOOT = 10000
SEED = 20261017
ALPHA = 0.05
TB = Path(__file__).resolve().parent
sys.path.insert(0, str(TB))
from render import POLICIES, RUNS, assignment  # noqa: E402


def sign_test(diff: pd.Series) -> dict:
    neg, pos = int((diff < 0).sum()), int((diff > 0).sum())
    p = binomtest(neg, neg + pos, 0.5, alternative="greater").pvalue if neg + pos else 1.0
    return {"favouring_pach": neg, "against": pos, "ties": int((diff == 0).sum()), "p_one_sided": float(p)}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def campaign_problems(runs: dict, pf: pd.DataFrame, functions: list[str]) -> list[str]:
    problems = []
    expected = {f"run{i}" for i in RUNS}
    if set(runs) != expected:
        problems.append(f"runs {sorted(runs)} instead of {sorted(expected)}")
    for name, r in runs.items():
        if not r["valid"]:
            problems.append(f"{name} is invalid")
        i = int(name[3:]) if name.startswith("run") and name[3:].isdigit() else None
        if i in RUNS and (r["assignment"] != assignment(i) or str(r["day"]) != str(RUNS[i]["day"])):
            problems.append(f"{name}: assignment or day differ from the protocol")
    for name, g in pf.groupby("run"):
        keys = set(zip(g.function, g.policy))
        want = {(f, p) for f in functions for p in POLICIES}
        if keys != want or len(g) != len(want):
            problems.append(f"{name}: {len(g)} function-policy rows, expected {len(want)}")
    return problems


def h3_tests(pf: pd.DataFrame, rng: np.random.Generator) -> dict:
    lo = pf.groupby(["function", "policy"]).overload_lo.mean().unstack()
    hi = pf.groupby(["function", "policy"]).overload_hi.mean().unstack()
    tests = {}
    for other in ("conf", "b1"):
        d_lo, d_hi = lo["pach"] - hi[other], hi["pach"] - lo[other]
        worst, best = sign_test(d_hi), sign_test(d_lo)
        idx = rng.integers(0, len(d_lo), size=(BOOT, len(d_lo)))
        tests[f"pach_vs_{other}"] = {
            "difference_interval_by_function": {f: [float(a), float(b)] for f, a, b in zip(d_lo.index, d_lo, d_hi)},
            "functions_certainly_favouring_pach": int((d_hi < 0).sum()), "functions_certainly_against": int((d_lo > 0).sum()),
            "functions_undetermined": int(((d_lo <= 0) & (d_hi >= 0) & ~((d_lo == 0) & (d_hi == 0))).sum()),
            "sign_test_worst_case": worst, "sign_test_best_case": best,
            "mean_difference_interval": [float(d_lo.mean()), float(d_hi.mean())],
            "bootstrap_95_outer": [float(np.quantile(d_lo.to_numpy()[idx].mean(axis=1), 0.025)),
                                   float(np.quantile(d_hi.to_numpy()[idx].mean(axis=1), 0.975))]}
    keys = list(tests)
    for case in ("worst_case", "best_case"):
        for k, p in zip(keys, holm([tests[k][f"sign_test_{case}"]["p_one_sided"] for k in keys])):
            tests[k][f"sign_test_{case}"]["p_holm"] = p
    for k in keys:
        pw, pb = tests[k]["sign_test_worst_case"]["p_holm"], tests[k]["sign_test_best_case"]["p_holm"]
        tests[k]["conclusion"] = "supported" if pw < ALPHA else ("not supported" if pb >= ALPHA else "undetermined")
    return tests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--plan", default=str(TB / "plan" / "plan.json"))
    ap.add_argument("--descriptive", action="store_true")
    a = ap.parse_args()
    dirs = [Path(d) for d in a.dirs]
    functions = [f["name"] for f in json.loads(Path(a.plan).read_text())["functions"]]
    frames, runs = [], {}
    for d in dirs:
        s = json.loads((d / "summary.json").read_text())
        v = json.loads((d / "validity.json").read_text())
        f = pd.read_csv(d / "per_function.csv")
        f["run"], f["day"] = s["run"], s["day"]
        frames.append(f)
        runs[s["run"]] = {"day": s["day"], "assignment": s["assignment"], "valid": v["valid"], "problems": v["problems"]}
    pf = pd.concat(frames, ignore_index=True)
    problems = campaign_problems(runs, pf, functions)
    if problems and not a.descriptive:
        raise SystemExit("incomplete campaign, nothing combined:\n" + "\n".join(problems))
    out = {"confirmatory": not problems, "campaign_problems": problems, "runs": runs, "h1": {}, "h2": {}, "h3": {},
           "repeatability": {}}
    for pol, g in pf[pf.policy.isin(["pach", "conf", "b1"])].groupby("policy"):
        if "h1" not in g:
            continue
        share = float((g.h1 == "within").mean())
        ratios = {}
        for run, gr in g.groupby("run"):
            model = (gr.ready_pod_minutes_lo / gr.h1_ready_ratio_lo).sum()
            ratios[run] = [float(gr.ready_pod_minutes_lo.sum() / model), float(gr.ready_pod_minutes_hi.sum() / model)]
        ok_ratio = all(H1_RATIO[0] <= lo and hi <= H1_RATIO[1] for lo, hi in ratios.values())
        out["h1"][pol] = {"pairs": int(len(g)), "within": int((g.h1 == "within").sum()), "outside": int((g.h1 == "outside").sum()),
                          "undetermined": int((g.h1 == "undetermined").sum()), "share_within": share,
                          "ready_ratio_by_run": ratios, "criterion_met": bool(share >= H1_SHARE and ok_ratio),
                          "fifo_tau0_overload": int(g.fifo_tau0_overload.sum()), "fifo_tau1_overload": int(g.fifo_tau1_overload.sum()),
                          "model_overload": int(g.model_overload.sum()), "observed_overload_lo": int(g.overload_lo.sum()),
                          "observed_overload_hi": int(g.overload_hi.sum())}
    for pol, g in pf.groupby("policy"):
        out["h2"][pol] = g.h2.value_counts().to_dict()
    out["h3"] = h3_tests(pf, np.random.default_rng(SEED))
    for day, g in pf.groupby("day"):
        runs_of_day = sorted(g.run.unique())
        if len(runs_of_day) == 2:
            a, b = (g[g.run == r].set_index(["function", "policy"]) for r in runs_of_day)
            out["repeatability"][str(day)] = {
                "runs": runs_of_day,
                "abs_diff_overload_lo_mean": float((a.overload_lo - b.overload_lo).abs().mean()),
                "abs_diff_ready_pod_minutes_rel_mean": float(((a.ready_pod_minutes_lo - b.ready_pod_minutes_lo).abs() / a.ready_pod_minutes_lo).mean())}
    base = dirs[0].parent
    stem = "combined" if out["confirmatory"] else "combined_descriptive"
    pf.to_csv(base / f"{stem}_per_function.csv", index=False)
    (base / f"{stem}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
