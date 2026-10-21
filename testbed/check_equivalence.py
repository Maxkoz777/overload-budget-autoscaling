#!/usr/bin/env python3
"""Check that the testbed decision rules reproduce the replay of the article decision by decision.

The replay functions are taken verbatim from analysis/audit (pac_window_grid.py, external_delay.py,
external_replication.py) and compared with policies.PACH and policies.ReactiveB1, minute by minute,
on the first test day of every unit of the three confirmatory Huawei periods. For B1 the replay
function returns only totals, so a copy instrumented with one added line records the capacity it
applies at every step; the decision logic itself is unchanged.

Usage (from the repository root):  python3 testbed/check_equivalence.py
The trace folder defaults to experiments/data/external_traces; set TESTBED_DATA to use another one.
"""
from __future__ import annotations

import ast
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "testbed"))
from policies import PACH, ReactiveB1, conformal_rank, pac_rank  # noqa: E402

AUDIT = ROOT / "analysis" / "audit"
DATA = Path(os.environ.get("TESTBED_DATA", ROOT / "experiments" / "data" / "external_traces")) / "huawei2023_confirmatory"
RUNS = ("A_days000_018", "B_days147_165", "C_days168_184")
M = 1440
TIE_REL = 1e-12
SELECTED_W = {("A_days000_018", 0.01): 1440, ("B_days147_165", 0.01): 1440, ("C_days168_184", 0.01): 2880,
              ("A_days000_018", 0.05): 2880, ("B_days147_165", 0.05): 2880, ("C_days168_184", 0.05): 1440}


RECORD = "over += dem[:, s] > mu * cap"


def replay_functions() -> dict:
    wanted = {"pac_window_grid.py": ("pac_rank", "margins", "evaluate"), "external_delay.py": ("closed_loop_reactive",),
              "external_replication.py": ("conf_rank",)}
    ns = {"np": np, "binom": binom, "math": math, "_caps": []}
    for fname, names in wanted.items():
        text = (AUDIT / fname).read_text()
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                src = ast.get_source_segment(text, node)
                if node.name == "closed_loop_reactive":
                    assert src.count(RECORD) == 1
                    src = src.replace(RECORD, "_caps.append(cap.copy()); " + RECORD)
                exec(compile(src, fname, "exec"), ns)
    return ns


def load(run: str) -> dict:
    z = np.load(DATA / run / "series.npz", allow_pickle=True)
    units = pd.read_csv(DATA / run / "units.csv", float_precision="round_trip")
    X = z["demand"].astype(np.float64)
    mu = units.mu_p50.to_numpy(float) * (1.0 + TIE_REL)
    t0 = 10 * M
    return dict(uid=units.unit_id.to_numpy(), X=X, y=X[:, :t0 + M], t0=t0, T=M, mu=mu)


def check_pac(f: dict, D: dict, W: int, delta: float, tau: int, rank: str = "pac") -> int:
    if rank == "pac":
        r = f["pac_rank"](W, delta, 0.05)
        assert r == pac_rank(W, delta, 0.05), (W, delta)
    else:
        r = f["conf_rank"](W, delta)
        assert r == conformal_rank(W, delta), (W, delta)
    h = tau + 1
    m = f["margins"](D, W, h, (r,), start=0)[r]
    forecast = D["y"][:, D["t0"] - 1: D["t0"] + D["T"] - 1]
    nominal = np.maximum(np.ceil((forecast + m) / D["mu"][:, None]), 1.0)
    bad = 0
    for k in range(len(D["uid"])):
        y = D["y"][k]
        ctl = PACH(W, delta, 0.05, tau, D["mu"][k], list(y[:D["t0"]]), rank)
        got = [ctl.initial()] + [ctl.decide(y[D["t0"] + i]) for i in range(D["T"] - 1)]
        bad += int(not np.array_equal(np.array(got, float), nominal[k]))
    return bad


def check_b1(f: dict, D: dict, u: float, c: int) -> int:
    X, t0, T, mu = D["X"], D["t0"], D["T"], D["mu"]
    n = len(mu)
    cap0 = np.maximum(1.0, np.ceil(X[:, t0 - 1] / (0.8 * mu)))
    f["_caps"].clear()
    f["closed_loop_reactive"](X[:, t0 - 1], cap0, X[:, t0:t0 + T], mu, np.full(n, u), np.full(n, float(c)), 0)
    replay = np.stack(f["_caps"], axis=1)
    assert replay.shape == (n, T)
    bad = 0
    for k in range(n):
        ctl = ReactiveB1(u, c, mu[k], X[k, t0 - 1])
        caps = np.array([ctl.initial()] + [ctl.decide(X[k, t0 + i]) for i in range(T - 1)], float)
        bad += int(not np.array_equal(caps, replay[k]))
    return bad


def main() -> int:
    f = replay_functions()
    total = 0
    for run in RUNS:
        D = load(run)
        for delta in (0.01, 0.05):
            W = SELECTED_W[(run, delta)]
            for rank in ("pac", "conformal"):
                for tau in (0, 1):
                    bad = check_pac(f, D, W, delta, tau, rank)
                    total += bad
                    print(f"[{'OK' if bad == 0 else 'MISMATCH'}] {rank} rank {run} delta={delta} W={W} tau={tau}: {len(D['uid']) - bad}/{len(D['uid'])} units identical")
        for u, c in ((0.5, 1), (0.7, 3), (0.9, 5), (0.99, 10)):
            bad = check_b1(f, D, u, c)
            total += bad
            print(f"[{'OK' if bad == 0 else 'MISMATCH'}] B1 {run} u={u} cooldown={c}: {len(D['uid']) - bad}/{len(D['uid'])} units identical")
    print("all decisions identical" if total == 0 else f"{total} mismatching unit checks")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
