"""Compare a re-run of the adaptive comparison with the frozen v1 outputs.

Usage: python3 compare_outputs.py <rerun out_v1 folder> [frozen out_v1 folder]
Unit counts within budget must match exactly; overload fractions and costs within 1e-9 relative tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

new = Path(sys.argv[1])
old = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "out_v1"
ok = True


def report(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("MATCH " if cond else "DIFFER ") + name + (f"  [{info}]" if info else ""))


for f in ("cal_traces.csv", "cal_periods.csv"):
    src = new / f.replace(".csv", "_rerun.csv")
    if src.exists():
        a, b = pd.read_csv(old / f), pd.read_csv(src)
        same = list(a.columns) == list(b.columns) and len(a) == len(b) and (a.select_dtypes(exclude="number") == b.select_dtypes(exclude="number")).all().all()
        num = a.select_dtypes("number").columns
        close = np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), rtol=1e-9, atol=1e-12)
        report(f"calibration {f}", same and close)

keys = ["trace", "budget", "tau", "policy", "unit_id"]
for f in ("test_per_unit.csv.gz", "confirm_per_unit.csv.gz"):
    if not (new / f).exists():
        report(f, False, "missing in re-run")
        continue
    a = pd.read_csv(old / f).set_index(keys).sort_index()
    b = pd.read_csv(new / f).set_index(keys).sort_index()
    report(f"{f} rows", a.index.equals(b.index), f"{len(a)} vs {len(b)}")
    if not a.index.equals(b.index):
        continue
    for c in ("overload_fraction", "relative_cost", "resource", "churn", "penalty"):
        d = np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)) / np.maximum(1.0, np.abs(a[c].to_numpy(float)))
        report(f"{f} {c}", np.nanmax(d) <= 1e-9, f"max rel diff {np.nanmax(d):.2e}")
    within_a = (a.overload_fraction <= a.index.get_level_values("budget") + 1e-12).groupby(level=[0, 1, 2, 3]).sum()
    within_b = (b.overload_fraction <= b.index.get_level_values("budget") + 1e-12).groupby(level=[0, 1, 2, 3]).sum()
    report(f"{f} units within budget per cell", within_a.equals(within_b), f"{int((within_a != within_b).sum())} cells differ")

print("RESULT", "REPRODUCED" if ok else "DIFFERENCES FOUND")
sys.exit(0 if ok else 1)
