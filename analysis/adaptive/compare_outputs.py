"""Compare a re-run of the adaptive comparison with the frozen v1 outputs.

Usage: python3 compare_outputs.py <re-run out_v1 folder> [frozen out_v1 folder]

Fails (exit code 1) unless all of the following hold:
- the re-run folder contains cal_traces_rerun.csv, cal_periods_rerun.csv, test_per_unit.csv.gz and confirm_per_unit.csv.gz;
- every file has the same columns as its frozen counterpart and unique identifying keys;
- the same rows exist in both, and in every column the missing-value masks agree, text values are equal,
  and numeric values agree within a relative tolerance of 1e-9;
- the five outcome metrics are finite in every row of both files;
- the number of units within budget agrees in every cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RTOL, ATOL = 1e-9, 1e-12
METRICS = ("overload_fraction", "relative_cost", "resource", "churn", "penalty")
UNIT_KEYS = ["trace", "budget", "tau", "policy", "unit_id"]
CAL_KEYS = ["unit", "budget", "gamma"]
failures: list[str] = []


def report(name: str, cond: bool, info: str = "") -> None:
    print(("MATCH " if cond else "DIFFER ") + name + (f"  [{info}]" if info else ""))
    if not cond:
        failures.append(name)


def compare_frames(name: str, a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> bool:
    if list(a.columns) != list(b.columns):
        report(f"{name} columns", False, f"{list(a.columns)} vs {list(b.columns)}")
        return False
    report(f"{name} columns", True, f"{len(a.columns)}")
    for label, f in (("frozen", a), ("re-run", b)):
        dup = int(f.duplicated(keys).sum())
        report(f"{name} unique keys ({label})", dup == 0, f"{dup} duplicates")
        if dup:
            return False
    a = a.set_index(keys).sort_index()
    b = b.set_index(keys).sort_index()
    same_rows = a.index.equals(b.index)
    report(f"{name} rows", same_rows, f"{len(a)} vs {len(b)}")
    if not same_rows:
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        nx, ny = x.isna().to_numpy(), y.isna().to_numpy()
        if not np.array_equal(nx, ny):
            report(f"{name} {c}", False, f"missing-value masks differ in {int((nx != ny).sum())} rows")
            continue
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float)[~nx], y.to_numpy(float)[~ny]
            ok = np.allclose(xv, yv, rtol=RTOL, atol=ATOL)
            diff = float(np.max(np.abs(xv - yv) / np.maximum(1.0, np.abs(xv)))) if len(xv) else 0.0
            report(f"{name} {c}", ok, f"max rel diff {diff:.2e}")
        else:
            ok = bool((x.astype(str)[~nx].to_numpy() == y.astype(str)[~ny].to_numpy()).all())
            report(f"{name} {c}", ok, "text")
    return True


def main(new: Path, old: Path) -> int:
    expected = ["cal_traces_rerun.csv", "cal_periods_rerun.csv", "test_per_unit.csv.gz", "confirm_per_unit.csv.gz"]
    for f in expected:
        report(f"present {f}", (new / f).exists())
    for f in ("cal_traces.csv", "cal_periods.csv"):
        src = new / f.replace(".csv", "_rerun.csv")
        if src.exists():
            compare_frames(f"calibration {f}", pd.read_csv(old / f), pd.read_csv(src), CAL_KEYS)
    for f in ("test_per_unit.csv.gz", "confirm_per_unit.csv.gz"):
        if not (new / f).exists():
            continue
        a, b = pd.read_csv(old / f), pd.read_csv(new / f)
        for label, frame in (("frozen", a), ("re-run", b)):
            fin = all(np.isfinite(frame[m].to_numpy(float)).all() for m in METRICS)
            report(f"{f} outcome metrics finite ({label})", fin)
        if not compare_frames(f, a, b, UNIT_KEYS):
            continue
        wa = (a.overload_fraction <= a.budget + 1e-12).groupby([a.trace, a.budget, a.tau, a.policy]).sum()
        wb = (b.overload_fraction <= b.budget + 1e-12).groupby([b.trace, b.budget, b.tau, b.policy]).sum()
        report(f"{f} units within budget per cell", wa.equals(wb), f"{int((wa != wb).sum())} cells differ" if wa.index.equals(wb.index) else "cells differ")
    print("RESULT", "REPRODUCED" if not failures else f"DIFFERENCES FOUND ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    new = Path(sys.argv[1])
    old = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "out_v1"
    sys.exit(main(new, old))
