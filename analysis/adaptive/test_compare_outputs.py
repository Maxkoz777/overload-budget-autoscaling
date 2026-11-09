"""Negative and positive checks of compare_outputs.py on small copies of the frozen outputs.

Usage: python3 test_compare_outputs.py   (needs only numpy and pandas)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "out_v1"


def make(root: Path) -> tuple[Path, Path]:
    old, new = root / "old", root / "new"
    old.mkdir(); new.mkdir()
    for f in ("cal_traces.csv", "cal_periods.csv"):
        shutil.copy(FROZEN / f, old / f)
        shutil.copy(FROZEN / f, new / f.replace(".csv", "_rerun.csv"))
    for f in ("test_per_unit.csv.gz", "confirm_per_unit.csv.gz"):
        d = pd.read_csv(FROZEN / f)
        d = pd.concat([d[d.policy == p].head(40) for p in d.policy.unique()])
        d.to_csv(old / f, index=False); d.to_csv(new / f, index=False)
    return old, new


def run(old: Path, new: Path) -> int:
    return subprocess.run([sys.executable, str(HERE / "compare_outputs.py"), str(new), str(old)], capture_output=True).returncode


def case(name, mutate, expect_ok):
    with tempfile.TemporaryDirectory() as t:
        old, new = make(Path(t))
        mutate(new)
        ok = run(old, new) == 0
        good = ok == expect_ok
        print(("PASS " if good else "FAIL ") + name)
        return good


def edit(f, fn):
    def m(new):
        d = pd.read_csv(new / f); fn(d); d.to_csv(new / f, index=False)
    return m


cases = [
    ("identical outputs are accepted", lambda n: None, True),
    ("missing calibration re-run is rejected", lambda n: (n / "cal_traces_rerun.csv").unlink(), False),
    ("NaN cost is rejected", edit("test_per_unit.csv.gz", lambda d: d.__setitem__("relative_cost", d.relative_cost.where(d.index != d.index[3], np.nan))), False),
    ("changed cost is rejected", edit("test_per_unit.csv.gz", lambda d: d.__setitem__("relative_cost", d.relative_cost.where(d.index != d.index[3], d.relative_cost * 1.01))), False),
    ("changed gamma is rejected", edit("test_per_unit.csv.gz", lambda d: d.__setitem__("gamma", d.gamma.where(d.gamma.isna() | (d.index != d[d.gamma.notna()].index[0]), 0.123))), False),
    ("missing-value mask change is rejected", edit("test_per_unit.csv.gz", lambda d: d.__setitem__("formal_err", d.formal_err.where(d.index != d[d.formal_err.notna()].index[0], np.nan))), False),
    ("duplicated row is rejected", lambda n: pd.concat([pd.read_csv(n / "confirm_per_unit.csv.gz")] * 2).to_csv(n / "confirm_per_unit.csv.gz", index=False), False),
    ("missing row is rejected", edit("confirm_per_unit.csv.gz", lambda d: d.drop(d.index[0], inplace=True)), False),
    ("missing column is rejected", edit("test_per_unit.csv.gz", lambda d: d.drop(columns=["churn"], inplace=True)), False),
    ("changed calibration is rejected", edit("cal_traces_rerun.csv", lambda d: d.__setitem__("mean_cost", d.mean_cost * 1.001)), False),
]
results = [case(*c) for c in cases]
print("ALL COMPARATOR CHECKS PASSED" if all(results) else "COMPARATOR CHECKS FAILED")
sys.exit(0 if all(results) else 1)
