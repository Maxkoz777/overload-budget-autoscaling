"""Data access for the adaptive-baseline study (v1).

Reuses the frozen loaders of the article's repository unchanged:
pac_window_grid.load (three traces) and confirmatory_replay.phase (Huawei periods).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("OBA_REPO", "/home/claude/ws/overload-budget-autoscaling")).resolve()
sys.path.insert(0, str(REPO / "analysis" / "audit"))

try:
    import pyarrow  # noqa: F401
    READER = "pyarrow"
except ImportError:
    from . import pq
    _cols = ["timestamp", "replica_count", "cpu_sum", "was_missing"]
    pd.read_parquet = lambda p, *a, **k: pd.DataFrame(pq.read(str(p), _cols))
    READER = "adaptive/pq.py (pyarrow unavailable)"

from canonical_inputs import install  # noqa: E402
install()
import pac_window_grid as r7  # noqa: E402
import confirmatory_replay as r8  # noqa: E402

TRACES = ("alibaba200", "huawei2023", "azure2019")
PERIODS = r8.RUNS
PAC_W = {("alibaba200", 0.01): 480, ("alibaba200", 0.05): 1440,
         ("huawei2023", 0.01): 2880, ("huawei2023", 0.05): 2880,
         ("azure2019", 0.01): 480, ("azure2019", 0.05): 240}
PAC_W_CONF = {("A_days000_018", 0.01): 1440, ("B_days147_165", 0.01): 1440, ("C_days168_184", 0.01): 2880,
              ("A_days000_018", 0.05): 2880, ("B_days147_165", 0.05): 2880, ("C_days168_184", 0.05): 1440}


def physical_cap(D: dict) -> np.ndarray:
    """Common replica bound M_u = max(1, ceil(2 * max demand before the evaluated period / mu))."""
    return np.maximum(np.ceil(2.0 * D["y"][:, :D["t0"]].max(1) / D["mu"]), 1.0)


def load(name: str, phase: str) -> dict:
    if name in TRACES:
        D = r7.load(name, phase)
    else:
        D = r8.phase(name, phase)
        D.pop("X", None)
    D["name"] = name
    D["M"] = physical_cap(D)
    return D
