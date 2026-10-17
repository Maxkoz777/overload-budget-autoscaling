#!/usr/bin/env python3
"""Run delay_and_reactive_grid.py (R1-R3) with any multiprocessing start method.

delay_and_reactive_grid.py belongs to the frozen confirmatory code and is kept byte for byte
(verify_confirmatory_inputs.py checks its hash). Its workers read module state that the parent
sets before it creates the pool: the service list, the reactive parameters, the conformal budget,
and the data folders. Only the fork start method copies that state into the workers; spawn (the
default on macOS and Windows) and forkserver do not. This runner hands the parent's state to every
worker when the pool starts and then runs the requested parts as the script's own entry point does.

Usage (from the analysis directory):  python3 audit/run_delay_and_reactive_grid.py [r1] [r2] [r3]
"""
from __future__ import annotations

import json
import multiprocessing
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delay_and_reactive_grid as rev  # noqa: E402


def _state() -> dict:
    return {"services": rev.SERVICES, "params": rev.PARAMS, "data": rev.g.DATA,
            "delta": rev.g.DELTA, "data_root": rev.rb.DATA_ROOT}


def _init_worker(state: dict) -> None:
    rev.SERVICES = state["services"]
    rev.PARAMS = state["params"]
    rev.g.DATA = state["data"]
    rev.g.DELTA = state["delta"]
    rev.rb.DATA_ROOT = state["data_root"]


def _pool(processes: int):
    return multiprocessing.Pool(processes, initializer=_init_worker, initargs=(_state(),))


def main() -> None:
    rev.Pool = _pool
    todo = set(sys.argv[1:]) or {"r1", "r2", "r3"}
    if "r2" in todo:
        rev.run_r2()
    if "r1" in todo:
        rev.run_r1()
    if "r3" in todo:
        rev.run_r3()
    meta = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "parts_run": sorted(todo), "protocol": "verified_results/delay_pac_study/protocol.json"}
    (rev.OUT / f"methodology_{'_'.join(sorted(todo))}.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
