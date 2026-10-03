#!/usr/bin/env python3
"""Same-environment ARIMA check for the MS_25320 pre-test fill.

Runs the causal one-step ARIMA replay of ``recompute_arima_causal.run_service``
for MS_25320 on the historical and on the past-only input in the *same*
environment and writes the held-out B6 outcomes to
``audit/verified_results/focused_causal_refit/arima_fill_check.csv``.  The
reported ARIMA replay is not replaced (it was not bit-reproducible across
the two software environments used); this check only isolates the imputation effect.
"""
from __future__ import annotations

import ast
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "audit"))
sys.path.insert(0, str(PAPER.parent / "experiments" / "src"))
import recompute_arima_causal as arima  # noqa: E402

SERVICE = "MS_25320"
OUT = PAPER / "audit/verified_results/focused_causal_refit/arima_fill_check.csv"


def policy(result: dict) -> pd.DataFrame:
    frame = pd.DataFrame(result["policy_rows"])
    frame = frame[frame.warm_start.eq("causal_arima_days_8_9") & frame.evaluation.eq("full_test")]
    return frame.sort_values("delta").reset_index(drop=True)


def main() -> None:
    log = pd.read_csv(arima.LOG_PATH)
    row = log[log.msname.eq(SERVICE)].iloc[0]
    split = json.loads(arima.SPLIT_PATH.read_text())
    task = (SERVICE, int(row.best_k), ast.literal_eval(str(row.arima_order)), split)
    arima.DATA = arima.EXP / "data" / "service_timeseries"
    old = arima.run_service(task)
    arima.DATA = arima.EXP / "data" / "service_timeseries_200_verified"
    new = arima.run_service(task)
    a, b = policy(old), policy(new)
    out = pd.DataFrame({
        "delta": a.delta,
        "overload_historical_input": a.overload_fraction,
        "overload_past_only_input": b.overload_fraction,
        "relative_cost_historical_input": a.relative_cost_vs_observed,
        "relative_cost_past_only_input": b.relative_cost_vs_observed,
        "within_historical_input": a.within_delta,
        "within_past_only_input": b.within_delta,
    })
    out["test_mae_historical_input"] = float(old["diagnostics"]["mae"])
    out["test_mae_past_only_input"] = float(new["diagnostics"]["mae"])
    out["platform"] = platform.platform()
    out.to_csv(OUT, index=False)
    assert (out.within_historical_input == out.within_past_only_input).all()
    assert np.allclose(out.overload_historical_input, out.overload_past_only_input, atol=1e-12, rtol=0)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
