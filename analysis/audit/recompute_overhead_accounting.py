#!/usr/bin/env python3
"""Reconcile focused-tier timing logs with replay-only costs.

The saved policy costs contain no predictor charge.  The optional sensitivity
uses an *unspecified* conversion coefficient kappa (surrogate replica-minutes
per recorded wall-clock minute); it is not a measured cloud-billing result.
Historical ARIMA timing is reported only as timing, never paired with the
corrected causal ARIMA replay.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER.parent / "experiments" / "results"
OUT = PAPER / "audit" / "overhead_results"
KAPPAS = (1.0, 10.0, 100.0)


def csv_text(columns: list[str], rows: list[dict[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def load_timing(model: str) -> pd.DataFrame:
    log = pd.read_csv(RESULTS / f"{model}_training_log.csv")
    assert len(log) == 20 and log.msname.nunique() == 20
    assert (log.test_rows == 4320).all()
    search = "k_selection_seconds" if model == "arima" else "search_seconds"
    for column in (search, "train_seconds", "inference_seconds_total"):
        assert np.isfinite(log[column]).all() and (log[column] >= 0).all()
    np.testing.assert_allclose(
        log.inference_seconds_per_step * log.test_rows,
        log.inference_seconds_total,
        rtol=1e-8,
        atol=1e-6,
    )
    return log.set_index("msname").sort_index()


def make_outputs() -> dict[str, str]:
    for filename in (
        "baseline_replay_per_service.csv",
        "risk_policy_replay_per_service.csv",
        "advanced_policy_replay_per_service.csv",
    ):
        saved = pd.read_csv(RESULTS / filename)
        assert (saved.c_inf == 0).all() and (saved.c_train == 0).all(), filename
        np.testing.assert_allclose(
            saved.c_total, saved.c_res + saved.c_act + saved.c_vio,
            rtol=0,
            atol=1e-6,
            err_msg=filename,
        )

    stage_rows: list[dict[str, object]] = []
    timing: dict[str, pd.DataFrame] = {}
    for model in ("arima", "lstm", "xgb"):
        log = load_timing(model)
        timing[model] = log
        search = "k_selection_seconds" if model == "arima" else "search_seconds"
        stage_rows.append({
            "model": model,
            "protocol": "archived_daily_block_stress" if model == "arima" else "focused_causal_one_step",
            "services": len(log),
            "test_steps_per_service": 4320,
            "search_minutes": float(log[search].sum() / 60),
            "final_fit_minutes": float(log.train_seconds.sum() / 60),
            "test_inference_minutes": float(log.inference_seconds_total.sum() / 60),
        })

    policies = pd.read_csv(RESULTS / "advanced_policy_replay_per_service.csv")
    per_service: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for model in ("lstm", "xgb"):
        replay = policies.loc[
            policies.policy.eq(f"{model}_conformal_guardrail_d0.05")
        ].set_index("msname").sort_index()
        log = timing[model]
        assert len(replay) == 20 and replay.index.is_unique
        assert replay.index.equals(log.index)
        assert (replay.n_intervals == 4320).all()
        assert (replay.c_inf == 0).all() and (replay.c_train == 0).all()
        np.testing.assert_allclose(
            replay.c_total, replay.c_res + replay.c_act + replay.c_vio,
            rtol=0,
            atol=1e-6,
        )
        for service in log.index:
            t = log.loc[service]
            c = float(replay.loc[service, "c_total"])
            assert c > 0
            per_service.append({
                "model": model,
                "service_id": service,
                "delta": 0.05,
                "c_replay_surrogate_replica_min": c,
                "search_wall_min": float(t.search_seconds / 60),
                "fit_wall_min": float(t.train_seconds / 60),
                "inference_wall_min": float(t.inference_seconds_total / 60),
            })
        matched = [r for r in per_service if r["model"] == model]
        for include_search in (False, True):
            for kappa in KAPPAS:
                uplift = np.array([
                    100 * kappa * (
                        r["fit_wall_min"] + r["inference_wall_min"]
                        + (r["search_wall_min"] if include_search else 0)
                    ) / r["c_replay_surrogate_replica_min"]
                    for r in matched
                ])
                summary.append({
                    "model": model,
                    "delta": 0.05,
                    "include_one_search": include_search,
                    "kappa_replica_min_per_wall_min": kappa,
                    "median_uplift_pct": float(np.median(uplift)),
                    "p90_uplift_pct": float(np.quantile(uplift, 0.9)),
                })

    return {
        "stage_totals.csv": csv_text(
            ["model", "protocol", "services", "test_steps_per_service", "search_minutes", "final_fit_minutes", "test_inference_minutes"],
            stage_rows,
        ),
        "per_service.csv": csv_text(
            ["model", "service_id", "delta", "c_replay_surrogate_replica_min", "search_wall_min", "fit_wall_min", "inference_wall_min"],
            per_service,
        ),
        "sensitivity_summary.csv": csv_text(
            ["model", "delta", "include_one_search", "kappa_replica_min_per_wall_min", "median_uplift_pct", "p90_uplift_pct"],
            summary,
        ),
    }


# Check mode.  Stored tables are compared semantically:
#   * exact: file schema (column names and order), row order, key and
#     category columns, discrete counts, and the replay costs that are copied
#     verbatim from the replay CSV;
#   * numerical: the wall-time minutes and uplift percentages that are
#     *computed* here (sums and ratios of logged seconds).  Their last binary
#     digits depend on the floating-point summation/division order of the
#     NumPy/pandas build.  The largest observed cross-environment difference
#     was 7.1e-15 min.  FLOAT_RTOL/FLOAT_ATOL below allow such rounding only;
#     any difference visible at the reported precision (>= 1e-3) fails.
FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-12
EXACT_COLUMNS = {
    "stage_totals.csv": ["model", "protocol", "services", "test_steps_per_service"],
    "per_service.csv": ["model", "service_id", "delta", "c_replay_surrogate_replica_min"],
    "sensitivity_summary.csv": ["model", "delta", "include_one_search", "kappa_replica_min_per_wall_min"],
}
FLOAT_COLUMNS = {
    "stage_totals.csv": ["search_minutes", "final_fit_minutes", "test_inference_minutes"],
    "per_service.csv": ["search_wall_min", "fit_wall_min", "inference_wall_min"],
    "sensitivity_summary.csv": ["median_uplift_pct", "p90_uplift_pct"],
}


def compare_table(name: str, stored_text: str, expected_text: str) -> list[str]:
    """Return a list of human-readable differences (empty when equivalent)."""
    stored = pd.read_csv(io.StringIO(stored_text), dtype=str, keep_default_na=False)
    expected = pd.read_csv(io.StringIO(expected_text), dtype=str, keep_default_na=False)
    problems: list[str] = []
    if list(stored.columns) != list(expected.columns):
        return [f"{name}: columns {list(stored.columns)} != {list(expected.columns)}"]
    if set(expected.columns) != set(EXACT_COLUMNS[name]) | set(FLOAT_COLUMNS[name]):
        return [f"{name}: column classification is incomplete"]
    if len(stored) != len(expected):
        return [f"{name}: {len(stored)} rows != {len(expected)}"]
    for column in EXACT_COLUMNS[name]:
        bad = stored[column] != expected[column]
        if bad.any():
            problems.append(f"{name}: exact column {column} differs in {int(bad.sum())} rows")
    for column in FLOAT_COLUMNS[name]:
        a = stored[column].astype(float).to_numpy()
        b = expected[column].astype(float).to_numpy()
        ok = np.isclose(a, b, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
        if not ok.all():
            worst = float(np.max(np.abs(a - b)))
            problems.append(f"{name}: numeric column {column} differs beyond rounding "
                            f"in {int((~ok).sum())} rows (max abs diff {worst:.3g})")
    return problems


def self_test(outputs: dict[str, str]) -> None:
    """The comparison must accept rounding noise and reject real changes."""
    for name, text in outputs.items():
        assert not compare_table(name, text, text), name
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        column = FLOAT_COLUMNS[name][0]
        noisy = frame.copy()
        noisy[column] = [repr(float(v) * (1 + 4e-16)) for v in frame[column]]
        assert not compare_table(name, noisy.to_csv(index=False, lineterminator="\n"), text), name
        broken = frame.copy()
        broken.loc[0, column] = repr(float(frame.loc[0, column]) * (1 + 1e-6) + 1e-6)
        assert compare_table(name, broken.to_csv(index=False, lineterminator="\n"), text), name
        relabel = frame.copy()
        key = EXACT_COLUMNS[name][0]
        relabel.loc[0, key] = relabel.loc[0, key] + "_x"
        assert compare_table(name, relabel.to_csv(index=False, lineterminator="\n"), text), name


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = make_outputs()
    if args.write:
        OUT.mkdir(exist_ok=True)
        for name, value in outputs.items():
            (OUT / name).write_text(value, encoding="utf-8")
        print(f"wrote {len(outputs)} overhead accounting tables to {OUT}")
        return 0
    self_test(outputs)
    failures: list[str] = []
    for name, value in outputs.items():
        path = OUT / name
        if not path.exists():
            failures.append(f"{path} is missing")
            continue
        failures += compare_table(name, path.read_text(encoding="utf-8"), value)
    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print("PASS: overhead tables match raw timing logs and replay-only costs "
          f"(exact keys/counts/costs; computed minutes within rtol={FLOAT_RTOL:g}, atol={FLOAT_ATOL:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
