#!/usr/bin/env python3
"""Audit future-looking gap filling in the 200-service replay corpus.

The archived preprocessing used ``ffill``/``bfill`` for replica counts and the
maximum of the adjacent observations for resource columns.  This script does
not mutate that corpus.  It reconstructs a causal forward-fill alternative,
replays the persistence conformal policy, and also scores the archived replay
after excluding (i) imputed test rows and (ii) each imputed row plus the next
W observations in whose rolling history that row can appear.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
EXP = RESEARCH / "experiments"
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from exp_core import load_service_data, simulate_policy  # noqa: E402


DATA = EXP / "data" / "service_timeseries_200"
SPLIT = EXP / "data" / "splits" / "split_definition.json"
SELECTION = EXP / "data" / "splits" / "selected_services_200.csv"
FOCUSED = EXP / "data" / "service_timeseries"
SAVED = EXP / "results" / "large_scale" / "analysis" / "ls_frontier_per_service_200.csv"
OUT = PAPER / "audit" / "statistical_results"
DELTA = 0.05
WINDOW = 240


def causal_forward_fill(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace synthetic rows by past observations only."""
    result = frame.copy()
    missing = result.was_missing.to_numpy(bool)
    # Only these two columns enter the persistence policy and its reference
    # cost. Other archived feature columns may legitimately contain NaNs and
    # must not be mistaken for leading gaps in the evaluated demand signal.
    numeric = ["replica_count", "cpu_sum"]
    result.loc[missing, numeric] = np.nan
    result[numeric] = result[numeric].ffill()
    leading_mask = result[numeric].isna().any(axis=1)
    leading = int(leading_mask.sum())
    # A strictly causal policy cannot borrow the first future observation.
    # Before a service is first observed, use the explicit idle/minimum state.
    result.loc[leading_mask, "cpu_sum"] = 0.0
    result.loc[leading_mask, "replica_count"] = 1.0
    result["_causal_default"] = leading_mask
    if result[numeric].isna().any().any():
        raise AssertionError("causal reconstruction still contains missing values")
    return result, leading


def contamination_mask(missing: np.ndarray, window: int) -> np.ndarray:
    contaminated = np.zeros(len(missing), dtype=bool)
    for index in np.flatnonzero(missing):
        contaminated[index : min(len(missing), index + window + 1)] = True
    return contaminated


def metric_row(overload: np.ndarray, mask: np.ndarray) -> tuple[int, float, bool]:
    selected = overload[mask]
    if not len(selected):
        raise AssertionError("sensitivity mask removed an entire service horizon")
    fraction = float(selected.mean())
    return len(selected), fraction, bool(fraction <= DELTA)


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (subset, scenario), group in long.groupby(["subset", "scenario"], sort=False):
        rows.append(
            {
                "subset": subset,
                "scenario": scenario,
                "services": int(group.service_id.nunique()),
                "total_scored_intervals": int(group.scored_intervals.sum()),
                "filled_test_intervals": int(group.filled_test_intervals.sum()),
                "excluded_intervals": int(group.excluded_intervals.sum()),
                "median_overload_fraction": float(group.overload_fraction.median()),
                "mean_overload_fraction": float(group.overload_fraction.mean()),
                "service_compliance": float(group.within_delta.mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text())
    selected = pd.read_csv(SELECTION)
    strata = selected.set_index("service_id").stratum.to_dict()
    focused_ids = {path.stem for path in FOCUSED.glob("MS_*.parquet")}
    saved = pd.read_csv(SAVED)
    saved = saved[np.isclose(saved.delta, DELTA) & saved.model.eq("persistence")].set_index("service_id")

    per_service: list[dict] = []
    long_rows: list[dict] = []
    total_leading = 0
    total_leading_test = 0
    for service in selected.service_id:
        frame = pd.read_parquet(DATA / f"{service}.parquet").sort_values("timestamp").reset_index(drop=True)
        causal, leading = causal_forward_fill(frame)
        total_leading += leading
        history, test = load_service_data(DATA / f"{service}.parquet", split)

        # Split the in-memory causal reconstruction using the same definition.
        specs = {item["name"]: item for item in split["splits"]}
        history_end = specs["calibration"]["timestamp_end_exclusive"]
        test_spec = specs["test"]
        causal_history = causal[causal.timestamp.lt(history_end)].copy()
        causal_test = causal[
            causal.timestamp.ge(test_spec["timestamp_start"])
            & causal.timestamp.lt(test_spec["timestamp_end_exclusive"])
        ].copy()
        total_leading_test += int(causal_test["_causal_default"].sum())

        archived = simulate_policy(history, test, delta=DELTA, W=WINDOW, use_margin=True, use_guardrail=True)
        causal_replay = simulate_policy(
            causal_history,
            causal_test,
            delta=DELTA,
            W=WINDOW,
            use_margin=True,
            use_guardrail=True,
        )
        if not np.isclose(archived["overload_fraction"], saved.loc[service, "overload_fraction"]):
            raise AssertionError(f"archived replay mismatch for {service}")

        missing = test.was_missing.to_numpy(bool)
        contaminated = contamination_mask(missing, WINDOW)
        masks = {
            "archived_adjacent_fill": np.ones(len(test), dtype=bool),
            "causal_forward_fill": np.ones(len(test), dtype=bool),
            "archived_observed_rows_only": ~missing,
            "archived_outside_fill_plus_W": ~contaminated,
        }
        overloads = {
            "archived_adjacent_fill": np.asarray(archived["overload_arr"], dtype=bool),
            "causal_forward_fill": np.asarray(causal_replay["overload_arr"], dtype=bool),
            "archived_observed_rows_only": np.asarray(archived["overload_arr"], dtype=bool),
            "archived_outside_fill_plus_W": np.asarray(archived["overload_arr"], dtype=bool),
        }
        subset_values = ["all200"] + (["focused20"] if service in focused_ids else [])
        for scenario, mask in masks.items():
            n, fraction, compliant = metric_row(overloads[scenario], mask)
            for subset in subset_values:
                long_rows.append(
                    {
                        "subset": subset,
                        "scenario": scenario,
                        "service_id": service,
                        "stratum": strata[service],
                        "scored_intervals": n,
                        "filled_test_intervals": int(missing.sum()),
                        "excluded_intervals": int(len(mask) - mask.sum()),
                        "overload_fraction": fraction,
                        "within_delta": compliant,
                    }
                )
        per_service.append(
            {
                "service_id": service,
                "stratum": strata[service],
                "focused20": service in focused_ids,
                "filled_test_intervals": int(missing.sum()),
                "fill_plus_W_contaminated_intervals": int(contaminated.sum()),
                "archived_overload_fraction": float(archived["overload_fraction"]),
                "causal_forward_fill_overload_fraction": float(causal_replay["overload_fraction"]),
                "difference_causal_minus_archived": float(
                    causal_replay["overload_fraction"] - archived["overload_fraction"]
                ),
            }
        )

    long = pd.DataFrame(long_rows)
    service_frame = pd.DataFrame(per_service)
    clean = long[
        long.subset.eq("all200")
        & long.scenario.eq("archived_adjacent_fill")
        & long.service_id.isin(service_frame.loc[service_frame.filled_test_intervals.eq(0), "service_id"])
    ].copy()
    clean["scenario"] = "clean_services_only_archived"
    clean["subset"] = "all200"
    long = pd.concat([long, clean], ignore_index=True)
    summary = summarize(long)

    service_frame.to_csv(OUT / "preprocessing_fill_per_service.csv", index=False)
    long.to_csv(OUT / "preprocessing_fill_scenarios.csv", index=False)
    summary.to_csv(OUT / "preprocessing_fill_summary.csv", index=False)
    methodology = {
        "delta": DELTA,
        "window": WINDOW,
        "archived_fill": "capacity columns use ffill then bfill; resource columns use max(ffill,bfill)",
        "causal_sensitivity": "replace synthetic rows by NaN and forward-fill from past observations; before the first observation use zero demand and one capacity unit",
        "exclusion_sensitivity": "exclude each filled test row and, conservatively, that row plus the next W rows",
        "future_value_backfill_used": False,
        "leading_rows_using_causal_default": total_leading,
        "leading_test_rows_using_causal_default": total_leading_test,
        "interpretation": "retrospective robustness analysis; it does not prove exchangeability or production SLA compliance",
    }
    (OUT / "preprocessing_methodology.json").write_text(json.dumps(methodology, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"\nAffected test rows: {service_frame.filled_test_intervals.sum()} across "
          f"{int((service_frame.filled_test_intervals > 0).sum())} services")


def verified_main() -> None:
    """Score causal verified input and exclusions without replaying old inputs."""
    verified = EXP / "data" / "service_timeseries_200_verified"
    out = PAPER / "audit" / "verified_results" / "preprocessing"
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT.read_text())
    selected = pd.read_csv(SELECTION)
    reference = pd.read_csv(PAPER / "audit" / "verified_results" / "comparative" / "margin_comparison_per_service.csv")
    reference = reference[
        reference.subset.eq("all200") & reference.delta.eq(DELTA)
        & reference.margin_family.eq("conformal") & reference.guard.astype(bool)
    ].set_index("service_id")
    old = pd.read_csv(SAVED)
    old = old[old.model.eq("persistence") & old.delta.eq(DELTA)].set_index("service_id")
    long_rows: list[dict] = []
    rows: list[dict] = []
    for index, item in enumerate(selected.itertuples(), 1):
        service = item.service_id
        history, test = load_service_data(verified / f"{service}.parquet", split)
        replay = simulate_policy(history, test, delta=DELTA, W=WINDOW, use_margin=True, use_guardrail=True)
        if not np.isclose(replay["overload_fraction"], reference.loc[service, "overload_fraction"], atol=1e-12):
            raise AssertionError(f"verified replay mismatch for {service}")
        missing = test.was_missing.to_numpy(bool)
        contaminated = contamination_mask(missing, WINDOW)
        overload = np.asarray(replay["overload_arr"], dtype=bool)
        scenarios = {
            "verified_all_rows": np.ones(len(test), dtype=bool),
            "verified_observed_rows_only": ~missing,
            "verified_outside_fill_plus_W": ~contaminated,
        }
        for name, mask in scenarios.items():
            n, fraction, compliant = metric_row(overload, mask)
            for subset in (["all200", "focused20"] if item.is_focused_20 else ["all200"]):
                long_rows.append({
                    "subset": subset, "scenario": name, "service_id": service,
                    "stratum": item.stratum, "scored_intervals": n,
                    "filled_test_intervals": int(missing.sum()),
                    "excluded_intervals": int(len(mask) - mask.sum()),
                    "overload_fraction": fraction, "within_delta": compliant,
                })
        rows.append({
            "service_id": service, "stratum": item.stratum,
            "focused20": bool(item.is_focused_20),
            "filled_test_intervals": int(missing.sum()),
            "fill_plus_W_contaminated_intervals": int(contaminated.sum()),
            "historical_overload_fraction": float(old.loc[service, "overload_fraction"]),
            "verified_overload_fraction": float(replay["overload_fraction"]),
            "difference_verified_minus_historical": float(replay["overload_fraction"] - old.loc[service, "overload_fraction"]),
        })
        if index % 50 == 0:
            print(f"verified preprocessing {index}/200", flush=True)
    long = pd.DataFrame(long_rows)
    by_service = pd.DataFrame(rows)
    clean = long[long.subset.eq("all200") & long.scenario.eq("verified_all_rows")
                 & long.service_id.isin(by_service.loc[by_service.filled_test_intervals.eq(0), "service_id"])].copy()
    clean["scenario"] = "verified_services_without_filled_test_rows"
    long = pd.concat([long, clean], ignore_index=True)
    summary = summarize(long)
    by_service.to_csv(out / "preprocessing_fill_per_service.csv", index=False)
    long.to_csv(out / "preprocessing_fill_scenarios.csv", index=False)
    summary.to_csv(out / "preprocessing_fill_summary.csv", index=False)
    (out / "methodology.json").write_text(json.dumps({
        "input_series": "service_timeseries_200_verified",
        "historical_role": "read-only per-service outcome comparison",
        "exclusions": "filled test rows and, conservatively, each filled row plus the next W scored intervals",
        "window": WINDOW, "delta": DELTA,
        "selection_warning": "exclusion is a retrospective sensitivity, not an alternative test population",
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    verified_main() if "--verified" in sys.argv else main()
