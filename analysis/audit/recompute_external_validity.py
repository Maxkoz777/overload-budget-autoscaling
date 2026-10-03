#!/usr/bin/env python3
"""Recompute the evidence used to scope the paper's external-validity claims.

The primary estimand is performance on the fixed 200-service Alibaba suite.
The focused 20 are embedded in that suite; the remaining 180 are therefore
reported separately.  Workload-size quartiles use days 0--9 only and are
revision-defined sensitivity analyses, not an independent replication.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from recompute_guardrail_actuation import load_services, simulate_conformal


PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
EXP = RESEARCH / "experiments"
OUT = PAPER / "audit" / "external_validity_results"
FRONTIER = EXP / "results" / "large_scale" / "analysis" / "ls_frontier_per_service_200.csv"
BASELINES = EXP / "results" / "large_scale" / "analysis" / "ls_baselines_per_service_for_stats_200.csv"
DELTA = 0.05
W = 240
H = 2


def q10(series: pd.Series) -> float:
    return float(series.quantile(0.10))


def q90(series: pd.Series) -> float:
    return float(series.quantile(0.90))


def simulate_mu_sensitivity(history: pd.DataFrame, test: pd.DataFrame, mu: float) -> dict[str, float | bool]:
    """Replay persistence B6 while varying only surrogate capacity mu."""
    history_y = history.cpu_sum.to_numpy(float)
    demand = test.cpu_sum.to_numpy(float)
    previous = float(history_y[-1])
    residuals = np.maximum(np.diff(history_y), 0.0).tolist()
    overload_history: list[bool] = []
    soft_history: list[bool] = []
    capacities: list[int] = []
    triggers: list[bool] = []
    for actual in demand:
        window = np.asarray(residuals[-W:], dtype=float)
        level = min(len(window), int(np.ceil((len(window) + 1) * (1.0 - DELTA))))
        margin = float(np.partition(window, level - 1)[level - 1])
        nominal = max(1, int(np.ceil((previous + margin) / mu)))
        trigger = len(overload_history) >= H and (
            all(overload_history[-H:]) or all(soft_history[-H:])
        )
        capacity = nominal + int(trigger)
        overload = bool(actual > mu * capacity)
        soft = bool(actual > 0.70 * mu * capacity)
        capacities.append(capacity)
        triggers.append(trigger)
        overload_history.append(overload)
        soft_history.append(soft)
        residuals.append(max(float(actual - previous), 0.0))
        previous = float(actual)
    cap = np.asarray(capacities, dtype=int)
    overload = demand > mu * cap
    return {
        "overload_fraction": float(overload.mean()),
        "within_delta": bool(overload.mean() <= DELTA),
        "mean_capacity": float(cap.mean()),
        "fraction_capacity_one": float((cap == 1).mean()),
        "guard_activation": float(np.mean(triggers)),
    }


def summarise_cohort(frame: pd.DataFrame, name: str) -> dict[str, object]:
    return {
        "cohort": name,
        "services": len(frame),
        "median_selection_mean_demand_full13": frame.selection_mean_demand_full13.median(),
        "median_pretest_mean_demand": frame.pretest_mean_demand.median(),
        "p10_pretest_mean_demand": q10(frame.pretest_mean_demand),
        "p90_pretest_mean_demand": q90(frame.pretest_mean_demand),
        "median_test_mean_demand": frame.test_mean_demand.median(),
        "median_test_max_demand": frame.test_max_demand.median(),
        "median_mean_applied_capacity": frame.mean_applied_capacity.median(),
        "services_mean_capacity_below_1_5": int((frame.mean_applied_capacity < 1.5).sum()),
        "median_fraction_capacity_one": frame.fraction_capacity_one.median(),
        "services_always_at_capacity_one": int(frame.always_capacity_one.sum()),
        "services_test_max_demand_at_most_one": int((frame.test_max_demand <= 1.0).sum()),
        "median_guard_activation": frame.guard_activation.median(),
        "p10_guard_activation": q10(frame.guard_activation),
        "p90_guard_activation": q90(frame.guard_activation),
        "b6_service_compliance": frame.within_delta.mean(),
        "median_b6_overload": frame.overload_fraction.median(),
        "median_b6_relative_cost": frame.relative_cost.median(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection, services = load_services()
    selection = selection.copy()
    selection["service_id"] = selection.service_id.astype(str)
    if selection.is_focused_20.dtype != bool:
        selection["is_focused_20"] = selection.is_focused_20.astype(str).str.lower().isin({"true", "1"})
    selection_mean = dict(zip(selection.service_id, selection.mean_demand, strict=True))

    records: list[dict[str, object]] = []
    for service in services:
        history = service["history"]
        calibration = service["calibration"]
        test = service["test"]
        replay = simulate_conformal(history, test, correction="fixed_plus1", rho=0.70)
        cap = np.asarray(replay["capacity"], dtype=int)
        records.append(
            {
                "service_id": service["service_id"],
                "focused20": bool(service["focused"]),
                "new180": not bool(service["focused"]),
                "selection_stratum_full13": service["stratum"],
                "pretest_size_quartile": service["scale_quartile"],
                "selection_mean_demand_full13": float(selection_mean[service["service_id"]]),
                "pretest_mean_demand": float(history.cpu_sum.mean()),
                "pretest_median_demand": float(history.cpu_sum.median()),
                "pretest_max_demand": float(history.cpu_sum.max()),
                "calibration_max_demand": float(calibration.cpu_sum.max()),
                "test_mean_demand": float(test.cpu_sum.mean()),
                "test_max_demand": float(test.cpu_sum.max()),
                "test_fraction_demand_gt_one": float((test.cpu_sum > 1.0).mean()),
                "mean_applied_capacity": float(cap.mean()),
                "fraction_capacity_one": float((cap == 1).mean()),
                "always_capacity_one": bool(np.all(cap == 1)),
                "guard_activation": float(replay["trigger_rate"]),
                "overload_fraction": float(replay["overload_fraction"]),
                "within_delta": bool(replay["within_delta"]),
                "relative_cost": float(replay["relative_cost"]),
            }
        )
    characteristics = pd.DataFrame(records).sort_values("service_id")

    baselines = pd.read_csv(BASELINES)
    b2_name = "B2_pure_predictive" if DATA_VARIANT == "verified" else "B2: pure predictive"
    b2 = baselines[
        baselines.policy.eq(b2_name)
        & np.isclose(baselines.delta.astype(float), DELTA)
    ][["service_id", "overload_fraction"]].rename(columns={"overload_fraction": "b2_test_overload"})
    characteristics = characteristics.merge(b2, on="service_id", how="left", validate="one_to_one")
    characteristics["test_selected_mean_capacity_ge_1_5"] = characteristics.mean_applied_capacity >= 1.5
    characteristics["test_selected_b2_at_risk"] = characteristics.b2_test_overload > DELTA / 2
    characteristics["retrospective_calibration_max_gt_one"] = characteristics.calibration_max_demand > 1.0
    characteristics.to_csv(OUT / "service_characteristics.csv", index=False)

    cohort_rows = [summarise_cohort(characteristics, "all200")]
    cohort_rows.append(summarise_cohort(characteristics[characteristics.focused20], "focused20_embedded"))
    cohort_rows.append(summarise_cohort(characteristics[characteristics.new180], "new180"))
    pd.DataFrame(cohort_rows).to_csv(OUT / "cohort_summary.csv", index=False)

    strata_rows: list[dict[str, object]] = []
    for quartile, frame in characteristics.groupby("pretest_size_quartile", observed=True, sort=True):
        row = summarise_cohort(frame, str(quartile))
        row.update(
            {
                "pretest_mean_demand_min": frame.pretest_mean_demand.min(),
                "pretest_mean_demand_max": frame.pretest_mean_demand.max(),
                "focused20_services": int(frame.focused20.sum()),
                "new180_services": int(frame.new180.sum()),
            }
        )
        strata_rows.append(row)
    pd.DataFrame(strata_rows).to_csv(OUT / "pretest_size_strata_summary.csv", index=False)

    frontier = pd.read_csv(FRONTIER)
    if DATA_VARIANT == "verified":
        frontier = frontier[
            np.isclose(frontier.delta.astype(float), DELTA)
            & frontier.margin_family.eq("conformal")
            & frontier.guard.astype(bool)
            & frontier.subset.eq("all200")
        ].copy()
        frontier["model"] = "persistence"
        frontier["rel_cost"] = frontier.relative_cost
        frontier["guardrail_rate"] = frontier.guard_rate
    else:
        frontier = frontier[
            np.isclose(frontier.delta.astype(float), DELTA)
            & frontier.model.isin(["persistence", "lstm", "xgb"])
        ]
    frontier = frontier.merge(characteristics[["service_id", "focused20", "new180", "pretest_size_quartile"]], on="service_id")
    forecaster_rows: list[dict[str, object]] = []
    for cohort, mask in {
        "all200": np.ones(len(frontier), dtype=bool),
        "focused20_embedded": frontier.focused20,
        "new180": frontier.new180,
    }.items():
        for model, frame in frontier[mask].groupby("model", sort=True):
            forecaster_rows.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "services": frame.service_id.nunique(),
                    "service_compliance": frame.within_delta.mean(),
                    "median_overload": frame.overload_fraction.median(),
                    "median_relative_cost": frame.rel_cost.median(),
                    "median_guard_activation": frame.guardrail_rate.median(),
                }
            )
    pd.DataFrame(forecaster_rows).to_csv(OUT / "cohort_forecaster_summary.csv", index=False)

    size_forecaster_rows: list[dict[str, object]] = []
    for (quartile, model), frame in frontier.groupby(["pretest_size_quartile", "model"], sort=True):
        size_forecaster_rows.append(
            {
                "pretest_size_quartile": quartile,
                "model": model,
                "services": frame.service_id.nunique(),
                "service_compliance": frame.within_delta.mean(),
                "median_overload": frame.overload_fraction.median(),
                "median_relative_cost": frame.rel_cost.median(),
            }
        )
    pd.DataFrame(size_forecaster_rows).to_csv(OUT / "pretest_size_forecaster_summary.csv", index=False)

    mu_rows: list[dict[str, object]] = []
    for service in services:
        for mu in (0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
            result = simulate_mu_sensitivity(service["history"], service["test"], mu)
            mu_rows.append(
                {
                    "service_id": service["service_id"],
                    "focused20": bool(service["focused"]),
                    "new180": not bool(service["focused"]),
                    "mu": mu,
                    **result,
                }
            )
    mu_per_service = pd.DataFrame(mu_rows)
    mu_per_service.to_csv(OUT / "mu_sensitivity_per_service.csv", index=False)
    mu_summary_rows: list[dict[str, object]] = []
    for cohort, mask in {
        "all200": np.ones(len(mu_per_service), dtype=bool),
        "focused20_embedded": mu_per_service.focused20,
        "new180": mu_per_service.new180,
    }.items():
        for mu, frame in mu_per_service[mask].groupby("mu", sort=True):
            mu_summary_rows.append(
                {
                    "cohort": cohort,
                    "mu": mu,
                    "services": frame.service_id.nunique(),
                    "service_compliance": frame.within_delta.mean(),
                    "median_overload": frame.overload_fraction.median(),
                    "median_mean_capacity": frame.mean_capacity.median(),
                    "median_fraction_capacity_one": frame.fraction_capacity_one.median(),
                    "services_always_at_capacity_one": int((frame.fraction_capacity_one == 1.0).sum()),
                    "median_guard_activation": frame.guard_activation.median(),
                }
            )
    pd.DataFrame(mu_summary_rows).to_csv(OUT / "mu_sensitivity_summary.csv", index=False)

    provenance = pd.DataFrame(
        [
            ("all200", 200, "40 per full-13-day burstiness stratum; focused 20 force-included", "primary finite-suite evaluation"),
            ("focused20", int(characteristics.focused20.sum()), "selected earlier for mechanism-level analysis; all are in all200", "embedded diagnostic cohort"),
            ("new180", int(characteristics.new180.sum()), "random fill within full-13-day burstiness strata, seed 42", "separate finite-suite sensitivity"),
            ("mean_capacity_ge_1_5", int(characteristics.test_selected_mean_capacity_ge_1_5.sum()), "B6 mean applied capacity on held-out days 10-12 >= 1.5", "test-informed exploratory subset; no headline inference"),
            ("b2_at_risk", int(characteristics.test_selected_b2_at_risk.sum()), "B2 held-out overload exceeds delta/2=0.025", "test-informed exploratory subset; no headline inference"),
            ("calibration_max_gt_one", int(characteristics.retrospective_calibration_max_gt_one.sum()), "maximum demand on calibration days 8-9 exceeds one", "revision-defined retrospective sensitivity"),
        ],
        columns=["subset", "services", "definition", "permitted_role"],
    )
    provenance.to_csv(OUT / "subset_provenance.csv", index=False)

    methodology = {
        "input_series": DATA_VARIANT,
        "forecaster_scope": "persistence_only" if DATA_VARIANT == "verified" else "historical_including_unauthenticated_learned",
        "primary_estimand": "performance on the fixed 200-service Alibaba suite",
        "independent_external_replication": False,
        "focused20_relation": "force-included in all200",
        "new180_relation": "all200 excluding focused20",
        "selection_strata": "burstiness computed over all 13 trace days, including the held-out horizon",
        "selection_strata_role": "benchmark descriptors only; not prospective or test-independent strata",
        "pretest_size_strata": "quartiles of mean cpu_sum over days 0-9, rank(method=first), 50 services each",
        "pretest_size_strata_role": "revision-defined sensitivity added after review",
        "demand_semantics": "sum of Alibaba min-max-normalised per-container CPU values",
        "capacity_semantics": "mu=1 is a surrogate replay unit, not measured physical CPU-request capacity",
        "mu_sensitivity": "fixed demand proxy with mu in {0.50,0.75,1.00,1.25,1.50,2.00}; this varies a modelling assumption rather than changing units",
        "outcome_semantics": "numerical demand-capacity threshold exceedance; latency and SLO outcomes are not observed",
        "random_fill_seed": 42,
        "delta": DELTA,
    }
    (OUT / "methodology.json").write_text(json.dumps(methodology, indent=2) + "\n")

    print(pd.DataFrame(cohort_rows).to_string(index=False))
    print("\nSubset provenance")
    print(provenance.to_string(index=False))


if __name__ == "__main__":
    import sys
    DATA_VARIANT = "verified" if "--verified" in sys.argv else "historical"
    if DATA_VARIANT == "verified":
        import recompute_guardrail_actuation as guard
        guard.DATA = EXP / "data" / "service_timeseries_200_verified"
        OUT = PAPER / "audit" / "verified_results" / "external_validity"
        FRONTIER = PAPER / "audit" / "verified_results" / "comparative" / "margin_comparison_per_service.csv"
        BASELINES = PAPER / "audit" / "verified_results" / "baselines" / "persistence_baselines_per_service.csv"
    main()
