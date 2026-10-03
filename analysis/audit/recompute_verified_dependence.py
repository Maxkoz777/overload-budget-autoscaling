#!/usr/bin/env python3
"""Dependence-aware finite-suite uncertainty from verified persistence replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
EXP = PAPER.parent / "experiments"
sys.path.insert(0, str(PAPER / "audit"))

import recompute_margin_comparison as margin  # noqa: E402
import recompute_reactive_baseline as reactive  # noqa: E402
import statistical_robustness as stats  # noqa: E402

DATA = EXP / "data" / "service_timeseries_200_verified"
OUT = PAPER / "audit" / "verified_results" / "statistical"
BLOCKS = (30, 60, 120, 240, 480, 1440)
SEED = 20260920


def main(replicates: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    margin.DATA_ROOT = DATA
    selection = pd.read_csv(EXP / "data" / "splits" / "selected_services_200.csv")
    split = json.loads((EXP / "data" / "splits" / "split_definition.json").read_text())
    params = pd.read_csv(PAPER / "audit" / "verified_results" / "comparative" / "reactive_selected_parameters.csv").set_index("service_id")
    saved = pd.read_csv(PAPER / "audit" / "verified_results" / "comparative" / "margin_comparison_per_service.csv")
    saved = saved[saved.subset.eq("all200") & saved.guard.astype(bool)]
    selected_reactive = pd.read_csv(PAPER / "audit" / "verified_results" / "comparative" / "reactive_test_per_service.csv").set_index("service_id")
    arrays: dict[str, list[np.ndarray]] = {"conformal_005": [], "gaussian_005": [],
                                            "conformal_001": [], "gaussian_001": [], "reactive_005": []}
    ids: list[str] = []
    for index, item in enumerate(selection.itertuples(), 1):
        sid = item.service_id
        history, test = margin.load_service(sid, split)
        ids.append(sid)
        for delta, suffix in ((0.05, "005"), (0.01, "001")):
            conf, gauss = stats.conformal_gaussian_overload_pair(history, test, delta)
            arrays[f"conformal_{suffix}"].append(conf)
            arrays[f"gaussian_{suffix}"].append(gauss)
            for name, values in (("conformal", conf), ("gaussian", gauss)):
                row = saved[saved.service_id.eq(sid) & saved.delta.eq(delta) & saved.margin_family.eq(name)]
                if len(row) != 1 or not np.isclose(values.mean(), row.overload_fraction.iloc[0], atol=1e-12):
                    raise AssertionError(f"verified margin array mismatch: {sid}/{delta}/{name}")
        capacity = reactive.reactive_capacity(history, test, threshold=float(params.loc[sid, "threshold"]),
                                              cooldown=int(params.loc[sid, "cooldown"]))
        react = test.cpu_sum.to_numpy(float) > capacity
        if not np.isclose(react.mean(), selected_reactive.loc[sid, "overload_fraction"], atol=1e-12):
            raise AssertionError(f"verified reactive array mismatch: {sid}")
        arrays["reactive_005"].append(react)
        if index % 25 == 0:
            print(f"verified dependence replay {index}/200", flush=True)
    strata = selection.set_index("service_id").loc[ids].stratum.to_numpy(str)
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for policy in ("conformal_005", "gaussian_005", "conformal_001", "gaussian_001", "reactive_005"):
        delta = 0.01 if policy.endswith("001") else 0.05
        point = np.asarray([x.mean() for x in arrays[policy]])
        for block in BLOCKS:
            for scheme, sync in (("independent_within_service", False), ("synchronised_across_services", True)):
                temporal = stats.temporal_fraction_matrix(arrays[policy], block, replicates, rng, sync)
                temporal_only = np.mean(temporal <= delta, axis=0)
                sampled = temporal[stats.stratified_indices(strata, replicates, rng), np.arange(replicates)[:, None]]
                hierarchical = np.mean(sampled <= delta, axis=1)
                mean_overload = np.mean(sampled, axis=1)
                t_lo, t_hi = stats.quantile_interval(temporal_only)
                h_lo, h_hi = stats.quantile_interval(hierarchical)
                m_lo, m_hi = stats.quantile_interval(mean_overload)
                rows.append({"policy": policy, "delta": delta, "resampling_scheme": scheme,
                             "block_minutes": block, "services": len(ids), "test_steps": 4320,
                             "point_compliance": float(np.mean(point <= delta)),
                             "temporal_only_ci95_lo": t_lo, "temporal_only_ci95_hi": t_hi,
                             "hierarchical_ci95_lo": h_lo, "hierarchical_ci95_hi": h_hi,
                             "point_mean_overload": float(point.mean()),
                             "hierarchical_mean_overload_ci95_lo": m_lo,
                             "hierarchical_mean_overload_ci95_hi": m_hi,
                             "bootstrap_replicates": replicates})
    pd.DataFrame(rows).to_csv(OUT / "temporal_block_bootstrap.csv", index=False)
    contrasts = (("conformal_005", "gaussian_005"), ("conformal_001", "gaussian_001"),
                 ("conformal_005", "reactive_005"))
    pair_rows: list[dict] = []
    for left, right in contrasts:
        point = float(np.mean([a.mean() - b.mean() for a, b in zip(arrays[left], arrays[right])]))
        for block in BLOCKS:
            for scheme, sync in (("independent_within_service", False), ("synchronised_across_services", True)):
                draw = stats.paired_temporal_mean_difference(arrays[left], arrays[right], strata,
                                                             block, replicates, rng, sync)
                lo, hi = stats.quantile_interval(draw)
                pair_rows.append({"policy_a": left, "policy_b": right,
                                  "delta": 0.01 if left.endswith("001") else 0.05,
                                  "resampling_scheme": scheme, "block_minutes": block,
                                  "services": len(ids), "point_mean_overload_difference": point,
                                  "bootstrap_ci95_lo": lo, "bootstrap_ci95_hi": hi,
                                  "bootstrap_replicates": replicates})
    pd.DataFrame(pair_rows).to_csv(OUT / "paired_temporal_bootstrap.csv", index=False)
    service_rows = []
    for policy, values in arrays.items():
        delta = 0.01 if policy.endswith("001") else 0.05
        successes = np.asarray([x.mean() <= delta for x in values])
        lo, hi = stats.wilson_interval(int(successes.sum()), len(successes))
        service_rows.append({"policy": policy, "delta": delta, "within_budget": int(successes.sum()),
                             "services": len(successes), "compliance": float(successes.mean()),
                             "wilson95_lo": lo, "wilson95_hi": hi})
    pd.DataFrame(service_rows).to_csv(OUT / "service_compliance_intervals.csv", index=False)
    (OUT / "methodology.json").write_text(json.dumps({
        "input_series": "service_timeseries_200_verified", "forecaster": "persistence_only",
        "seed": SEED, "replicates": replicates, "block_minutes": BLOCKS,
        "resampling": "non-circular moving blocks within services, then five-stratum service resampling; independent and synchronised clock schemes",
        "paired_policy_arrays": "same block and service draws for both arms; no independent-minute assumption",
        "scope": "fixed-suite retrospective sensitivity, not exchangeability proof or population SLA interval",
    }, indent=2) + "\n")
    print(pd.DataFrame(service_rows).to_string(index=False))
    print(pd.DataFrame(pair_rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=2000)
    main(parser.parse_args().replicates)
