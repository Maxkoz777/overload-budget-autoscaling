#!/usr/bin/env python3
"""Rebuild the bounded 1% comparison from independently checked replay rows.

W=240 is the original policy setting. W=1440, Gaussian z=4, and reactive
calibration at 0.25% are retrospective sensitivities, never test-selected
confirmatory policies. The focused 20 are nested in the fixed all-200 suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "review/strict_budget_crosscheck/independent_replay.csv"
RANK = ROOT / "audit/verified_results/rank_ablation/rank_ablation_per_service.csv"
CAL = ROOT / "audit/verified_results/comparative/reactive_calibration_grid_per_service.csv"
TEST = ROOT / "audit/verified_results/comparative/reactive_test_grid_per_service.csv"
META = ROOT / "audit/verified_results/external_validity/service_characteristics.csv"
INDEPENDENT_REACTIVE = ROOT / "review/strict_budget_crosscheck/reactive_reselection.csv"
OUT = ROOT / "audit/verified_results/strict_budget"
KEY = ["service_id", "policy", "W", "guard", "calibration_budget"]


def choose_reactive(calibration: pd.DataFrame, test: pd.DataFrame, budget: float) -> pd.DataFrame:
    rows = []
    for service_id, group in calibration.groupby("service_id", sort=True):
        feasible = group[group.overload_fraction.le(budget)]
        if len(feasible):
            ranked = feasible.sort_values(
                ["relative_cost", "overload_fraction", "threshold", "cooldown"],
                ascending=[True, True, False, True],
            )
        else:
            ranked = group.sort_values(
                ["overload_fraction", "relative_cost", "threshold", "cooldown"],
                ascending=[True, True, False, True],
            )
        selected = ranked.iloc[0]
        rows.append({"service_id": service_id, "threshold": selected.threshold,
                     "cooldown": selected.cooldown, "calibration_feasible": bool(len(feasible))})
    selected = pd.DataFrame(rows)
    result = selected.merge(test, on=["service_id", "threshold", "cooldown"], validate="one_to_one")
    assert len(result) == 200 and result.service_id.is_unique
    return result


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, subset in [("all200", frame), ("focused20", frame[frame.focused]),
                           *[(f"Q{i}", frame[frame.pretest_size_quartile.eq(f"Q{i}")]) for i in range(1, 5)]]:
        for (policy, window, guard, cal_budget), group in subset.groupby(
            ["policy", "W", "guard", "calibration_budget"], dropna=False, sort=True
        ):
            rows.append({"cohort": cohort, "policy": policy, "W": window, "guard": guard,
                         "calibration_budget": cal_budget, "n": len(group),
                         "compliant": int(group.within_budget.sum()),
                         "mean_realised_overload": float(group.overload_fraction.mean()),
                         "median_realised_overload": float(group.overload_fraction.median()),
                         "median_relative_cost": float(group.relative_cost.median()),
                         "mean_relative_cost": float(group.relative_cost.mean())})
    return pd.DataFrame(rows)


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policies = [
        ("conformal", 240, True, np.nan),
        ("conformal", 1440, False, np.nan),
        ("gaussian", 240, True, np.nan),
        ("inverse_cdf", 240, True, np.nan),
        ("reactive", 0, False, .0025),
    ]
    for cohort, subset in [("all200", frame), ("focused20", frame[frame.focused]),
                           *[(f"Q{i}", frame[frame.pretest_size_quartile.eq(f"Q{i}")]) for i in range(1, 5)]]:
        baseline = subset[subset.policy.eq("reactive") & subset.calibration_budget.eq(.01)]
        assert len(baseline) == len(subset.service_id.unique())
        for policy, window, guard, cal_budget in policies:
            candidate = subset[subset.policy.eq(policy) & subset.W.eq(window) & subset.guard.eq(guard)]
            if policy == "reactive":
                candidate = candidate[candidate.calibration_budget.eq(cal_budget)]
            merged = candidate.merge(baseline, on="service_id", validate="one_to_one", suffixes=("_candidate", "_reactive"))
            assert len(merged) == len(baseline)
            wins = int((merged.within_budget_candidate & ~merged.within_budget_reactive).sum())
            losses = int((~merged.within_budget_candidate & merged.within_budget_reactive).sum())
            rows.append({"cohort": cohort, "candidate": policy, "W": window, "guard": guard,
                         "candidate_calibration_budget": cal_budget, "n": len(merged),
                         "candidate_only_compliant": wins, "reactive_only_compliant": losses,
                         "exact_paired_compliance_p_unadjusted":
                         float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0,
                         "mean_overload_difference_candidate_minus_reactive":
                         float((merged.overload_fraction_candidate - merged.overload_fraction_reactive).mean()),
                         "mean_relative_cost_difference_candidate_minus_reactive":
                         float((merged.relative_cost_candidate - merged.relative_cost_reactive).mean())})
    return pd.DataFrame(rows)


def main() -> None:
    replay = pd.read_csv(REPLAY)
    replay = replay[replay.delta.eq(.01)].copy()
    replay = replay.rename(columns={"ol": "overload_fraction", "rc": "relative_cost"})
    assert replay.service_id.nunique() == 200
    assert not replay.duplicated(["service_id", "W", "method", "guard"]).any()
    rank = pd.read_csv(RANK)
    rank = rank[rank.delta.eq(.01) & rank.method.isin(["conformal", "gaussian", "empirical_inverse_cdf"])]
    rank["method"] = rank.method.replace({"empirical_inverse_cdf": "invcdf"})
    original = replay[replay.W.eq(240) & replay.method.isin(["conformal", "gaussian", "invcdf"])].merge(
        rank, on=["service_id", "method", "guard"], validate="one_to_one", suffixes=("_independent", "_canonical")
    )
    assert len(original) == 1200
    for field in ("overload_fraction", "relative_cost"):
        assert np.allclose(original[f"{field}_independent"], original[f"{field}_canonical"], rtol=0, atol=1e-12)
    assert replay[replay.W.eq(1440) & replay.method.eq("conformal")].shape[0] == 400

    meta = pd.read_csv(META)[["service_id", "pretest_size_quartile", "focused20"]]
    assert len(meta) == 200 and meta.service_id.is_unique
    assert meta.pretest_size_quartile.value_counts().to_dict() == {f"Q{i}": 50 for i in range(1, 5)}
    calibration = pd.read_csv(CAL)
    test = pd.read_csv(TEST)
    independent_reactive = pd.read_csv(INDEPENDENT_REACTIVE)
    frames = []
    for budget in (.01, .0025):
        selected = choose_reactive(calibration, test, budget)
        assert int((~selected.calibration_feasible).sum()) == (3 if budget == .01 else 9)
        for cohort, group in (("all200", selected), ("focused20", selected[selected.is_focused_20])):
            expected = independent_reactive[
                independent_reactive.cohort.eq(cohort) & independent_reactive.calibration_budget.eq(budget)
            ].iloc[0]
            assert int((group.overload_fraction.le(.01)).sum()) == int(expected.compliant_at_01)
            assert np.isclose(group.overload_fraction.mean(), expected.mean_ol, atol=1e-12)
            assert np.isclose(group.relative_cost.mean(), expected.mean_rc, atol=1e-12)
        selected["policy"] = "reactive"
        selected["W"] = 0
        selected["guard"] = False
        selected["calibration_budget"] = budget
        frames.append(selected[["service_id", "policy", "W", "guard", "calibration_budget",
                                "overload_fraction", "relative_cost"]])

    replay["policy"] = replay.method.replace({"invcdf": "inverse_cdf", "gaussian_z4": "gaussian_z4"})
    replay["calibration_budget"] = np.nan
    frames.insert(0, replay[["service_id", "policy", "W", "guard", "calibration_budget",
                             "overload_fraction", "relative_cost"]])
    data = pd.concat(frames, ignore_index=True).merge(meta, on="service_id", validate="many_to_one")
    data = data.rename(columns={"focused20": "focused"})
    data["within_budget"] = data.overload_fraction.le(.01)
    assert not data.duplicated(KEY).any()
    summary = summarise(data)
    pairs = paired(data)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "strict_budget_per_service.csv", index=False)
    summary.to_csv(OUT / "strict_budget_summary.csv", index=False)
    pairs.to_csv(OUT / "strict_budget_paired.csv", index=False)
    (OUT / "protocol.json").write_text(json.dumps({
        "budget_evaluated": .01,
        "original_window": 240,
        "retrospective_window_sensitivity": [1440],
        "retrospective_gaussian_z": [4],
        "retrospective_reactive_calibration_budget": [.0025],
        "reactive_original_calibration_budget": .01,
        "reactive_selection": "per-service calibration-feasible minimum cost, then overload, decreasing threshold and increasing cooldown; if none feasible minimum overload, then cost, decreasing threshold and increasing cooldown",
        "cohorts": "all 200 fixed services, nested focused 20, and four pre-test-size quartiles of 50 services each",
        "overload": "mean realised overload across services, not theorem-level expected overload",
        "inference": "paired exact p-values are unadjusted retrospective descriptives, not confirmatory tests after sensitivity selection",
        "source": [str(REPLAY.relative_to(ROOT)), str(RANK.relative_to(ROOT)), str(CAL.relative_to(ROOT)),
                   str(TEST.relative_to(ROOT)), str(META.relative_to(ROOT))],
    }, indent=2) + "\n")
    print(summary[summary.cohort.isin(["all200", "focused20"])].to_string(index=False))
    print("strict-budget replay/canonical and reactive independent cross-checks passed")


if __name__ == "__main__":
    main()
