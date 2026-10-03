#!/usr/bin/env python3
"""Describe frozen 1% replay outcomes; no fitting, retuning, or new replay.

Q3-Q4 is an exploratory subset defined by pre-test demand. The finite-grid
cost-cap envelope is selected retrospectively on test summaries, never a
deployable comparator. All costs are means of per-service relative costs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "audit/verified_results/strict_budget/strict_budget_per_service.csv"
OUT = ROOT / "audit/verified_results/budget_utilisation"
KEY = ["policy", "W", "guard", "calibration_budget"]


def main():
    data = pd.read_csv(SOURCE)
    assert not data.duplicated(["service_id", *KEY]).any()
    assert np.isfinite(data[["overload_fraction", "relative_cost"]]).all().all()
    assert data.within_budget.eq(data.overload_fraction.le(.01)).all()
    rows = []
    for cohort, subset in (("all200", data), ("focused20", data[data.focused]),
                           ("Q3-Q4", data[data.pretest_size_quartile.isin(["Q3", "Q4"])])):
        expected = {"all200": 200, "focused20": 20, "Q3-Q4": 100}[cohort]
        for key, group in subset.groupby(KEY, dropna=False, sort=True):
            assert len(group) == expected and group.service_id.is_unique
            ratio = group.overload_fraction / .01
            rows.append(dict(cohort=cohort, **dict(zip(KEY, key)), n=len(group),
                             compliant=int(group.within_budget.sum()),
                             mean_relative_cost=group.relative_cost.mean(),
                             mean_budget_utilisation=ratio.mean(),
                             p10_budget_utilisation=ratio.quantile(.1),
                             median_budget_utilisation=ratio.median(),
                             p90_budget_utilisation=ratio.quantile(.9),
                             mean_excess_ratio=np.maximum(ratio-1, 0).mean(),
                             mean_unused_ratio=np.maximum(1-ratio, 0).mean()))
    summary = pd.DataFrame(rows)
    envelopes = []
    for cohort, group in summary.groupby("cohort"):
        ref = group[group.policy.eq("conformal") & group.W.eq(240) & group.guard].iloc[0]
        for fraction in (1., 1.0025):
            cap = ref.mean_relative_cost * fraction
            feasible = group[group.mean_relative_cost.le(cap + 1e-12)]
            for objective, order, asc in (
                ("max_compliant", ["compliant", "mean_relative_cost"], [False, True]),
                ("min_mean_overload", ["mean_budget_utilisation", "mean_relative_cost"], [True, True]),
            ):
                best = feasible.sort_values(order, ascending=asc).iloc[0]
                envelopes.append(dict(cohort=cohort, cost_cap=cap,
                                      cap_relative_to_reference=fraction, objective=objective,
                                      **{k:best[k] for k in KEY}, compliant=int(best.compliant),
                                      mean_relative_cost=best.mean_relative_cost,
                                      mean_budget_utilisation=best.mean_budget_utilisation))
    OUT.mkdir(exist_ok=True, parents=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(envelopes).to_csv(OUT / "cost_cap_envelope.csv", index=False)
    (OUT / "methodology.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "delta": .01,
        "ratio": "realised per-service overload / delta, not an estimator of a marginal coverage error",
        "cost": "mean of per-service costs normalised to each service's observed-capacity replay cost",
        "quantiles": "pandas linear interpolation",
        "selection": "descriptive test-selected envelope of 18 existing configurations; no interpolation or randomisation",
        "subset": "Q3-Q4 uses pre-test mean demand, chosen for this retrospective description",
        "cost_caps": "reference conformal W=240+guard mean cost and 1.0025 times that mean",
    }, indent=2) + "\n")
    print(summary.query("cohort == 'all200'").to_string(index=False))
    print(pd.DataFrame(envelopes).to_string(index=False))


if __name__ == "__main__":
    main()
