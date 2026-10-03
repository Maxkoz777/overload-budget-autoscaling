#!/usr/bin/env python3
"""Exact envelope.

The protocol-fixed envelope in envelope.csv samples 30 equally spaced caps.
Drawing those samples as a step line misstates the family minimum between
caps.  This script evaluates the same definition exactly, at every
fleet-uniform configuration's own resource value (all families of a cohort),
from the unchanged summary.csv.  No replay, no new configuration, no
interpolation; envelope.csv and protocol.json are left unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
EXT = PAPER / "audit/verified_results/final_extension_2026-09-29"
FAMILIES = ("conformal", "gaussian", "reactive", "pure_predictive")


def exact_envelope(summary: pd.DataFrame) -> pd.DataFrame:
    fu = summary[summary.point_type.eq("fleet_uniform")]
    rows = []
    for (cohort, budget), part in fu.groupby(["cohort", "evaluation_budget"], sort=True):
        caps = np.unique(part.mean_norm_resource.to_numpy())
        for fam in FAMILIES:
            f = part[part.family.eq(fam)]
            for cap in caps:
                ok = f[f.mean_norm_resource <= cap]
                if ok.empty:
                    rows.append({"cohort": cohort, "evaluation_budget": budget, "family": fam, "resource_cap": cap,
                                 "available": False, "best_mean_overload": np.nan, "best_mean_overload_config": "",
                                 "best_mean_overload_resource": np.nan, "best_compliance": np.nan})
                    continue
                best = ok.sort_values(["mean_overload", "mean_norm_resource", "config_id"]).iloc[0]
                rows.append({"cohort": cohort, "evaluation_budget": budget, "family": fam, "resource_cap": cap,
                             "available": True, "best_mean_overload": best.mean_overload,
                             "best_mean_overload_config": best.config_id,
                             "best_mean_overload_resource": best.mean_norm_resource,
                             "best_compliance": ok.compliance.max()})
    return pd.DataFrame(rows)


def comparisons(env: pd.DataFrame) -> pd.DataFrame:
    """Where do family envelopes cross, on the common reachable region?"""
    out = []
    e = env[env.evaluation_budget.eq(0.01)]  # mean overload does not depend on the budget
    for cohort, part in e.groupby("cohort"):
        wide = part.pivot(index="resource_cap", columns="family", values="best_mean_overload")
        for a, b in (("conformal", "reactive"), ("gaussian", "reactive"), ("conformal", "gaussian")):
            both = wide[[a, b]].dropna()
            out.append({"cohort": cohort, "family_a": a, "family_b": b, "shared_caps": len(both),
                        "shared_from": float(both.index.min()), "shared_to": float(both.index.max()),
                        "caps_a_lower": int((both[a] < both[b]).sum()), "caps_b_lower": int((both[b] < both[a]).sum()),
                        "caps_equal": int((both[a] == both[b]).sum()),
                        "max_a_minus_b_pp": float(100 * (both[a] - both[b]).max()),
                        "min_a_minus_b_pp": float(100 * (both[a] - both[b]).min())})
    return pd.DataFrame(out)


def main() -> None:
    summary = pd.read_csv(EXT / "summary.csv")
    env = exact_envelope(summary)
    # property: at each configuration's own resource, its family line is not above it,
    # and equals the direct minimum over admissible rows
    fu = summary[summary.point_type.eq("fleet_uniform")]
    ok = True
    for r in fu.itertuples(index=False):
        line = env[(env.cohort == r.cohort) & (env.evaluation_budget == r.evaluation_budget)
                   & (env.family == r.family) & np.isclose(env.resource_cap, r.mean_norm_resource, rtol=0, atol=0)]
        direct = fu[(fu.cohort == r.cohort) & (fu.evaluation_budget == r.evaluation_budget) & (fu.family == r.family)
                    & (fu.mean_norm_resource <= r.mean_norm_resource)].mean_overload.min()
        ok &= len(line) == 1 and line.best_mean_overload.iloc[0] <= r.mean_overload and line.best_mean_overload.iloc[0] == direct
    assert ok
    env.to_csv(EXT / "envelope_exact.csv", index=False)
    comp = comparisons(env)
    comp.to_csv(EXT / "envelope_exact_comparisons.csv", index=False)
    (EXT / "envelope_exact_methodology.json").write_text(json.dumps({
        "reason": "independent review 2026-09-29, item 1: step line through 30 sampled caps misstated the family minimum between caps",
        "definition": "same as protocol envelope; caps = every unique fleet-uniform mean_norm_resource in the cohort (all families)",
        "source": "summary.csv (unchanged)", "unchanged_files": ["protocol.json", "envelope.csv"],
        "property_check": "line at each configuration's resource <= its overload and equals the direct admissible minimum",
        "property_check_passed": bool(ok), "status": "dated post-review addendum, descriptive and test-selected"}, indent=2) + "\n")
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
