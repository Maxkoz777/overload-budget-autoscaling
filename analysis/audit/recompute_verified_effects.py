#!/usr/bin/env python3
"""Matched non-learned policy effects on verified service series."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER / "audit"))
import recompute_margin_comparison as margin  # noqa: E402

BASE = PAPER / "audit" / "verified_results" / "baselines"
OUT = PAPER / "audit" / "verified_results" / "statistical"
COMPARATORS = (
    "B2_pure_predictive", "B8_guard_only", "B7_conformal_only",
    "B4_gaussian_guard", "B9_linear_percentile_guard", "B3_fixed_margin",
)


def main() -> None:
    frame = pd.read_csv(BASE / "persistence_baselines_per_service.csv")
    frame = frame[frame.delta.eq(0.05)].copy()
    baseline = frame[frame.policy.eq("B6_conformal_guard")].set_index("service_id")
    rows = []
    for index, name in enumerate(COMPARATORS):
        other = frame[frame.policy.eq(name)].set_index("service_id")
        joined = baseline[["stratum", "overload_fraction", "relative_cost"]].join(
            other[["overload_fraction", "relative_cost"]], lsuffix="_b6", rsuffix="_other", validate="one_to_one"
        )
        if len(joined) != 200 or joined.isna().any().any():
            raise AssertionError(f"incomplete matched comparison: {name}")
        for metric in ("overload_fraction", "relative_cost"):
            a = joined[f"{metric}_b6"]
            b = joined[f"{metric}_other"]
            boot = joined[["stratum"]].copy()
            boot["a"] = a
            boot["b"] = b
            point, lo, hi, family_lo, family_hi = margin.bootstrap_mean_difference(
                boot, "a", "b", alpha=0.05 / len(COMPARATORS), seed=20260920 + 10 * index + int(metric == "relative_cost")
            )
            difference = a - b
            p = 1.0 if np.allclose(difference, 0) else float(wilcoxon(difference).pvalue)
            rows.append({"comparison": f"B6_minus_{name}", "metric": metric,
                         "services": len(joined), "mean_difference": point,
                         "pointwise_ci95_lo": lo, "pointwise_ci95_hi": hi,
                         "familywise_ci_lo": family_lo, "familywise_ci_hi": family_hi,
                         "wilcoxon_two_sided_p": p,
                         "family_size": len(COMPARATORS), "bonferroni_alpha": 0.05 / len(COMPARATORS)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "paired_policy_effects.csv", index=False)
    (OUT / "paired_policy_methodology.json").write_text(json.dumps({
        "input": "verified_results/baselines/persistence_baselines_per_service.csv",
        "delta": 0.05, "comparators": COMPARATORS, "paired_service_bootstrap": 20000,
        "strata": "five fixed full-period burstiness strata; retrospective benchmark descriptors",
        "familywise_interval": "percentile CI with alpha=0.05/6 per metric family",
        "test": "two-sided paired Wilcoxon; compare to Bonferroni alpha=0.05/6",
    }, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
