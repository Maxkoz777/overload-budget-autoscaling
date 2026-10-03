#!/usr/bin/env python3
"""Pre-test 'non-trivial' subset services whose maximum demand on the
calibration days 8-9 exceeds one capacity unit (mu = 1). Uses only canonical per-service rows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
EXP = PAPER.parent / "experiments"
VER = HERE / "verified_results"
OUT = VER / "delay_pac_study"

spec = {s["name"]: s for s in json.loads((EXP / "data/splits/split_definition.json").read_text())["splits"]}["calibration"]
sel = pd.read_csv(EXP / "data/splits/selected_services_200.csv")
rows = []
for sid in sel.service_id.astype(str):
    f = pd.read_parquet(EXP / "data/service_timeseries_200_verified" / f"{sid}.parquet")
    cal = f[(f.timestamp >= spec["timestamp_start"]) & (f.timestamp < spec["timestamp_end_exclusive"])]
    assert len(cal) == spec["expected_rows"]
    rows.append({"service_id": sid, "calibration_max_demand": float(cal.cpu_sum.max())})
crit = pd.DataFrame(rows)
crit["nontrivial_pretest"] = crit.calibration_max_demand > 1.0

sb = pd.read_csv(VER / "strict_budget/strict_budget_per_service.csv")
ra = pd.read_csv(VER / "rank_ablation/rank_ablation_per_service.csv")
rt = pd.read_csv(VER / "comparative/reactive_test_per_service.csv")
arms = {
    (0.01, "B6"): sb[(sb.policy == "conformal") & (sb.W == 240) & sb.guard],
    (0.01, "B7"): sb[(sb.policy == "conformal") & (sb.W == 240) & ~sb.guard],
    (0.01, "Inverse-CDF+offset"): sb[(sb.policy == "inverse_cdf") & (sb.W == 240) & sb.guard],
    (0.01, "Reactive cal. 1%"): sb[(sb.policy == "reactive") & (sb.calibration_budget == 0.01)],
    (0.01, "Reactive cal. 0.25%"): sb[(sb.policy == "reactive") & (sb.calibration_budget == 0.0025)],
    (0.01, "Conformal W1440"): sb[(sb.policy == "conformal") & (sb.W == 1440) & ~sb.guard],
    (0.01, "Conformal W1440+offset"): sb[(sb.policy == "conformal") & (sb.W == 1440) & sb.guard],
    (0.01, "Gaussian+offset"): sb[(sb.policy == "gaussian") & (sb.W == 240) & sb.guard],
    (0.01, "Gaussian z=4+offset"): sb[(sb.policy == "gaussian_z4") & (sb.W == 240) & sb.guard],
    (0.05, "B6"): ra[(ra.delta == 0.05) & (ra.method == "conformal") & ra.guard],
    (0.05, "B7"): ra[(ra.delta == 0.05) & (ra.method == "conformal") & ~ra.guard],
    (0.05, "Reactive cal. 5%"): rt,
}
out, per = [], []
for (d, name), fr in arms.items():
    fr = fr[["service_id", "overload_fraction", "relative_cost"]].merge(crit, on="service_id", validate="one_to_one")
    assert len(fr) == 200
    fr["within"] = fr.overload_fraction <= d
    per.append(fr.assign(budget=d, policy=name))
    for cohort, part in (("all200", fr), ("nontrivial_pretest", fr[fr.nontrivial_pretest]), ("rest", fr[~fr.nontrivial_pretest])):
        out.append({"budget": d, "policy": name, "cohort": cohort, "n": len(part), "compliant": int(part.within.sum()),
                    "mean_relative_cost": part.relative_cost.mean(), "median_relative_cost": part.relative_cost.median(),
                    "mean_overload": part.overload_fraction.mean()})
summ = pd.DataFrame(out)
per = pd.concat(per, ignore_index=True)
pairs = []
for d, a, b in ((0.01, "B6", "Reactive cal. 1%"), (0.01, "B6", "Reactive cal. 0.25%"), (0.01, "B7", "Reactive cal. 1%"),
                (0.01, "B6", "Inverse-CDF+offset"), (0.05, "B6", "Reactive cal. 5%")):
    x = per[(per.budget == d) & (per.policy == a) & per.nontrivial_pretest].set_index("service_id")
    y = per[(per.budget == d) & (per.policy == b) & per.nontrivial_pretest].set_index("service_id").loc[x.index]
    w, l = int((x.within & ~y.within).sum()), int((~x.within & y.within).sum())
    pairs.append({"budget": d, "a": a, "b": b, "n": len(x), "a_only": w, "b_only": l,
                  "exact_p_unadjusted": binomtest(w, w + l).pvalue if w + l else 1.0,
                  "mean_cost_ratio": x.relative_cost.mean() / y.relative_cost.mean(),
                  "paired_mean_cost_diff": (x.relative_cost - y.relative_cost).mean()})
crit.to_csv(OUT / "subset_pretest_criterion.csv", index=False)
summ.to_csv(OUT / "subset_pretest_summary.csv", index=False)
pd.DataFrame(pairs).to_csv(OUT / "subset_pretest_paired.csv", index=False)
print(int(crit.nontrivial_pretest.sum()), "services meet the pre-test criterion")
print(summ.to_string(index=False)); print(pd.DataFrame(pairs).to_string(index=False))
