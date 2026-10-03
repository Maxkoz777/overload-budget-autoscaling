"""C3: conditional / at-risk compliance (reaggregation, no replay)."""
from pathlib import Path
import pandas as pd, numpy as np
A = Path("experiments/results/large_scale/analysis")
b = pd.read_csv(A / "ls_baselines_per_service_for_stats_200.csv")
f = pd.read_csv(A / "ls_frontier_per_service_200.csv")

# at-risk = services where pure-predictive (B2) overload exceeds delta/2 at delta=0.05
b2 = b[b.policy.str.startswith("B2")][["service_id", "stratum", "overload_fraction"]]
thr = 0.05 / 2
at_risk = set(b2[b2.overload_fraction > thr]["service_id"])
at_risk0 = set(b2[b2.overload_fraction > 0]["service_id"])
print(f"at-risk (B2 OL > {thr}): {len(at_risk)} / 200 services")
print(f"any-risk (B2 OL > 0):   {len(at_risk0)} / 200")

# B6 (full policy) compliance on full set vs at-risk subset, by forecaster x delta
rows = []
for model in ["persistence", "lstm", "xgb", "arima"]:
    for d in [0.05, 0.03, 0.01]:
        sub = f[(f.model == model) & (np.isclose(f.delta, d))]
        full = (sub.within_delta == True).mean() if "within_delta" in sub else (sub.overload_fraction <= d).mean()
        ar = sub[sub.service_id.isin(at_risk)]
        ar_frac = (ar.overload_fraction <= d).mean()
        rows.append(dict(model=model, delta=d,
                         frac_full=round(float((sub.overload_fraction <= d).mean()), 3),
                         frac_at_risk=round(float(ar_frac), 3),
                         n_at_risk=len(ar)))
c3 = pd.DataFrame(rows)
c3.to_csv(A / "ls_c3_conditional_compliance_200.csv", index=False)
print("\n=== B6 compliance: full set vs at-risk subset ===")
print(c3.to_string(index=False))

# per-stratum B6 persistence compliance at delta=0.01 (full + at-risk)
rows2 = []
for st in ["G1", "G2", "G3", "G4", "G5"]:
    sub = f[(f.model == "persistence") & (np.isclose(f.delta, 0.01)) & (f.stratum == st)]
    ar = sub[sub.service_id.isin(at_risk)]
    rows2.append(dict(stratum=st,
                      frac_full=round(float((sub.overload_fraction <= 0.01).mean()), 3),
                      n_full=len(sub),
                      frac_at_risk=(round(float((ar.overload_fraction <= 0.01).mean()), 3) if len(ar) else float("nan")),
                      n_at_risk=len(ar)))
c3s = pd.DataFrame(rows2)
c3s.to_csv(A / "ls_c3_stratum_delta01_200.csv", index=False)
print("\n=== B6 persistence @delta=0.01 by stratum (full vs at-risk) ===")
print(c3s.to_string(index=False))
