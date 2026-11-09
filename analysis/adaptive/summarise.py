"""Cell summaries from the per-unit outputs (reporting only; controllers are not re-run).

Usage: python3 summarise.py   -> out_v1/test_summary.csv, out_v1/confirm_summary.csv,
                                  out_v1/confirm_pooled_summary.csv
Reference for every cost ratio: PAC-h under the common resource bound ("PAC-h").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive.data import load
from adaptive.env import applied, outcome

RES = "out_v1"


def feasible_under_bound(names) -> pd.DataFrame:
    """Units whose budget is attainable at the bound: overload with constant capacity M (post hoc diagnostic)."""
    rows = []
    for nm in names:
        D = load(nm, "test")
        o = outcome(D, applied(D, np.repeat(D["M"][:, None], D["T"], 1), 0, True))
        for u, f in zip(D["uid"], o["overload_fraction"]):
            rows.append(dict(trace=nm, unit_id=u, T=D["T"], overload_at_M=f))
    return pd.DataFrame(rows)


def summarise(d: pd.DataFrame, feas: pd.DataFrame, keys) -> pd.DataFrame:
    T = feas.set_index(["trace", "unit_id"])["T"]
    fz = feas.set_index(["trace", "unit_id"]).overload_at_M
    ref = d[d.policy == "PAC-h"].set_index(["trace", "budget", "tau", "unit_id"])
    out = []
    for key, g in d.groupby(keys, sort=False):
        k = dict(zip(keys, key))
        g = g.set_index(["trace", "budget", "tau", "unit_id"]); p = ref.loc[g.index]
        b = k["budget"]
        ok = g.overload_fraction <= b + 1e-12; okp = p.overload_fraction <= b + 1e-12; both = ok & okp
        r = (g.relative_cost / p.relative_cost)[both]
        tu = g.index.droplevel(["budget", "tau"])
        feasible = (fz.loc[tu].to_numpy() <= b + 1e-12)
        row = dict(k, n=len(g), within=int(ok.sum()), pac_within=int(okp.sum()), both=int(both.sum()),
                   only_policy=int((ok & ~okp).sum()), only_pac=int((~ok & okp).sum()),
                   feasible_under_bound=int(feasible.sum()),
                   paired_median_ratio=r.median(), ratio_of_means=g.relative_cost.mean() / p.relative_cost.mean(),
                   ratio_p90=r.quantile(0.9), ratio_max=r.max(),
                   mean_resource=g.resource.mean(), mean_churn=g.churn.mean(), mean_penalty=g.penalty.mean())
        if "formal_err" in g and g.formal_err.notna().any():
            unit_minutes = T.loc[tu].sum()
            row.update(max_formal_err=g.formal_err.max(), units_formal_err_over=int((g.formal_err > b + 1e-12).sum()),
                       infinite_set_share=g.inf_min.sum() / unit_minutes,
                       mean_infinite_minutes=g.inf_min.mean(),
                       units_exceptional_overload=int(((g.over_inf + g.over_bound + g.over_prefix) > 0).sum()),
                       units_over_budget_with_formal_err_within=int(((~ok) & (g.formal_err <= b + 1e-12)).sum()))
        if "max_step" in g and g.max_step.notna().any():
            row.update(max_issued_step=g.max_step.max())
        if "gamma" in g and g.gamma.notna().any():
            row.update(gamma=g.gamma.iloc[0], alpha_star=g.alpha_star.iloc[0])
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    test = pd.read_csv(f"{RES}/test_per_unit.csv.gz")
    conf = pd.read_csv(f"{RES}/confirm_per_unit.csv.gz")
    feas = pd.concat([feasible_under_bound(sorted(test.trace.unique())), feasible_under_bound(sorted(conf.trace.unique()))])
    feas.to_csv(f"{RES}/feasibility_under_bound.csv.gz", index=False)
    summarise(test, feas, ["trace", "budget", "tau", "policy"]).to_csv(f"{RES}/test_summary.csv", index=False)
    summarise(conf, feas, ["trace", "budget", "tau", "policy"]).to_csv(f"{RES}/confirm_summary.csv", index=False)
    summarise(conf, feas, ["budget", "tau", "policy"]).to_csv(f"{RES}/confirm_pooled_summary.csv", index=False)
    print("written")
