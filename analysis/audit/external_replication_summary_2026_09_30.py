#!/usr/bin/env python3
"""Summaries for the external replication (called by external_replication_2026_09_30.py summarise)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

PAPER = Path(__file__).resolve().parents[1]
EXT = PAPER.parent / "experiments" / "data" / "external_traces"
OUT = PAPER / "audit" / "verified_results" / "external_replication_2026-09-30"
BUDGETS = (0.01, 0.05)
RUNS = (("huawei2023", "primary"), ("huawei2023", "p90"), ("azure2019", "primary"), ("azure2019", "K5"), ("azure2019", "K20"))


def named(d: float) -> dict:
    z = f"{float(norm.ppf(1 - d)):.6f}"
    return {"B2": "pure_predictive_g0", "B8": "pure_predictive_g1",
            "B7": f"conformal_W240_d{d:g}_a1_g0", "B6": f"conformal_W240_d{d:g}_a1_g1",
            "invcdf": f"invcdf_W240_d{d:g}_g0", "invcdf_G": f"invcdf_W240_d{d:g}_g1",
            "B4": f"gaussian_W240_z{z}_g0", "B4G": f"gaussian_W240_z{z}_g1",
            "conf1440": f"conformal_W1440_d{d:g}_a1_g0", "conf1440_G": f"conformal_W1440_d{d:g}_a1_g1"}


def strata(family: str) -> pd.DataFrame:
    u = pd.read_csv(EXT / family / "units.csv")
    if family == "huawei2023":
        units = u.mean_requests_pre / u.mu_p50
        u["stratum"] = np.where(units >= units.median(), "upper_half", "lower_half")
        u["pretest_mean_units"] = units
    else:
        u["stratum"] = u.group
        u["pretest_mean_units"] = 10.0
    return u[["unit_id", "stratum", "pretest_mean_units"]]


def load(family, mu):
    per = pd.read_csv(OUT / f"per_unit_{family}_{mu}.csv.gz")
    return per.merge(strata(family), on="unit_id", validate="many_to_one")


def policy_rows(per: pd.DataFrame, d: float) -> pd.DataFrame:
    test = per[per.phase == "test"]
    frames = []
    for name, cid in named(d).items():
        frames.append(test[test.config_id == cid].assign(policy=name))
    cal = per[per.phase == "calibration"]
    rows = []
    for uid, g in cal.groupby("unit_id", sort=True):
        feas = g[g.overload_fraction <= d]
        if len(feas):
            r = feas.sort_values(["relative_cost", "overload_fraction", "threshold", "cooldown"], ascending=[True, True, False, True]).iloc[0]
        else:
            r = g.sort_values(["overload_fraction", "relative_cost", "threshold", "cooldown"], ascending=[True, True, False, True]).iloc[0]
        rows.append((uid, r.config_id, bool(len(feas))))
    choice = pd.DataFrame(rows, columns=["unit_id", "config_id", "calibration_feasible"])
    b1 = test.merge(choice, on=["unit_id", "config_id"], validate="one_to_one").assign(policy="B1")
    assert len(b1) == test.unit_id.nunique()
    frames.append(b1)
    out = pd.concat(frames, ignore_index=True)
    out["budget"] = d
    out["within"] = out.overload_fraction <= d
    return out


def agg(g: pd.DataFrame) -> dict:
    r = {"n": len(g), "compliant": int(g.within.sum()), "compliance": float(g.within.mean()),
         "mean_overload": float(g.overload_fraction.mean()), "median_overload": float(g.overload_fraction.median()),
         "mean_relative_cost": float(g.relative_cost.mean()), "median_relative_cost": float(g.relative_cost.median()),
         "mean_norm_resource": float(g.norm_resource.mean()), "mean_floor_fraction": float(g.floor_fraction.mean())}
    if "resource_vs_observed" in g:
        r["mean_resource_vs_observed"] = float(g.resource_vs_observed.mean())
    if "calibration_feasible" in g and g.calibration_feasible.notna().any():
        r["calibration_infeasible"] = int((g.calibration_feasible == False).sum())  # noqa: E712
    return r


def paired(pol: pd.DataFrame, a: str, b: str) -> dict:
    x = pol[pol.policy == a].set_index("unit_id"); y = pol[pol.policy == b].set_index("unit_id")
    x, y = x.loc[x.index.intersection(y.index)], y.loc[x.index.intersection(y.index)]
    wins = int((x.within & ~y.within).sum()); losses = int((~x.within & y.within).sum())
    return {"a": a, "b": b, "n": len(x), "a_only": wins, "b_only": losses,
            "exact_p_unadjusted": float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0,
            "mean_overload_diff": float((x.overload_fraction - y.overload_fraction).mean()),
            "mean_relative_cost_diff": float((x.relative_cost - y.relative_cost).mean()),
            "relative_cost_ratio_of_means": float(x.relative_cost.mean() / y.relative_cost.mean())}


def exact_envelope(fu: pd.DataFrame) -> pd.DataFrame:
    caps = np.unique(fu.mean_norm_resource)
    rows = []
    for fam in ("conformal", "gaussian", "reactive", "pure_predictive"):
        f = fu[fu.family == fam]
        for cap in caps:
            ok = f[f.mean_norm_resource <= cap]
            rows.append({"family": fam, "resource_cap": cap, "available": bool(len(ok)),
                         "best_mean_overload": float(ok.mean_overload.min()) if len(ok) else np.nan,
                         "best_config": ok.sort_values(["mean_overload", "mean_norm_resource"]).config_id.iloc[0] if len(ok) else ""})
    return pd.DataFrame(rows)


def summarise() -> None:
    main, pairs, strat, fleet, envs, comps = [], [], [], [], [], []
    for family, mu in RUNS:
        per = load(family, mu)
        for d in BUDGETS:
            pol = policy_rows(per, d)
            pol.to_csv(OUT / f"policy_rows_{family}_{mu}_d{d:g}.csv.gz", index=False)
            for p, g in pol.groupby("policy", sort=False):
                main.append({"family": family, "mu": mu, "budget": d, "policy": p, **agg(g)})
                for s, gs in g.groupby("stratum"):
                    strat.append({"family": family, "mu": mu, "budget": d, "policy": p, "stratum": s, **agg(gs)})
            for a, b in (("B6", "invcdf_G"), ("B7", "invcdf"), ("B6", "B1"), ("B6", "B4G"), ("B7", "B4"), ("conf1440", "B1")):
                pairs.append({"family": family, "mu": mu, "budget": d, "stratum": "all", **paired(pol, a, b)})
                for s in sorted(pol.stratum.unique()):
                    pairs.append({"family": family, "mu": mu, "budget": d, "stratum": s, **paired(pol[pol.stratum == s], a, b)})
        if mu == "primary":
            test = per[per.phase == "test"]
            fam_map = test.family.replace({"inverse_cdf": "other"})
            keep = test[(test.W == 240) | test.W.isna()]
            keep = keep[keep.family.isin(["conformal", "gaussian", "reactive", "pure_predictive"])]
            for cohort, part in (("all", keep), *[(s, keep[keep.stratum == s]) for s in sorted(keep.stratum.unique())]):
                fu = part.groupby(["family", "config_id"]).agg(mean_overload=("overload_fraction", "mean"),
                                                              mean_norm_resource=("norm_resource", "mean")).reset_index()
                fu["family_trace"], fu["cohort"] = family, cohort
                fleet.append(fu)
                env = exact_envelope(fu); env["family_trace"], env["cohort"] = family, cohort
                envs.append(env)
                w = env.pivot(index="resource_cap", columns="family", values="best_mean_overload")
                for a, b in (("conformal", "reactive"), ("gaussian", "reactive"), ("conformal", "gaussian")):
                    lo = max(fu[fu.family == a].mean_norm_resource.min(), fu[fu.family == b].mean_norm_resource.min())
                    hi = min(fu[fu.family == a].mean_norm_resource.max(), fu[fu.family == b].mean_norm_resource.max())
                    x = w[(w.index >= lo) & (w.index <= hi)][[a, b]].dropna()
                    dd = 100 * (x[a] - x[b])
                    comps.append({"family_trace": family, "cohort": cohort, "a": a, "b": b, "overlap_lo": lo, "overlap_hi": hi,
                                  "caps": len(x), "a_lower": int((dd < 0).sum()), "b_lower": int((dd > 0).sum()),
                                  "median_abs_diff_pp": float(dd.abs().median()) if len(x) else np.nan})
    pd.DataFrame(main).to_csv(OUT / "summary_policies.csv", index=False)
    pd.DataFrame(strat).to_csv(OUT / "summary_strata.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUT / "paired_compliance.csv", index=False)
    pd.concat(fleet).to_csv(OUT / "fleet_uniform_primary.csv", index=False)
    pd.concat(envs).to_csv(OUT / "envelope_exact_primary.csv", index=False)
    pd.DataFrame(comps).to_csv(OUT / "envelope_comparisons_primary.csv", index=False)
    meta = {"protocol_sha256": hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest(),
            "runs": [f"{f}/{m}" for f, m in RUNS],
            "inputs": {f: hashlib.sha256((EXT / f / "series.npz").read_bytes()).hexdigest() for f in ("huawei2023", "azure2019")},
            "alibaba_regression": json.loads((OUT / "alibaba_regression.json").read_text())}
    (OUT / "methodology.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("summaries written")
