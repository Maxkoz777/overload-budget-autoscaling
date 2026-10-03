#!/usr/bin/env python3
"""Bounded persistence extension: resource axis, scale strata, and cost decomposition.

Implements exactly the analysis fixed in
audit/verified_results/resource_axis_study/protocol.json:

* margin families (conformal alpha grid, Gaussian z grid, pure prediction),
  each with and without the shared guard, replayed with the causal one-step
  persistence forecast on the upstream-verified 200-service series;
* the existing reactive threshold--cooldown trajectories (reused, not
  re-implemented);
* per-service cost components, a normalised resource axis with a common
  per-service denominator, pre-test-selected points (days 8--9 only), and a
  descriptive test-selected envelope over fleet-uniform configurations.

No forecaster is trained, no original result file is overwritten, and no
hypothesis test is added.  Run from the paper repository:

    python3 audit/resource_axis_extension.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PAPER = Path(__file__).resolve().parents[1]
EXP = PAPER.parent / "experiments"
DATA = EXP / "data" / "service_timeseries_200_verified"
SPLIT = EXP / "data" / "splits" / "split_definition.json"
SELECTION = EXP / "data" / "splits" / "selected_services_200.csv"
CHAR = PAPER / "audit/verified_results/external_validity/service_characteristics.csv"
RANK = PAPER / "audit/verified_results/rank_ablation/rank_ablation_per_service.csv"
REACTIVE_CAL = PAPER / "audit/verified_results/comparative/reactive_calibration_grid_per_service.csv"
REACTIVE_TEST = PAPER / "audit/verified_results/comparative/reactive_test_grid_per_service.csv"
REACTIVE_SEL5 = PAPER / "audit/verified_results/comparative/reactive_selected_parameters.csv"
STRICT = PAPER / "audit/verified_results/strict_budget/strict_budget_per_service.csv"
OUT = PAPER / "audit/verified_results/resource_axis_study"
PROTOCOL = OUT / "protocol.json"
FIG = PAPER / "figures/verified"

W = 240
C_RES, C_ACT, C_VIO = 1.0, 0.05, 10.0
RHO, H, GAMMA = 0.70, 2, 1
BUDGETS = (0.01, 0.05)
ALPHAS = (1.0, 1.25, 1.5, 2.0)
Z_GRID = (1.28, 1.64, float(norm.ppf(0.95)), 1.96, float(norm.ppf(0.99)), 3.0, 4.0)
N_CAPS = 30
COHORTS = ("all200", "Q3Q4", "Q4", "focused20")


def guarded(nominal: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """h=2, rho=0.70, gamma=1; identical to recompute_verified_rank_ablation.guarded."""
    soft0 = (demand > RHO * nominal).tolist()
    soft1 = (demand > RHO * (nominal + GAMMA)).tolist()
    trig = [False] * len(demand)
    s2 = s1 = False
    for t in range(len(demand)):
        tr = t >= 2 and s1 and s2
        trig[t] = tr
        s = soft1[t] if tr else soft0[t]
        s2, s1 = s1, s
    return nominal + GAMMA * np.asarray(trig, dtype=nominal.dtype)


def windows_for(history: np.ndarray, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forecast = np.r_[history[-1], demand[:-1]]
    residual = np.diff(np.r_[history, demand])
    n = len(history)
    win = np.lib.stride_tricks.sliding_window_view(residual, W)[n - 1 - W: n - 1 - W + len(demand)]
    # causality: the window used at step t ends with the residual realised at t-1
    assert np.array_equal(win[0], np.diff(history)[-W:])
    assert win[1, -1] == demand[0] - history[-1]
    assert forecast[1] == demand[0]
    return forecast, win


def components(capacity: np.ndarray, demand: np.ndarray) -> dict:
    over = demand > capacity
    churn = float(np.abs(np.diff(capacity)).sum())
    c_res = C_RES * float(capacity.sum())
    c_act = C_ACT * churn
    c_vio = C_VIO * float(over.sum())
    return {"overload_count": int(over.sum()), "overload_fraction": float(over.mean()),
            "mean_capacity": float(capacity.mean()), "churn": churn,
            "floor_fraction": float((capacity == 1).mean()),
            "c_res": c_res, "c_act": c_act, "c_vio": c_vio, "total_cost": c_res + c_act + c_vio}


def margin_configs():
    for rd in BUDGETS:
        for a in ALPHAS:
            for g in (False, True):
                yield {"family": "conformal", "rank_delta": rd, "alpha": a, "z": np.nan, "guard": g,
                       "config_id": f"conformal_d{rd:g}_a{a:g}_g{int(g)}"}
    for z in Z_GRID:
        for g in (False, True):
            yield {"family": "gaussian", "rank_delta": np.nan, "alpha": np.nan, "z": z, "guard": g,
                   "config_id": f"gaussian_z{z:.6f}_g{int(g)}"}
    yield {"family": "pure_predictive", "rank_delta": np.nan, "alpha": np.nan, "z": np.nan, "guard": False,
           "config_id": "pure_predictive_g0"}


def replay_phase(history, demand, replicas):
    forecast, win = windows_for(history, demand)
    ordered = np.sort(np.maximum(win, 0.0), axis=1)
    mean, std = win.mean(axis=1), win.std(axis=1, ddof=0)
    observed = np.maximum(np.ceil(replicas), 1)
    obs = components(observed, demand)
    rows = []
    for cfg in margin_configs():
        if cfg["family"] == "conformal":
            rank = min(W, math.ceil((W + 1) * (1 - cfg["rank_delta"])))
            margin = cfg["alpha"] * ordered[:, rank - 1]
        elif cfg["family"] == "gaussian":
            margin = np.maximum(0.0, mean + cfg["z"] * std)
        else:
            margin = np.zeros(len(demand))
        nominal = np.maximum(np.ceil(forecast + margin), 1)
        capacity = guarded(nominal, demand) if cfg["guard"] else nominal
        rows.append({**cfg, **components(capacity, demand),
                     "c_res_observed": obs["c_res"], "replay_observed": obs["total_cost"]})
    return rows


def pick(group: pd.DataFrame, budget: float, param: str) -> pd.Series:
    feasible = group[group.overload_fraction <= budget]
    if len(feasible):
        return feasible.sort_values(["relative_cost", "overload_fraction", param], ascending=[True, True, True]).iloc[0]
    return group.sort_values(["overload_fraction", "relative_cost", param], ascending=[True, True, False]).iloc[0]


def pick_reactive(group: pd.DataFrame, budget: float) -> pd.Series:
    feasible = group[group.overload_fraction <= budget]
    if len(feasible):
        return feasible.sort_values(["relative_cost", "overload_fraction", "threshold", "cooldown"],
                                    ascending=[True, True, False, True]).iloc[0]
    return group.sort_values(["overload_fraction", "relative_cost", "threshold", "cooldown"],
                             ascending=[True, True, False, True]).iloc[0]


def cohort_masks(frame: pd.DataFrame) -> dict:
    return {"all200": np.ones(len(frame), bool),
            "Q3Q4": frame.pretest_size_quartile.isin(["Q3", "Q4"]).to_numpy(),
            "Q4": frame.pretest_size_quartile.eq("Q4").to_numpy(),
            "focused20": frame.focused.to_numpy(bool)}


def aggregate(frame: pd.DataFrame, budget: float) -> dict:
    return {"services": len(frame),
            "mean_overload": float(frame.overload_fraction.mean()),
            "compliant": int((frame.overload_fraction <= budget).sum()),
            "compliance": float((frame.overload_fraction <= budget).mean()),
            "mean_norm_resource": float(frame.norm_resource.mean()),
            "pooled_norm_resource": float(frame.c_res.sum() / frame.c_res_observed.sum()),
            "mean_norm_actuation": float(frame.norm_actuation.mean()),
            "mean_relative_cost": float(frame.relative_cost.mean()),
            "median_relative_cost": float(frame.relative_cost.median()),
            "mean_floor_fraction": float(frame.floor_fraction.mean()) if "floor_fraction" in frame and frame.floor_fraction.notna().all() else np.nan}


def main() -> None:
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    split = {s["name"]: s for s in json.loads(SPLIT.read_text())["splits"]}
    selection = pd.read_csv(SELECTION)
    char = pd.read_csv(CHAR)[["service_id", "pretest_size_quartile", "focused20"]]
    assert len(selection) == 200 and selection.service_id.is_unique

    rows = []
    for i, item in enumerate(selection.itertuples(index=False), 1):
        f = pd.read_parquet(DATA / f"{item.service_id}.parquet").sort_values("timestamp")
        ts = f.timestamp
        train = f[ts < split["train"]["timestamp_end_exclusive"]]
        cal = f[(ts >= split["calibration"]["timestamp_start"]) & (ts < split["calibration"]["timestamp_end_exclusive"])]
        pre = f[ts < split["test"]["timestamp_start"]]
        test = f[(ts >= split["test"]["timestamp_start"]) & (ts < split["test"]["timestamp_end_exclusive"])]
        assert (len(train), len(cal), len(pre), len(test)) == (11520, 2880, 14400, 4320)
        for phase, hist, ev in (("calibration", train, cal), ("test", pre, test)):
            for r in replay_phase(hist.cpu_sum.to_numpy(float), ev.cpu_sum.to_numpy(float), ev.replica_count.to_numpy(float)):
                rows.append({"service_id": item.service_id, "stratum": item.stratum, "phase": phase, **r})
        if i % 25 == 0:
            print(f"extension replay {i}/200", flush=True)
    per = pd.DataFrame(rows).merge(char, on="service_id", validate="many_to_one").rename(columns={"focused20": "focused"})
    per["relative_cost"] = per.total_cost / per.replay_observed
    per["norm_resource"] = per.c_res / per.c_res_observed
    per["norm_actuation"] = per.c_act / per.c_res_observed

    # Reproduction of original rows (alpha=1 conformal, nominal-z Gaussian)
    rank = pd.read_csv(RANK)
    test = per[per.phase.eq("test")]
    checks = {}
    conf = test[test.family.eq("conformal") & test.alpha.eq(1.0)].merge(
        rank[rank.method.eq("conformal")], left_on=["service_id", "rank_delta", "guard"],
        right_on=["service_id", "delta", "guard"], validate="one_to_one", suffixes=("", "_orig"))
    gauss = []
    for d in BUDGETS:
        z = float(norm.ppf(1 - d))
        g = test[test.family.eq("gaussian") & np.isclose(test.z, z, rtol=0, atol=1e-12)].merge(
            rank[rank.method.eq("gaussian") & rank.delta.eq(d)], on=["service_id", "guard"],
            validate="one_to_one", suffixes=("", "_orig"))
        gauss.append(g)
    gauss = pd.concat(gauss)
    for name, m in (("conformal", conf), ("gaussian", gauss)):
        ok = (len(m) == 800 and np.allclose(m.overload_fraction, m.overload_fraction_orig, rtol=0, atol=1e-12)
              and np.allclose(m.relative_cost, m.relative_cost_orig, rtol=0, atol=1e-12))
        checks[f"reproduces_rank_ablation_{name}"] = bool(ok)
        assert ok, name

    # Reactive: reuse saved trajectories, derive components, check identities
    obs = per[per.config_id.eq("pure_predictive_g0")][["service_id", "phase", "c_res_observed", "replay_observed"]]
    reactive = []
    for phase, path, T in (("calibration", REACTIVE_CAL, 2880), ("test", REACTIVE_TEST, 4320)):
        r = pd.read_csv(path).rename(columns={"is_focused_20": "focused_saved", "scaling_churn": "churn"})
        r["phase"] = phase
        r["c_res"] = C_RES * r.mean_capacity * T
        r["c_act"] = C_ACT * r.churn
        r["overload_count"] = np.rint(r.overload_fraction * T).astype(int)
        r["c_vio"] = C_VIO * r.overload_count
        ident = np.allclose(r.c_res + r.c_act + r.c_vio, r.total_cost, rtol=1e-12, atol=1e-6)
        checks[f"reactive_{phase}_cost_identity"] = bool(ident)
        assert ident
        r = r.merge(obs[obs.phase.eq(phase)].drop(columns="phase"), on="service_id", validate="many_to_one")
        rel = np.allclose(r.total_cost / r.replay_observed, r.relative_cost, rtol=1e-12, atol=1e-12)
        checks[f"reactive_{phase}_denominator_matches"] = bool(rel)
        assert rel
        reactive.append(r)
    reactive = pd.concat(reactive).merge(char, on="service_id", validate="many_to_one").rename(columns={"focused20": "focused"})
    reactive["family"] = "reactive"
    reactive["guard"] = False
    reactive["config_id"] = [f"reactive_u{u:g}_c{c:d}" for u, c in zip(reactive.threshold, reactive.cooldown)]
    reactive["norm_resource"] = reactive.c_res / reactive.c_res_observed
    reactive["norm_actuation"] = reactive.c_act / reactive.c_res_observed
    reactive["floor_fraction"] = np.nan
    keep = ["service_id", "stratum", "phase", "family", "config_id", "rank_delta", "alpha", "z", "threshold", "cooldown",
            "guard", "overload_count", "overload_fraction", "mean_capacity", "churn", "floor_fraction", "c_res", "c_act",
            "c_vio", "total_cost", "c_res_observed", "replay_observed", "relative_cost", "norm_resource",
            "norm_actuation", "pretest_size_quartile", "focused"]
    for col in ("threshold", "cooldown"):
        per[col] = np.nan
    for col in ("rank_delta", "alpha", "z"):
        reactive[col] = np.nan
    allrows = pd.concat([per[keep], reactive[keep]], ignore_index=True)
    assert not allrows.duplicated(["service_id", "phase", "config_id"]).any()

    # Pre-test-selected points
    cal = allrows[allrows.phase.eq("calibration")]
    tst = allrows[allrows.phase.eq("test")].set_index(["service_id", "config_id"])
    selected = []
    for budget in BUDGETS:
        for sid, g in cal.groupby("service_id", sort=True):
            choices = {}
            for guard in (False, True):
                cg = g[g.family.eq("conformal") & g.rank_delta.eq(budget) & g.guard.eq(guard)]
                choices[f"conformal_selected_g{int(guard)}"] = pick(cg, budget, "alpha")
                gg = g[g.family.eq("gaussian") & g.guard.eq(guard)]
                choices[f"gaussian_selected_g{int(guard)}"] = pick(gg, budget, "z")
            choices["reactive_selected"] = pick_reactive(g[g.family.eq("reactive")], budget)
            for label, c in choices.items():
                t = tst.loc[(sid, c.config_id)]
                selected.append({"service_id": sid, "evaluation_budget": budget, "selected_point": label,
                                 "config_id": c.config_id, "calibration_overload": c.overload_fraction,
                                 "calibration_feasible": bool(c.overload_fraction <= budget),
                                 **{k: t[k] for k in ("overload_fraction", "c_res", "c_act", "c_vio", "relative_cost",
                                                      "norm_resource", "norm_actuation", "c_res_observed", "replay_observed",
                                                      "floor_fraction", "pretest_size_quartile", "focused")}})
    selected = pd.DataFrame(selected)
    # reactive selection reproduces the saved B1 choices
    s5 = pd.read_csv(REACTIVE_SEL5)
    r5 = selected[selected.selected_point.eq("reactive_selected") & selected.evaluation_budget.eq(0.05)]
    exp5 = [f"reactive_u{u:g}_c{c:d}" for u, c in zip(s5.sort_values("service_id").threshold, s5.sort_values("service_id").cooldown)]
    checks["reactive_selection_5pct_reproduced"] = bool(list(r5.sort_values("service_id").config_id) == exp5)
    strict = pd.read_csv(STRICT)
    s1 = strict[strict.policy.eq("reactive") & strict.calibration_budget.eq(0.01)].set_index("service_id")
    r1 = selected[selected.selected_point.eq("reactive_selected") & selected.evaluation_budget.eq(0.01)].set_index("service_id")
    checks["reactive_selection_1pct_reproduced"] = bool(
        np.allclose(r1.loc[s1.index, "overload_fraction"], s1.overload_fraction, rtol=0, atol=1e-12)
        and np.allclose(r1.loc[s1.index, "relative_cost"], s1.relative_cost, rtol=0, atol=1e-12))
    assert all(checks.values()), checks

    # Summaries of fleet-uniform configurations and selected points
    test_rows = allrows[allrows.phase.eq("test")].reset_index(drop=True)
    masks = cohort_masks(test_rows)
    sel_masks = cohort_masks(selected)
    summary = []
    for cohort in COHORTS:
        part = test_rows[masks[cohort]]
        for budget in BUDGETS:
            for cid, g in part.groupby("config_id", sort=True):
                first = g.iloc[0]
                summary.append({"cohort": cohort, "evaluation_budget": budget, "point_type": "fleet_uniform",
                                "family": first.family, "config_id": cid, "guard": bool(first.guard),
                                "rank_delta": first.rank_delta, "alpha": first.alpha, "z": first.z,
                                "threshold": first.threshold, "cooldown": first.cooldown, **aggregate(g, budget)})
            sp = selected[sel_masks[cohort] & selected.evaluation_budget.eq(budget)]
            for label, g in sp.groupby("selected_point", sort=True):
                fam = label.split("_")[0]
                summary.append({"cohort": cohort, "evaluation_budget": budget, "point_type": "pretest_selected",
                                "family": fam, "config_id": label, "guard": label.endswith("g1"),
                                "calibration_infeasible_services": int((~g.calibration_feasible).sum()),
                                **aggregate(g, budget)})
    summary = pd.DataFrame(summary)

    # Descriptive envelope (test-selected, fleet-uniform only)
    env = []
    fu = summary[summary.point_type.eq("fleet_uniform")]
    for cohort in COHORTS:
        c = fu[fu.cohort.eq(cohort)]
        caps = np.linspace(c.mean_norm_resource.min(), c.mean_norm_resource.max(), N_CAPS)
        for budget in BUDGETS:
            cb = c[c.evaluation_budget.eq(budget)]
            for fam in ("conformal", "gaussian", "reactive", "pure_predictive"):
                cf = cb[cb.family.eq(fam)]
                for k, cap in enumerate(caps):
                    ok = cf[cf.mean_norm_resource <= cap + 1e-15]
                    env.append({"cohort": cohort, "evaluation_budget": budget, "family": fam, "cap_index": k,
                                "resource_cap": float(cap), "available": bool(len(ok)),
                                "best_mean_overload": float(ok.mean_overload.min()) if len(ok) else np.nan,
                                "best_mean_overload_config": ok.sort_values(["mean_overload", "mean_norm_resource"]).config_id.iloc[0] if len(ok) else "",
                                "best_compliance": float(ok.compliance.max()) if len(ok) else np.nan,
                                "best_compliance_config": ok.sort_values(["compliance", "mean_norm_resource"], ascending=[False, True]).config_id.iloc[0] if len(ok) else ""})
    env = pd.DataFrame(env)

    # Cost decomposition (B1): named policies, per-service shares of the observed replay cost
    named = {"B2_pure_predictive": lambda d: "pure_predictive_g0",
             "B7_conformal_margin": lambda d: f"conformal_d{d:g}_a1_g0",
             "B6_conformal_plus_offset": lambda d: f"conformal_d{d:g}_a1_g1",
             "B4_gaussian_nominal": lambda d: f"gaussian_z{float(norm.ppf(1-d)):.6f}_g0",
             "B4G_gaussian_nominal_plus_offset": lambda d: f"gaussian_z{float(norm.ppf(1-d)):.6f}_g1"}
    decomp = []
    for cohort in COHORTS:
        part = test_rows[masks[cohort]]
        for budget in BUDGETS:
            frames = {k: part[part.config_id.eq(f(budget))] for k, f in named.items()}
            sp = selected[sel_masks[cohort] & selected.evaluation_budget.eq(budget) & selected.selected_point.eq("reactive_selected")]
            frames["B1_reactive_pretest_selected"] = sp
            for policy, g in frames.items():
                denom = g
                res = (denom.c_res / denom.replay_observed)
                act = (denom.c_act / denom.replay_observed)
                vio = (denom.c_vio / denom.replay_observed)
                decomp.append({"cohort": cohort, "evaluation_budget": budget, "policy": policy, "services": len(denom),
                               "mean_resource_share": float(res.mean()), "mean_actuation_share": float(act.mean()),
                               "mean_violation_share": float(vio.mean()),
                               "mean_relative_cost": float((res + act + vio).mean()),
                               "median_relative_cost": float((res + act + vio).median()),
                               "mean_overload": float(denom.overload_fraction.mean()),
                               "compliant": int((denom.overload_fraction <= budget).sum())})
    decomp = pd.DataFrame(decomp)
    # shares sum exactly to relative cost (linear decomposition of a mean of ratios)
    checks["decomposition_sums_to_relative_cost"] = bool(np.allclose(
        decomp.mean_resource_share + decomp.mean_actuation_share + decomp.mean_violation_share, decomp.mean_relative_cost))

    OUT.mkdir(parents=True, exist_ok=True)
    allrows.to_csv(OUT / "per_service.csv", index=False)
    selected.to_csv(OUT / "pretest_selected_per_service.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    env.to_csv(OUT / "envelope.csv", index=False)
    decomp.to_csv(OUT / "cost_decomposition.csv", index=False)
    (OUT / "methodology.json").write_text(json.dumps({
        "protocol_sha256": protocol_hash, "checks": checks,
        "exact_nominal_z": {"0.95": float(norm.ppf(0.95)), "0.99": float(norm.ppf(0.99))},
        "rows": {"per_service": len(allrows), "pretest_selected_per_service": len(selected),
                 "summary": len(summary), "envelope": len(env), "cost_decomposition": len(decomp)},
        "reactive_floor_fraction": "not stored in the saved reactive trajectories; reported as missing",
        "scope": "retrospective replay on the already inspected held-out days; descriptive only"}, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
