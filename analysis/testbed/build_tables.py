"""Tables of the Kubernetes validation (T: defined by PROTOCOL.md; X: exploratory, added after the campaign).

Reads testbed/results (analysed outputs, combined.json) and data/ (from extract.py). Writes tables/*.csv and *.tex."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from common import (BUDGET, CUSTOM, DATA, LABEL, MINUTES, MU, POLICIES, RES, RUNS, TB, TEST_MINUTES, WARM, combined, delays,
                    jload, meta, per_function, per_minute, plan, save_table, summary, validity)

OKB = ["ok_le5", "ok_le10", "ok_le25", "ok_le50", "ok_le100", "ok_le250", "ok_le500", "ok_le1000", "ok_le2000", "ok_gt2000"]
EDGES = [5, 10, 25, 50, 100, 250, 500, 1000, 2000, math.inf]
PF = per_function()
PF["node"] = [meta(r)["assignment"][p] for r, p in zip(PF.run, PF.policy)]


def msk(t: float) -> str:
    return pd.Timestamp(t, unit="s", tz="UTC").tz_convert("Europe/Moscow").strftime("%Y-%m-%d %H:%M")


def bucket_q(row: pd.Series, q: float) -> str:
    c = row[OKB].to_numpy(float)
    i = int(np.searchsorted(np.cumsum(c) / c.sum(), q))
    return f"<= {EDGES[i]}" if EDGES[i] != math.inf else "> 2000"


def attempts() -> pd.DataFrame:
    return pd.read_csv(RES / "attempts.tsv", sep="\t", header=None, names=["time", "name", "folder", "reason"])


def t01_campaign() -> pd.DataFrame:
    att = attempts()
    corr = pd.read_csv(DATA / "attribution_down_corrected.csv").set_index("run")
    rows = []
    for r in RUNS:
        m, s, v = meta(r), summary(r), validity(r)
        a = s["delay_attribution"]
        g = PF[PF.run == r]
        reasons = att[att.name == r].reason.tolist()
        ce = sum(1 for _ in open(RES / r / "collector_errors.csv")) - 1
        rows.append({"run": r, "test_day": m["day"], "load_seed": m["seed"],
                     "assignment": ", ".join(f"{LABEL[p]} {m['assignment'][p]}" for p in POLICIES),
                     "minute0_msk": msk(m["t0"]), "test_start_msk": msk(m["t0"] + WARM * 60),
                     "end_msk": msk(m["t0"] + MINUTES * 60), "valid": v["valid"], "problems": len(v["problems"]),
                     "dispatch_lag_p99_ms": round(v["dispatch_lag_p99_ms"], 2),
                     "requests_per_policy": int(g[g.policy == "pach"].requests.sum()),
                     "requests_all_policies": int(g.requests.sum()),
                     "scale_up_requested": a["up_requested"], "scale_down_requested": a["down_requested"],
                     "up_attributed": a["up_attributed"], "down_attributed_analyze": a["down_attributed"],
                     "attributed_twice_analyze": a["pods_attributed_twice"],
                     "down_attributed_corrected": int(corr.loc[r, "down_attributed"]), "attributed_twice_corrected": int(corr.loc[r, "attributed_twice"]),
                     "censored": a["up_censored"] + a["down_censored"],
                     "unknown_minutes_all_series": int(g.unknown_minutes.sum()),
                     "implementation_exact": bool(g[g.policy.isin(CUSTOM)].implementation_exact.all()),
                     "scale_calls_failed": int(g[g.policy.isin(CUSTOM)].scale_calls_failed.sum()),
                     "collector_errors": ce,
                     "fetch": "recovered, sha256 verified" if "fetched_recovered" in reasons else "direct"})
    df = pd.DataFrame(rows)
    save_table(df, "T01_campaign", "Main runs: schedule, assignment and validity checks (PROTOCOL Sect. 7). Scale-down attribution as computed by the frozen analyze.py and after the post-freeze correction of overlapping windows (attribution_fix.py; one pod in run2 was attributed twice).")
    return df


def t02_parameters() -> pd.DataFrame:
    p, act = plan(), jload(TB / "actuator_model.json")
    cap = p["capacity"]
    cl = jload(RES / "run1" / "cluster.json")
    rows = [
        ("Provider / service", "VK Cloud Managed Kubernetes, zone GZ1 (Moscow)"),
        ("Kubernetes", cl["version"]["gitVersion"]),
        ("Nodes", "5 x STD3-4-8 (Intel Xeon Icelake, 4 vCPU, 8 GB), 1 driver + 4 workers; allocatable 3920m CPU"),
        ("Container runtime / OS", "cri-o 1.35.5 / AlmaLinux 9.7; CNI Calico"),
        ("Functions", f"{len(p['functions'])} (Huawei 2023 confirmatory periods; terciles of burstiness)"),
        ("Test days", "10 and 11 (1,440 test minutes after 30 warm-up minutes)"),
        ("Pod", f"CPU request = limit = {p['pod_cpu_millicores']}m; ITER = {cap['iter']}; warm-up 15 s; readiness probe 2 s"),
        ("Capacity mu", f"{cap['rate_per_second']} requests/s per pod ({int(cap['mu_phys_per_minute'])}/min), verified under amendment A1; r* = {cap['r_star']}"),
        ("CPU utilisation at mu, c", f"{cap['cpu_utilisation_at_capacity']} (CPU per request implied: {cap['cpu_utilisation_at_capacity'] * p['pod_cpu_millicores'] / cap['rate_per_second']:.3f} ms)"),
        ("Risk level delta / budget", f"{p['delta']} -> {BUDGET} overload minutes per function-run (H2: <= 14)"),
        ("PAC-h", f"tau = {p['tau']}, eta = {p['eta']}, W = 1440 (periods A, B) or 2880 (C)"),
        ("HPA", "default behaviour; target round(100 u c) = 19-37%; minReplicas 1, maxReplicas 40"),
        ("Actuator model (pilot fit)", f"d_up = {act['d_up_s']:.2f} s, d_down = {act['d_down_s']:.2f} s (n = {act['n_up']}/{act['n_down']})"),
        ("Bad request", "503, timeout (3 s), connection error, or successful response > 500 ms"),
        ("Freeze", "local branch testbed-cloud, commits 0971244 / 1211dc1; 123 input hashes verified before every run"),
    ]
    df = pd.DataFrame(rows, columns=["parameter", "value"])
    save_table(df, "T02_parameters", "Testbed and policy parameters (fixed before the main runs).")
    return df


def t03_functions() -> pd.DataFrame:
    p = plan()
    rows = []
    for f in p["functions"]:
        rec = {"function": f["name"], "period": f["run"].split("_")[0], "unit": f["unit_id"], "stratum": f["stratum"],
               "burstiness": f["burstiness"], "W": f["W"], "b1_u": f["b1_threshold"], "b1_cooldown": f["b1_cooldown"],
               "hpa_target_pct": f["hpa_target_cpu_percent"]}
        for d in (10, 11):
            dem = pd.read_csv(TB / "plan" / f"{f['name']}_run_d{d}.csv").demand.to_numpy(float)[WARM:MINUTES]
            rec[f"d{d}_mean_pods"] = round(float(np.mean(dem) / MU), 2)
            rec[f"d{d}_peak_pods"] = int(np.ceil(dem.max() / MU))
            rec[f"d{d}_requests_M"] = round(float(dem.sum()) / 1e6, 3)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["_n"] = df.function.str[1:].astype(int)
    df = df.sort_values("_n").drop(columns="_n")
    save_table(df, "T03_functions", "The 12 functions: selection stratum, policy parameters and test-day demand in pods of capacity mu.")
    return df


def t04_h1() -> pd.DataFrame:
    c = combined()["h1"]
    rows = []
    for pol in CUSTOM:
        g = PF[PF.policy == pol]
        h = c[pol]
        rr = h["ready_ratio_by_run"]
        rows.append({"policy": LABEL[pol], "function_runs": h["pairs"], "within": h["within"], "outside": h["outside"],
                     "undetermined": h["undetermined"], "share_within": round(h["share_within"], 3),
                     "criterion_met": h["criterion_met"],
                     "ready_ratio_min": round(min(v[0] for v in rr.values()), 4), "ready_ratio_max": round(max(v[1] for v in rr.values()), 4),
                     "overload_model": h["model_overload"], "overload_observed_lo": h["observed_overload_lo"],
                     "overload_observed_hi": h["observed_overload_hi"], "overload_fifo_tau0": h["fifo_tau0_overload"],
                     "overload_fifo_tau1": h["fifo_tau1_overload"],
                     "h1_overload_within": int((g.h1_overload == "within").sum()), "h1_ready_within": int((g.h1_ready == "within").sum()),
                     "fp_minutes": int(g.fp_minutes.sum()), "fn_minutes": int(g.fn_minutes.sum()),
                     "f1_known_mean": round(float(g.f1_known.mean()), 3), "capacity_mae_pods_mean": round(float(g.capacity_mae_pods.mean()), 4)})
    df = pd.DataFrame(rows)
    save_table(df, "T04_h1", "H1 (actuator model): function-runs within tolerance, ready pod-minute ratios, and overload minutes of the model, the observation and the FIFO references (sums over 48 function-runs).")
    return df


def t05_h2() -> pd.DataFrame:
    rows = []
    for pol in POLICIES:
        for r in RUNS + ["all"]:
            g = PF[(PF.policy == pol) & ((PF.run == r) if r != "all" else True)]
            vc = g.h2.value_counts()
            rows.append({"policy": LABEL[pol], "run": r, "node": g.node.iloc[0] if r != "all" else "all",
                         "compliant": int(vc.get("compliant", 0)), "undetermined": int(vc.get("undetermined", 0)),
                         "violated": int(vc.get("violated", 0)), "overload_lo": int(g.overload_lo.sum()),
                         "overload_hi": int(g.overload_hi.sum()), "max_per_function_lo": int(g.overload_lo.max()),
                         "max_per_function_hi": int(g.overload_hi.max()),
                         "function_runs_with_overload": int((g.overload_lo > 0).sum()),
                         "budget_excess_min": int(np.maximum(g.overload_lo - 14, 0).sum())})
    df = pd.DataFrame(rows)
    save_table(df, "T05_h2", "H2 (overload budget, at most 14 of 1,440 minutes per function-run) and overload minutes per policy and run.")
    return df


def t06_h3() -> tuple[pd.DataFrame, pd.DataFrame]:
    h3 = combined()["h3"]
    rows = []
    for k, v in h3.items():
        w, b = v["sign_test_worst_case"], v["sign_test_best_case"]
        rows.append({"comparison": k.replace("pach_vs_", "PAC-h vs ").replace("conf", "Conformal").replace("b1", "B1"),
                     "favouring_pach": v["functions_certainly_favouring_pach"], "against": v["functions_certainly_against"],
                     "undetermined": v["functions_undetermined"],
                     "worst_case_counts": f"{w['favouring_pach']}/{w['against']}/{w['ties']}", "worst_p": w["p_one_sided"], "worst_p_holm": w["p_holm"],
                     "best_case_counts": f"{b['favouring_pach']}/{b['against']}/{b['ties']}", "best_p": b["p_one_sided"], "best_p_holm": b["p_holm"],
                     "mean_diff_lo": v["mean_difference_interval"][0], "mean_diff_hi": v["mean_difference_interval"][1],
                     "bootstrap95_lo": v["bootstrap_95_outer"][0], "bootstrap95_hi": v["bootstrap_95_outer"][1],
                     "conclusion": v["conclusion"]})
    df = pd.DataFrame(rows)
    save_table(df, "T06_h3", "H3 (exploratory): one-sided sign tests with the function as unit (counts favouring/against/ties), Holm-adjusted, and the mean difference in overload minutes per function-run with the outer bootstrap interval.",
               {"worst_p": "{:.4f}", "worst_p_holm": "{:.4f}", "best_p": "{:.4f}", "best_p_holm": "{:.4f}", "mean_diff_lo": "{:.2f}",
                "mean_diff_hi": "{:.2f}", "bootstrap95_lo": "{:.2f}", "bootstrap95_hi": "{:.2f}"})
    lo = PF.groupby(["function", "policy"]).overload_lo.mean().unstack()
    hi = PF.groupby(["function", "policy"]).overload_hi.mean().unstack()
    strata = {f["name"]: f["stratum"] for f in plan()["functions"]}
    fr = []
    for f in sorted(lo.index, key=lambda s: int(s[1:])):
        rec = {"function": f, "stratum": strata[f]}
        for p in POLICIES:
            rec[f"{p}_mean"] = f"{lo.loc[f, p]:.2f}" if lo.loc[f, p] == hi.loc[f, p] else f"{lo.loc[f, p]:.2f}-{hi.loc[f, p]:.2f}"
        for o in ("conf", "b1"):
            rec[f"pach_minus_{o}"] = f"[{lo.loc[f, 'pach'] - hi.loc[f, o]:.2f}, {hi.loc[f, 'pach'] - lo.loc[f, o]:.2f}]"
        fr.append(rec)
    dfb = pd.DataFrame(fr)
    save_table(dfb, "T06b_h3_by_function", "Overload minutes per function-run averaged over the four runs ([lo, hi] where unknown minutes exist) and the admissible differences PAC-h minus comparator.")
    return df, dfb


def t07_cost() -> pd.DataFrame:
    rows = []
    for r in RUNS:
        g = PF[PF.run == r]
        base = g[g.policy == "pach"].ready_pod_minutes_lo.sum()
        pods = pd.read_csv(DATA / f"pod_nodes_{r}.csv").set_index("namespace").pods
        for pol in POLICIES:
            x = g[g.policy == pol]
            rows.append({"policy": LABEL[pol], "run": r, "node": x.node.iloc[0],
                         "ready_pod_minutes": round(x.ready_pod_minutes_lo.sum()), "existing_pod_minutes": round(x.existing_pod_minutes_known.sum()),
                         "terminating_pod_minutes": round(x.terminating_pod_minutes_known.sum(), 1),
                         "zero_overload_reference": round(x.clairvoyant_pod_minutes.sum()),
                         "ready_over_reference": round(x.ready_pod_minutes_lo.sum() / x.clairvoyant_pod_minutes.sum(), 3),
                         "vs_pach_pct": round(100 * (x.ready_pod_minutes_lo.sum() / base - 1), 2),
                         "vs_b1_pct": round(100 * (x.ready_pod_minutes_lo.sum() / g[g.policy == "b1"].ready_pod_minutes_lo.sum() - 1), 2),
                         "pods_created": int(pods.get(pol, 0))})
    df = pd.DataFrame(rows)
    tot = df.groupby("policy", sort=False)[["ready_pod_minutes", "existing_pod_minutes", "zero_overload_reference"]].mean()
    mean = tot.reset_index()
    mean["ready_over_reference"] = (tot.ready_pod_minutes / tot.zero_overload_reference).round(3).values
    mean["vs_pach_pct"] = (100 * (tot.ready_pod_minutes / tot.loc["PAC-h", "ready_pod_minutes"] - 1)).round(2).values
    mean["vs_b1_pct"] = (100 * (tot.ready_pod_minutes / tot.loc["B1", "ready_pod_minutes"] - 1)).round(2).values
    mean.insert(1, "run", "mean")
    mean.insert(2, "node", "all")
    df = pd.concat([df, mean], ignore_index=True)
    df["pods_created"] = df.pods_created.astype("Int64")
    for c in ("ready_pod_minutes", "existing_pod_minutes", "zero_overload_reference"):
        df[c] = df[c].round().astype(int)
    save_table(df, "T07_cost", "Pod-time per policy and run (resource proxies, not cloud cost): ready, existing and terminating pod-minutes; zero-overload clairvoyant reference = sum over minutes of max(1, ceil(requests/mu)), i.e. the cheapest integer replica count per minute with no proxy overload (not a lower bound for policies allowed 14 overload minutes or sub-minute changes); differences relative to PAC-h and to B1 (base = the named policy). Mean rows: mean pod-minutes over the runs and ratios of these means.")
    return df


def t08_users() -> pd.DataFrame:
    rows = []
    for r in RUNS:
        h = pd.read_csv(DATA / f"latency_hist_{r}.csv").set_index("policy")
        s = summary(r)
        for pol in POLICIES:
            x = h.loc[pol]
            n = x.dispatched
            slow = x[["ok_le1000", "ok_le2000", "ok_gt2000"]].sum()
            pv = s["policies"][pol]["proxy_vs_users"]
            rows.append({"policy": LABEL[pol], "run": r, "node": meta(r)["assignment"][pol], "requests": int(n),
                         "bad_pct": round(100 * x.bad / n, 4), "slow_gt500ms_pct": round(100 * slow / n, 4),
                         "shed_503_pct": round(100 * x.shed / n, 4), "timeout_pct": round(100 * x.timeout / n, 4),
                         "conn_error_pct": round(100 * x.failed / n, 4), "ok_p50_ms": bucket_q(x, 0.5), "ok_p95_ms": bucket_q(x, 0.95),
                         "ok_p99_ms": bucket_q(x, 0.99), "minutes_over_bad": pv["over_bad"], "minutes_over_ok": pv["over_ok"],
                         "minutes_free_bad": pv["free_bad"], "minutes_free_ok": pv["free_ok"]})
    df = pd.DataFrame(rows)
    save_table(df, "T08_users", "User metrics over all requests of the test minutes, and minutes by proxy status (overloaded / not) against a bad-request share above 1% (unknown minutes excluded). Latency quantiles are histogram bucket bounds.")
    return df


def t09_delays() -> tuple[pd.DataFrame, pd.DataFrame]:
    act = jload(TB / "actuator_model.json")
    d = pd.concat([delays(r) for r in RUNS], ignore_index=True)
    up = d[d.kind == "up"].copy()
    up["endpoint_mid"] = (up.to_endpoint_lo + up.to_endpoint) / 2
    dn = pd.read_csv(DATA / "delays_down_corrected_all_runs.csv.gz")
    hp = d[d.kind == "hpa_pod"]

    def q(s, p):
        return round(float(s.quantile(p)), 2)

    rows = []
    for name, sel in [(r, r) for r in RUNS] + [("pooled", None)]:
        u = up if sel is None else up[up.run == sel]
        w = dn if sel is None else dn[dn.run == sel]
        hh = hp if sel is None else hp[hp.run == sel]
        rows.append({"run": name, "n_up": len(u), "created_med": q(u.to_created, .5), "started_med": q(u.to_started, .5),
                     "ready_p10": q(u.to_ready, .1), "ready_med": q(u.to_ready, .5), "ready_p90": q(u.to_ready, .9),
                     "endpoint_mid_p10": q(u.endpoint_mid, .1), "endpoint_mid_med": q(u.endpoint_mid, .5), "endpoint_mid_p90": q(u.endpoint_mid, .9),
                     "endpoint_mid_p99": q(u.endpoint_mid, .99), "endpoint_mid_max": round(float(u.endpoint_mid.max()), 2),
                     "first_served_med": q(u.to_first_served, .5), "first_served_p90": q(u.to_first_served, .9),
                     "n_down": len(w), "removed_mid_med": q(w.removed_mid, .5), "removed_mid_p90": q(w.removed_mid, .9),
                     "deleted_med": q(w.to_deleted, .5), "n_hpa_pods": len(hh), "hpa_ready_from_creation_med": q(hh.to_ready, .5)})
    df = pd.DataFrame(rows)
    df.attrs["model"] = act
    save_table(df, "T09_delays", f"Actuation delays (s) from the start of the scale call: scale-ups of the custom controllers (midpoint of the endpoint observation interval), scale-downs (leaving the endpoints), and HPA pods from creation. Model fitted on the pilot: d_up = {act['d_up_s']:.2f} s, d_down = {act['d_down_s']:.2f} s. Scale-downs use the corrected non-overlapping attribution (one duplicate record of run2 removed; post-freeze correction of a derived quantity).")
    pp = []
    for pol in CUSTOM:
        u, w = up[up.policy == pol], dn[dn.policy == pol]
        pp.append({"policy": LABEL[pol], "n_up": len(u), "endpoint_mid_med": q(u.endpoint_mid, .5), "endpoint_mid_p90": q(u.endpoint_mid, .9),
                   "endpoint_mid_p99": q(u.endpoint_mid, .99), "first_served_med": q(u.to_first_served, .5), "n_down": len(w),
                   "removed_mid_med": q(w.removed_mid, .5), "removed_mid_p90": q(w.removed_mid, .9)})
    dfp = pd.DataFrame(pp)
    save_table(dfp, "T09b_delays_by_policy", "Actuation delays (s) per custom controller, pooled over the four runs (scale-downs with the corrected attribution).")
    up[["run", "policy", "function", "minute", "to_created", "to_scheduled", "to_started", "to_ready", "to_endpoint_lo", "to_endpoint",
        "endpoint_mid", "to_first_served"]].to_csv(DATA / "delays_up_all_runs.csv.gz", index=False)
    dn[["run", "policy", "function", "uid", "minute", "to_deleted", "to_endpoint_removed_lo", "to_endpoint_removed", "removed_mid"]].to_csv(
        DATA / "delays_down_all_runs.csv.gz", index=False)
    return df, dfp


def calib_cpu_per_node() -> dict:
    p2 = jload(TB / "capacity.json")["measurements"]["phase2"]
    return {w: float(np.mean([x["cpu_millicores"] for x in v["repeats"] if x["rate"] == 4])) / 4 for w, v in p2.items()}


def t10_cpu() -> tuple[pd.DataFrame, pd.DataFrame]:
    cal = calib_cpu_per_node()
    common_cal = summary("run1")["cpu_ms_per_request_at_calibration"]
    rows = []
    for r in RUNS:
        s = summary(r)
        for pol in POLICIES:
            node = meta(r)["assignment"][pol]
            v = s["cpu_ms_per_request"][pol]
            rows.append({"run": r, "policy": LABEL[pol], "node": node, "cpu_ms_per_request": round(v, 3),
                         "vs_common_calibration_pct": round(100 * (v / common_cal - 1), 1),
                         "node_calibration_ms": round(cal[node], 3), "vs_node_calibration_pct": round(100 * (v / cal[node] - 1), 1)})
    df = pd.DataFrame(rows)
    save_table(df, "T10_cpu_per_request", f"CPU time per request (median over test minutes with at least 60 requests) against the common calibration value {common_cal:.3f} ms and the node's own phase-2 value at 4 requests/s.")
    piv = df.pivot(index="node", columns="run", values="cpu_ms_per_request")
    piv["mean"] = piv.mean(axis=1).round(3)
    piv["calibration_phase2"] = [round(cal[n], 3) for n in piv.index]
    pol = df.pivot(index="node", columns="run", values="policy")
    piv = piv.reset_index()
    for r in RUNS:
        piv[r] = [f"{v:.2f} ({p})" for v, p in zip(piv[r], pol[r].reindex(piv.node))]
    save_table(piv, "T10b_cpu_by_node", "CPU time per request (ms) by worker and run (policy on that worker in parentheses).")
    return df, piv


def t11_nodes() -> pd.DataFrame:
    snap = sorted(RES.glob("ops/node_snapshot_*/snapshot.jsonl"))[-1]
    lab = {v: k for k, v in {"w1": "testbed-workers-4", "w2": "testbed-workers-2", "w3": "testbed-workers-1", "w4": "testbed-workers-3",
                             "load": "testbed-workers-0"}.items()}
    rows = []
    for line in open(snap):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        rows.append({"role": lab[d["node"]], "node": d["node"], "cpu_model": d["model_name"], "mhz_reported": round(float(np.mean(d["cpu_mhz"]))),
                     "uptime_h": round(d["uptime_s"] / 3600, 1), "steal_since_boot_pct": round(100 * d["steal_fraction_since_boot"], 3),
                     "steal_during_bench_pct": round(100 * d["steal_fraction_during_bench"], 3),
                     "sha256_per_s_median": round(float(np.median(d["sha256_per_s_single_thread"]))),
                     "sha256_per_s_min": round(float(np.min(d["sha256_per_s_single_thread"]))),
                     "sha256_per_s_max": round(float(np.max(d["sha256_per_s_single_thread"])))})
    df = pd.DataFrame(rows).sort_values("role")
    w = df[df.role.isin(["w1", "w2", "w3"])].sha256_per_s_median.mean()
    df["vs_mean_w1_w3_pct"] = (100 * (df.sha256_per_s_median / w - 1)).round(1)
    save_table(df, "X06_node_snapshot", f"Post hoc, outside the protocol: node snapshot after the campaign ({snap.parent.name}; descriptive, outside the protocol): CPU model, steal time since boot (covers all main runs, does not exclude short pauses) and a single-thread SHA-256 benchmark (5 x 10 s).")
    return df


def t12_repeatability() -> pd.DataFrame:
    rows = []
    for day, (a, b) in {10: ("run1", "run3"), 11: ("run2", "run4")}.items():
        for pol in POLICIES:
            x = PF[(PF.run == a) & (PF.policy == pol)].set_index("function").sort_index()
            y = PF[(PF.run == b) & (PF.policy == pol)].set_index("function").sort_index()
            rows.append({"day": day, "runs": f"{a}/{b}", "policy": LABEL[pol], "nodes": f"{x.node.iloc[0]}/{y.node.iloc[0]}",
                         "overload_lo": f"{int(x.overload_lo.sum())}/{int(y.overload_lo.sum())}",
                         "functions_identical_overload": int((x.overload_lo == y.overload_lo).sum()),
                         "mean_abs_diff_overload": round(float((x.overload_lo - y.overload_lo).abs().mean()), 3),
                         "ready_pod_minutes_diff_pct": round(100 * (y.ready_pod_minutes_lo.sum() / x.ready_pod_minutes_lo.sum() - 1), 2),
                         "bad_pct": f"{100 * (x.bad_request_fraction * x.requests).sum() / x.requests.sum():.3f}/{100 * (y.bad_request_fraction * y.requests).sum() / y.requests.sum():.3f}"})
    df = pd.DataFrame(rows)
    save_table(df, "T12_repeatability", "Repeatability (descriptive): runs replaying identical requests on different workers.")
    return df


def t13_hpa() -> pd.DataFrame:
    tgt = {f["name"]: f["hpa_target_cpu_percent"] for f in plan()["functions"]}
    rows = []
    for r in RUNS:
        h = pd.read_csv(DATA / f"hpa_{r}.csv")
        h["ratio"] = h.mean_cpu_utilisation / h.function.map(tgt)
        g = PF[PF.run == r]
        hp, b1 = g[g.policy == "hpa"], g[g.policy == "b1"]
        rows.append({"run": r, "node": meta(r)["assignment"]["hpa"], "mean_replicas_sum": round(h.mean_current.sum(), 1),
                     "max_replicas": int(h.max_current.max()), "share_at_maxReplicas": float(h.share_at_max.max()),
                     "utilisation_over_target_mean": round(float(h.ratio.mean()), 3), "scale_changes": int(h.scale_changes.sum()),
                     "overload_lo": int(hp.overload_lo.sum()), "h2_violated": int((hp.h2 == "violated").sum()),
                     "ready_pod_minutes": round(hp.ready_pod_minutes_lo.sum()),
                     "bad_pct": round(100 * (hp.bad_request_fraction * hp.requests).sum() / hp.requests.sum(), 4),
                     "b1_overload_lo": int(b1.overload_lo.sum()), "b1_ready_pod_minutes": round(b1.ready_pod_minutes_lo.sum())})
    df = pd.DataFrame(rows)
    save_table(df, "T13_hpa", "HPA per run (H4, descriptive) next to B1: replicas, mean CPU utilisation relative to the target, overload, pod-time and bad requests.")
    return df


def t14_proxy_curve() -> pd.DataFrame:
    edges = [0, .5, .7, .8, .9, 1.0, 1.1, 1.25, np.inf]
    labels = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", "1.0-1.1", "1.1-1.25", ">1.25"]
    frames = []
    for r in RUNS:
        pm = per_minute(r)
        pm = pm[(pm.minute >= WARM) & (pm.known_seconds >= 60) & (pm.ready_lo > 0)]
        pm = pm.assign(rho=pm.requests / (MU * pm.ready_lo))
        frames.append(pm[["run", "policy", "function", "minute", "requests", "bad", "rho"]])
    a = pd.concat(frames)
    a["bin"] = pd.cut(a.rho, edges, labels=labels, right=True)
    rows = []
    for pol in POLICIES:
        for b in labels:
            x = a[(a.policy == pol) & (a.bin == b)]
            if len(x) == 0:
                rows.append({"policy": LABEL[pol], "load_ratio": b, "minutes": 0, "bad_pct": np.nan, "share_minutes_bad_gt1pct": np.nan})
                continue
            rows.append({"policy": LABEL[pol], "load_ratio": b, "minutes": len(x), "bad_pct": round(100 * x.bad.sum() / x.requests.sum(), 3),
                         "share_minutes_bad_gt1pct": round(float((x.bad > 0.01 * x.requests).mean()), 3)})
    df = pd.DataFrame(rows)
    save_table(df, "T14_proxy_vs_users_curve", "Bad-request share as a function of the load ratio requests / (mu x ready capacity) per minute (fully observed minutes, all runs).")
    a.to_csv(DATA / "minutes_load_ratio_all_runs.csv.gz", index=False)
    return df


def x01_synchrony() -> pd.DataFrame:
    rows = []
    for r in RUNS:
        s = pd.read_csv(DATA / f"seconds_{r}.csv.gz")
        n = len(s)
        marg = {p: float((s[f"bad_{p}"] > 0).mean()) for p in POLICIES}
        bad = sum(s[f"bad_{p}"].sum() for p in POLICIES)
        req = sum(s[f"dispatched_{p}"].sum() for p in POLICIES)
        sync = s.policies_bad >= 3
        s["hour"] = (s.second - WARM * 60) // 3600
        hourly = sum(len(g) * np.prod([float((g[f"bad_{p}"] > 0).mean()) for p in POLICIES]) for _, g in s.groupby("hour"))
        req_out = sum(s.loc[~sync, f"dispatched_{p}"].sum() for p in POLICIES)
        bad_out = sum(s.loc[~sync, f"bad_{p}"].sum() for p in POLICIES)
        rec = {"run": r, "test_seconds": n, "seconds_all4_bad_observed": int((s.policies_bad == 4).sum()),
               "expected_const_daily_rates": round(float(np.prod(list(marg.values())) * n), 1),
               "expected_const_hourly_rates": round(float(hourly), 1),
               "seconds_ge3_bad": int(sync.sum()),
               "request_share_in_ge3_seconds": round(float(1 - req_out / req), 4),
               "bad_share_in_ge3_seconds": round(float(1 - bad_out / bad), 3),
               "bad_pct_all": round(100 * bad / req, 3),
               "residual_contribution_pct": round(100 * bad_out / req, 3),
               "bad_pct_among_requests_outside_ge3": round(100 * bad_out / req_out, 3)}
        for p in POLICIES:
            rec[f"{p}_bad_second_share"] = round(marg[p], 3)
        rows.append(rec)
    df = pd.DataFrame(rows)
    save_table(df, "X01_bad_second_synchrony", "Post hoc descriptive diagnostic: seconds in which bad requests occur simultaneously in several policies (each policy on a different worker; all policies receive identical request schedules from one load generator). Reference counts assume independence with constant per-policy rates over the day or within each hour; they ignore load and serial dependence and are not a test of a cause. residual_contribution_pct = bad requests outside seconds with >= 3 policies bad / all requests (a component of bad_pct_all); bad_pct_among_requests_outside_ge3 uses the requests outside those seconds as denominator. No request is removed from any protocol metric.")
    return df


def x02_scale_events() -> pd.DataFrame:
    rows = []
    for r in RUNS:
        pm = per_minute(r)
        pm = pm[pm.minute >= WARM]
        sc = pd.read_csv(DATA / f"scale_{r}.csv.gz")
        c = sc[sc.policy.isin(CUSTOM)].copy()
        c["minute"] = c.minute + 1
        sc2 = pd.concat([c, sc[sc.policy == "hpa"]])
        m = pm.merge(sc2[["policy", "function", "minute", "change"]], on=["policy", "function", "minute"], how="left")
        m["kind"] = np.select([m.change > 0, m.change < 0], ["after scale-up", "after scale-down"], "no change")
        m["run"] = r
        rows.append(m[["run", "policy", "kind", "requests", "bad"]])
    a = pd.concat(rows)
    g = a.groupby(["policy", "kind"]).agg(minutes=("bad", "size"), requests=("requests", "sum"), bad=("bad", "sum")).reset_index()
    g["bad_pct"] = (100 * g.bad / g.requests).round(3)
    tot = g.groupby("policy").bad.transform("sum")
    g["share_of_policy_bad"] = (g.bad / tot).round(3)
    g["policy"] = g.policy.map(LABEL)
    save_table(g, "X02_scale_events", "Exploratory: bad-request share in minutes in which the replica count of the function changed (custom controllers: the minute after the decision; HPA: observed change), pooled over the four runs.")
    return g


def x03_node_users() -> pd.DataFrame:
    rows = []
    for r in RUNS:
        s = pd.read_csv(DATA / f"seconds_{r}.csv.gz")
        sync = s.policies_bad >= 3
        sm = summary(r)
        for p in POLICIES:
            rows.append({"run": r, "node": meta(r)["assignment"][p], "policy": LABEL[p],
                         "bad_pct": round(100 * s[f"bad_{p}"].sum() / s[f"dispatched_{p}"].sum(), 3),
                         "residual_contribution_pct": round(100 * s.loc[~sync, f"bad_{p}"].sum() / s[f"dispatched_{p}"].sum(), 3),
                         "bad_pct_among_requests_outside_ge3": round(100 * s.loc[~sync, f"bad_{p}"].sum() / s.loc[~sync, f"dispatched_{p}"].sum(), 3),
                         "share_seconds_with_bad": round(float((s[f"bad_{p}"] > 0).mean()), 3),
                         "cpu_ms_per_request": round(sm["cpu_ms_per_request"][p], 3)})
    df = pd.DataFrame(rows)
    df["rank_in_run_bad_pct"] = df.groupby("run").bad_pct.rank().astype(int)
    save_table(df, "X03_node_users", "Post hoc descriptive: bad requests by worker and run (rank 1 = fewest in the run); residual contribution and conditional share defined as in X01.")
    return df


def x04_hourly() -> pd.DataFrame:
    rows = []
    for r in RUNS:
        t0 = meta(r)["t0"]
        pm = per_minute(r)
        pm = pm[pm.minute >= WARM]
        pm["hour"] = (pm.minute - WARM) // 60
        g = pm.groupby(["hour", "policy"]).agg(requests=("requests", "sum"), bad=("bad", "sum")).reset_index()
        g["run"] = r
        g["start_utc"] = pd.to_datetime(t0 + (WARM + 60 * g.hour) * 60, unit="s", utc=True)
        rows.append(g)
    df = pd.concat(rows, ignore_index=True)
    df["bad_pct"] = (100 * df.bad / df.requests).round(4)
    df.to_csv(DATA / "hourly_bad_by_policy.csv", index=False)
    s = df.groupby("run").agg(hours=("hour", "nunique"), bad=("bad", "sum"), requests=("requests", "sum")).reset_index()
    hp = df.groupby(["run", "hour"]).agg(bad=("bad", "sum"), requests=("requests", "sum")).reset_index()
    hp["pct"] = 100 * hp.bad / hp.requests
    s["bad_pct"] = (100 * s.bad / s.requests).round(3)
    s["hourly_median_pct"] = hp.groupby("run").pct.median().round(3).values
    s["hourly_p90_pct"] = hp.groupby("run").pct.quantile(.9).round(3).values
    s["hourly_max_pct"] = hp.groupby("run").pct.max().round(3).values
    s["start_msk"] = [msk(meta(r)["t0"] + WARM * 60) for r in s.run]
    save_table(s, "X04_drift", "Exploratory: bad-request share over all policies by run in chronological order, with the distribution over hours.")
    return s


def x05_cpu_hourly() -> pd.DataFrame:
    rows = []
    for r in RUNS:
        t0 = meta(r)["t0"]
        c = pd.read_csv(DATA / f"cpu_minutes_{r}.csv.gz").rename(columns={"namespace": "policy"})
        pm = per_minute(r)[["policy", "function", "minute", "requests"]]
        m = c.merge(pm, on=["policy", "function", "minute"])
        m = m[m.requests >= 60]
        m["ms_per_request"] = m.cpu_millicores * 60 / m.requests
        m["hour"] = (m.minute - WARM) // 60
        g = m.groupby(["policy", "hour"]).ms_per_request.median().reset_index()
        g["run"] = r
        g["node"] = g.policy.map(meta(r)["assignment"])
        g["start_utc"] = pd.to_datetime(t0 + (WARM + 60 * g.hour) * 60, unit="s", utc=True)
        rows.append(g)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(DATA / "hourly_cpu_by_node.csv", index=False)
    s = df.groupby(["node", "run"]).ms_per_request.agg(["median", "min", "max"]).round(3).reset_index()
    s["policy"] = [LABEL[[p for p, n in meta(r)["assignment"].items() if n == w][0]] for w, r in zip(s.node, s.run)]
    save_table(s, "X05_cpu_hourly_by_node", "Exploratory: hourly medians of CPU time per request (ms) by worker and run: median, minimum and maximum over the 24 hours.")
    return s


if __name__ == "__main__":
    for fn in (t01_campaign, t02_parameters, t03_functions, t04_h1, t05_h2, t06_h3, t07_cost, t08_users, t09_delays, t10_cpu, t11_nodes,
               t12_repeatability, t13_hpa, t14_proxy_curve, x01_synchrony, x02_scale_events, x03_node_users, x04_hourly, x05_cpu_hourly):
        fn()
        print("ok", fn.__name__, flush=True)
    PF.to_csv(DATA / "function_runs.csv", index=False)
