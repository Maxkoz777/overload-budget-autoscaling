#!/usr/bin/env python3
"""Analyse one testbed run (hypotheses and validity rules in PROTOCOL.md).

Inputs: the run's result folder (meta.json, load.csv, served.csv, counts.csv, pods.csv, hpa.csv,
cpu.csv, decisions_<policy>.csv), testbed/plan, and, except for the pilot, the frozen actuator
model testbed/actuator_model.json. Missing or inconsistent data are never filled in: unknown
capacity gives bounds on overload, and a run whose measurements are incomplete is marked invalid.

Outputs in the result folder: validity.json, per_minute.csv.gz, per_function.csv, delays.csv,
delay_attribution.csv, summary.json; with --fit-actuator (pilot only) also testbed/actuator_model.json.

Usage (from the repository root):
  python3 testbed/analyze.py RESULTS_DIR [--fit-actuator]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "testbed"
sys.path.insert(0, str(TB))
from policies import PACH, ReactiveB1  # noqa: E402
from render import MAX_REPLICAS  # noqa: E402

POLICIES = ("pach", "conf", "b1", "hpa")
CUSTOM = ("pach", "conf", "b1")
BOUNDS = (5, 10, 25, 50, 100, 250, 500, 1000, 2000)
OK_COLS = [f"ok_le{b}" for b in BOUNDS] + ["ok_gt2000"]
SLOW_COLS = ["ok_le1000", "ok_le2000", "ok_gt2000"]
MAX_GAP_S = 2.5
COVERAGE_MIN = 0.95
COVERAGE_MAX_BAD_MINUTES = 0.01
LAG_P99_MAX_MS = 1000.0
H1_OVERLOAD_TOL = 2
H1_POD_MINUTES = (0.95, 1.05)
BOUND_LOOKBACK_S = 10.0


def history_file(name: str, day) -> str:
    return f"{name}_pilot_history.csv" if day == "pilot" else f"{name}_history_d{day}.csv"


def run_file(name: str, day) -> str:
    return f"{name}_pilot.csv" if day == "pilot" else f"{name}_run_d{day}.csv"


def offline_decisions(policy: str, fn: dict, plan: dict, hist: list[float], demand: np.ndarray) -> np.ndarray:
    mu = plan["mu_phys_per_minute"]
    if policy == "b1":
        ctl = ReactiveB1(fn["b1_threshold"], fn["b1_cooldown"], mu, hist[-1])
    else:
        ctl = PACH(fn["W"], plan["delta"], plan["eta"], plan["tau"], mu, hist, "pac" if policy == "pach" else "conformal")
    return np.array([ctl.initial()] + [ctl.decide(d) for d in demand[:-1]], float)


def integrate(obs_t: np.ndarray, obs_v: np.ndarray, t0: float, minutes: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-minute integral of a piecewise-constant signal over its known time, and the known time.

    An observation holds until the next one if they are at most MAX_GAP_S apart, otherwise for one
    second; the rest of the time is unknown."""
    integral = np.zeros(minutes)
    known = np.zeros(minutes)
    if len(obs_t) == 0:
        return integral, known
    ends = np.append(obs_t[1:], obs_t[-1] + 1.0)
    ends = np.where(ends - obs_t <= MAX_GAP_S, ends, obs_t + 1.0)
    for a, b, v in zip(obs_t - t0, ends - t0, obs_v):
        a, b = max(a, 0.0), min(b, minutes * 60.0)
        while a < b:
            m = int(a // 60)
            e = min(b, (m + 1) * 60.0)
            integral[m] += v * (e - a)
            known[m] += e - a
            a = e
    return integral, known


def model_ready(dec: pd.DataFrame, r0: float, t_start: float, model: dict, t0: float, minutes: int) -> np.ndarray:
    """Per-minute mean ready pods predicted by the actuator model from the logged scale calls.

    A call's time is the start of its successful attempt (called_at; decided_at in logs without it)."""
    state = {"ready": int(round(r0)), "pending": [], "removals": []}
    timeline = [(t_start, state["ready"])]

    def advance(until: float) -> None:
        due = sorted([(x, 1) for x in state["pending"] if x <= until] + [(x, -1) for x in state["removals"] if x <= until])
        for x, step in due:
            state["ready"] += step
            timeline.append((x, state["ready"]))
        state["pending"] = [x for x in state["pending"] if x > until]
        state["removals"] = [x for x in state["removals"] if x > until]

    for t, k in sorted(zip(dec.command_at.to_numpy(float), dec.replicas.to_numpy(int))):
        advance(t)
        live = state["ready"] + len(state["pending"]) - len(state["removals"])
        if k > live:
            state["pending"] += [t + model["d_up_s"]] * (k - live)
        elif k < live:
            cancel = min(live - k, len(state["pending"]))
            state["pending"] = sorted(state["pending"])[:len(state["pending"]) - cancel]
            state["removals"] += [t + model["d_down_s"]] * (live - k - cancel)
    advance(np.inf)
    ts = np.array([x for x, _ in timeline], float)
    vs = np.array([v for _, v in timeline], float)
    order = np.argsort(ts, kind="stable")
    ts, vs = ts[order], vs[order]
    grid = t0 + np.arange(minutes * 60) + 0.5
    idx = np.searchsorted(ts, grid, side="right") - 1
    per_sec = np.where(idx >= 0, vs[np.clip(idx, 0, None)], np.nan)
    return per_sec.reshape(minutes, 60).mean(axis=1)


def replica_bound(dec: pd.DataFrame | None, t0: float, minutes: int, fallback: float) -> np.ndarray:
    """Upper bound on ready endpoints per minute for time without telemetry.

    For a custom policy: the largest replica count acknowledged by the API that was in effect at any
    time from BOUND_LOOKBACK_S before the minute to its end (ready endpoints never exceed the
    Deployment's replica count except while a removal is propagating; analyze.py checks this on the
    observed time). Without a decision log, or for the HPA, the bound is the HPA's maxReplicas."""
    bound = np.full(minutes, float(fallback))
    if dec is None:
        return bound
    d = dec[dec.status == "ok"].sort_values("command_at")
    t, k = d.command_at.to_numpy(float), d.acknowledged.to_numpy(float)
    for m in range(minutes):
        a, b = t0 + 60.0 * m - BOUND_LOOKBACK_S, t0 + 60.0 * (m + 1)
        before = k[t <= a]
        vals = list(k[(t > a) & (t < b)]) + ([before[-1]] if len(before) else [])
        if vals and len(before):
            bound[m] = max(vals)
    return bound


def neighbour_obs(tt: np.ndarray, x: float, side: str) -> float:
    """The observation before or after the one at time x (counts.csv rounds times to 1 ms)."""
    if np.isnan(x):
        return np.nan
    if side == "before":
        i = np.searchsorted(tt, x - 0.005, side="left") - 1
        return float(tt[i]) if i >= 0 else np.nan
    i = np.searchsorted(tt, x + 0.005, side="right")
    return float(tt[i]) if i < len(tt) else np.nan


def cpu_per_request(res: Path, per_minute: pd.DataFrame, t0: float, test: slice) -> dict:
    """Median CPU time per request (ms) per policy over test minutes with at least 60 requests.

    For every namespace, function and metrics snapshot (one API response), the pods' CPU usage is
    summed; the per-minute mean of these sums (ms of CPU per second) times 60, divided by the
    minute's requests. Files without a snapshot column (collector before 18 October 2026) are
    grouped into snapshots by gaps of more than 2 s between consecutive rows of a namespace."""
    if not (res / "cpu.csv").exists():
        return {}
    cpu = pd.read_csv(res / "cpu.csv")
    if cpu.empty:
        return {}
    if "snapshot" not in cpu:
        cpu = cpu.sort_values(["namespace", "t"])
        cpu["snapshot"] = cpu.groupby("namespace").t.transform(lambda t: (t.diff() > 2.0).cumsum())
    first = cpu.groupby(["namespace", "snapshot"]).t.transform("min")
    cpu["minute"] = np.floor((first - t0) / 60).astype(int)
    per_sample = cpu.groupby(["namespace", "function", "minute", "snapshot"]).millicores.sum().reset_index()
    per_min = per_sample.groupby(["namespace", "function", "minute"]).millicores.mean().rename("cpu").reset_index()
    pm = per_minute.rename(columns={"policy": "namespace"}).merge(per_min, on=["namespace", "function", "minute"])
    pm = pm[(pm.minute >= test.start) & (pm.minute < test.stop) & (pm.requests >= 60)]
    return {pol: float((g.cpu * 60 / g.requests).median()) for pol, g in pm.groupby("namespace")}


def quantile_from_buckets(counts: np.ndarray, q: float) -> float:
    total = counts.sum()
    if total == 0:
        return float("nan")
    i = int(np.searchsorted(np.cumsum(counts) / total, q))
    return float(BOUNDS[i]) if i < len(BOUNDS) else float("inf")


def f1(pred: np.ndarray, obs: np.ndarray) -> float:
    tp, fp, fn = int((pred & obs).sum()), int((pred & ~obs).sum()), int((~pred & obs).sum())
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--plan", default=str(TB / "plan"))
    ap.add_argument("--fit-actuator", action="store_true")
    a = ap.parse_args()
    res, pdir = Path(a.results), Path(a.plan)
    plan = json.loads((pdir / "plan.json").read_text())
    meta = json.loads((res / "meta.json").read_text())
    t0, minutes, warm, day = float(meta["t0"]), int(meta["minutes"]), int(meta["warmup_minutes"]), meta["day"]
    mu, delta = plan["mu_phys_per_minute"], plan["delta"]
    test = slice(warm, minutes)
    n_test = minutes - warm
    model = None if a.fit_actuator else json.loads((TB / "actuator_model.json").read_text())
    problems: list[str] = []

    load = pd.read_csv(res / "load.csv")
    load["minute"] = load.second // 60
    load["slow"] = load[SLOW_COLS].sum(axis=1)
    counts = pd.read_csv(res / "counts.csv")
    pods = pd.read_csv(res / "pods.csv")
    served = pd.read_csv(res / "served.csv") if (res / "served.csv").exists() else pd.DataFrame(columns=["target", "pod", "first_ok"])
    expected_targets = {f"{pol}-{fn['name']}" for pol in POLICIES for fn in plan["functions"]}
    present = set(load.target.unique())
    if present != expected_targets:
        problems.append(f"load.csv targets differ: missing {sorted(expected_targets - present)}")
    dup = load.duplicated(["second", "target"]).sum()
    if dup:
        problems.append(f"load.csv has {dup} duplicate rows")
    per_target_seconds = load.groupby("target").second.agg(["min", "max", "count"])
    short = per_target_seconds[(per_target_seconds["count"] < minutes * 60) | (per_target_seconds["max"] < minutes * 60 - 1)]
    if len(short):
        problems.append(f"load.csv incomplete for {len(short)} targets")
    lag = load[load.minute.between(warm, minutes - 1) & (load.dispatched > 0)].lag_ms_max
    lag_p99 = float(lag.quantile(0.99)) if len(lag) else float("nan")
    if not lag_p99 <= LAG_P99_MAX_MS:
        problems.append(f"dispatch lag p99 {lag_p99:.0f} ms exceeds {LAG_P99_MAX_MS:.0f} ms")

    rows, per_fn, delay_rows, attribution = [], [], [], []
    coverage_bad = {}
    for pol in POLICIES:
        for fn in plan["functions"]:
            c = counts[(counts.namespace == pol) & (counts.function == fn["name"])]
            obs_times = np.sort(c.t.unique())
            _, known_s = integrate(obs_times, np.ones(len(obs_times)), t0, minutes)
            frac_bad = float((known_s[test] / 60.0 < COVERAGE_MIN).mean())
            coverage_bad[f"{pol}-{fn['name']}"] = frac_bad
            if frac_bad > COVERAGE_MAX_BAD_MINUTES:
                problems.append(f"{pol}-{fn['name']}: telemetry coverage below {COVERAGE_MIN:.0%} in {frac_bad:.1%} of test minutes")
    for fn in plan["functions"]:
        name = fn["name"]
        hist = pd.read_csv(pdir / history_file(name, day)).demand.astype(float).tolist()
        schedule = pd.read_csv(pdir / run_file(name, day)).demand.to_numpy(float)[:minutes]
        for pol in POLICIES:
            target = f"{pol}-{name}"
            lm = load[load.target == target].groupby("minute")[["planned", "dispatched", "ok", "shed", "timeout", "failed", "slow"] + OK_COLS].sum()
            lm = lm.reindex(range(minutes))
            planned = lm.planned.to_numpy(float)
            dispatched = lm.dispatched.to_numpy(float)
            completed = (lm.ok + lm.shed + lm.timeout + lm.failed).to_numpy(float)
            if not np.array_equal(planned, schedule):
                problems.append(f"{target}: planned requests differ from the schedule")
            if not np.array_equal(dispatched, planned):
                problems.append(f"{target}: dispatched differ from planned in {int(np.sum(dispatched != planned))} minutes")
            if not np.array_equal(completed, dispatched, equal_nan=True):
                problems.append(f"{target}: {int(np.nansum(dispatched - completed))} requests without outcome")
            c = counts[(counts.namespace == pol) & (counts.function == name)].sort_values("t")
            tt = c.t.to_numpy(float)
            ready_int, known = integrate(tt, c.endpoints_ready.to_numpy(float), t0, minutes)
            exist_int, _ = integrate(tt, c.pods_existing.to_numpy(float), t0, minutes)
            term_int, _ = integrate(tt, c.pods_terminating.to_numpy(float), t0, minutes)
            unknown_s = 60.0 - known
            dec = None
            if pol in CUSTOM:
                dec = pd.read_csv(res / f"decisions_{pol}.csv")
                dec = dec[dec.function == name].sort_values("minute")
                dec["command_at"] = dec.called_at if "called_at" in dec else dec.decided_at
            cap_max = replica_bound(dec, t0, minutes, MAX_REPLICAS)
            obs_m = np.floor((tt - t0) / 60).astype(int)
            inside = (obs_m >= 0) & (obs_m < minutes)
            above = int((c.endpoints_ready.to_numpy(float)[inside] > cap_max[obs_m[inside]]).sum())
            if above:
                problems.append(f"{target}: {above} observations with more ready endpoints than the replica bound")
            ready_lo = ready_int / 60.0
            ready_hi = (ready_int + unknown_s * cap_max) / 60.0
            over_definite = dispatched > mu * ready_hi
            not_over = dispatched <= mu * ready_lo
            status = np.where(over_definite, 1, np.where(not_over, 0, -1))
            frame = pd.DataFrame({"function": name, "policy": pol, "minute": range(minutes), "requests": dispatched,
                                  "ready_mean_known": np.where(known > 0, ready_int / np.maximum(known, 1e-9), np.nan),
                                  "ready_lo": ready_lo, "ready_hi": ready_hi, "known_seconds": known,
                                  "existing_pod_seconds": exist_int, "terminating_pod_seconds": term_int,
                                  "overload_status": status,
                                  "bad": (lm.shed + lm.timeout + lm.failed + lm.slow).to_numpy(float)})
            st = status[test]
            lo, hi = int((st == 1).sum()), int((st != 0).sum())
            budget = delta * n_test
            rec = {"function": name, "policy": pol, "overload_lo": lo, "overload_hi": hi, "unknown_minutes": int((st == -1).sum()),
                   "h2": "compliant" if hi <= budget else ("violated" if lo > budget else "undetermined"),
                   "ready_pod_minutes_lo": float(ready_lo[test].sum()), "ready_pod_minutes_hi": float(ready_hi[test].sum()),
                   "existing_pod_minutes_known": float(exist_int[test].sum() / 60), "terminating_pod_minutes_known": float(term_int[test].sum() / 60),
                   "clairvoyant_pod_minutes": float(np.maximum(np.ceil(dispatched[test] / mu), 1).sum()),
                   "requests": int(np.nansum(dispatched[test]))}
            rec["bad_request_fraction"] = float(frame.bad.iloc[test].sum() / max(rec["requests"], 1))
            okb = lm[OK_COLS].iloc[test].sum().to_numpy(float)
            rec["ok_p95_ms"], rec["ok_p99_ms"] = quantile_from_buckets(okb, 0.95), quantile_from_buckets(okb, 0.99)
            rec["shed_fraction"] = float(lm.shed.iloc[test].sum() / max(rec["requests"], 1))
            rec["timeout_or_error_fraction"] = float((lm.timeout + lm.failed).iloc[test].sum() / max(rec["requests"], 1))
            known_m = st != -1
            badm = (frame.bad.iloc[test] > 0.01 * np.maximum(dispatched[test], 1)).to_numpy()
            rec["proxy_vs_users"] = {"over_bad": int(((st == 1) & badm).sum()), "over_ok": int(((st == 1) & ~badm).sum()),
                                     "free_bad": int(((st == 0) & badm).sum()), "free_ok": int(((st == 0) & ~badm).sum())}
            if pol in CUSTOM:
                expected_minutes = list(range(-1, minutes - 1))
                if dec.minute.tolist() != expected_minutes:
                    problems.append(f"{target}: decision log has {len(dec)} of {len(expected_minutes)} decisions")
                    rec["implementation_exact"] = False
                elif np.isnan(dispatched).any():
                    problems.append(f"{target}: decisions not checked, the load record is incomplete")
                    rec["implementation_exact"] = False
                else:
                    off = offline_decisions(pol, fn, plan, hist, dispatched)
                    logged_demand = dec.demand.iloc[1:].to_numpy(float)
                    rec["implementation_exact"] = bool(np.array_equal(dec.replicas.to_numpy(float), off)
                                                       and np.array_equal(logged_demand, dispatched[:-1]))
                    if not rec["implementation_exact"]:
                        problems.append(f"{target}: logged decisions differ from the offline rule")
                rec["scale_calls_failed"] = int((dec.status != "ok").sum())
                if len(dec) == len(expected_minutes):
                    k = dec.replicas.to_numpy(float)
                    for tau in (0, 1):
                        cap = np.concatenate([np.repeat(k[0], tau), k[:len(k) - tau]])
                        frame[f"fifo_tau{tau}_overload"] = dispatched > mu * cap
                        rec[f"fifo_tau{tau}_overload"] = int(frame[f"fifo_tau{tau}_overload"].iloc[test].sum())
                        rec[f"fifo_tau{tau}_ready_pod_minutes"] = float(cap[test].sum())
                    if model is not None:
                        first = c.iloc[0] if len(c) else None
                        r0 = float(first.endpoints_ready) if first is not None else float(k[0])
                        pred = model_ready(dec, r0, float(first.t) if first is not None else t0, model, t0, minutes)
                        frame["model_ready"] = pred
                        pred_over = (dispatched > mu * pred)[test]
                        p = int(pred_over.sum())
                        rec["model_overload"] = p
                        dev = max(abs(p - lo), abs(p - hi))
                        rec["h1_overload"] = "within" if dev <= H1_OVERLOAD_TOL else (
                            "outside" if min(abs(p - x) for x in range(lo, hi + 1)) > H1_OVERLOAD_TOL else "undetermined")
                        mp = float(pred[test].sum())
                        rlo, rhi = rec["ready_pod_minutes_lo"] / mp, rec["ready_pod_minutes_hi"] / mp
                        rec["h1_ready_ratio_lo"], rec["h1_ready_ratio_hi"] = rlo, rhi
                        inside = H1_POD_MINUTES[0] <= rlo and rhi <= H1_POD_MINUTES[1]
                        outside = rhi < H1_POD_MINUTES[0] or rlo > H1_POD_MINUTES[1]
                        rec["h1_ready"] = "within" if inside else ("outside" if outside else "undetermined")
                        rec["h1"] = "within" if rec["h1_overload"] == rec["h1_ready"] == "within" else (
                            "outside" if "outside" in (rec["h1_overload"], rec["h1_ready"]) else "undetermined")
                        kn = known_m
                        obs_over = (st == 1)
                        rec["capacity_mae_pods"] = float(np.nanmean(np.abs(frame.ready_mean_known.iloc[test] - pred[test])[known[test] >= 59]))
                        rec["fp_minutes"] = int((pred_over & ~obs_over & kn).sum())
                        rec["fn_minutes"] = int((~pred_over & obs_over & kn).sum())
                        rec["f1_known"] = f1(pred_over[kn], obs_over[kn])
                    ups = dec[dec.replicas.diff() > 0]
                    nxt = dec.command_at.shift(-1)
                    fpods = pods[(pods.namespace == pol) & (pods.function == name)]
                    removed = fpods.deleted.fillna(fpods.gone) if "gone" in fpods else fpods.deleted
                    first = dec.command_at.iloc[1] - 1.0
                    att = {"policy": pol, "function": name, "up_requested": int(dec.replicas.diff().clip(lower=0).sum()),
                           "down_requested": int(-dec.replicas.diff().clip(upper=0).sum()),
                           "pods_created_in_test": int((fpods.created >= first).sum()),
                           "pods_removed_in_test": int((removed >= first).sum()),
                           "up_attributed": 0, "up_censored": 0, "down_attributed": 0, "down_censored": 0}
                    for r in ups.itertuples():
                        until = nxt.loc[r.Index] if pd.notna(nxt.loc[r.Index]) else np.inf
                        new = fpods[(fpods.created >= r.command_at - 1.0) & (fpods.created < until - 1.0)]
                        for p in new.itertuples():
                            fs = served[(served.target == target) & (served.pod == p.pod)].first_ok
                            before = neighbour_obs(tt, p.endpoint_first, "before")
                            att["up_attributed"] += 1
                            att["up_censored"] += int(pd.isna(p.endpoint_first))
                            delay_rows.append({"policy": pol, "function": name, "kind": "up", "uid": p.uid, "minute": r.minute,
                                               "command_at": r.command_at,
                                               "to_created": p.created - r.command_at, "to_scheduled": p.scheduled - r.command_at,
                                               "to_started": p.started - r.command_at, "to_ready": p.ready - r.command_at,
                                               "to_endpoint_lo": max(before - r.command_at, 0.0) if pd.notna(before) else np.nan,
                                               "to_endpoint": p.endpoint_first - r.command_at,
                                               "to_first_served": (fs.iloc[0] - r.command_at) if len(fs) else np.nan})
                    downs = dec[dec.replicas.diff() < 0]
                    for r in downs.itertuples():
                        until = nxt.loc[r.Index] if pd.notna(nxt.loc[r.Index]) else np.inf
                        gone = fpods[(removed >= r.command_at - 1.0) & (removed < until)]
                        for p in gone.itertuples():
                            after = neighbour_obs(tt, p.endpoint_last, "after")
                            att["down_attributed"] += 1
                            att["down_censored"] += int(pd.isna(p.endpoint_last) or pd.isna(after))
                            delay_rows.append({"policy": pol, "function": name, "kind": "down", "uid": p.uid, "minute": r.minute,
                                               "command_at": r.command_at, "to_deleted": p.deleted - r.command_at,
                                               "to_endpoint_removed_lo": max(p.endpoint_last - r.command_at, 0.0) if pd.notna(p.endpoint_last) else np.nan,
                                               "to_endpoint_removed": (after - r.command_at) if pd.notna(after) else np.nan})
                    attribution.append(att)
            else:
                new = pods[(pods.namespace == pol) & (pods.function == name)]
                for p in new.itertuples():
                    fs = served[(served.target == target) & (served.pod == p.pod)].first_ok
                    delay_rows.append({"policy": pol, "function": name, "kind": "hpa_pod", "command_at": p.created,
                                       "to_scheduled": p.scheduled - p.created, "to_started": p.started - p.created,
                                       "to_ready": p.ready - p.created, "to_endpoint": p.endpoint_first - p.created,
                                       "to_first_served": (fs.iloc[0] - p.created) if len(fs) else np.nan})
            per_fn.append(rec)
            rows.append(frame)
    per_minute = pd.concat(rows, ignore_index=True)
    per_minute.to_csv(res / "per_minute.csv.gz", index=False)
    pf = pd.DataFrame(per_fn)
    pf.drop(columns=["proxy_vs_users"]).to_csv(res / "per_function.csv", index=False)
    dl = pd.DataFrame(delay_rows)
    dl.to_csv(res / "delays.csv", index=False)
    att = pd.DataFrame(attribution)
    att.to_csv(res / "delay_attribution.csv", index=False)
    dup_uids = int(dl[dl.kind.isin(["up", "down"])].duplicated(["kind", "uid"]).sum()) if len(dl) else 0
    validity = {"valid": not problems, "problems": problems, "dispatch_lag_p99_ms": lag_p99,
                "coverage_fraction_of_minutes_below_threshold": coverage_bad}
    (res / "validity.json").write_text(json.dumps(validity, indent=1) + "\n")

    if a.fit_actuator:
        if dl.empty or not {"up", "down"} <= set(dl.kind):
            raise SystemExit("cannot fit the actuator model: the run has no complete scale-up and scale-down records")
        if dup_uids:
            raise SystemExit(f"cannot fit the actuator model: {dup_uids} pods are attributed to more than one scale call")
        u = dl[dl.kind == "up"].dropna(subset=["to_endpoint_lo", "to_endpoint"])
        d = dl[dl.kind == "down"].dropna(subset=["to_endpoint_removed_lo", "to_endpoint_removed"])
        up_mid = (u.to_endpoint_lo + u.to_endpoint) / 2
        down_mid = (d.to_endpoint_removed_lo + d.to_endpoint_removed) / 2
        fitted = {"d_up_s": float(up_mid.median()), "d_down_s": float(down_mid.median()), "n_up": int(len(u)), "n_down": int(len(d)),
                  "d_up_interval_medians_s": [float(u.to_endpoint_lo.median()), float(u.to_endpoint.median())],
                  "d_down_interval_medians_s": [float(d.to_endpoint_removed_lo.median()), float(d.to_endpoint_removed.median())],
                  "up_quantiles": {q: float(up_mid.quantile(q)) for q in (0.1, 0.5, 0.9)},
                  "attribution": {k: int(v) for k, v in att.drop(columns=["policy", "function"]).sum().items()} if len(att) else {},
                  "rule": "scale-up: new pods become ready endpoints d_up_s after the scale call; scale-down: pending pods "
                          "are cancelled first (latest first), other pods leave the endpoints d_down_s after the call; "
                          "each delay is the midpoint of the interval between the last observation before and the first "
                          "observation after the change",
                  "source": str(res)}
        (TB / "actuator_model.json").write_text(json.dumps(fitted, indent=1) + "\n")
        print("actuator model:", json.dumps(fitted))

    summary = {"run": meta.get("run"), "day": day, "seed": meta.get("seed"), "assignment": meta.get("assignment"),
               "valid": validity["valid"], "budget_minutes": delta * n_test, "policies": {},
               "delay_attribution": {**({k: int(v) for k, v in att.drop(columns=["policy", "function"]).sum().items()} if len(att) else {}),
                                     "pods_attributed_twice": dup_uids}}
    for pol in POLICIES:
        g = pf[pf.policy == pol]
        s = {"h2": g.h2.value_counts().to_dict(), "overload_lo": g.overload_lo.tolist(), "overload_hi": g.overload_hi.tolist(),
             "ready_pod_minutes_lo": float(g.ready_pod_minutes_lo.sum()), "existing_pod_minutes_known": float(g.existing_pod_minutes_known.sum()),
             "bad_request_fraction": float((g.bad_request_fraction * g.requests).sum() / max(g.requests.sum(), 1)),
             "proxy_vs_users": {k: int(sum(d[k] for d in g.proxy_vs_users)) for k in ("over_bad", "over_ok", "free_bad", "free_ok")}}
        if pol in CUSTOM:
            s["implementation_exact"] = bool(g.implementation_exact.all())
            s["scale_calls_failed"] = int(g.scale_calls_failed.sum())
            if "h1" in g:
                s["h1"] = g.h1.value_counts().to_dict()
        summary["policies"][pol] = s
    cap = plan.get("capacity", {})
    summary["cpu_ms_per_request"] = cpu_per_request(res, per_minute, t0, test)
    if cap.get("rate_per_second"):
        summary["cpu_ms_per_request_at_calibration"] = cap["cpu_utilisation_at_capacity"] * plan["pod_cpu_millicores"] / cap["rate_per_second"]
    if len(dl):
        summary["delays_s"] = {k: {q: float(dl[dl.kind == "up"][k].quantile(v)) for q, v in (("p10", .1), ("median", .5), ("p90", .9))}
                               for k in ("to_created", "to_scheduled", "to_started", "to_ready", "to_endpoint_lo", "to_endpoint",
                                         "to_first_served") if k in dl}
    (res / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps(validity, indent=1))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
