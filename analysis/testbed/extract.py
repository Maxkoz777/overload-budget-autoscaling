"""Reduce the raw run files (load.csv, hpa.csv, pods.csv, decisions_*.csv) to small derived files in data/.

Read-only on testbed/results. Run once; build_tables.py and build_figures.py read data/ and the analysed outputs."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import DATA, POLICIES, RES, RUNS, CUSTOM, WARM, MINUTES, meta

OKB = ["ok_le5", "ok_le10", "ok_le25", "ok_le50", "ok_le100", "ok_le250", "ok_le500", "ok_le1000", "ok_le2000", "ok_gt2000"]
SLOW = ["ok_le1000", "ok_le2000", "ok_gt2000"]


def load_reduce(run: str) -> None:
    cols = ["second", "target", "dispatched", "ok", "shed", "timeout", "failed", "lag_ms_max"] + OKB
    dt = {c: "int32" for c in cols if c not in ("target", "lag_ms_max")}
    dt["target"] = "category"
    dt["lag_ms_max"] = "float32"
    ld = pd.read_csv(RES / run / "load.csv", usecols=cols, dtype=dt)
    ld["policy"] = ld.target.str.split("-").str[0].astype("category")
    ld["bad"] = (ld.shed + ld.timeout + ld.failed + ld[SLOW].sum(axis=1)).astype("int32")
    test = (ld.second >= WARM * 60) & (ld.second < MINUTES * 60)
    t = ld[test]
    hist = t.groupby("policy", observed=True)[["dispatched", "ok", "shed", "timeout", "failed", "bad"] + OKB].sum()
    hist.insert(0, "run", run)
    hist.reset_index().to_csv(DATA / f"latency_hist_{run}.csv", index=False)
    sec = t.groupby(["second", "policy"], observed=True)[["dispatched", "bad"]].sum().unstack("policy").fillna(0).astype(int)
    sec.columns = [f"{a}_{b}" for a, b in sec.columns]
    nb = t[t.bad > 0].groupby("second").target.nunique().rename("targets_bad")
    npol = t[t.bad > 0].groupby("second").policy.nunique().rename("policies_bad")
    sec = sec.join(nb).join(npol).fillna(0)
    sec[["targets_bad", "policies_bad"]] = sec[["targets_bad", "policies_bad"]].astype(int)
    sec.reset_index().to_csv(DATA / f"seconds_{run}.csv.gz", index=False)
    tb = t.groupby(["target", "second"], observed=True).bad.sum().reset_index()
    tb = tb[tb.bad > 0]
    tb.to_csv(DATA / f"bad_target_seconds_{run}.csv.gz", index=False)
    print(run, "load reduced", len(ld), "rows", flush=True)


def hpa_reduce(run: str) -> None:
    h = pd.read_csv(RES / run / "hpa.csv")
    t0 = meta(run)["t0"]
    h["minute"] = np.floor((h.t - t0) / 60).astype(int)
    h = h[(h.minute >= WARM) & (h.minute < MINUTES)]
    g = h.groupby("function")
    out = pd.DataFrame({"mean_current": g.current.mean(), "max_current": g.current.max(), "min_current": g.current.min(),
                        "share_at_max": g.current.apply(lambda s: float((s >= 40).mean())),
                        "mean_cpu_utilisation": g.cpu_utilisation.mean(),
                        "utilisation_missing_share": g.cpu_utilisation.apply(lambda s: float(s.isna().mean())),
                        "scale_changes": g.current.apply(lambda s: int((s.diff().fillna(0) != 0).sum()))}).reset_index()
    out.insert(0, "run", run)
    out.to_csv(DATA / f"hpa_{run}.csv", index=False)
    pm = h.groupby(["function", "minute"]).agg(current=("current", "mean"), util=("cpu_utilisation", "mean")).reset_index()
    pm.to_csv(DATA / f"hpa_minutes_{run}.csv.gz", index=False)
    print(run, "hpa reduced", flush=True)


def pods_reduce(run: str) -> None:
    p = pd.read_csv(RES / run / "pods.csv")
    p = p[p.namespace.isin(POLICIES)]
    a = p.groupby(["namespace", "node"]).size().rename("pods").reset_index()
    a.insert(0, "run", run)
    a["assigned"] = a.namespace.map(meta(run)["assignment"])
    a.to_csv(DATA / f"pod_nodes_{run}.csv", index=False)
    print(run, "pods reduced", flush=True)


def scale_reduce(run: str) -> None:
    frames = []
    for pol in CUSTOM:
        d = pd.read_csv(RES / run / f"decisions_{pol}.csv").sort_values(["function", "minute"])
        d["change"] = d.groupby("function").replicas.diff()
        d["policy"] = pol
        frames.append(d[["policy", "function", "minute", "replicas", "change", "status"]])
    pm = pd.read_csv(DATA / f"hpa_minutes_{run}.csv.gz")
    pm = pm.sort_values(["function", "minute"])
    pm["replicas"] = pm.current.round()
    pm["change"] = pm.groupby("function").replicas.diff()
    pm["policy"] = "hpa"
    pm["status"] = "observed"
    frames.append(pm[["policy", "function", "minute", "replicas", "change", "status"]])
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "run", run)
    out.to_csv(DATA / f"scale_{run}.csv.gz", index=False)
    print(run, "scale reduced", flush=True)


def cpu_reduce(run: str) -> None:
    cpu = pd.read_csv(RES / run / "cpu.csv", usecols=["t", "snapshot", "namespace", "function", "millicores"])
    t0 = meta(run)["t0"]
    first = cpu.groupby(["namespace", "snapshot"]).t.transform("min")
    cpu["minute"] = np.floor((first - t0) / 60).astype(int)
    s = cpu.groupby(["namespace", "function", "minute", "snapshot"]).millicores.sum().reset_index()
    m = s.groupby(["namespace", "function", "minute"]).millicores.mean().rename("cpu_millicores").reset_index()
    m = m[(m.minute >= WARM) & (m.minute < MINUTES)]
    m.to_csv(DATA / f"cpu_minutes_{run}.csv.gz", index=False)
    print(run, "cpu reduced", flush=True)


if __name__ == "__main__":
    runs = sys.argv[1:] or RUNS
    for r in runs:
        load_reduce(r)
        hpa_reduce(r)
        pods_reduce(r)
        scale_reduce(r)
        cpu_reduce(r)
