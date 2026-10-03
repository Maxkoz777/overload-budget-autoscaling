#!/usr/bin/env python3
"""Build minute series for the external replication (protocol in verified_results/external_replication_study/).

Huawei Cloud 2023 (private): per-function requests and recorded pods,
relative days 0-32 (= trace days 28-60).  Azure Functions 2019: per-application
invocations, days 0-13.  Selection and capacity units use pre-test days 0-7
only, exactly as fixed in audit/verified_results/external_replication_study/protocol.json.

Usage (from the paper repository):
    python3 audit/prepare_external_traces.py --family huawei2023
    python3 audit/prepare_external_traces.py --family azure2019 --stage days --days 1-5
    python3 audit/prepare_external_traces.py --family azure2019 --stage days --days 6-10
    python3 audit/prepare_external_traces.py --family azure2019 --stage days --days 11-14
    python3 audit/prepare_external_traces.py --family azure2019 --stage build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parents[1]
SHARED = RESEARCH / "shared-data"
OUT = PAPER.parent / "experiments" / "data" / "external_traces"
CACHE = Path(os.environ.get("EXT_CACHE", str(Path.home() / "ext_cache_overload_budget")))
M = 1440


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def huawei() -> None:
    raw = SHARED / "huawei_2023" / "private"
    first, last = 28, 60
    def load(metric):
        blocks, files = [], []
        for d in range(first, last + 1):
            p = raw / metric / f"day_{d:03d}.csv"
            df = pd.read_csv(p)
            assert len(df) == M, (metric, d, len(df))
            blocks.append(df.iloc[:, 2:].to_numpy(np.float64))
            files.append((str(p.relative_to(RESEARCH)), sha(p)))
            cols = list(df.columns[2:])
        return np.vstack(blocks).T, cols, files
    R, cols, fr = load("requests_minute")
    P, cols_p, fp = load("instances_minute")
    assert cols == cols_p
    pre = slice(0, 8 * M)
    req_present = (~np.isnan(R[:, pre])).mean(1)
    pods_present = (np.nan_to_num(P[:, pre]) >= 1).mean(1)
    Rz = np.nan_to_num(R, nan=0.0)
    Pz = np.nan_to_num(P, nan=0.0)
    mean_pre = Rz[:, pre].mean(1)
    elig = (req_present >= 0.99) & (mean_pre >= 1.0) & (pods_present >= 0.99)
    per_pod = np.where(Pz[:, pre] >= 1, Rz[:, pre] / np.maximum(Pz[:, pre], 1), np.nan)
    mu50 = np.nanmedian(per_pod, axis=1)
    mu90 = np.nanpercentile(per_pod, 90, axis=1)
    table = pd.DataFrame({"function": cols, "request_present_pre": req_present, "pods_present_pre": pods_present,
                          "mean_requests_pre": mean_pre, "mu_p50": mu50, "mu_p90": mu90, "eligible": elig})
    sel = table[table.eligible & (table.mu_p50 > 0)].reset_index(drop=True)
    sel.insert(0, "unit_id", [f"HW_{int(f):03d}" for f in sel.function])
    idx = [cols.index(f) for f in sel.function]
    out = OUT / "huawei2023"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "series.npz", demand=Rz[idx].astype(np.float64), observed=Pz[idx].astype(np.float64),
                        unit_id=sel.unit_id.to_numpy())
    table.to_csv(out / "eligibility.csv", index=False)
    sel.to_csv(out / "units.csv", index=False)
    (out / "inputs_sha256.json").write_text(json.dumps({"requests": fr, "instances": fp}, indent=1) + "\n")
    print(f"huawei2023: {int(elig.sum())} eligible of {len(cols)}; saved {len(sel)} units")


def azure_days(days: range) -> None:
    raw = SHARED / "azure_functions_2019" / "raw"
    CACHE.mkdir(parents=True, exist_ok=True)
    minute_cols = [str(i) for i in range(1, M + 1)]
    for k in days:
        p = raw / f"invocations_per_function_md.anon.d{k:02d}.csv"
        df = pd.read_csv(p, engine="pyarrow")  # the C parser segfaults on some day files
        df.columns = [str(c) for c in df.columns]
        vals = df[minute_cols].to_numpy(np.float32)
        g = pd.DataFrame(vals, index=df.HashApp).groupby(level=0).sum().astype(np.float32)
        tv = pd.DataFrame({"app": df.HashApp, "trigger": df.Trigger, "vol": vals.sum(1, dtype=np.float64)})
        tv = tv.groupby(["app", "trigger"]).vol.sum().reset_index()
        g.columns = [str(c) for c in g.columns]
        g.reset_index(names="app").to_parquet(CACHE / f"azure_app_day{k:02d}.parquet", index=False)
        tv.to_parquet(CACHE / f"azure_trigger_day{k:02d}.parquet", index=False)
        (CACHE / f"azure_sha_day{k:02d}.txt").write_text(f"{p.relative_to(RESEARCH)} {sha(p)}\n")
        print(f"day {k:02d}: {len(g)} apps", flush=True)
        del df, vals, g


def azure_build() -> None:
    day = lambda k: CACHE / f"azure_app_day{k:02d}.parquet"
    present = [set(pd.read_parquet(day(k), columns=["app"]).app) for k in range(1, 15)]
    full = sorted(set.intersection(*present))
    pos = {a: i for i, a in enumerate(full)}
    # pre-test statistics (days 0-7) streamed one day at a time
    ssum = np.zeros(len(full)); nz = np.zeros(len(full))
    for k in range(1, 9):
        f = pd.read_parquet(day(k)).set_index("app").reindex(full)
        v = f.to_numpy(np.float64)
        ssum += v.sum(1); nz += (v > 0).sum(1)
        del f, v
    mean_pre = ssum / (8 * M); nz_pre = nz / (8 * M)
    trig = pd.concat([pd.read_parquet(CACHE / f"azure_trigger_day{k:02d}.parquet") for k in range(1, 15)])
    trig = trig[trig.app.isin(pos)]
    tvol = trig.groupby(["app", "trigger"]).vol.sum().unstack(fill_value=0.0).reindex(full).fillna(0.0)
    dom = tvol.idxmax(axis=1)
    elig = (mean_pre >= 1.0) & (nz_pre >= 0.5)
    table = pd.DataFrame({"app": full, "mean_invocations_pre": mean_pre, "nonzero_pre": nz_pre,
                          "dominant_trigger": dom.to_numpy(), "eligible": elig})
    table["group"] = table.dominant_trigger.where(table.dominant_trigger.isin(["http", "timer", "queue"]), "other")
    sel = table[table.eligible].sort_values(["group", "app"]).reset_index(drop=True)
    sel.insert(0, "unit_id", [f"AZ_{i:04d}" for i in range(len(sel))])
    for K in (5, 10, 20):
        sel[f"mu_K{K}"] = sel.mean_invocations_pre / K
    X = np.zeros((len(sel), 14 * M), np.float64)
    for k in range(1, 15):
        f = pd.read_parquet(day(k)).set_index("app").reindex(sel.app)
        X[:, (k - 1) * M:k * M] = f.to_numpy(np.float64)
        del f
    assert np.isfinite(X).all()
    out = OUT / "azure2019"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "series.npz", demand=X, unit_id=sel.unit_id.to_numpy())
    table.to_csv(out / "eligibility.csv", index=False)
    sel.to_csv(out / "units.csv", index=False)
    shas = [(CACHE / f"azure_sha_day{k:02d}.txt").read_text().split() for k in range(1, 15)]
    (out / "inputs_sha256.json").write_text(json.dumps({"invocations": shas}, indent=1) + "\n")
    print(f"azure2019: {len(full)} apps in all days, {len(sel)} eligible", sel.group.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["huawei2023", "azure2019"])
    ap.add_argument("--stage", default="all", choices=["all", "days", "build"])
    ap.add_argument("--days", default="1-14")
    a = ap.parse_args()
    if a.family == "huawei2023":
        huawei()
    else:
        lo, hi = map(int, a.days.split("-"))
        if a.stage in ("all", "days"):
            azure_days(range(lo, hi + 1))
        if a.stage in ("all", "build"):
            azure_build()
