#!/usr/bin/env python3
"""Build the untouched Huawei 2023 runs for the confirmatory replay (R8).

Contiguous runs of the private Huawei Cloud 2023 trace that no earlier analysis used
(the external replication used trace days 28-60): days 0-18, 147-165, 168-184.
Unit selection and capacity units follow prepare_external_traces.huawei() exactly,
applied to relative days 0-7 of each run.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent.parent
SHARED = RESEARCH / "shared-data"
OUT = PAPER.parent / "experiments" / "data" / "external_traces" / "huawei2023_confirmatory"
M = 1440
RUNS = {"A_days000_018": (0, 18), "B_days147_165": (147, 165), "C_days168_184": (168, 184)}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(name: str, first: int, last: int) -> None:
    raw = SHARED / "huawei_2023" / "private"
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
    Rz = np.nan_to_num(R, nan=0.0); Pz = np.nan_to_num(P, nan=0.0)
    mean_pre = Rz[:, pre].mean(1)
    elig = (req_present >= 0.99) & (mean_pre >= 1.0) & (pods_present >= 0.99)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_pod = np.where(Pz[:, pre] >= 1, Rz[:, pre] / np.maximum(Pz[:, pre], 1), np.nan)
    mu50 = np.nanmedian(per_pod, axis=1); mu90 = np.nanpercentile(per_pod, 90, axis=1)
    table = pd.DataFrame({"function": cols, "request_present_pre": req_present, "pods_present_pre": pods_present,
                          "mean_requests_pre": mean_pre, "mu_p50": mu50, "mu_p90": mu90, "eligible": elig})
    sel = table[table.eligible & (table.mu_p50 > 0)].reset_index(drop=True)
    sel.insert(0, "unit_id", [f"HW_{int(f):03d}" for f in sel.function])
    idx = [cols.index(f) for f in sel.function]
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "series.npz", demand=Rz[idx].astype(np.float64), observed=Pz[idx].astype(np.float64),
                        unit_id=sel.unit_id.to_numpy())
    table.to_csv(out / "eligibility.csv", index=False)
    sel.to_csv(out / "units.csv", index=False)
    (out / "inputs_sha256.json").write_text(json.dumps({"trace_days": [first, last], "requests": fr, "instances": fp}, indent=1) + "\n")
    print(f"{name}: {last-first+1} days, {int(elig.sum())} eligible of {len(cols)}; saved {len(sel)} units", flush=True)


if __name__ == "__main__":
    for n, (a, b) in RUNS.items():
        build(n, a, b)
