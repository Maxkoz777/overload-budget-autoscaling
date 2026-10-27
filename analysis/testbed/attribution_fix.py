"""Post-freeze correction of a derived quantity: attribution of removed pods to scale-down calls.

The frozen analyze.py attributes a removed pod to a scale-down call if its removal time lies in
[command_at - 1 s, next_command_at). Removal times have 1 s resolution, so this window overlaps the
next call's window by one second, and one pod (run2, B1, f9) was attributed to two calls. Scale-ups
already use the non-overlapping upper bound next_command_at - 1 s. This script applies the same
rule to scale-downs, recomputes the scale-down delay records exactly as analyze.py does otherwise,
and lists every difference to the original delays.csv. analyze.py, the raw files and the original
outputs are not changed; H1-H3 do not use these records."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CUSTOM, DATA, RES, RUNS, plan


def neighbour_after(tt: np.ndarray, x: float) -> float:
    if np.isnan(x):
        return np.nan
    i = np.searchsorted(tt, x + 0.005, side="right")
    return float(tt[i]) if i < len(tt) else np.nan


def downs(run: str, overlap: bool, cache: dict) -> tuple[list[dict], list[dict]]:
    if run not in cache:
        counts = pd.read_csv(RES / run / "counts.csv", usecols=["t", "namespace", "function"])
        cache[run] = ({k: np.sort(g.t.to_numpy(float)) for k, g in counts.groupby(["namespace", "function"])},
                      pd.read_csv(RES / run / "pods.csv"),
                      {pol: pd.read_csv(RES / run / f"decisions_{pol}.csv") for pol in CUSTOM})
    times, pods, decs = cache[run]
    rows, att = [], []
    for pol in CUSTOM:
        dec_all = decs[pol]
        for fn in plan()["functions"]:
            name = fn["name"]
            dec = dec_all[dec_all.function == name].sort_values("minute").copy()
            dec["command_at"] = dec.called_at if "called_at" in dec else dec.decided_at
            tt = times[(pol, name)]
            nxt = dec.command_at.shift(-1)
            fpods = pods[(pods.namespace == pol) & (pods.function == name)]
            removed = fpods.deleted.fillna(fpods.gone)
            first = dec.command_at.iloc[1] - 1.0
            n_att = 0
            for r in dec[dec.replicas.diff() < 0].itertuples():
                until = nxt.loc[r.Index] if pd.notna(nxt.loc[r.Index]) else np.inf
                upper = until if overlap else until - 1.0
                gone = fpods[(removed >= r.command_at - 1.0) & (removed < upper)]
                for p in gone.itertuples():
                    after = neighbour_after(tt, p.endpoint_last)
                    n_att += 1
                    rows.append({"run": run, "policy": pol, "function": name, "uid": p.uid, "minute": r.minute, "command_at": r.command_at,
                                 "to_deleted": p.deleted - r.command_at,
                                 "to_endpoint_removed_lo": max(p.endpoint_last - r.command_at, 0.0) if pd.notna(p.endpoint_last) else np.nan,
                                 "to_endpoint_removed": (after - r.command_at) if pd.notna(after) else np.nan})
            att.append({"run": run, "policy": pol, "function": name, "pods_removed_in_test": int((removed >= first).sum()), "down_attributed": n_att})
    return rows, att


if __name__ == "__main__":
    corr, orig_check, atts, diffs = [], [], [], []
    for run in RUNS:
        cache = {}
        c, a = downs(run, overlap=False, cache=cache)
        o, _ = downs(run, overlap=True, cache=cache)
        corr += c
        orig_check += o
        atts += a
        orig = pd.read_csv(RES / run / "delays.csv")
        orig = orig[orig.kind == "down"]
        oc = pd.DataFrame(o)
        k = ["policy", "function", "uid", "minute"]
        mm = orig.merge(oc, on=k, suffixes=("_o", "_n"))
        same = len(orig) == len(oc) == len(mm) and all(
            np.allclose(mm[f"{c}_o"].fillna(-1).to_numpy(), mm[f"{c}_n"].fillna(-1).to_numpy())
            for c in ("to_deleted", "to_endpoint_removed_lo", "to_endpoint_removed"))
        print(run, "re-derived original scale-down records match delays.csv:", same, flush=True)
        key = ["uid", "minute"]
        cc = pd.DataFrame(c)
        m = oc.merge(cc[key], on=key, how="left", indicator=True)
        for r in m[m._merge == "left_only"].itertuples():
            diffs.append({"run": run, "change": "removed", "policy": r.policy, "function": r.function, "uid": r.uid, "minute": r.minute,
                          "to_endpoint_removed": r.to_endpoint_removed, "to_deleted": r.to_deleted})
        m2 = cc.merge(oc[key], on=key, how="left", indicator=True)
        for r in m2[m2._merge == "left_only"].itertuples():
            diffs.append({"run": run, "change": "added", "policy": r.policy, "function": r.function, "uid": r.uid, "minute": r.minute,
                          "to_endpoint_removed": r.to_endpoint_removed, "to_deleted": r.to_deleted})
    cd = pd.DataFrame(corr)
    cd["removed_mid"] = (cd.to_endpoint_removed_lo + cd.to_endpoint_removed) / 2
    cd.to_csv(DATA / "delays_down_corrected_all_runs.csv.gz", index=False)
    at = pd.DataFrame(atts)
    s = at.groupby("run")[["pods_removed_in_test", "down_attributed"]].sum()
    s["unique_uids"] = cd.groupby("run").uid.nunique()
    s["attributed_twice"] = cd.groupby("run").uid.apply(lambda u: int(u.duplicated().sum()))
    s.reset_index().to_csv(DATA / "attribution_down_corrected.csv", index=False)
    pd.DataFrame(diffs).to_csv(DATA / "attribution_correction_diff.csv", index=False)
    print(s.to_string())
    print(pd.DataFrame(diffs).to_string())
