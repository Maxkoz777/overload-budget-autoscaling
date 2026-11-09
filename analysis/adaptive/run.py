"""v1 runner. Usage: python3 run.py cal | test | confirm"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive.data import load, PAC_W, PAC_W_CONF, TRACES, PERIODS
from adaptive.env import applied, outcome
from adaptive import policies as P

OUT = Path("out_v1"); OUT.mkdir(exist_ok=True)
GAMMAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
BUDGETS = (0.01, 0.05); TAUS = (0, 1, 5)
BACC_V = {"P": {}, "V1": dict(c_mult=0.1), "V2": dict(c_mult=10.0), "V3": dict(limit=np.inf),
          "V4": dict(limit="hpa"), "V5": dict(A_mode="missed"), "V6": dict(trigger=False), "V9": dict(replan="mult")}


def calibrate(names, window) -> pd.DataFrame:
    rows = []
    for nm in names:
        D = load(nm, "cal"); T = D["T"]
        for d in BUDGETS:
            W = window[(nm, d)]
            tr = [(g, P.alpha_star(d, g, 0, T)) for g in GAMMAS]
            tr = [t for t in tr if t[1] > 0]
            R = P.aci(D, W, 0, tr)
            for k, (g, ast) in enumerate(tr):
                o = outcome(D, applied(D, R["dec"][k], 0, True))
                rows.append(dict(unit=nm, budget=d, gamma=g, alpha_star=ast,
                                 within=int((o["overload_fraction"] <= d + 1e-12).sum()),
                                 mean_cost=float(o["relative_cost"].mean())))
            print(nm, d, "calibrated", flush=True)
    cal = pd.DataFrame(rows)
    cal["order"] = cal.sort_values(["within", "mean_cost"], ascending=[False, True]).groupby(["unit", "budget"]).cumcount()
    return cal


def unit_rows(meta, D, o, extra=None):
    rows = []
    for u in range(len(D["uid"])):
        r = dict(meta, unit_id=D["uid"][u], overload_fraction=o["overload_fraction"][u], relative_cost=o["relative_cost"][u],
                 resource=o["resource"][u], churn=o["churn"][u], penalty=o["penalty"][u])
        if extra:
            r.update({k: v[u] for k, v in extra.items()})
        rows.append(r)
    return rows


def aci_extra(D, R, k, tau, cap_bound=True):
    inf = R["inf"][k]; nomv = R["nominal"][k]
    bound = (~inf) & (nomv > D["M"][:, None])
    prefix = np.zeros_like(inf)
    if not D["warm"]:
        prefix[:, :tau] = True
    o = outcome(D, applied(D, R["dec"][k], tau, cap_bound))
    ov = o["over"]
    return o, dict(formal_err=R["err"][k].mean(1), inf_min=inf.sum(1), empty_min=R["emp"][k].sum(1),
                   bound_min=bound.sum(1), over_inf=(ov & inf & ~prefix).sum(1), over_bound=(ov & bound & ~prefix).sum(1),
                   over_prefix=(ov & prefix).sum(1), over_regular=(ov & ~inf & ~bound & ~prefix).sum(1))


def replay(names, window, cal, tag):
    rows = []
    sel = {}
    for (nm, d), g in cal.groupby(["unit", "budget"]):
        sel[(nm, d)] = g.sort_values("order")[["gamma", "alpha_star"]].to_records(index=False).tolist()
    for nm in names:
        D = load(nm, "test"); T = D["T"]
        for d in BUDGETS:
            W = window[(nm, d)]
            ranked = [g for g, _ in sel[(nm, d)]]
            for tau in TAUS:
                t = time.time()
                base = dict(trace=nm, budget=d, tau=tau)
                dec = P.pac_h(D, W, d, tau)
                for cb in (False, True):
                    rows += unit_rows(dict(base, policy="PAC-h" + ("" if cb else " (no bound)")), D, outcome(D, applied(D, dec, tau, cb)))
                g_primary = ranked[0]
                g_r1 = next((g for g in ranked if P.alpha_star(d, g, tau, T) > 0), None)
                tracks = [(0.005, d)]; labels = ["ACI"]
                if P.alpha_star(d, g_primary, tau, T) > 0:
                    tracks.append((g_primary, P.alpha_star(d, g_primary, tau, T))); labels.append("ACI-G")
                if g_r1 is not None and g_r1 != g_primary:
                    tracks.append((g_r1, P.alpha_star(d, g_r1, tau, T))); labels.append("ACI-G (R1)")
                R = P.aci(D, W, tau, tracks)
                for k, lab in enumerate(labels):
                    o, ex = aci_extra(D, R, k, tau)
                    rows += unit_rows(dict(base, policy=lab, gamma=tracks[k][0], alpha_star=tracks[k][1]), D, o, ex)
                if "ACI-G" not in labels:
                    print(f"  {nm} d={d} tau={tau}: primary gamma {g_primary} has alpha*<=0 (not applicable); R1 gamma {g_r1}", flush=True)
                if "ACI-G (R1)" not in labels and g_r1 == g_primary and g_r1 is not None:
                    rows += [dict(r, policy="ACI-G (R1)") for r in rows if r.get("policy") == "ACI-G" and r["trace"] == nm and r["budget"] == d and r["tau"] == tau]
                for vn, kw in (BACC_V.items() if tag == "test" else [("P", {})]):
                    rb = P.bacc(D, d, tau, **kw)
                    o = outcome(D, applied(D, rb["dec"], tau, True))
                    rows += unit_rows(dict(base, policy="BACC-r" + ("" if vn == "P" else f" {vn}"), max_step=rb["max_step"]), D, o)
                print(f"{nm} d={d} tau={tau} done [{time.time()-t:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"{tag}_per_unit.csv.gz", index=False)
    return pd.DataFrame(rows)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    step = sys.argv[1]
    man = json.loads(Path("manifest_v1.json").read_text()) if step != "cal" and Path("manifest_v1.json").exists() else None
    if step == "cal":
        calibrate(TRACES, PAC_W).to_csv(OUT / "cal_traces.csv", index=False)
        calibrate(PERIODS, PAC_W_CONF).to_csv(OUT / "cal_periods.csv", index=False)
    elif step in ("test", "confirm"):
        assert man is not None, "freeze the manifest first"
        f = OUT / ("cal_traces.csv" if step == "test" else "cal_periods.csv")
        assert man["selected_parameters"][f.name] == sha(f), "calibration file changed after freeze"
        for k, h in man["code"].items():
            assert sha(k) == h, f"code changed after freeze: {k}"
        cal = pd.read_csv(f)
        if step == "test":
            replay(TRACES, PAC_W, cal, "test")
        else:
            replay(PERIODS, PAC_W_CONF, cal, "confirm")
    print("DONE", step, flush=True)
