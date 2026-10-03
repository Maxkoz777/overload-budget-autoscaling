"""C1 (supplement): ACI (B10) and weighted conformal (B11) vs fixed-delta (B6).
Replay on focused tier (20 services) x 4 forecasters x delta. No training.
Key question: does ACI recover compliance where fixed-delta ARIMA breaks?"""
import sys, json; sys.path.insert(0, "experiments/src")
from pathlib import Path
import numpy as np, pandas as pd
from exp_core import load_all_services, load_forecast, simulate_policy, SPLITS_PATH, _runlengths

RES = Path("experiments/results"); MODELS = ["persistence", "arima", "xgb", "lstm"]
DELTAS = [0.10, 0.05, 0.01]; W = 240; MU = 1.0; RHO = 0.70; H = 2
KAPPA = 0.01; LAM = 0.99
sd = json.load(open(SPLITS_PATH)); SERVICES = load_all_services(sd)
FC = {m: load_forecast(m) for m in MODELS if m != "persistence"}

def fc_arr(model, name, test):
    if model == "persistence": return None
    d = FC[model]; d = d[d.msname == name]
    return test[["timestamp"]].merge(d[["timestamp","forecast"]], on="timestamp", how="left")["forecast"].ffill().bfill().to_numpy(float)

def weighted_quantile(vals, weights, q):
    order = np.argsort(vals); v = np.asarray(vals)[order]; w = np.asarray(weights)[order]
    cw = np.cumsum(w); cw /= cw[-1]
    return float(v[np.searchsorted(cw, q, side="left").clip(0, len(v)-1)])

def simulate(history, test, fa, delta, mode):
    hist_y = history["cpu_sum"].to_numpy(float); test_y = test["cpu_sum"].to_numpy(float)
    prev = hist_y[-1]; residuals = list(np.maximum(hist_y[1:]-hist_y[:-1], 0.0))
    ol_hist, soft_hist, caps = [], [], []
    delta_t = delta
    for i in range(len(test_y)):
        demand = float(test_y[i]); forecast = max(prev,0.0) if fa is None else max(float(fa[i]),0.0)
        win = residuals[-W:]; w = len(win)
        if mode == "aci":
            lvl = min(w, int(np.ceil((w+1)*(1.0-delta_t)))); margin = float(sorted(win)[lvl-1]) if w else 0.0
        elif mode == "weighted":
            ages = np.arange(w)[::-1]; wts = LAM**ages
            margin = weighted_quantile(win, wts, 1.0-delta) if w else 0.0
        cap = max(int(np.ceil((forecast+margin)/MU)), 1)
        if len(ol_hist) >= H and (all(ol_hist[-H:]) or all(soft_hist[-H:])): cap += 1
        ol = demand > MU*cap; soft = demand > RHO*MU*cap
        residuals.append(max(demand-forecast,0.0)); ol_hist.append(bool(ol)); soft_hist.append(bool(soft))
        caps.append(cap); prev = demand
        if mode == "aci":
            delta_t = min(max(delta_t + KAPPA*(delta - (1.0 if ol else 0.0)), 1e-4), 0.5)
    cap = np.array(caps); ol_arr = test_y > MU*cap
    den_cap = np.maximum(np.ceil(test["replica_count"].to_numpy(float)),1)
    den = cap_cost(den_cap, test_y)
    return dict(overload=float(ol_arr.mean()), rel_cost=cap_cost(cap, test_y)/den,
                churn=int(np.abs(np.diff(cap)).sum()), max_run=max(_runlengths(ol_arr) or [0]))

def cap_cost(cap, demand):
    actions = np.abs(np.diff(cap, prepend=cap[0]))
    return 1.0*cap.sum() + 0.05*actions.sum() + 10.0*(demand > MU*cap).sum()

rows = []
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
_modes = [a for a in _args if a in ("aci", "weighted")] or ["aci", "weighted"]
_mods = [a for a in _args if a in MODELS] or MODELS
for mode in _modes:
    for model in _mods:
        for d in DELTAS:
            ol, rc = [], []
            for name, hist, test in SERVICES:
                r = simulate(hist, test, fc_arr(model,name,test), d, mode)
                ol.append(r["overload"]); rc.append(r["rel_cost"])
            ol = np.array(ol)
            rows.append(dict(method={"aci":"B10_ACI","weighted":"B11_weighted"}[mode],
                             model=model, delta=d,
                             frac_within_delta=round(float((ol<=d).mean()),3),
                             median_overload=round(float(np.median(ol)),4),
                             median_rel_cost=round(float(np.median(rc)),4)))
df = pd.DataFrame(rows)
_out = RES/"c1_adaptive_conformal.csv"
if _out.exists() and "--fresh" not in sys.argv:
    df = pd.concat([pd.read_csv(_out), df], ignore_index=True)
df.to_csv(_out, index=False)
print(f"(kappa={KAPPA}, lambda={LAM})")
print(df.to_string(index=False))
