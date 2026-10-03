"""P3: does a COST-OPTIMISING predictive agent under-provision when the penalty
is weak? (Uses an optimiser rather than the feed-forward B2 replay.)

Predictive policy family k_{t+1} = ceil(theta * d_hat_{t+1} / mu) (persistence
forecast), theta a tunable provisioning factor. For each c_vio, select theta
minimising fleet C_total = C_res + C_act + c_vio * (#overload intervals); report
the selected theta's budget compliance. Focused tier (20 bursty services)."""
import sys, json; sys.path.insert(0, "experiments/src")
from pathlib import Path
import numpy as np, pandas as pd
from exp_core import load_all_services, SPLITS_PATH, _runlengths

RES = Path("experiments/results"); MU = 1.0; CRES, CACT = 1.0, 0.05; D = 0.05
THETA = [0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]; CVIO = [1, 5, 10, 50, 100]
sd = json.load(open(SPLITS_PATH)); SERVICES = load_all_services(sd)

def cost_components(cap, demand):
    actions = np.abs(np.diff(cap, prepend=cap[0]))
    olc = int((demand > MU * cap).sum())
    return CRES * cap.sum() + CACT * actions.sum(), olc, float((demand > MU * cap).mean())

stat = {th: {"base": 0.0, "olc": 0.0, "olf": []} for th in THETA}
for name, hist, test in SERVICES:
    y = test["cpu_sum"].to_numpy(float)
    prev = float(hist["cpu_sum"].iloc[-1]); dhat = np.concatenate([[prev], y[:-1]])
    for th in THETA:
        cap = np.maximum(np.ceil(th * dhat / MU), 1).astype(int)
        base, olc, olf = cost_components(cap, y)
        stat[th]["base"] += base; stat[th]["olc"] += olc; stat[th]["olf"].append(olf)

rows = []
for cv in CVIO:
    tot = {th: stat[th]["base"] + cv * stat[th]["olc"] for th in THETA}
    sel = min(tot, key=tot.get)
    olf = np.array(stat[sel]["olf"])
    rows.append(dict(c_vio=cv, selected_theta=sel,
                     frac_within_delta=round(float((olf <= D).mean()), 3),
                     median_overload=round(float(np.median(olf)), 4)))
df = pd.DataFrame(rows)
df.to_csv(RES / "p3_penalty_optimizer.csv", index=False)
print("Cost-optimising predictive agent: selected theta and compliance vs c_vio")
print(df.to_string(index=False))
print("\nProposed policy (B6) keeps frac_within_delta=1.0 at every c_vio by construction.")
