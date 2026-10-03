"""C5: densify the focused-tier cost-risk frontier (delta = 6 points). Replay only."""
import sys, json; sys.path.insert(0, "experiments/src")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp_core import load_all_services, load_forecast, simulate_policy, SPLITS_PATH

RESULTS = Path("experiments/results"); FIGS = Path("experiments/figures")
DELTAS = [0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
MODELS = ["persistence", "arima", "xgb", "lstm"]
MU, CRES, CACT, CVIO = 1.0, 1.0, 0.05, 10.0
sd = json.load(open(SPLITS_PATH))
SERVICES = load_all_services(sd)
FCAST = {m: load_forecast(m) for m in MODELS if m != "persistence"}

def observed_cost(test):
    cap = np.maximum(np.ceil(test["replica_count"].to_numpy(float)), 1).astype(int)
    actions = np.abs(np.diff(cap, prepend=cap[0]))
    ol = test["cpu_sum"].to_numpy(float) > MU * cap
    return CRES*cap.sum() + CACT*actions.sum() + CVIO*ol.sum()

def fc_array(model, name, test):
    if model == "persistence": return None
    df = FCAST[model]; df = df[df["msname"] == name]
    m = test[["timestamp"]].merge(df[["timestamp", "forecast"]], on="timestamp", how="left")
    return m["forecast"].ffill().bfill().to_numpy(float)

rows = []
for model in MODELS:
    for d in DELTAS:
        rc, ol = [], []
        for name, hist, test in SERVICES:
            fa = fc_array(model, name, test)
            r = simulate_policy(hist, test, fa, delta=d, W=240, alpha=1.0, mu=MU,
                                use_margin=True, use_guardrail=True,
                                c_res=CRES, c_act=CACT, c_vio=CVIO)
            denom = observed_cost(test)
            rc.append(r["c_total"]/denom); ol.append(r["overload_fraction"])
        rows.append(dict(model=model, delta=d,
                         median_rel_cost=float(np.median(rc)),
                         median_overload=float(np.median(ol)),
                         p10_overload=float(np.quantile(ol,0.1)),
                         p90_overload=float(np.quantile(ol,0.9))))
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "c5_frontier_density.csv", index=False)
print(df.to_string(index=False))

PAL = {"lstm":"#2196F3","xgb":"#4CAF50","arima":"#FF9800","persistence":"#9C27B0"}
LAB = {"lstm":"LSTM + conformal","xgb":"XGBoost + conformal",
       "arima":"ARIMA + conformal","persistence":"Persistence + conformal"}
plt.rcParams.update({"axes.grid":True,"grid.alpha":0.3,"grid.linestyle":"--",
                     "axes.spines.top":False,"axes.spines.right":False})
fig, ax = plt.subplots(figsize=(7,4.6))
for m in MODELS:
    s = df[df.model==m].sort_values("median_overload")
    ax.plot(s["median_overload"]*100, s["median_rel_cost"], "o-",
            color=PAL[m], label=LAB[m], ms=6, lw=1.8,
            ls="--" if m=="persistence" else "-")
ax.axhline(0.196, color="black", ls=":", lw=1.4, label="Oracle lower bound (0.196)")
ax.plot(0.0, 0.280, "P", color="#F44336", ms=12, label="Reactive threshold (0.280)")
ax.set_xlabel("Median Overload Fraction (%)")
ax.set_ylabel("Median Relative Cost\n(fraction of observed capacity cost)")
ax.set_title("")
ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
fig.tight_layout()
fig.savefig(FIGS/"fig1_cost_risk_frontier.pdf", bbox_inches="tight")
fig.savefig(FIGS/"fig1_cost_risk_frontier.png", dpi=200, bbox_inches="tight")
print("fig1 regenerated with 6 delta points")
