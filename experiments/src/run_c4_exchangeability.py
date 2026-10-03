"""C4 (supplement): residual exchangeability diagnostics vs compliance.
Per service x forecaster (focused tier): underprediction rate, signed bias,
lag-1..5 residual autocorrelation. Relate to within-delta compliance (delta=0.05).
No simulation/training - pure residual statistics on saved forecasts."""
import sys, json; sys.path.insert(0, "experiments/src")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp_core import load_all_services, load_forecast, SPLITS_PATH

RES = Path("experiments/results"); FIGS = Path("experiments/figures")
MODELS = ["persistence", "arima", "xgb", "lstm"]
sd = json.load(open(SPLITS_PATH)); SERVICES = load_all_services(sd)
FC = {m: load_forecast(m) for m in MODELS if m != "persistence"}
# Within-delta compliance on the focused tier, from the EXP-2 coverage CSV
cov = pd.read_csv(RES / "exp2_coverage_calibration.csv")

def fc_arr(model, name, test):
    if model == "persistence":
        return np.concatenate([[test["cpu_sum"].iloc[0]], test["cpu_sum"].to_numpy(float)[:-1]])
    d = FC[model]; d = d[d.msname == name]
    return test[["timestamp"]].merge(d[["timestamp","forecast"]], on="timestamp", how="left")["forecast"].ffill().bfill().to_numpy(float)

def acf(x, lag):
    x = x - x.mean()
    if x.std() == 0: return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0,1])

rows = []
for model in MODELS:
    for name, hist, test in SERVICES:
        y = test["cpu_sum"].to_numpy(float); f = fc_arr(model, name, test)
        r = y - f
        rpos = np.maximum(r, 0.0)
        rows.append(dict(model=model, service=name,
                         underpred_rate=float((r > 0).mean()),
                         bias=float(r.mean()),
                         norm_bias=float(r.mean() / (np.abs(y).mean() + 1e-9)),
                         acf1=acf(rpos, 1),
                         acf_mean_1_5=float(np.mean([acf(rpos, l) for l in range(1, 6)]))))
df = pd.DataFrame(rows)
df.to_csv(RES / "c4_exchangeability_diagnostics.csv", index=False)

# per-model aggregate + merge with compliance at delta=0.05
agg = df.groupby("model").agg(underpred_rate=("underpred_rate","median"),
                              norm_bias=("norm_bias","median"),
                              acf1=("acf1","median"),
                              acf_mean_1_5=("acf_mean_1_5","median")).reset_index()
c05 = cov[np.isclose(cov.delta, 0.05)][["model","frac_within_delta"]]
agg = agg.merge(c05, on="model", how="left")
agg.to_csv(RES / "c4_diag_by_model.csv", index=False)
print(agg.to_string(index=False))

# Figure: mean lag-1..5 residual ACF vs compliance, one point per model (medians)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
PAL = {"persistence":"#9C27B0","arima":"#FF9800","xgb":"#4CAF50","lstm":"#2196F3"}
for _, row in agg.iterrows():
    ax.scatter(row["acf_mean_1_5"], row["frac_within_delta"]*100, s=120,
               color=PAL[row["model"]], label=row["model"], zorder=3)
    ax.annotate(row["model"], (row["acf_mean_1_5"], row["frac_within_delta"]*100),
                textcoords="offset points", xytext=(6,4), fontsize=8)
ax.axhline(95, color="grey", ls="--", lw=1, label="95% compliance")
ax.set_xlabel("Median residual autocorrelation (mean lag 1–5)")
ax.set_ylabel("Services within δ=0.05 (%)")
ax.grid(True, alpha=0.3, ls="--")
fig.tight_layout()
fig.savefig(FIGS/"exp_c4_exchangeability.pdf", bbox_inches="tight")
fig.savefig(FIGS/"exp_c4_exchangeability.png", dpi=200, bbox_inches="tight")
print("C4 figure saved: exp_c4_exchangeability")
