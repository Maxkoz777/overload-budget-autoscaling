"""C6: paired bootstrap CI on large-scale relative-cost differences. Replay."""
import sys; sys.path.insert(0, "experiments/src")
from pathlib import Path
import numpy as np, pandas as pd
from exp_core import simulate_policy
from exp_core_ls import load_all_services_200, load_split_def, RESULTS_LS
from run_ls_baselines import observed_cost, metrics_from_capacity
from policies import reactive_threshold_capacity

rng = np.random.default_rng(42)
sd = load_split_def()
SERVICES = load_all_services_200(sd)
C = dict(c_res=1.0, c_act=0.05, c_vio=10.0)
D = 0.05

b6, b7, b1 = [], [], []
for name, hist, test in SERVICES:
    den = observed_cost(test)
    r6 = simulate_policy(hist, test, None, delta=D, W=240, alpha=1.0, mu=1.0,
                         use_margin=True, use_guardrail=True, **C)
    r7 = simulate_policy(hist, test, None, delta=D, W=240, alpha=1.0, mu=1.0,
                         use_margin=True, use_guardrail=False, **C)
    cap1 = reactive_threshold_capacity(hist, test, mu=1.0, target_utilization=0.8, cooldown=3)
    c1 = metrics_from_capacity(test, cap1)["c_total"]
    b6.append(r6["c_total"]/den); b7.append(r7["c_total"]/den); b1.append(c1/den)
b6, b7, b1 = map(np.array, (b6, b7, b1))

def boot(diff, n=10000):
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

rows = []
for lab, a, bb in [("B6 - B1(u=0.8)", b6, b1), ("B6 - B7(margin only)", b6, b7)]:
    d = a - bb
    m, lo, hi = boot(d)
    rows.append(dict(comparison=lab, mean_rel_cost_diff=round(m, 5),
                     ci_lo=round(lo, 5), ci_hi=round(hi, 5),
                     median_a=round(float(np.median(a)), 4),
                     median_b=round(float(np.median(bb)), 4)))
df = pd.DataFrame(rows)
df.to_csv(RESULTS_LS / "ls_c6_cost_bootstrap_ci_200.csv", index=False)
print(f"median rel_cost: B6={np.median(b6):.4f} B1(u0.8)={np.median(b1):.4f} B7={np.median(b7):.4f}")
print(df.to_string(index=False))
