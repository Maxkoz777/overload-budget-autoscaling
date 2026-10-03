"""C2: actuation delay x conservativeness alpha (B6, large-scale, persistence).
Mitigation check: can inflating alpha restore compliance lost to delay tau?
Usage: python run_c2_delay_alpha.py 1.0 1.25   (alphas via argv; appends CSV)."""
import sys; sys.path.insert(0, "experiments/src")
from pathlib import Path
import numpy as np, pandas as pd
from exp_core import simulate_policy
from exp_core_ls import load_all_services_200, load_split_def, RESULTS_LS
from run_ls_actuation_delay import (apply_actuation_delay, metrics_from_capacity,
                                    observed_cost, starting_capacity)

D = 0.05; TAUS = [0, 1, 2, 5]
alphas = [float(a) for a in sys.argv[1:]] or [1.0, 1.25, 1.5, 2.0]
sd = load_split_def(); SERVICES = load_all_services_200(sd)
C = dict(c_res=1.0, c_act=0.05, c_vio=10.0)
out = RESULTS_LS / "ls_c2_delay_alpha_200.csv"

rows = []
for a in alphas:
    # per (service, alpha): plan once, apply each tau
    per = {t: {"ol": [], "rc": []} for t in TAUS}
    for name, hist, test in SERVICES:
        r = simulate_policy(hist, test, None, delta=D, W=240, alpha=a, mu=1.0,
                            use_margin=True, use_guardrail=True, **C)
        planned = r["capacities"]; init = starting_capacity(hist); den = observed_cost(test)
        for t in TAUS:
            applied = apply_actuation_delay(planned, t, init)
            m = metrics_from_capacity(test, applied)
            per[t]["ol"].append(m["overload_fraction"]); per[t]["rc"].append(m["c_total"]/den)
    for t in TAUS:
        ol = np.array(per[t]["ol"]); rc = np.array(per[t]["rc"])
        rows.append(dict(alpha=a, tau=t,
                         frac_within_delta=round(float((ol <= D).mean()), 3),
                         median_overload=round(float(np.median(ol)), 4),
                         p90_overload=round(float(np.quantile(ol, 0.9)), 4),
                         median_rel_cost=round(float(np.median(rc)), 4)))
df = pd.DataFrame(rows)
if out.exists() and "--fresh" not in sys.argv:
    df = pd.concat([pd.read_csv(out), df], ignore_index=True)
df.to_csv(out, index=False)
print(df.to_string(index=False))
