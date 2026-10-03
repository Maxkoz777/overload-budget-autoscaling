"""Run EXP-9: Penalty Sensitivity"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'experiments/src')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from exp_core import RESULTS_DIR, SPLITS_PATH, load_all_services, simulate_policy

FIGURES = Path('experiments/figures'); FIGURES.mkdir(parents=True, exist_ok=True)
PAL = {'persistence': '#9C27B0', 'arima': '#FF9800'}
SPLIT_DEF = json.loads(SPLITS_PATH.read_text())
SERVICES  = load_all_services(SPLIT_DEF)

c_vio_values = [1, 5, 10, 50, 100]
delta = 0.05
W     = 240
base_per = pd.read_csv(RESULTS_DIR / 'baseline_replay_per_service.csv')
obs_cost = (base_per[base_per['policy'] == 'observed_capacity'].set_index('msname')['c_total'])

rows = []
for c_vio in c_vio_values:
    for variant, (use_margin, use_guard) in {'B6_proposed': (True, True), 'B2_pure_predict': (False, False)}.items():
        per_ol, per_cost = [], []
        for name, hist, test in SERVICES:
            r = simulate_policy(hist, test, delta=delta, W=W, use_margin=use_margin, use_guardrail=use_guard, c_vio=float(c_vio))
            per_ol.append(r['overload_fraction'])
            oc = obs_cost.get(name, float('nan'))
            per_cost.append(r['c_total'] / oc if not pd.isna(oc) else float('nan'))
        rows.append({'c_vio': c_vio, 'variant': variant, 'median_ol': float(np.median(per_ol)), 'median_rel_cost': float(np.nanmedian(per_cost))})
    print(f'  c_vio={c_vio:3d}  B6 ol={rows[-2]["median_ol"]:.4f} cost={rows[-2]["median_rel_cost"]:.4f}  B2 ol={rows[-1]["median_ol"]:.4f} cost={rows[-1]["median_rel_cost"]:.4f}')

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / 'exp9_penalty_sensitivity.csv', index=False)
print('CSV saved.')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))
for variant, col, ls in [('B6_proposed', PAL['persistence'], '-'), ('B2_pure_predict', PAL['arima'], '--')]:
    sub = df[df['variant'] == variant].sort_values('c_vio')
    lbl = {'B6_proposed': 'B6: Proposed (margin+guardrail)', 'B2_pure_predict': 'B2: Pure predictive (no margin)'}[variant]
    ax1.plot(sub['c_vio'], sub['median_rel_cost'], color=col, marker='o', linewidth=1.8, markersize=7, linestyle=ls, label=lbl)
    ax2.plot(sub['c_vio'], sub['median_ol'] * 100, color=col, marker='o', linewidth=1.8, markersize=7, linestyle=ls, label=lbl)
ax1.set_xlabel('Violation penalty c_vio'); ax1.set_ylabel('Median relative cost'); ax1.set_title('Resource Cost vs c_vio'); ax1.legend(fontsize=8); ax1.set_xscale('log')
ax2.axhline(delta * 100, color='black', linestyle=':', linewidth=1.0, label=f'Nominal delta = {delta*100:.0f}%')
ax2.set_xlabel('Violation penalty c_vio'); ax2.set_ylabel('Median overload fraction (%)'); ax2.set_title('Overload Rate vs c_vio'); ax2.legend(fontsize=8); ax2.set_xscale('log')
fig.suptitle('EXP-9: Penalty Sensitivity -- c_vio Sweep (delta=0.05, W=240)', fontsize=10)
fig.tight_layout()
fig.savefig(FIGURES / 'exp9_penalty_sensitivity.pdf')
fig.savefig(FIGURES / 'exp9_penalty_sensitivity.png', dpi=200)
plt.close(fig)
print('Figure saved.')
