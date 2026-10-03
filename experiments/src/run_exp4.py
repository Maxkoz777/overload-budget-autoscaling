"""Run EXP-4: Conservativeness alpha Sweep"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'experiments/src')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from exp_core import RESULTS_DIR, SPLITS_PATH, load_all_services, simulate_policy

FIGURES = Path('experiments/figures'); FIGURES.mkdir(parents=True, exist_ok=True)
PAL = {'persistence': '#9C27B0', 'xgb': '#4CAF50'}
SPLIT_DEF = json.loads(SPLITS_PATH.read_text())
SERVICES  = load_all_services(SPLIT_DEF)

alpha_values = [1.0, 1.1, 1.25, 1.5, 2.0]
delta = 0.01
W     = 240

base = pd.read_csv(RESULTS_DIR / 'baseline_replay_per_service.csv')
obs_cost = (base[base['policy'] == 'observed_capacity'].set_index('msname')['c_total'])

rows = []
for alpha in alpha_values:
    per_ol, per_cost, per_margin = [], [], []
    for name, hist, test in SERVICES:
        r = simulate_policy(hist, test, delta=delta, W=W, alpha=alpha,
                            use_margin=True, use_guardrail=False)
        per_ol.append(r['overload_fraction'])
        oc = obs_cost.get(name, float('nan'))
        per_cost.append(r['c_total'] / oc if not pd.isna(oc) else float('nan'))
        per_margin.append(r['margin_mean'])
    rows.append({'alpha': alpha,
                 'median_ol':       float(np.median(per_ol)),
                 'p10_ol':          float(np.quantile(per_ol, 0.10)),
                 'p90_ol':          float(np.quantile(per_ol, 0.90)),
                 'median_rel_cost': float(np.nanmedian(per_cost)),
                 'p10_rel_cost':    float(np.nanquantile(per_cost, 0.10)),
                 'p90_rel_cost':    float(np.nanquantile(per_cost, 0.90)),
                 'median_margin':   float(np.median(per_margin))})
    print(f'  alpha={alpha:.2f}  overload={rows[-1]["median_ol"]:.4f}  cost={rows[-1]["median_rel_cost"]:.4f}')

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / 'exp4_alpha_sweep.csv', index=False)
print('CSV saved.')

fig, ax1 = plt.subplots(figsize=(6.0, 4.0))
ax2 = ax1.twinx()
ax1.plot(df['alpha'], df['median_ol'] * 100, color=PAL['persistence'], marker='o', linewidth=1.8, markersize=7, label='Overload (%)')
ax1.fill_between(df['alpha'], df['p10_ol']*100, df['p90_ol']*100, alpha=0.2, color=PAL['persistence'])
ax1.axhline(delta * 100, color=PAL['persistence'], linestyle='--', linewidth=1.0, label=f'Nominal delta={delta*100:.0f}%')
ax1.set_xlabel('Conservativeness factor alpha')
ax1.set_ylabel('Median overload fraction (%)', color=PAL['persistence'])
ax2.plot(df['alpha'], df['median_rel_cost'], color=PAL['xgb'], marker='s', linewidth=1.8, markersize=7, label='Relative cost')
ax2.set_ylabel('Median relative cost', color=PAL['xgb'])
ax1.set_title('')
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, loc='upper left')
fig.tight_layout()
fig.savefig(FIGURES / 'exp4_alpha_sweep.pdf')
fig.savefig(FIGURES / 'exp4_alpha_sweep.png', dpi=200)
plt.close(fig)
print('Figure saved.')
