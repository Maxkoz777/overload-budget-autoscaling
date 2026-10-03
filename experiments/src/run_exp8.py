"""Run EXP-8: Cost Decomposition"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'experiments/src')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from exp_core import RESULTS_DIR, SPLITS_PATH, load_all_services, load_forecast, simulate_policy

FIGURES = Path('experiments/figures'); FIGURES.mkdir(parents=True, exist_ok=True)
SPLIT_DEF = json.loads(SPLITS_PATH.read_text())
SERVICES  = load_all_services(SPLIT_DEF)

inference_per_step = {'persistence': 0.0, 'arima': 1.6*60/4320/60, 'xgb': 14.0*60/4320/60, 'lstm': 1.1*60/4320/60}
training_per_service = {'persistence': 0.0, 'arima': 68.0/20, 'xgb': 537.7/20, 'lstm': 302.6/20}
delta = 0.05
W     = 240
fc_data = {m: load_forecast(m) for m in ['arima', 'xgb', 'lstm']}
base_per = pd.read_csv(RESULTS_DIR / 'baseline_replay_per_service.csv')
obs_cost = (base_per[base_per['policy'] == 'observed_capacity'].set_index('msname')['c_total'])

rows = []
for model in ['persistence', 'arima', 'xgb', 'lstm']:
    c_inf_ps  = inference_per_step[model]
    c_train_s = training_per_service[model]
    totals = {'c_res': [], 'c_act': [], 'c_vio': [], 'c_inf': [], 'c_train': []}
    for name, hist, test in SERVICES:
        if model == 'persistence':
            fc = None
        else:
            sub = fc_data[model]
            sub = sub[sub['msname'] == name].sort_values('timestamp')
            fc  = sub['forecast'].to_numpy() if len(sub) == len(test) else None
        r = simulate_policy(hist, test, fc, delta=delta, W=W, use_margin=True, use_guardrail=True,
                            c_inf_per_step=c_inf_ps, c_train_total=c_train_s)
        for k in totals:
            totals[k].append(r[k])
    oc = obs_cost.reindex([s[0] for s in SERVICES]).values
    c_total_arr = np.array(totals['c_res']) + np.array(totals['c_act']) + np.array(totals['c_vio']) + np.array(totals['c_inf']) + np.array(totals['c_train'])
    rel_cost = c_total_arr / np.where(oc > 0, oc, np.nan)
    rows.append({'model': model,
                 'median_c_res_pct':   float(np.median([totals['c_res'][i]/c_total_arr[i]*100 for i in range(len(SERVICES))])),
                 'median_c_act_pct':   float(np.median([totals['c_act'][i]/c_total_arr[i]*100 for i in range(len(SERVICES))])),
                 'median_c_vio_pct':   float(np.median([totals['c_vio'][i]/c_total_arr[i]*100 for i in range(len(SERVICES))])),
                 'median_c_inf_pct':   float(np.median([totals['c_inf'][i]/c_total_arr[i]*100 for i in range(len(SERVICES))])),
                 'median_c_train_pct': float(np.median([totals['c_train'][i]/c_total_arr[i]*100 for i in range(len(SERVICES))])),
                 'median_rel_cost':    float(np.nanmedian(rel_cost)),
                 'c_inf_per_step': c_inf_ps, 'c_train_per_svc': c_train_s})
    print(f'  {model:<12s}  C_res={rows[-1]["median_c_res_pct"]:.1f}%  C_act={rows[-1]["median_c_act_pct"]:.1f}%  C_vio={rows[-1]["median_c_vio_pct"]:.1f}%')

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / 'exp8_cost_decomposition.csv', index=False)
print('CSV saved.')

fig, ax = plt.subplots(figsize=(7.5, 4.2))
models = ['persistence', 'arima', 'xgb', 'lstm']
comp_colors = {'C_res': '#1565C0', 'C_act': '#43A047', 'C_vio': '#E53935', 'C_inf': '#FB8C00', 'C_train': '#8E24AA'}
bottoms = np.zeros(len(models))
for comp, col in comp_colors.items():
    col_key = f'median_{comp.lower()}_pct'
    vals = [df[df['model'] == m][col_key].values[0] for m in models]
    ax.bar(models, vals, bottom=bottoms, color=col, alpha=0.85, label=comp, edgecolor='white')
    for i, v in enumerate(vals):
        if v > 0.5:
            ax.text(i, bottoms[i] + v/2, f'{v:.1f}%', ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
    bottoms += np.array(vals)
ax.set_ylabel('Cost component share (%)')
ax.set_title('EXP-8: Cost Decomposition (delta=0.05, W=240, guardrail)\nC_inf and C_train calibrated from measured M1 inference/training times')
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
ax.set_ylim(0, 110)
fig.tight_layout()
fig.savefig(FIGURES / 'exp8_cost_decomposition.pdf')
fig.savefig(FIGURES / 'exp8_cost_decomposition.png', dpi=200)
plt.close(fig)
print('Figure saved.')
