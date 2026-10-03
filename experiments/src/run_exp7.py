"""Run EXP-7: Run-Length Analysis"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'experiments/src')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from exp_core import RESULTS_DIR, SPLITS_PATH, load_all_services, simulate_policy, _runlengths

FIGURES = Path('experiments/figures'); FIGURES.mkdir(parents=True, exist_ok=True)
PAL = {'persistence': '#9C27B0', 'arima': '#FF9800', 'guardrail_only': '#FF9800'}
SPLIT_DEF = json.loads(SPLITS_PATH.read_text())
SERVICES  = load_all_services(SPLIT_DEF)

delta = 0.05
W     = 240

all_runs = {'B1_reactive': [], 'B6_conformal': [], 'B2_pure_predictive': []}
for name, hist, test in SERVICES:
    test_y = test['cpu_sum'].to_numpy(dtype=float)
    r6 = simulate_policy(hist, test, delta=delta, W=W, use_margin=True, use_guardrail=True)
    all_runs['B6_conformal'].extend(r6['run_lengths'])
    r2 = simulate_policy(hist, test, delta=delta, W=W, use_margin=False, use_guardrail=False)
    all_runs['B2_pure_predictive'].extend(r2['run_lengths'])
    prev = float(hist['cpu_sum'].iloc[-1])
    mu   = 1.0
    caps = []
    for demand in test_y:
        cap = max(int(np.ceil(prev / (0.70 * mu))), 1)
        caps.append(cap)
        prev = demand
    cap_arr = np.array(caps)
    ol_arr  = test_y > mu * cap_arr
    all_runs['B1_reactive'].extend(_runlengths(ol_arr))

rows = []
for policy, runs in all_runs.items():
    if not runs:
        runs = [0]
    rows.append({'policy': policy, 'n_runs': len(runs), 'mean_run': float(np.mean(runs)),
                 'median_run': float(np.median(runs)), 'p95_run': float(np.quantile(runs, 0.95)),
                 'max_run': int(np.max(runs))})
    print(f'  {policy:<25s}  n_runs={rows[-1]["n_runs"]:5d}  mean={rows[-1]["mean_run"]:.2f}  max={rows[-1]["max_run"]:4d}')

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / 'exp7_runlength.csv', index=False)
print('CSV saved.')

fig, ax = plt.subplots(figsize=(6.5, 4.2))
policy_labels = {
    'B1_reactive':        ('Reactive threshold (B1)', '#FF9800', '-'),
    'B2_pure_predictive': ('Pure predictive (B2)',    '#FF9800', '--'),
    'B6_conformal':       ('Conformal policy (B6)',   '#9C27B0', '-'),
}
for key, (label, col, ls) in policy_labels.items():
    runs = sorted(all_runs[key])
    if not runs: continue
    max_r = max(runs)
    x = np.arange(1, max_r + 2)
    y = [float(np.mean(np.array(runs) >= xi)) for xi in x]
    ax.step(x, y, color=col, linewidth=1.8, linestyle=ls, label=label, where='post')

ax.set_xlabel('Overload run length (consecutive steps)')
ax.set_ylabel('Survival probability P(run > x)')
ax.set_title('EXP-7: Overload Run-Length Survival Function\n(delta=0.05, W=240; all services combined)')
ax.set_yscale('log')
ax.legend()
ax.set_xlim(left=1)
fig.tight_layout()
fig.savefig(FIGURES / 'exp7_runlength.pdf')
fig.savefig(FIGURES / 'exp7_runlength.png', dpi=200)
plt.close(fig)
print('Figure saved.')
