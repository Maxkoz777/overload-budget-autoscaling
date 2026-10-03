"""Run EXP-10: Recovery-Time Analysis"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'experiments/src')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from exp_core import RESULTS_DIR, SPLITS_PATH, load_all_services, simulate_policy

FIGURES = Path('experiments/figures'); FIGURES.mkdir(parents=True, exist_ok=True)
PAL = {'guardrail': '#4CAF50', 'margin_only': '#2196F3'}
SPLIT_DEF = json.loads(SPLITS_PATH.read_text())
SERVICES  = load_all_services(SPLIT_DEF)

delta        = 0.05
W            = 240
magnitudes   = [0.30, 0.50, 1.00]
RECOVERY_WIN = 30

def recovery_steps(ol_arr, shift_start, delta_, window):
    for t in range(shift_start, len(ol_arr) - window + 1):
        if np.mean(ol_arr[t: t + window]) <= delta_:
            return t - shift_start
    return len(ol_arr) - shift_start

rows = []
for mag in magnitudes:
    for variant, (use_margin, use_guard) in {'B6_full': (True, True), 'B7_margin_only': (True, False)}.items():
        rec_times = []
        for name, hist, test in SERVICES:
            test_y  = test['cpu_sum'].to_numpy(dtype=float)
            n       = len(test_y)
            shifted = test_y.copy()
            mid     = n // 2
            shifted[mid: mid + 60] *= (1.0 + mag)
            r = simulate_policy(hist, test, demand_override=shifted, delta=delta, W=W,
                                use_margin=use_margin, use_guardrail=use_guard)
            rec = recovery_steps(r['overload_arr'], mid + 60, delta, RECOVERY_WIN)
            rec_times.append(rec)
        rows.append({'magnitude': mag, 'variant': variant,
                     'median_rec': float(np.median(rec_times)),
                     'p10_rec':    float(np.quantile(rec_times, 0.10)),
                     'p90_rec':    float(np.quantile(rec_times, 0.90)),
                     'max_rec':    int(np.max(rec_times))})
        print(f'  mag={mag:.0%}  {variant:<18s}  median_rec={rows[-1]["median_rec"]:.0f}  max_rec={rows[-1]["max_rec"]}')

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / 'exp10_recovery_time.csv', index=False)
print('CSV saved.')

fig, ax = plt.subplots(figsize=(6.5, 4.0))
for variant, col, ls, mk in [('B6_full', PAL['guardrail'], '-', 'o'), ('B7_margin_only', PAL['margin_only'], '--', 's')]:
    sub = df[df['variant'] == variant].sort_values('magnitude')
    ax.errorbar(sub['magnitude'] * 100, sub['median_rec'],
                yerr=[sub['median_rec'] - sub['p10_rec'], sub['p90_rec'] - sub['median_rec']],
                color=col, marker=mk, markersize=7, linewidth=1.8, capsize=4, linestyle=ls,
                label=variant.replace('_', ' ').title())
ax.set_xlabel('Demand shift magnitude (%)')
ax.set_ylabel(f'Median recovery steps\n(steps until rolling {RECOVERY_WIN}-step OL <= delta)')
ax.set_title('EXP-10: Recovery Time vs Shift Magnitude\n(delta=0.05, W=240, persistence forecast)')
ax.legend(); ax.set_xticks([30, 50, 100]); ax.set_xticklabels(['30%', '50%', '100%'])
fig.tight_layout()
fig.savefig(FIGURES / 'exp10_recovery_time.pdf')
fig.savefig(FIGURES / 'exp10_recovery_time.png', dpi=200)
plt.close(fig)
print('Figure saved.')
