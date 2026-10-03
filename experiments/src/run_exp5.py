"""Run EXP-5: Guard-Rail Ablation"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'experiments/src')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from exp_core import RESULTS_DIR, SPLITS_PATH, load_all_services, simulate_policy

FIGURES = Path('experiments/figures'); FIGURES.mkdir(parents=True, exist_ok=True)
PAL = {'margin_only': '#2196F3', 'guardrail': '#4CAF50', 'guardrail_only': '#FF9800'}
SPLIT_DEF = json.loads(SPLITS_PATH.read_text())
SERVICES  = load_all_services(SPLIT_DEF)

delta = 0.05
W     = 240

VARIANTS = {
    'B7_margin_only':    (True,  False),
    'B6_full':           (True,  True),
    'B8_guardrail_only': (False, True),
}

# Part A: Natural trace
rows_nat = []
for variant, (use_margin, use_guardrail) in VARIANTS.items():
    per_ol, per_cost = [], []
    for name, hist, test in SERVICES:
        r = simulate_policy(hist, test, delta=delta, W=W, use_margin=use_margin, use_guardrail=use_guardrail)
        per_ol.append(r['overload_fraction'])
        per_cost.append(r['c_total'])
    rows_nat.append({'variant': variant, 'setting': 'natural',
                     'median_ol':   float(np.median(per_ol)),
                     'p10_ol':      float(np.quantile(per_ol, 0.10)),
                     'p90_ol':      float(np.quantile(per_ol, 0.90)),
                     'median_cost': float(np.median(per_cost))})
    print(f'  {variant:<25s}  overload={rows_nat[-1]["median_ol"]:.4f}')

# Part B: Synthetic shift +50%
rows_shift = []
for variant, (use_margin, use_guardrail) in VARIANTS.items():
    per_ol_total, per_ol_shift, per_max_run = [], [], []
    for name, hist, test in SERVICES:
        test_y  = test['cpu_sum'].to_numpy(dtype=float)
        n       = len(test_y)
        shifted = test_y.copy()
        mid     = n // 2
        shifted[mid: mid + 60] *= 1.5
        r = simulate_policy(hist, test, demand_override=shifted, delta=delta, W=W,
                            use_margin=use_margin, use_guardrail=use_guardrail)
        per_ol_total.append(r['overload_fraction'])
        per_ol_shift.append(float(np.mean(r['overload_arr'][mid: mid + 60])))
        per_max_run.append(r['max_overload_run'])
    rows_shift.append({'variant': variant, 'setting': 'synthetic_shift_50pct',
                       'median_ol_total': float(np.median(per_ol_total)),
                       'median_ol_shift': float(np.median(per_ol_shift)),
                       'p10_ol_shift':    float(np.quantile(per_ol_shift, 0.10)),
                       'p90_ol_shift':    float(np.quantile(per_ol_shift, 0.90)),
                       'median_max_run':  float(np.median(per_max_run))})
    print(f'  {variant:<25s}  ol_total={rows_shift[-1]["median_ol_total"]:.4f}  ol_shift={rows_shift[-1]["median_ol_shift"]:.4f}')

# Part C: Magnitude sweep
rows_mag = []
magnitudes = [0.0, 0.30, 0.50, 1.00]
for mag in magnitudes:
    per_ol_b6, per_ol_b7 = [], []
    for name, hist, test in SERVICES:
        test_y  = test['cpu_sum'].to_numpy(dtype=float)
        shifted = test_y.copy()
        if mag > 0:
            mid = len(test_y) // 2
            shifted[mid: mid + 60] *= (1.0 + mag)
        for use_margin, use_guard, storage in [(True, True, per_ol_b6), (True, False, per_ol_b7)]:
            r = simulate_policy(hist, test, demand_override=shifted, delta=delta, W=W,
                                use_margin=use_margin, use_guardrail=use_guard)
            mid_s  = len(test_y) // 2
            ol_win = float(np.mean(r['overload_arr'][mid_s: mid_s + 60]))
            storage.append(ol_win)
    rows_mag.append({'magnitude': mag,
                     'B6_full_ol_shift':      float(np.median(per_ol_b6)),
                     'B7_margin_only_ol_shift': float(np.median(per_ol_b7))})

df_nat   = pd.DataFrame(rows_nat)
df_shift = pd.DataFrame(rows_shift)
df_mag   = pd.DataFrame(rows_mag)
df_nat.to_csv(RESULTS_DIR   / 'exp5a_guardrail_natural.csv',   index=False)
df_shift.to_csv(RESULTS_DIR / 'exp5b_guardrail_shift.csv',     index=False)
df_mag.to_csv(RESULTS_DIR   / 'exp5c_guardrail_magnitude.csv', index=False)
print('CSVs saved.')

# Figure
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
variants  = ['B7_margin_only', 'B6_full', 'B8_guardrail_only']
var_labels = ['Margin only\n(B7)', 'Margin +\nGuardrail (B6)', 'Guardrail only\n(B8)']
cols      = [PAL['margin_only'], PAL['guardrail'], PAL['guardrail_only']]
ax = axes[0]
nat_ols = [df_nat[df_nat['variant'] == v]['median_ol'].values[0] * 100 for v in variants]
bars = ax.bar(var_labels, nat_ols, color=cols, alpha=0.85, edgecolor='white')
ax.axhline(delta * 100, color='black', linestyle='--', linewidth=1.0, label=f'delta = {delta}')
for bar, v in zip(bars, nat_ols):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
ax.set_title('A. Natural Trace'); ax.set_ylabel('Median overload (%)'); ax.legend(fontsize=8)
shift_ols = [df_shift[df_shift['variant'] == v]['median_ol_shift'].values[0] * 100 for v in variants]
bars2 = axes[1].bar(var_labels, shift_ols, color=cols, alpha=0.85, edgecolor='white')
for bar, v in zip(bars2, shift_ols):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
axes[1].set_title('B. Synthetic Shift Window\n(+50% demand, 60 steps)'); axes[1].set_ylabel('Median overload in shift window (%)')
axes[2].plot(df_mag['magnitude'] * 100, df_mag['B6_full_ol_shift'] * 100, color=PAL['guardrail'], marker='o', linewidth=1.8, markersize=7, label='B6: Margin + Guardrail')
axes[2].plot(df_mag['magnitude'] * 100, df_mag['B7_margin_only_ol_shift'] * 100, color=PAL['margin_only'], marker='s', linewidth=1.8, markersize=7, linestyle='--', label='B7: Margin only')
axes[2].set_xlabel('Demand shift magnitude (%)'); axes[2].set_ylabel('Median overload in shift window (%)'); axes[2].set_title('C. Recovery vs Shift Magnitude'); axes[2].legend()
fig.suptitle('EXP-5: Guard-Rail Ablation Study (delta=0.05, W=240, persistence)', fontsize=10)
fig.tight_layout()
fig.savefig(FIGURES / 'exp5_guardrail_ablation.pdf')
fig.savefig(FIGURES / 'exp5_guardrail_ablation.png', dpi=200)
plt.close(fig)
print('Figure saved.')
