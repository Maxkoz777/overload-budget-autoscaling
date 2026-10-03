"""Run EXP-3: Window-Size Sensitivity"""
import sys, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'experiments/src')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from exp_core import RESULTS_DIR, SPLITS_PATH, load_all_services, simulate_policy

FIGURES = Path('experiments/figures'); FIGURES.mkdir(parents=True, exist_ok=True)
PAL = {'persistence': '#9C27B0'}
SPLIT_DEF = json.loads(SPLITS_PATH.read_text())
SERVICES  = load_all_services(SPLIT_DEF)
print(f'Services: {len(SERVICES)}')

W_values = [30, 60, 120, 240, 480, 960, 1440, 2880]
delta = 0.05
rows = []
for W in W_values:
    per_service_ol = [simulate_policy(hist, test, delta=delta, W=W, alpha=1.0,
                      use_margin=True, use_guardrail=False)['overload_fraction']
                      for name, hist, test in SERVICES]
    med = float(np.median(per_service_ol))
    rows.append({'W': W, 'median_ol': med,
                 'p10_ol': float(np.quantile(per_service_ol, 0.10)),
                 'p90_ol': float(np.quantile(per_service_ol, 0.90)),
                 'std_ol': float(np.std(per_service_ol)),
                 'abs_dev_median': abs(med - delta),
                 'frac_above_delta': float(np.mean([x > delta for x in per_service_ol]))})
    print(f'  W={W:5d}  median_ol={med:.4f}  std={rows[-1]["std_ol"]:.4f}')

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / 'exp3_window_sensitivity.csv', index=False)
print('CSV saved.')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))
ax1.semilogx(df['W'], df['median_ol'], color=PAL['persistence'], marker='o', linewidth=1.8, markersize=7, label='Median realised OL')
ax1.fill_between(df['W'], df['p10_ol'], df['p90_ol'], alpha=0.25, color=PAL['persistence'], label='[p10, p90]')
ax1.axhline(delta, color='black', linestyle='--', linewidth=1.2, label=f'Nominal delta = {delta}')
ax1.set_xlabel('Calibration window W (log scale)')
ax1.set_ylabel('Realised overload fraction')
ax1.set_title('Coverage vs Window Size')
ax1.legend()
abs_devs = df['abs_dev_median'].clip(1e-6)
ax2.loglog(df['W'], abs_devs, color=PAL['persistence'], marker='o', linewidth=1.8, markersize=7, label='|realised - delta|')
Wref = np.array(W_values, dtype=float)
ref  = abs_devs.iloc[0] * np.sqrt(W_values[0] / Wref)
ax2.loglog(Wref, ref, 'k--', linewidth=1.2, label='1/sqrt(W) reference')
ax2.set_xlabel('Calibration window W (log scale)')
ax2.set_ylabel('|Realised - Nominal delta| (log)')
ax2.set_title('Deviation vs Window Size\n(DKW concentration)')
ax2.legend()
fig.tight_layout()
fig.savefig(FIGURES / 'exp3_window_sensitivity.pdf')
fig.savefig(FIGURES / 'exp3_window_sensitivity.png', dpi=200)
plt.close(fig)
print('Figure saved.')
