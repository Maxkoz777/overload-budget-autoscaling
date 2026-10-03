"""Re-plot EXP-3 (window sensitivity) from saved CSV, titleless. No recompute."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("experiments/results"); FIGS = Path("experiments/figures")
PAL = {'persistence': '#9C27B0'}
delta = 0.05
df = pd.read_csv(RESULTS / "exp3_window_sensitivity.csv").sort_values("W")
W_values = df["W"].to_numpy(dtype=float)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

ax1.semilogx(df['W'], df['median_ol'], color=PAL['persistence'], marker='o',
             linewidth=1.8, markersize=7, label='Median realised OL')
ax1.fill_between(df['W'], df['p10_ol'], df['p90_ol'], alpha=0.25,
                 color=PAL['persistence'], label='[p10, p90]')
ax1.axhline(delta, color='black', linestyle='--', linewidth=1.2, label=f'Nominal δ = {delta}')
ax1.set_xlabel('Calibration window W (log)')
ax1.set_ylabel('Realised overload fraction')
ax1.set_title('Coverage vs Window Size')
ax1.legend()
ax1.grid(True, alpha=0.3, linestyle='--')

abs_devs = df['abs_dev_median'].clip(1e-6)
ax2.loglog(df['W'], abs_devs, color=PAL['persistence'], marker='o',
           linewidth=1.8, markersize=7, label='|realised − δ|')
Wref = W_values
ref = abs_devs.iloc[0] * np.sqrt(W_values[0] / Wref)
ax2.loglog(Wref, ref, 'k--', linewidth=1.2, label='1/√W reference')
ax2.set_xlabel('Calibration window W (log)')
ax2.set_ylabel('|Realised − Nominal δ| (log)')
ax2.set_title('Deviation vs Window Size\n(DKW concentration)')
ax2.legend()
ax2.grid(True, alpha=0.3, linestyle='--', which='both')

fig.tight_layout()
fig.savefig(FIGS / 'exp3_window_sensitivity.pdf', bbox_inches='tight')
fig.savefig(FIGS / 'exp3_window_sensitivity.png', dpi=200, bbox_inches='tight')
print("exp3 re-plotted titleless from CSV")
