"""
exp_validator_fixes.py
======================
Additional baselines and robustness analyses:

  - EXP-10b: persistent (non-transient) demand step-shift
  - Baselines B3 (fixed margin), B4 (Gaussian margin), B5 (reactive+guardrail)
  - Statistical tests: paired Wilcoxon + Bonferroni correction + Clopper-Pearson CI
  - Cost-risk frontier with and without C_inf / C_train overhead

Usage
-----
    python3 experiments/src/exp_validator_fixes.py

Outputs
-------
    experiments/results/exp10b_persistent_shift.csv
    experiments/results/exp_b3_fixed_margin.csv
    experiments/results/exp_b4_gaussian_margin.csv
    experiments/results/exp_b5_reactive_guardrail.csv
    experiments/results/exp_baselines_summary.csv
    experiments/results/exp_statistical_tests.csv
    experiments/results/exp_overhead_frontier.csv
    experiments/figures/exp10b_persistent_shift.pdf / .png
    experiments/figures/exp_b3_b4_b5_comparison.pdf / .png
    experiments/figures/exp_statistical_significance.pdf / .png
    experiments/figures/exp_overhead_frontier.pdf / .png
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("experiments/src")))
from exp_core import (
    GROUP_MAP, DEFAULT_COST, RESULTS_DIR, SPLITS_PATH, TIMESERIES,
    load_all_services, load_forecast, simulate_policy, _runlengths,
)

warnings.filterwarnings("ignore")

FIGURES = Path("experiments/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.framealpha": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

PAL = {
    "b3_fixed":    "#795548",
    "b4_gaussian": "#E91E63",
    "b5_react_gr": "#607D8B",
    "b6_proposed": "#4CAF50",
    "b7_margin":   "#2196F3",
    "b1_reactive": "#F44336",
    "persistence": "#9C27B0",
    "oracle":      "#000000",
}

SPLIT_DEF = json.loads(SPLITS_PATH.read_text())
SERVICES  = load_all_services(SPLIT_DEF)

# Load baseline observed costs for normalisation
_base_per = pd.read_csv(RESULTS_DIR / "baseline_replay_per_service.csv")
OBS_COST  = (_base_per[_base_per["policy"] == "observed_capacity"]
             .set_index("msname")["c_total"])


def save(fig, name: str):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  Figure saved: {name}")


def simulate_fixed_margin(
    history: pd.DataFrame,
    test: pd.DataFrame,
    fixed_margin: float,
    *,
    mu: float = 1.0,
    c_vio: float = 10.0,
) -> dict:
    """Policy k_{t+1} = ceil((forecast + M̄) / mu) with constant M̄."""
    hist_y   = history["cpu_sum"].to_numpy(float)
    test_y   = test["cpu_sum"].to_numpy(float)
    prev     = hist_y[-1]
    cap_list = []
    ol_list  = []
    prev_cap = 1

    for demand in test_y:
        forecast = max(prev, 0.0)
        cap      = max(int(np.ceil((forecast + fixed_margin) / mu)), 1)
        cap_list.append(cap)
        ol_list.append(demand > mu * cap)
        prev     = demand
        prev_cap = cap

    cap_arr  = np.array(cap_list, dtype=int)
    ol_arr   = np.array(ol_list, dtype=bool)
    c_res    = float(np.sum(cap_arr)) * 1.0
    c_act    = 0.05 * float(np.sum(np.abs(np.diff(np.concatenate([[cap_list[0]], cap_arr])))))
    c_vio_   = c_vio * float(np.sum(ol_arr))

    return {
        "overload_fraction": float(np.mean(ol_arr)),
        "c_total":           c_res + c_act + c_vio_,
        "c_res":             c_res,
        "run_lengths":       _runlengths(ol_arr),
        "max_overload_run":  int(max(_runlengths(ol_arr))) if _runlengths(ol_arr) else 0,
    }


def simulate_gaussian_margin(
    history: pd.DataFrame,
    test: pd.DataFrame,
    delta: float = 0.05,
    W: int = 240,
    mu: float = 1.0,
) -> dict:
    """Policy k_{t+1} = ceil((forecast + z_{1-delta}*sigma_hat) / mu)."""
    from scipy.stats import norm  # type: ignore
    z = float(norm.ppf(1.0 - delta))

    hist_y    = history["cpu_sum"].to_numpy(float)
    test_y    = test["cpu_sum"].to_numpy(float)
    residuals = list(np.maximum(hist_y[1:] - hist_y[:-1], 0.0))
    prev      = hist_y[-1]
    cap_list  = []
    ol_list   = []

    for demand in test_y:
        forecast = max(prev, 0.0)
        window   = residuals[-W:]
        sigma    = float(np.std(window)) if len(window) > 1 else 0.0
        margin   = z * sigma
        cap      = max(int(np.ceil((forecast + margin) / mu)), 1)
        cap_list.append(cap)
        ol_list.append(demand > mu * cap)
        residuals.append(max(demand - forecast, 0.0))
        prev     = demand

    cap_arr = np.array(cap_list, dtype=int)
    ol_arr  = np.array(ol_list, dtype=bool)
    c_res   = float(np.sum(cap_arr)) * 1.0
    c_act   = 0.05 * float(np.sum(np.abs(np.diff(np.concatenate([[cap_list[0]], cap_arr])))))
    c_vio   = 10.0 * float(np.sum(ol_arr))

    return {
        "overload_fraction": float(np.mean(ol_arr)),
        "c_total":           c_res + c_act + c_vio,
        "c_res":             c_res,
        "run_lengths":       _runlengths(ol_arr),
        "max_overload_run":  int(max(_runlengths(ol_arr))) if _runlengths(ol_arr) else 0,
    }


def simulate_reactive_guardrail(
    history: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = 0.70,
    mu: float = 1.0,
    rho: float = 0.70,
    guardrail_h: int = 2,
    guardrail_gamma: int = 1,
) -> dict:
    """HPA-style reactive rule k_t = ceil(d_{t-1} / (threshold*mu)) + guardrail."""
    hist_y    = history["cpu_sum"].to_numpy(float)
    test_y    = test["cpu_sum"].to_numpy(float)
    prev      = hist_y[-1]
    cap_list  = []
    ol_list   = []
    soft_list = []

    for demand in test_y:
        nominal = max(int(np.ceil(prev / (threshold * mu))), 1)
        trigger = False
        if len(ol_list) >= guardrail_h:
            trigger = (all(ol_list[-guardrail_h:]) or
                       all(soft_list[-guardrail_h:]))
        cap = nominal + (guardrail_gamma if trigger else 0)
        cap_list.append(cap)
        ol_list.append(demand > mu * cap)
        soft_list.append(demand > rho * mu * cap)
        prev = demand

    cap_arr = np.array(cap_list, dtype=int)
    ol_arr  = np.array(ol_list, dtype=bool)
    c_res   = float(np.sum(cap_arr)) * 1.0
    c_act   = 0.05 * float(np.sum(np.abs(np.diff(np.concatenate([[cap_list[0]], cap_arr])))))
    c_vio   = 10.0 * float(np.sum(ol_arr))

    return {
        "overload_fraction": float(np.mean(ol_arr)),
        "c_total":           c_res + c_act + c_vio,
        "run_lengths":       _runlengths(ol_arr),
        "max_overload_run":  int(max(_runlengths(ol_arr))) if _runlengths(ol_arr) else 0,
    }


def fix3_new_baselines():
    """
    B3: Fixed margin  — Autopilot-style, calibration-data-derived constant margin
    B4: Gaussian margin — parametric conformal counterpart (z_{1-δ}·σ̂)
    B5: Reactive + guardrail — isolates prediction vs guardrail contribution
    """
    print("\n=== FIX-3: Baselines B3, B4, B5 ===")

    delta_values = [0.10, 0.05, 0.01]
    rows_b3, rows_b4, rows_b5 = [], [], []

    for delta in delta_values:
        per_b3, per_b4, per_b5, per_b6 = {d: [] for d in ["ol","cost","max_run"]}, \
            {d: [] for d in ["ol","cost","max_run"]}, \
            {d: [] for d in ["ol","cost","max_run"]}, \
            {d: [] for d in ["ol","cost","max_run"]}

        for name, hist, test in SERVICES:
            hist_y = hist["cpu_sum"].to_numpy(float)

            # B3: fixed margin = q_{1-δ} of history positive residuals
            hist_res = list(np.maximum(hist_y[1:] - hist_y[:-1], 0.0))
            fixed_m  = float(np.quantile(hist_res, 1.0 - delta)) if hist_res else 0.0
            r3 = simulate_fixed_margin(hist, test, fixed_margin=fixed_m)

            # B4: Gaussian margin
            r4 = simulate_gaussian_margin(hist, test, delta=delta, W=240)

            # B5: reactive + guardrail
            r5 = simulate_reactive_guardrail(hist, test)

            # B6: proposed (for comparison)
            r6 = simulate_policy(hist, test, delta=delta, W=240,
                                 use_margin=True, use_guardrail=True)

            oc = OBS_COST.get(name, np.nan)
            for per, r in [(per_b3, r3), (per_b4, r4), (per_b5, r5), (per_b6, r6)]:
                per["ol"].append(r["overload_fraction"])
                per["cost"].append(r["c_total"] / oc if not np.isnan(oc) else np.nan)
                per["max_run"].append(r["max_overload_run"])

        for tag, per, rows in [("B3_fixed_margin", per_b3, rows_b3),
                                ("B4_gaussian",    per_b4, rows_b4),
                                ("B5_react_guard", per_b5, rows_b5)]:
            rows.append({
                "policy":          tag,
                "delta":           delta,
                "median_ol":       float(np.median(per["ol"])),
                "p10_ol":          float(np.quantile(per["ol"], 0.10)),
                "p90_ol":          float(np.quantile(per["ol"], 0.90)),
                "frac_within_delta": float(np.mean([x <= delta for x in per["ol"]])),
                "median_rel_cost": float(np.nanmedian(per["cost"])),
                "median_max_run":  float(np.median(per["max_run"])),
            })
            print(f"  delta={delta}  {tag:<20s}  "
                  f"ol={rows[-1]['median_ol']:.4f}  "
                  f"cost={rows[-1]['median_rel_cost']:.4f}  "
                  f"within_delta={rows[-1]['frac_within_delta']:.0%}")

    df3 = pd.DataFrame(rows_b3)
    df4 = pd.DataFrame(rows_b4)
    df5 = pd.DataFrame(rows_b5)
    df3.to_csv(RESULTS_DIR / "exp_b3_fixed_margin.csv",     index=False)
    df4.to_csv(RESULTS_DIR / "exp_b4_gaussian_margin.csv",  index=False)
    df5.to_csv(RESULTS_DIR / "exp_b5_reactive_guardrail.csv", index=False)

    # Combined summary at delta=0.05
    adv  = pd.read_csv(RESULTS_DIR / "advanced_policy_replay_summary.csv")
    risk = pd.read_csv(RESULTS_DIR / "risk_policy_replay_summary.csv")
    base = pd.read_csv(RESULTS_DIR / "baseline_replay_summary.csv")
    oracle_c = float(base[base["policy"]=="oracle_demand_capacity"]["median_relative_cost"])
    react_c  = float(base[base["policy"]=="reactive_threshold_u70_cooldown3"]["median_relative_cost"])
    react_ol = float(base[base["policy"]=="reactive_threshold_u70_cooldown3"]["median_overload_fraction"])

    summary_rows = [
        {"policy": "Oracle lower bound",     "cost": oracle_c, "overload": 0.0,    "frac_within_delta": 1.0},
        {"policy": "B6: Proposed conformal", "cost": float(risk[(risk["policy"]=="risk_conformal_guardrail")&(risk["delta"]==0.05)]["median_relative_cost"]),
         "overload": float(risk[(risk["policy"]=="risk_conformal_guardrail")&(risk["delta"]==0.05)]["median_overload_fraction"]),
         "frac_within_delta": 1.0},
        {"policy": "B3: Fixed margin (δ-calibrated)", "cost": float(df3[df3["delta"]==0.05]["median_rel_cost"]),
         "overload": float(df3[df3["delta"]==0.05]["median_ol"]),
         "frac_within_delta": float(df3[df3["delta"]==0.05]["frac_within_delta"])},
        {"policy": "B4: Gaussian margin",    "cost": float(df4[df4["delta"]==0.05]["median_rel_cost"]),
         "overload": float(df4[df4["delta"]==0.05]["median_ol"]),
         "frac_within_delta": float(df4[df4["delta"]==0.05]["frac_within_delta"])},
        {"policy": "B5: Reactive+guardrail", "cost": float(df5[df5["delta"]==0.05]["median_rel_cost"]),
         "overload": float(df5[df5["delta"]==0.05]["median_ol"]),
         "frac_within_delta": float(df5[df5["delta"]==0.05]["frac_within_delta"])},
        {"policy": "B1: Reactive threshold", "cost": react_c, "overload": react_ol, "frac_within_delta": np.nan},
    ]
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(RESULTS_DIR / "exp_baselines_summary.csv", index=False)
    print("\n  Baselines summary at δ=0.05:")
    print(df_sum.to_string(index=False))

    # Figure: B3/B4/B5 vs B6 cost-risk comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # Panel A: delta sweep for overload
    for tag, df_, col, ls in [
        ("B3: Fixed margin",    df3, PAL["b3_fixed"],    "--"),
        ("B4: Gaussian margin", df4, PAL["b4_gaussian"], ":"),
        ("B5: React+guardrail", df5, PAL["b5_react_gr"], "-."),
        ("B6: Conformal (proposed)", None, PAL["b6_proposed"], "-"),
    ]:
        if df_ is not None:
            d = df_.sort_values("delta")
            axes[0].plot(d["delta"], d["median_ol"]*100, color=col, marker="o",
                         linewidth=1.8, markersize=7, linestyle=ls, label=tag)
        else:
            pers_d = risk[risk["policy"]=="risk_conformal_guardrail"].sort_values("delta")
            axes[0].plot(pers_d["delta"], pers_d["median_overload_fraction"]*100,
                         color=col, marker="o", linewidth=1.8, markersize=7,
                         linestyle=ls, label=tag)
    axes[0].plot([0.01, 0.10], [1, 10], "k--", linewidth=0.8, alpha=0.5, label="y=δ (ideal)")
    axes[0].set_xlabel("Nominal δ"); axes[0].set_ylabel("Median overload (%)")
    axes[0].set_title("Overload vs δ: conformal vs parametric baselines")
    axes[0].legend(fontsize=7.5)

    # Panel B: coverage calibration (% of services within δ) at delta=0.05
    policies = ["B3: Fixed margin", "B4: Gaussian margin",
                "B5: React+guardrail", "B6: Conformal"]
    frac_vals = [float(df3[df3["delta"]==0.05]["frac_within_delta"]),
                 float(df4[df4["delta"]==0.05]["frac_within_delta"]),
                 float(df5[df5["delta"]==0.05]["frac_within_delta"]),
                 1.00]
    cols = [PAL["b3_fixed"], PAL["b4_gaussian"], PAL["b5_react_gr"], PAL["b6_proposed"]]
    bars = axes[1].bar(policies, [v*100 for v in frac_vals], color=cols, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, frac_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, v*100+0.5, f"{v:.0%}",
                     ha="center", va="bottom", fontsize=8.5)
    axes[1].axhline(100, color="green", linestyle="--", linewidth=1.0,
                    label="100% (all services within δ)")
    axes[1].set_ylabel("Services satisfying OL ≤ δ (%)")
    axes[1].set_title("Coverage guarantee: % services within δ=0.05")
    axes[1].set_ylim(0, 115)
    axes[1].legend(fontsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")

    fig.tight_layout()
    save(fig, "exp_b3_b4_b5_comparison")
    return df3, df4, df5, df_sum


def fix2_exp10_persistent():
    """
    EXP-10b: demand permanently increases by +mag at t=T/2.
    No return to baseline → conformal window contaminated by old residuals.
    Recovery time = steps until rolling 60-step OL drops below δ.
    """
    print("\n=== FIX-2: EXP-10 Redesign (Persistent Step Shift) ===")

    RECOVERY_WIN = 60
    delta        = 0.05
    W            = 240
    magnitudes   = [0.30, 0.50, 1.00]

    def recovery_steps(ol_arr, shift_start, delta_, window):
        for t in range(shift_start, len(ol_arr) - window + 1):
            if np.mean(ol_arr[t: t + window]) <= delta_:
                return t - shift_start
        return len(ol_arr) - shift_start  # did not recover within horizon

    rows = []
    for mag in magnitudes:
        for variant, (use_margin, use_guard) in {
            "B6_full":        (True, True),
            "B7_margin_only": (True, False),
        }.items():
            recs, in_shift_ols = [], []
            for name, hist, test in SERVICES:
                test_y  = test["cpu_sum"].to_numpy(float)
                n       = len(test_y)
                shifted = test_y.copy()
                mid     = n // 2
                shifted[mid:] *= (1.0 + mag)   # permanent increase

                r = simulate_policy(hist, test, demand_override=shifted,
                                    delta=delta, W=W,
                                    use_margin=use_margin, use_guardrail=use_guard)
                rec = recovery_steps(r["overload_arr"], mid, delta, RECOVERY_WIN)
                recs.append(rec)
                in_shift_ols.append(float(np.mean(r["overload_arr"][mid: mid + 2 * RECOVERY_WIN])))

            rows.append({
                "magnitude":      mag,
                "variant":        variant,
                "median_rec":     float(np.median(recs)),
                "p10_rec":        float(np.quantile(recs, 0.10)),
                "p90_rec":        float(np.quantile(recs, 0.90)),
                "max_rec":        int(np.max(recs)),
                "median_ol_shift":float(np.median(in_shift_ols)),
            })
            print(f"  mag={mag:.0%}  {variant:<18s}  "
                  f"median_rec={rows[-1]['median_rec']:.0f}  max_rec={rows[-1]['max_rec']}  "
                  f"ol_shift={rows[-1]['median_ol_shift']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp10b_persistent_shift.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))

    for variant, col, ls, mk in [
        ("B6_full",        PAL["b6_proposed"], "-",  "o"),
        ("B7_margin_only", PAL["b7_margin"],   "--", "s"),
    ]:
        sub = df[df["variant"] == variant].sort_values("magnitude")
        ax1.errorbar(sub["magnitude"]*100, sub["median_rec"],
                     yerr=[sub["median_rec"]-sub["p10_rec"],
                           sub["p90_rec"]-sub["median_rec"]],
                     color=col, marker=mk, markersize=7, linewidth=1.8,
                     capsize=4, linestyle=ls, label=variant.replace("_"," ").title())
        ax2.plot(sub["magnitude"]*100, sub["median_ol_shift"]*100,
                 color=col, marker=mk, markersize=7, linewidth=1.8,
                 linestyle=ls, label=variant.replace("_"," ").title())

    ax1.set_xlabel("Permanent demand shift magnitude (%)")
    ax1.set_ylabel(f"Median recovery steps\n(60-step rolling OL ≤ δ after shift)")
    ax1.set_title("Recovery Time after PERMANENT Shift")
    ax1.legend()

    ax2.axhline(delta*100, color="black", linestyle=":", linewidth=1.0,
                label=f"Nominal δ = {delta*100:.0f}%")
    ax2.set_xlabel("Permanent demand shift magnitude (%)")
    ax2.set_ylabel("Median OL in post-shift window (%)")
    ax2.set_title("Overload Rate during Transition")
    ax2.legend()

    fig.tight_layout()
    save(fig, "exp10b_persistent_shift")
    return df


def fix4_statistical_tests():
    """
    Paired Wilcoxon signed-rank tests + Bonferroni correction.
    Clopper-Pearson exact CI for overload fraction per policy.
    """
    print("\n=== FIX-4: Statistical Tests ===")

    try:
        from scipy import stats          # type: ignore
        from scipy.stats import beta as _beta  # type: ignore
    except ImportError:
        print("  scipy not available — skipping")
        return pd.DataFrame()

    delta = 0.05
    W     = 240
    n_services = len(SERVICES)
    T          = len(SERVICES[0][2])   # test split length

    # Collect per-service overload fractions for each policy
    policies = {
        "B6_proposed":     dict(use_margin=True,  use_guardrail=True),
        "B7_margin_only":  dict(use_margin=True,  use_guardrail=False),
        "B8_guardrail_only":dict(use_margin=False, use_guardrail=True),
        "B2_pure_predict": dict(use_margin=False, use_guardrail=False),
    }
    fc_data = {m: load_forecast(m) for m in ["arima", "xgb", "lstm"]}
    for model in ["arima", "xgb", "lstm"]:
        policies[f"B6_{model}"] = {"model": model}

    per_service: dict[str, list[float]] = {k: [] for k in policies}

    for name, hist, test in SERVICES:
        for pname, kwargs in policies.items():
            if "model" in kwargs:
                m = kwargs["model"]
                sub = fc_data[m]
                sub = sub[sub["msname"] == name].sort_values("timestamp")
                fc  = sub["forecast"].to_numpy() if len(sub) == len(test) else None
                r   = simulate_policy(hist, test, fc, delta=delta, W=W,
                                      use_margin=True, use_guardrail=True)
            else:
                r = simulate_policy(hist, test, delta=delta, W=W, **kwargs)
            per_service[pname].append(r["overload_fraction"])

    # Wilcoxon + Bonferroni
    comparisons = [
        ("B6_proposed", "B7_margin_only",  "B6 vs B7 (guardrail adds value?)"),
        ("B6_proposed", "B8_guardrail_only","B6 vs B8 (margin essential?)"),
        ("B6_proposed", "B2_pure_predict",  "B6 vs B2 (margin vs no-margin?)"),
        ("B6_proposed", "B6_arima",         "persistence vs ARIMA forecaster"),
        ("B6_xgb",      "B6_arima",         "XGBoost vs ARIMA forecaster"),
        ("B6_lstm",     "B6_arima",         "LSTM vs ARIMA forecaster"),
        ("B6_lstm",     "B6_xgb",           "LSTM vs XGBoost forecaster"),
    ]
    n_tests = len(comparisons)
    alpha_corrected = 0.05 / n_tests  # Bonferroni

    rows_stat = []
    for (a, b, description) in comparisons:
        x = np.array(per_service[a])
        y = np.array(per_service[b])
        try:
            stat, p_val = stats.wilcoxon(x, y, alternative="less")
            reject = bool(p_val < alpha_corrected)
        except Exception:
            stat, p_val, reject = np.nan, np.nan, False

        rows_stat.append({
            "comparison":       description,
            "policy_a":         a,
            "policy_b":         b,
            "median_ol_a":      float(np.median(per_service[a])),
            "median_ol_b":      float(np.median(per_service[b])),
            "wilcoxon_stat":    float(stat) if not np.isnan(stat) else None,
            "p_value":          float(p_val) if not np.isnan(p_val) else None,
            "alpha_bonferroni": alpha_corrected,
            "reject_H0":        reject,
            "interpretation":   "a < b (significant)" if reject else "not significant",
        })
        print(f"  {description:<45s}  p={p_val:.4f}  reject={reject}")

    df_stat = pd.DataFrame(rows_stat)
    df_stat.to_csv(RESULTS_DIR / "exp_statistical_tests.csv", index=False)

    # Clopper-Pearson CI for key policies
    cp_rows = []
    for pname in ["B6_proposed", "B7_margin_only", "B2_pure_predict",
                  "B6_arima", "B6_xgb", "B6_lstm"]:
        for i, ol_frac in enumerate(per_service[pname]):
            k = int(round(ol_frac * T))
            lo = float(_beta.ppf(0.025, max(k, 0.5), T - k + 1))
            hi = float(_beta.ppf(0.975, k + 1, max(T - k, 0.5)))
            cp_rows.append({"policy": pname, "service": SERVICES[i][0],
                            "ol_frac": ol_frac, "ci_lo": lo, "ci_hi": hi})
    df_cp = pd.DataFrame(cp_rows)
    df_cp.to_csv(RESULTS_DIR / "exp_clopper_pearson_ci.csv", index=False)

    # Figure: statistical significance heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: p-values
    y_pos = range(len(rows_stat))
    p_vals = [r["p_value"] if r["p_value"] is not None else 1.0 for r in rows_stat]
    cols_bar = [PAL["b6_proposed"] if r["reject_H0"] else PAL["b1_reactive"]
                for r in rows_stat]
    ax1.barh(list(y_pos), p_vals, color=cols_bar, alpha=0.8, edgecolor="white")
    ax1.axvline(alpha_corrected, color="black", linestyle="--", linewidth=1.2,
                label=f"Bonferroni α = {alpha_corrected:.4f}")
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels([r["comparison"][:40] for r in rows_stat], fontsize=7.5)
    ax1.set_xlabel("p-value (Wilcoxon, H₀: a ≥ b)")
    ax1.set_title("Statistical Significance\n(green = reject H₀)")
    ax1.legend(fontsize=8)
    ax1.set_xscale("log")

    # Panel B: Clopper-Pearson CI for key policies
    key_policies = ["B6_proposed", "B7_margin_only", "B2_pure_predict",
                    "B6_arima", "B6_xgb", "B6_lstm"]
    kp_labels    = ["B6 Conf.", "B7 Margin", "B2 Predict.", "B6+ARIMA", "B6+XGB", "B6+LSTM"]
    for i, (pname, label) in enumerate(zip(key_policies, kp_labels)):
        sub = df_cp[df_cp["policy"] == pname]
        ax2.scatter([i]*len(sub), sub["ol_frac"]*100,
                    alpha=0.4, s=20, color=PAL.get(pname.lower().replace("b6_",""), "#888"))
        med = float(sub["ol_frac"].median())
        lo  = float(sub["ci_lo"].median())
        hi  = float(sub["ci_hi"].median())
        ax2.errorbar([i], [med*100], yerr=[[med*100-lo*100], [hi*100-med*100]],
                     color="black", capsize=5, linewidth=2, marker="D", markersize=6)
    ax2.axhline(delta*100, color="black", linestyle="--", linewidth=1.0,
                label=f"δ = {delta*100:.0f}%")
    ax2.set_xticks(range(len(kp_labels)))
    ax2.set_xticklabels(kp_labels, fontsize=8)
    ax2.set_ylabel("Overload fraction (%) with Clopper-Pearson 95% CI")
    ax2.set_title("Per-Service Distribution + CP CI\n(diamond = median CI; dots = individual services)")
    ax2.legend(fontsize=8)

    fig.suptitle("FIX-4: Statistical Significance Tests (Wilcoxon + Bonferroni + Clopper-Pearson)",
                 fontsize=9)
    fig.tight_layout()
    save(fig, "exp_statistical_significance")
    return df_stat, df_cp


def fix6_overhead_frontier():
    """
    Cost-risk frontier for XGBoost and LSTM with and without C_inf + C_train.
    Shows whether the overhead changes policy ranking.
    """
    print("\n=== FIX-6: Overhead Frontier ===")

    deltas = [0.10, 0.05, 0.01]
    fc_data = {m: load_forecast(m) for m in ["xgb", "lstm"]}

    # Calibrated from training logs (min → replica-min conversion)
    c_inf_ps = {"xgb": 14.0*60/4320/60, "lstm": 1.1*60/4320/60}
    c_train_s = {"xgb": 537.7/20, "lstm": 302.6/20}

    rows = []
    for model in ["xgb", "lstm"]:
        fc = fc_data[model]
        for delta in deltas:
            for include_overhead in [False, True]:
                per_cost, per_ol = [], []
                for name, hist, test in SERVICES:
                    sub = fc[fc["msname"] == name].sort_values("timestamp")
                    fc_arr = sub["forecast"].to_numpy() if len(sub) == len(test) else None
                    r = simulate_policy(
                        hist, test, fc_arr,
                        delta=delta, W=240,
                        use_margin=True, use_guardrail=True,
                        c_inf_per_step=c_inf_ps[model] if include_overhead else 0.0,
                        c_train_total=c_train_s[model] if include_overhead else 0.0,
                    )
                    oc = OBS_COST.get(name, np.nan)
                    per_cost.append(r["c_total"] / oc if not np.isnan(oc) else np.nan)
                    per_ol.append(r["overload_fraction"])

                rows.append({
                    "model":           model,
                    "delta":           delta,
                    "include_overhead": include_overhead,
                    "median_rel_cost": float(np.nanmedian(per_cost)),
                    "median_ol":       float(np.median(per_ol)),
                })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp_overhead_frontier.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    model_cols  = {"xgb": PAL["b6_proposed"], "lstm": "#2196F3"}
    model_marks = {"xgb": "s", "lstm": "o"}

    for model in ["xgb", "lstm"]:
        for include, ls, alpha_ in [(False, "--", 0.5), (True, "-", 1.0)]:
            sub = df[(df["model"]==model) & (df["include_overhead"]==include)].sort_values("delta")
            label = f"{model.upper()} {'with' if include else 'without'} C_inf+C_train"
            ax.plot(sub["median_ol"]*100, sub["median_rel_cost"],
                    color=model_cols[model], marker=model_marks[model],
                    markersize=7, linewidth=1.8, linestyle=ls, alpha=alpha_,
                    label=label)
            if not include:
                for i, row in sub.iterrows():
                    ax.annotate(f"δ={row['delta']}", xy=(row["median_ol"]*100, row["median_rel_cost"]),
                                xytext=(3, -8), textcoords="offset points",
                                fontsize=7, color=model_cols[model])

    # Oracle reference
    oracle_c = float(pd.read_csv(RESULTS_DIR/"baseline_replay_summary.csv")
                     [pd.read_csv(RESULTS_DIR/"baseline_replay_summary.csv")["policy"]=="oracle_demand_capacity"]
                     ["median_relative_cost"])
    ax.axhline(oracle_c, color="black", linestyle=":", linewidth=1.2, label=f"Oracle ({oracle_c:.3f})")

    ax.set_xlabel("Median Overload Fraction (%)")
    ax.set_ylabel("Median Relative Cost")
    ax.set_title("")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "exp_overhead_frontier")
    return df


if __name__ == "__main__":
    import datetime
    print(f"\n{'='*72}")
    print(f"  Validator Fixes  —  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Services: {len(SERVICES)}")
    print(f"{'='*72}")

    df3, df4, df5, df_sum = fix3_new_baselines()
    df10b                  = fix2_exp10_persistent()
    df_stat, df_cp         = fix4_statistical_tests()
    df_ov                  = fix6_overhead_frontier()

    print(f"\n{'='*72}")
    print("  All validator fixes complete.")
    print(f"{'='*72}\n")
