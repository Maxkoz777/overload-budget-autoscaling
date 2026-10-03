"""
run_supplemental_experiments.py
================================
Runs all supplemental experiments EXP-2 through EXP-10 and writes results to
experiments/results/exp_*.csv and experiments/figures/exp_*.pdf/.png.

Usage
-----
    python3 experiments/src/run_supplemental_experiments.py

Experiments
-----------
  EXP-2   Coverage calibration: realised vs nominal δ
  EXP-3   Window-size sensitivity: W sweep, checks the DKW quantile-localization rate
  EXP-4   Conservativeness α sweep: cost-SLA trade-off knob
  EXP-5   Guard-rail ablation: stationary + synthetic demand shift
  EXP-7   Run-length analysis: overload clustering statistics
  EXP-8   Cost decomposition: C_res / C_act / C_vio / C_inf / C_train
  EXP-9   Penalty sensitivity: c_vio sweep
  EXP-10  Recovery-time analysis: time to restore coverage after regime change
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
    load_all_services, load_forecast, simulate_policy,
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
    "persistence": "#9C27B0", "lstm": "#2196F3",
    "xgb": "#4CAF50", "arima": "#FF9800",
    "margin_only": "#2196F3", "guardrail": "#4CAF50",
    "guardrail_only": "#FF9800",
}


def save(fig, name: str):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  Figure saved: {name}")


SPLIT_DEF = json.loads(SPLITS_PATH.read_text())
SERVICES  = load_all_services(SPLIT_DEF)   # list of (name, history, test)


def exp2_coverage_calibration():
    """
    Plot realised overload fraction vs nominal δ for all models + persistence.
    Checks the coverage guarantee E[OL_T] ≤ δ.
    """
    print("\n=== EXP-2: Coverage Calibration ===")

    deltas  = [0.10, 0.05, 0.01]
    models  = ["arima", "xgb", "lstm"]
    fc_data = {m: load_forecast(m) for m in models}

    rows = []
    for delta in deltas:
        # Persistence + conformal guardrail
        per_service_ol = []
        for name, hist, test in SERVICES:
            r = simulate_policy(hist, test, delta=delta, W=240, alpha=1.0,
                                use_margin=True, use_guardrail=True)
            per_service_ol.append(r["overload_fraction"])
        rows.append({"model": "persistence", "delta": delta,
                     "median_ol": float(np.median(per_service_ol)),
                     "p10_ol": float(np.quantile(per_service_ol, 0.10)),
                     "p90_ol": float(np.quantile(per_service_ol, 0.90)),
                     "frac_within_delta": float(np.mean(
                         [x <= delta for x in per_service_ol]))})

        for model in models:
            fc = fc_data[model]
            per_service_ol = []
            for name, hist, test in SERVICES:
                sub = fc[fc["msname"] == name].sort_values("timestamp")
                if len(sub) != len(test):
                    continue
                r = simulate_policy(hist, test, sub["forecast"].to_numpy(),
                                    delta=delta, W=240, alpha=1.0,
                                    use_margin=True, use_guardrail=True)
                per_service_ol.append(r["overload_fraction"])
            rows.append({"model": model, "delta": delta,
                         "median_ol": float(np.median(per_service_ol)),
                         "p10_ol": float(np.quantile(per_service_ol, 0.10)),
                         "p90_ol": float(np.quantile(per_service_ol, 0.90)),
                         "frac_within_delta": float(np.mean(
                             [x <= delta for x in per_service_ol]))})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp2_coverage_calibration.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    colors = {"persistence": PAL["persistence"], "arima": PAL["arima"],
              "xgb": PAL["xgb"], "lstm": PAL["lstm"]}

    ax.plot([0, 0.12], [0, 0.12], "k--", linewidth=1.2, label="Ideal (realised = nominal)")
    ax.fill_between([0, 0.12], [0, 0], [0, 0.12], alpha=0.04, color="green")
    ax.text(0.03, 0.005, "Conservative\n(realised < nominal)", fontsize=7.5,
            color="green", style="italic")

    for model, col in colors.items():
        sub = df[df["model"] == model].sort_values("delta")
        ax.errorbar(sub["delta"], sub["median_ol"],
                    yerr=[sub["median_ol"] - sub["p10_ol"],
                          sub["p90_ol"] - sub["median_ol"]],
                    color=col, marker="o", markersize=7,
                    capsize=4, linewidth=1.6,
                    label=model.capitalize())

    ax.set_xlabel("Nominal risk budget δ")
    ax.set_ylabel("Realised overload fraction\n(median ± [p10, p90] across services)")
    ax.set_title("")
    ax.legend()
    ax.set_xlim(0, 0.12)
    ax.set_ylim(0, 0.22)
    save(fig, "exp2_coverage_calibration")
    return df


def exp3_window_sensitivity():
    """
    Sweep W ∈ {30, 60, 120, 240, 480, 960, 1440, 2880}.
    Checks the DKW rate: |realised − nominal| ~ 1/√W.
    """
    print("\n=== EXP-3: Window-Size Sensitivity ===")

    W_values = [30, 60, 120, 240, 480, 960, 1440, 2880]
    delta    = 0.05

    rows = []
    for W in W_values:
        per_service_ol = []
        for name, hist, test in SERVICES:
            r = simulate_policy(hist, test, delta=delta, W=W, alpha=1.0,
                                use_margin=True, use_guardrail=False)
            per_service_ol.append(r["overload_fraction"])
        med = float(np.median(per_service_ol))
        rows.append({
            "W": W,
            "median_ol":       med,
            "p10_ol":          float(np.quantile(per_service_ol, 0.10)),
            "p90_ol":          float(np.quantile(per_service_ol, 0.90)),
            "std_ol":          float(np.std(per_service_ol)),
            "abs_dev_median":  abs(med - delta),
            "frac_above_delta": float(np.mean([x > delta for x in per_service_ol])),
        })
        print(f"  W={W:5d}  median_ol={med:.4f}  std={rows[-1]['std_ol']:.4f}  "
              f"above_δ={rows[-1]['frac_above_delta']:.2%}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp3_window_sensitivity.csv", index=False)

    # Figure: two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    ax1.semilogx(df["W"], df["median_ol"], color=PAL["persistence"],
                 marker="o", linewidth=1.8, markersize=7, label="Median realised OL")
    ax1.fill_between(df["W"], df["p10_ol"], df["p90_ol"],
                     alpha=0.25, color=PAL["persistence"], label="[p10, p90]")
    ax1.axhline(delta, color="black", linestyle="--", linewidth=1.2,
                label=f"Nominal δ = {delta}")
    ax1.set_xlabel("Calibration window W (log scale)")
    ax1.set_ylabel("Realised overload fraction")
    ax1.set_title("Coverage vs Window Size")
    ax1.legend()

    # Log-log: |realised - delta| vs W with 1/√W reference
    abs_devs = df["abs_dev_median"].clip(1e-6)
    ax2.loglog(df["W"], abs_devs, color=PAL["persistence"],
               marker="o", linewidth=1.8, markersize=7, label="|realised − δ|")
    Wref = np.array(W_values, dtype=float)
    ref  = abs_devs.iloc[0] * np.sqrt(W_values[0] / Wref)
    ax2.loglog(Wref, ref, "k--", linewidth=1.2, label="1/√W reference")
    ax2.set_xlabel("Calibration window W (log scale)")
    ax2.set_ylabel("|Realised − Nominal δ| (log)")
    ax2.set_title("Deviation vs Window Size\n(DKW concentration)")
    ax2.legend()

    fig.tight_layout()
    save(fig, "exp3_window_sensitivity")
    return df


def exp4_alpha_sweep():
    """
    Sweep α ∈ {1.0, 1.1, 1.25, 1.5, 2.0} at fixed δ=0.01, W=240.
    Shows α as a second tuning knob: cost vs. tighter coverage.
    """
    print("\n=== EXP-4: Conservativeness α Sweep ===")

    alpha_values = [1.0, 1.1, 1.25, 1.5, 2.0]
    delta = 0.01
    W     = 240

    # Load baseline observed costs for normalisation
    base = pd.read_csv(RESULTS_DIR / "baseline_replay_per_service.csv")
    obs_cost = (base[base["policy"] == "observed_capacity"]
                .set_index("msname")["c_total"])

    rows = []
    for alpha in alpha_values:
        per_ol, per_cost, per_margin = [], [], []
        for name, hist, test in SERVICES:
            r = simulate_policy(hist, test, delta=delta, W=W, alpha=alpha,
                                use_margin=True, use_guardrail=False)
            per_ol.append(r["overload_fraction"])
            oc = obs_cost.get(name, np.nan)
            per_cost.append(r["c_total"] / oc if not np.isnan(oc) else np.nan)
            per_margin.append(r["margin_mean"])
        rows.append({
            "alpha":            alpha,
            "median_ol":        float(np.median(per_ol)),
            "p10_ol":           float(np.quantile(per_ol, 0.10)),
            "p90_ol":           float(np.quantile(per_ol, 0.90)),
            "median_rel_cost":  float(np.nanmedian(per_cost)),
            "p10_rel_cost":     float(np.nanquantile(per_cost, 0.10)),
            "p90_rel_cost":     float(np.nanquantile(per_cost, 0.90)),
            "median_margin":    float(np.median(per_margin)),
        })
        print(f"  α={alpha:.2f}  overload={rows[-1]['median_ol']:.4f}  "
              f"cost={rows[-1]['median_rel_cost']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp4_alpha_sweep.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(6.0, 4.0))
    ax2 = ax1.twinx()

    ax1.plot(df["alpha"], df["median_ol"] * 100, color=PAL["persistence"],
             marker="o", linewidth=1.8, markersize=7, label="Overload (%)")
    ax1.fill_between(df["alpha"], df["p10_ol"]*100, df["p90_ol"]*100,
                     alpha=0.2, color=PAL["persistence"])
    ax1.axhline(delta * 100, color=PAL["persistence"], linestyle="--",
                linewidth=1.0, label=f"Nominal δ={delta*100:.0f}%")
    ax1.set_xlabel("Conservativeness factor α")
    ax1.set_ylabel("Median overload fraction (%)", color=PAL["persistence"])

    ax2.plot(df["alpha"], df["median_rel_cost"], color=PAL["xgb"],
             marker="s", linewidth=1.8, markersize=7, label="Relative cost")
    ax2.set_ylabel("Median relative cost", color=PAL["xgb"])

    ax1.set_title("")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left")
    fig.tight_layout()
    save(fig, "exp4_alpha_sweep")
    return df


def exp5_guardrail_ablation():
    """
    Compare B6 (margin+guardrail), B7 (margin-only), B8 (guardrail-only)
    on (a) natural trace and (b) synthetically shifted trace.
    """
    print("\n=== EXP-5: Guard-Rail Ablation ===")

    delta = 0.05
    W     = 240

    # Part A: natural trace
    rows_nat = []
    for variant, (use_margin, use_guardrail) in {
        "B7_margin_only":   (True,  False),
        "B6_full":          (True,  True),
        "B8_guardrail_only":(False, True),
    }.items():
        per_ol, per_cost = [], []
        for name, hist, test in SERVICES:
            r = simulate_policy(hist, test, delta=delta, W=W,
                                use_margin=use_margin,
                                use_guardrail=use_guardrail)
            per_ol.append(r["overload_fraction"])
            per_cost.append(r["c_total"])
        rows_nat.append({
            "variant": variant,
            "setting": "natural",
            "median_ol": float(np.median(per_ol)),
            "p10_ol":    float(np.quantile(per_ol, 0.10)),
            "p90_ol":    float(np.quantile(per_ol, 0.90)),
            "median_cost": float(np.median(per_cost)),
        })
        print(f"  {variant:<25s}  overload={rows_nat[-1]['median_ol']:.4f}")

    # Part B: synthetic shift (+50% at midpoint for 60 steps)
    rows_shift = []
    for variant, (use_margin, use_guardrail) in {
        "B7_margin_only":   (True,  False),
        "B6_full":          (True,  True),
        "B8_guardrail_only":(False, True),
    }.items():
        per_ol_total, per_ol_shift, per_max_run = [], [], []
        for name, hist, test in SERVICES:
            test_y  = test["cpu_sum"].to_numpy(dtype=float)
            n       = len(test_y)
            shifted = test_y.copy()
            mid     = n // 2
            shifted[mid: mid + 60] *= 1.5   # +50% demand burst

            r = simulate_policy(hist, test, demand_override=shifted,
                                delta=delta, W=W,
                                use_margin=use_margin,
                                use_guardrail=use_guardrail)
            per_ol_total.append(r["overload_fraction"])
            # Overload fraction specifically in the shift window
            ol_shift = float(np.mean(r["overload_arr"][mid: mid + 60]))
            per_ol_shift.append(ol_shift)
            per_max_run.append(r["max_overload_run"])

        rows_shift.append({
            "variant": variant,
            "setting": "synthetic_shift_50pct",
            "median_ol_total":  float(np.median(per_ol_total)),
            "median_ol_shift":  float(np.median(per_ol_shift)),
            "p10_ol_shift":     float(np.quantile(per_ol_shift, 0.10)),
            "p90_ol_shift":     float(np.quantile(per_ol_shift, 0.90)),
            "median_max_run":   float(np.median(per_max_run)),
        })
        print(f"  {variant:<25s}  ol_total={rows_shift[-1]['median_ol_total']:.4f}  "
              f"ol_shift_window={rows_shift[-1]['median_ol_shift']:.4f}  "
              f"max_run={rows_shift[-1]['median_max_run']:.1f}")

    # Also sweep shift magnitudes
    rows_mag = []
    magnitudes = [0.0, 0.30, 0.50, 1.00]
    for mag in magnitudes:
        per_ol_b6, per_ol_b7 = [], []
        for name, hist, test in SERVICES:
            test_y  = test["cpu_sum"].to_numpy(dtype=float)
            shifted = test_y.copy()
            if mag > 0:
                mid = len(test_y) // 2
                shifted[mid: mid + 60] *= (1.0 + mag)
            for use_margin, use_guard, storage in [
                (True, True,  per_ol_b6),
                (True, False, per_ol_b7),
            ]:
                r = simulate_policy(hist, test, demand_override=shifted,
                                    delta=delta, W=W,
                                    use_margin=use_margin,
                                    use_guardrail=use_guard)
                mid_s  = len(test_y) // 2
                ol_win = float(np.mean(r["overload_arr"][mid_s: mid_s + 60]))
                storage.append(ol_win)
        rows_mag.append({
            "magnitude": mag,
            "B6_full_ol_shift": float(np.median(per_ol_b6)),
            "B7_margin_only_ol_shift": float(np.median(per_ol_b7)),
        })

    df_nat   = pd.DataFrame(rows_nat)
    df_shift = pd.DataFrame(rows_shift)
    df_mag   = pd.DataFrame(rows_mag)
    df_nat.to_csv(RESULTS_DIR   / "exp5a_guardrail_natural.csv",   index=False)
    df_shift.to_csv(RESULTS_DIR / "exp5b_guardrail_shift.csv",     index=False)
    df_mag.to_csv(RESULTS_DIR   / "exp5c_guardrail_magnitude.csv", index=False)

    # Figure: three panels
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel A: natural trace
    variants  = ["B7_margin_only", "B6_full", "B8_guardrail_only"]
    var_labels = ["Margin only\n(B7)", "Margin +\nGuardrail (B6)", "Guardrail only\n(B8)"]
    cols      = [PAL["margin_only"], PAL["guardrail"], PAL["guardrail_only"]]
    ax = axes[0]
    nat_ols = [df_nat[df_nat["variant"] == v]["median_ol"].values[0] * 100 for v in variants]
    bars = ax.bar(var_labels, nat_ols, color=cols, alpha=0.85, edgecolor="white")
    ax.axhline(delta * 100, color="black", linestyle="--", linewidth=1.0,
               label=f"δ = {delta}")
    for bar, v in zip(bars, nat_ols):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_title("A. Natural Trace")
    ax.set_ylabel("Median overload (%)")
    ax.legend(fontsize=8)

    # Panel B: shift window
    shift_ols = [df_shift[df_shift["variant"] == v]["median_ol_shift"].values[0] * 100
                 for v in variants]
    bars2 = axes[1].bar(var_labels, shift_ols, color=cols, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars2, shift_ols):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.2, f"{v:.1f}%",
                     ha="center", va="bottom", fontsize=8)
    axes[1].set_title("B. Synthetic Shift Window\n(+50% demand, 60 steps)")
    axes[1].set_ylabel("Median overload in shift window (%)")

    # Panel C: shift magnitude
    axes[2].plot(df_mag["magnitude"] * 100, df_mag["B6_full_ol_shift"] * 100,
                 color=PAL["guardrail"], marker="o", linewidth=1.8,
                 markersize=7, label="B6: Margin + Guardrail")
    axes[2].plot(df_mag["magnitude"] * 100, df_mag["B7_margin_only_ol_shift"] * 100,
                 color=PAL["margin_only"], marker="s", linewidth=1.8,
                 markersize=7, linestyle="--", label="B7: Margin only")
    axes[2].set_xlabel("Demand shift magnitude (%)")
    axes[2].set_ylabel("Median overload in shift window (%)")
    axes[2].set_title("C. Recovery vs Shift Magnitude")
    axes[2].legend()

    fig.tight_layout()
    save(fig, "exp5_guardrail_ablation")
    return df_nat, df_shift, df_mag


def exp7_runlength_analysis():
    """
    Compare overload run-length distributions across policies matched to
    similar overload fractions: reactive vs. proposed conformal.
    """
    print("\n=== EXP-7: Run-Length Analysis ===")

    delta = 0.05
    W     = 240

    # Load baseline per-service (reactive threshold)
    base_per = pd.read_csv(RESULTS_DIR / "baseline_replay_per_service.csv")
    react_oc = (base_per[base_per["policy"] == "observed_capacity"]
                .set_index("msname")["c_total"])

    all_runs = {"B1_reactive": [], "B6_conformal": [], "B2_pure_predictive": []}

    for name, hist, test in SERVICES:
        test_y = test["cpu_sum"].to_numpy(dtype=float)

        # B6: Proposed conformal
        r6 = simulate_policy(hist, test, delta=delta, W=W,
                              use_margin=True, use_guardrail=True)
        all_runs["B6_conformal"].extend(r6["run_lengths"])

        # B2: Pure predictive (no margin, no guardrail)
        r2 = simulate_policy(hist, test, delta=delta, W=W,
                              use_margin=False, use_guardrail=False)
        all_runs["B2_pure_predictive"].extend(r2["run_lengths"])

        # B1: Reactive (simulate threshold u70)
        prev = float(hist["cpu_sum"].iloc[-1])
        mu   = 1.0
        caps = []
        for demand in test_y:
            cap = max(int(np.ceil(prev / (0.70 * mu))), 1)
            caps.append(cap)
            prev = demand
        cap_arr = np.array(caps)
        ol_arr  = test_y > mu * cap_arr
        from exp_core import _runlengths
        all_runs["B1_reactive"].extend(_runlengths(ol_arr))

    rows = []
    for policy, runs in all_runs.items():
        if not runs:
            runs = [0]
        rows.append({
            "policy":     policy,
            "n_runs":     len(runs),
            "mean_run":   float(np.mean(runs)),
            "median_run": float(np.median(runs)),
            "p95_run":    float(np.quantile(runs, 0.95)),
            "max_run":    int(np.max(runs)),
        })
        print(f"  {policy:<25s}  n_runs={rows[-1]['n_runs']:5d}  "
              f"mean={rows[-1]['mean_run']:.2f}  max={rows[-1]['max_run']:4d}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp7_runlength.csv", index=False)

    # Figure: survival function
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    policy_labels = {
        "B1_reactive":       ("Reactive threshold (B1)", PAL["guardrail_only"], "-"),
        "B2_pure_predictive":("Pure predictive (B2)",    PAL["arima"],          "--"),
        "B6_conformal":      ("Conformal policy (B6)",   PAL["persistence"],    "-"),
    }
    for key, (label, col, ls) in policy_labels.items():
        runs = sorted(all_runs[key])
        if not runs:
            continue
        max_r = max(runs)
        x = np.arange(1, max_r + 2)
        # P(run length > x)
        y = [float(np.mean(np.array(runs) >= xi)) for xi in x]
        ax.step(x, y, color=col, linewidth=1.8, linestyle=ls, label=label, where="post")

    ax.set_xlabel("Overload run length (consecutive steps)")
    ax.set_ylabel("Survival probability P(run > x)")
    ax.set_title("EXP-7: Overload Run-Length Survival Function\n"
                 "(δ=0.05, W=240; all services combined)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlim(left=1)
    fig.tight_layout()
    save(fig, "exp7_runlength")
    return df


def exp8_cost_decomposition():
    """
    Decompose C_total into C_res, C_act, C_vio, C_inf, C_train for each model.
    Uses measured inference times from training logs to calibrate c_inf.
    """
    print("\n=== EXP-8: Cost Decomposition ===")

    # Measured inference seconds per step (from training logs)
    # c_inf_per_step in replica-minutes = (inference_sec/step) / 60 * 1 CPU-core
    inference_per_step = {
        "persistence": 0.0,
        "arima":   1.6 * 60 / 4320 / 60,    # 1.6 min total / 4320 steps / 60 → replica-min/step
        "xgb":    14.0 * 60 / 4320 / 60,
        "lstm":    1.1 * 60 / 4320 / 60,
    }
    # c_train_per_service in replica-minutes (training time per service)
    training_per_service = {
        "persistence": 0.0,
        "arima":  68.0 / 20,     # total minutes / 20 services = min/service
        "xgb":   537.7 / 20,
        "lstm":  302.6 / 20,
    }

    delta = 0.05
    W     = 240
    fc_data = {m: load_forecast(m) for m in ["arima", "xgb", "lstm"]}

    base_per = pd.read_csv(RESULTS_DIR / "baseline_replay_per_service.csv")
    obs_cost = (base_per[base_per["policy"] == "observed_capacity"]
                .set_index("msname")["c_total"])

    rows = []
    for model in ["persistence", "arima", "xgb", "lstm"]:
        c_inf_ps  = inference_per_step[model]
        c_train_s = training_per_service[model]
        totals = {"c_res": [], "c_act": [], "c_vio": [], "c_inf": [], "c_train": []}
        for name, hist, test in SERVICES:
            if model == "persistence":
                fc = None
            else:
                sub = fc_data[model]
                sub = sub[sub["msname"] == name].sort_values("timestamp")
                fc  = sub["forecast"].to_numpy() if len(sub) == len(test) else None

            r = simulate_policy(hist, test, fc, delta=delta, W=W,
                                use_margin=True, use_guardrail=True,
                                c_inf_per_step=c_inf_ps,
                                c_train_total=c_train_s)
            for k in totals:
                totals[k].append(r[k])

        oc = obs_cost.reindex([s[0] for s in SERVICES]).values
        c_total_arr = np.array(totals["c_res"]) + np.array(totals["c_act"]) + \
                      np.array(totals["c_vio"]) + np.array(totals["c_inf"]) + \
                      np.array(totals["c_train"])
        rel_cost = c_total_arr / np.where(oc > 0, oc, np.nan)

        rows.append({
            "model":             model,
            "median_c_res_pct":  float(np.median([totals["c_res"][i] / c_total_arr[i] * 100
                                                   for i in range(len(SERVICES))])),
            "median_c_act_pct":  float(np.median([totals["c_act"][i] / c_total_arr[i] * 100
                                                   for i in range(len(SERVICES))])),
            "median_c_vio_pct":  float(np.median([totals["c_vio"][i] / c_total_arr[i] * 100
                                                   for i in range(len(SERVICES))])),
            "median_c_inf_pct":  float(np.median([totals["c_inf"][i] / c_total_arr[i] * 100
                                                   for i in range(len(SERVICES))])),
            "median_c_train_pct":float(np.median([totals["c_train"][i] / c_total_arr[i] * 100
                                                   for i in range(len(SERVICES))])),
            "median_rel_cost":   float(np.nanmedian(rel_cost)),
            "c_inf_per_step":    c_inf_ps,
            "c_train_per_svc":   c_train_s,
        })
        print(f"  {model:<12s}  C_res={rows[-1]['median_c_res_pct']:.1f}%  "
              f"C_act={rows[-1]['median_c_act_pct']:.1f}%  "
              f"C_vio={rows[-1]['median_c_vio_pct']:.1f}%  "
              f"C_inf={rows[-1]['median_c_inf_pct']:.3f}%  "
              f"C_train={rows[-1]['median_c_train_pct']:.3f}%")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp8_cost_decomposition.csv", index=False)

    # Figure: stacked bars
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    models = ["persistence", "arima", "xgb", "lstm"]
    comp_colors = {"C_res": "#1565C0", "C_act": "#43A047",
                   "C_vio": "#E53935", "C_inf": "#FB8C00", "C_train": "#8E24AA"}
    bottoms = np.zeros(len(models))
    for comp, col in comp_colors.items():
        col_key = f"median_{comp.lower()}_pct"
        vals = [df[df["model"] == m][col_key].values[0] for m in models]
        ax.bar(models, vals, bottom=bottoms, color=col, alpha=0.85,
               label=comp, edgecolor="white")
        # Annotate if > 0.5%
        for i, v in enumerate(vals):
            if v > 0.5:
                ax.text(i, bottoms[i] + v / 2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=7.5, color="white",
                        fontweight="bold")
        bottoms += np.array(vals)

    ax.set_ylabel("Cost component share (%)")
    ax.set_title("EXP-8: Cost Decomposition (δ=0.05, W=240, guardrail)\n"
                 "C_inf and C_train calibrated from measured M1 inference/training times")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    ax.set_ylim(0, 110)
    fig.tight_layout()
    save(fig, "exp8_cost_decomposition")
    return df


def exp9_penalty_sensitivity():
    """
    Sweep c_vio ∈ {1, 5, 10, 50, 100}.
    Shows that proposed policy's coverage is independent of c_vio,
    while pure predictive (B2) trades off cost vs violations based on c_vio.
    """
    print("\n=== EXP-9: Penalty Sensitivity (c_vio sweep) ===")

    c_vio_values = [1, 5, 10, 50, 100]
    delta = 0.05
    W     = 240

    base_per = pd.read_csv(RESULTS_DIR / "baseline_replay_per_service.csv")
    obs_cost = (base_per[base_per["policy"] == "observed_capacity"]
                .set_index("msname")["c_total"])

    rows = []
    for c_vio in c_vio_values:
        for variant, (use_margin, use_guard) in {
            "B6_proposed":     (True,  True),
            "B2_pure_predict": (False, False),
        }.items():
            per_ol, per_cost = [], []
            for name, hist, test in SERVICES:
                r = simulate_policy(hist, test, delta=delta, W=W,
                                    use_margin=use_margin,
                                    use_guardrail=use_guard,
                                    c_vio=float(c_vio))
                per_ol.append(r["overload_fraction"])
                oc = obs_cost.get(name, np.nan)
                per_cost.append(r["c_total"] / oc if not np.isnan(oc) else np.nan)

            rows.append({
                "c_vio":           c_vio,
                "variant":         variant,
                "median_ol":       float(np.median(per_ol)),
                "median_rel_cost": float(np.nanmedian(per_cost)),
            })
        print(f"  c_vio={c_vio:3d}  "
              f"B6 ol={rows[-2]['median_ol']:.4f} cost={rows[-2]['median_rel_cost']:.4f}  "
              f"B2 ol={rows[-1]['median_ol']:.4f} cost={rows[-1]['median_rel_cost']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp9_penalty_sensitivity.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    for variant, col, ls in [("B6_proposed", PAL["persistence"], "-"),
                              ("B2_pure_predict", PAL["arima"],    "--")]:
        sub = df[df["variant"] == variant].sort_values("c_vio")
        lbl = {"B6_proposed": "B6: Proposed (margin+guardrail)",
               "B2_pure_predict": "B2: Pure predictive (no margin)"}[variant]
        ax1.plot(sub["c_vio"], sub["median_rel_cost"], color=col,
                 marker="o", linewidth=1.8, markersize=7,
                 linestyle=ls, label=lbl)
        ax2.plot(sub["c_vio"], sub["median_ol"] * 100, color=col,
                 marker="o", linewidth=1.8, markersize=7,
                 linestyle=ls, label=lbl)

    ax1.set_xlabel("Violation penalty c_vio")
    ax1.set_ylabel("Median relative cost")
    ax1.set_title("Resource Cost vs c_vio")
    ax1.legend(fontsize=8)
    ax1.set_xscale("log")

    ax2.axhline(delta * 100, color="black", linestyle=":", linewidth=1.0,
                label=f"Nominal δ = {delta*100:.0f}%")
    ax2.set_xlabel("Violation penalty c_vio")
    ax2.set_ylabel("Median overload fraction (%)")
    ax2.set_title("Overload Rate vs c_vio")
    ax2.legend(fontsize=8)
    ax2.set_xscale("log")

    fig.suptitle("EXP-9: Penalty Sensitivity — c_vio Sweep (δ=0.05, W=240)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "exp9_penalty_sensitivity")
    return df


def exp10_recovery_time():
    """
    Measure the number of steps after a synthetic demand shift until the
    rolling 30-step overload fraction returns below δ.
    Compare B6 (full) vs B7 (margin-only).
    """
    print("\n=== EXP-10: Recovery-Time Analysis ===")

    delta        = 0.05
    W            = 240
    magnitudes   = [0.30, 0.50, 1.00]
    RECOVERY_WIN = 30    # rolling window for recovery check

    def recovery_steps(ol_arr: np.ndarray, shift_start: int,
                       delta_: float, window: int) -> int:
        """Steps after shift_start until rolling mean(ol[t:t+window]) < delta_."""
        for t in range(shift_start, len(ol_arr) - window + 1):
            if np.mean(ol_arr[t: t + window]) <= delta_:
                return t - shift_start
        return len(ol_arr) - shift_start  # did not recover

    rows = []
    for mag in magnitudes:
        for variant, (use_margin, use_guard) in {
            "B6_full":        (True,  True),
            "B7_margin_only": (True,  False),
        }.items():
            rec_times = []
            for name, hist, test in SERVICES:
                test_y  = test["cpu_sum"].to_numpy(dtype=float)
                n       = len(test_y)
                shifted = test_y.copy()
                mid     = n // 2
                shifted[mid: mid + 60] *= (1.0 + mag)

                r = simulate_policy(hist, test, demand_override=shifted,
                                    delta=delta, W=W,
                                    use_margin=use_margin,
                                    use_guardrail=use_guard)
                rec = recovery_steps(r["overload_arr"], mid + 60, delta, RECOVERY_WIN)
                rec_times.append(rec)

            rows.append({
                "magnitude":    mag,
                "variant":      variant,
                "median_rec":   float(np.median(rec_times)),
                "p10_rec":      float(np.quantile(rec_times, 0.10)),
                "p90_rec":      float(np.quantile(rec_times, 0.90)),
                "max_rec":      int(np.max(rec_times)),
            })
            print(f"  mag={mag:.0%}  {variant:<18s}  "
                  f"median_rec={rows[-1]['median_rec']:.0f}  "
                  f"max_rec={rows[-1]['max_rec']}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "exp10_recovery_time.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for variant, col, ls, mk in [
        ("B6_full",        PAL["guardrail"],    "-",  "o"),
        ("B7_margin_only", PAL["margin_only"], "--",  "s"),
    ]:
        sub = df[df["variant"] == variant].sort_values("magnitude")
        ax.errorbar(sub["magnitude"] * 100, sub["median_rec"],
                    yerr=[sub["median_rec"] - sub["p10_rec"],
                          sub["p90_rec"] - sub["median_rec"]],
                    color=col, marker=mk, markersize=7,
                    linewidth=1.8, capsize=4, linestyle=ls,
                    label=variant.replace("_", " ").title())

    ax.set_xlabel("Demand shift magnitude (%)")
    ax.set_ylabel(f"Median recovery steps\n(steps until rolling {RECOVERY_WIN}-step OL ≤ δ)")
    ax.set_title("EXP-10: Recovery Time vs Shift Magnitude\n"
                 "(δ=0.05, W=240, persistence forecast)")
    ax.legend()
    ax.set_xticks([30, 50, 100])
    ax.set_xticklabels(["30%", "50%", "100%"])
    fig.tight_layout()
    save(fig, "exp10_recovery_time")
    return df


if __name__ == "__main__":
    import datetime
    print(f"\n{'='*72}")
    print(f"  Supplemental Experiments  —  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Services loaded: {len(SERVICES)}")
    print(f"{'='*72}")

    df_exp2  = exp2_coverage_calibration()
    df_exp3  = exp3_window_sensitivity()
    df_exp4  = exp4_alpha_sweep()
    df5a, df5b, df5c = exp5_guardrail_ablation()
    df_exp7  = exp7_runlength_analysis()
    df_exp8  = exp8_cost_decomposition()
    df_exp9  = exp9_penalty_sensitivity()
    df_exp10 = exp10_recovery_time()

    print(f"\n{'='*72}")
    print("  All experiments complete.")
    print(f"  Results → {RESULTS_DIR}/")
    print(f"  Figures → {FIGURES}/")
    print(f"{'='*72}\n")
