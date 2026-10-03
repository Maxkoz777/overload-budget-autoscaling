"""
run_ls_baselines.py
===================
Large-scale baselines for 200 services.

The script compares tuned reactive HPA, pure predictive, fixed/Gaussian
margin baselines, conformal variants, guardrail-only, and an empirical
percentile margin baseline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("experiments/src")))

from exp_core_ls import (  # noqa: E402
    FIGS_LS,
    GROUP_MAP_200,
    RESULTS_LS,
    SIM_COST,
    load_all_services_200,
    load_split_def,
    simulate_policy,
)
from policies import reactive_threshold_capacity  # noqa: E402


MPLCONFIG = FIGS_LS / ".mplconfig"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIG))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402


DELTAS = [0.10, 0.05, 0.03, 0.01]
PRIMARY_DELTA = 0.05
HPA_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
W = 240
ALPHA = 1.0
RHO = 0.70
GUARDRAIL_H = 2
GUARDRAIL_GAMMA = 1
MU = 1.0

SUMMARY_PATH = RESULTS_LS / "ls_baselines_summary_200.csv"
HPA_GRID_PATH = RESULTS_LS / "ls_hpa_grid_200.csv"
PERCENTILE_PATH = RESULTS_LS / "ls_percentile_vs_conformal_200.csv"
RUNLENGTH_PATH = RESULTS_LS / "ls_runlength_200.csv"

PAL = {
    "B1": "#F44336",
    "B2": "#9E9E9E",
    "B3": "#795548",
    "B4": "#E91E63",
    "B6": "#4CAF50",
    "B7": "#2196F3",
    "B8": "#607D8B",
    "B9": "#FF9800",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def runlengths(flags: np.ndarray) -> list[int]:
    runs: list[int] = []
    cur = 0
    for flag in flags.astype(bool):
        if flag:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def observed_cost(test: pd.DataFrame) -> float:
    cap = np.maximum(np.ceil(test["replica_count"].to_numpy(dtype=float)).astype(int), 1)
    return metrics_from_capacity(test, cap)["c_total"]


def metrics_from_capacity(test: pd.DataFrame, capacity: np.ndarray) -> dict:
    demand = test["cpu_sum"].to_numpy(dtype=float)
    cap = np.maximum(np.ceil(capacity).astype(int), 1)
    overload = demand > (MU * cap)
    actions = np.abs(np.diff(cap, prepend=cap[0]))
    runs = runlengths(overload)
    c_res = SIM_COST["c_res"] * float(cap.sum())
    c_act = SIM_COST["c_act"] * float(actions.sum())
    c_vio = SIM_COST["c_vio"] * float(overload.sum())
    c_inf = SIM_COST["c_inf_per_step"] * float(len(demand))
    c_train = SIM_COST["c_train_total"]
    return {
        "capacities": cap,
        "overload_arr": overload,
        "run_lengths": runs,
        "overload_fraction": float(overload.mean()),
        "c_res": c_res,
        "c_act": c_act,
        "c_vio": c_vio,
        "c_inf": c_inf,
        "c_train": c_train,
        "c_total": c_res + c_act + c_vio + c_inf + c_train,
        "scaling_churn": int(actions.sum()),
        "max_overload_run": int(max(runs) if runs else 0),
    }


def simulate_fixed_margin_ls(history: pd.DataFrame, test: pd.DataFrame, delta: float) -> dict:
    """B3 formula from exp_validator_fixes: constant history residual quantile."""
    hist_y = history["cpu_sum"].to_numpy(dtype=float)
    residuals = np.maximum(hist_y[1:] - hist_y[:-1], 0.0)
    fixed_margin = float(np.quantile(residuals, 1.0 - delta)) if len(residuals) else 0.0
    prev = hist_y[-1]
    caps = []
    for demand in test["cpu_sum"].to_numpy(dtype=float):
        forecast = max(float(prev), 0.0)
        cap = max(int(np.ceil((forecast + fixed_margin) / MU)), 1)
        caps.append(cap)
        prev = demand
    out = metrics_from_capacity(test, np.asarray(caps, dtype=int))
    out["margin_mean"] = fixed_margin
    out["margin_p95"] = fixed_margin
    out["guardrail_rate"] = 0.0
    return out


def simulate_gaussian_margin_ls(history: pd.DataFrame, test: pd.DataFrame, delta: float) -> dict:
    """B4 formula from exp_validator_fixes: z_{1-delta} times rolling residual std."""
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - delta))
    hist_y = history["cpu_sum"].to_numpy(dtype=float)
    residuals = list(np.maximum(hist_y[1:] - hist_y[:-1], 0.0))
    prev = hist_y[-1]
    caps = []
    margins = []
    for demand in test["cpu_sum"].to_numpy(dtype=float):
        forecast = max(float(prev), 0.0)
        window = residuals[-W:]
        sigma = float(np.std(window)) if len(window) > 1 else 0.0
        margin = z * sigma
        caps.append(max(int(np.ceil((forecast + margin) / MU)), 1))
        margins.append(margin)
        residuals.append(max(float(demand) - forecast, 0.0))
        prev = demand
    out = metrics_from_capacity(test, np.asarray(caps, dtype=int))
    out["margin_mean"] = float(np.mean(margins)) if margins else 0.0
    out["margin_p95"] = float(np.quantile(margins, 0.95)) if margins else 0.0
    out["guardrail_rate"] = 0.0
    return out


def simulate_empirical_percentile_margin(history: pd.DataFrame, test: pd.DataFrame, delta: float) -> dict:
    """
    B9: rolling empirical percentile margin without conformal rank correction.

    margin = np.quantile(positive_residuals[-W:], 1-delta). No alpha multiplier,
    no ceil((W+1)(1-delta)) rank correction. Persistence forecast and guardrail
    logic otherwise match simulate_policy.
    """
    hist_y = history["cpu_sum"].to_numpy(dtype=float)
    test_y = test["cpu_sum"].to_numpy(dtype=float)
    residuals = list(np.maximum(hist_y[1:] - hist_y[:-1], 0.0))
    prev = hist_y[-1]
    overload_hist: list[bool] = []
    soft_hist: list[bool] = []
    caps = []
    margins = []

    for demand in test_y:
        forecast = max(float(prev), 0.0)
        window = residuals[-W:]
        margin = float(np.quantile(window, 1.0 - delta)) if window else 0.0
        margins.append(margin)
        nominal = max(int(np.ceil((forecast + margin) / MU)), 1)
        trigger = False
        if len(overload_hist) >= GUARDRAIL_H:
            trigger = all(overload_hist[-GUARDRAIL_H:]) or all(soft_hist[-GUARDRAIL_H:])
        cap = nominal + (GUARDRAIL_GAMMA if trigger else 0)
        caps.append(cap)
        overload = float(demand) > MU * cap
        soft = float(demand) > RHO * MU * cap
        residuals.append(max(float(demand) - forecast, 0.0))
        overload_hist.append(bool(overload))
        soft_hist.append(bool(soft))
        prev = demand

    out = metrics_from_capacity(test, np.asarray(caps, dtype=int))
    out["margin_mean"] = float(np.mean(margins)) if margins else 0.0
    out["margin_p95"] = float(np.quantile(margins, 0.95)) if margins else 0.0
    out["guardrail_rate"] = float(
        np.mean(
            [
                all(overload_hist[max(0, i - GUARDRAIL_H):i])
                or all(soft_hist[max(0, i - GUARDRAIL_H):i])
                for i in range(GUARDRAIL_H, len(test_y))
            ]
        )
    ) if len(test_y) > GUARDRAIL_H else 0.0
    return out


def row_from_result(
    *,
    service: str,
    stratum: str,
    policy: str,
    policy_family: str,
    delta: float,
    result: dict,
    observed_denominator: float,
    hpa_threshold: float | None = None,
) -> dict:
    return {
        "service_id": service,
        "stratum": stratum,
        "policy": policy,
        "policy_family": policy_family,
        "delta": delta,
        "hpa_threshold": hpa_threshold,
        "overload_fraction": result["overload_fraction"],
        "within_delta": bool(result["overload_fraction"] <= delta),
        "rel_cost": float(result["c_total"] / observed_denominator),
        "observed_c_total": observed_denominator,
        "c_total": result["c_total"],
        "c_res": result["c_res"],
        "c_act": result["c_act"],
        "c_vio": result["c_vio"],
        "c_inf": result["c_inf"],
        "c_train": result["c_train"],
        "margin_mean": result.get("margin_mean", 0.0),
        "margin_p95": result.get("margin_p95", 0.0),
        "guardrail_rate": result.get("guardrail_rate", 0.0),
        "max_overload_run": result["max_overload_run"],
        "scaling_churn": result["scaling_churn"],
        "run_lengths": result["run_lengths"],
    }


def q10(s: pd.Series) -> float:
    return float(s.quantile(0.10))


def q90(s: pd.Series) -> float:
    return float(s.quantile(0.90))


def run_stats(run_lists: pd.Series) -> dict:
    pooled = [int(x) for runs in run_lists for x in runs]
    if not pooled:
        return {
            "run_count": 0,
            "run_length_mean": 0.0,
            "run_length_median": 0.0,
            "run_length_p95": 0.0,
            "run_length_p99": 0.0,
            "run_length_max": 0,
        }
    arr = np.asarray(pooled, dtype=float)
    return {
        "run_count": int(len(arr)),
        "run_length_mean": float(np.mean(arr)),
        "run_length_median": float(np.median(arr)),
        "run_length_p95": float(np.quantile(arr, 0.95)),
        "run_length_p99": float(np.quantile(arr, 0.99)),
        "run_length_max": int(np.max(arr)),
    }


def summarize(per_service: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for keys, sub in per_service.groupby(["policy", "policy_family", "delta"], sort=False, dropna=False):
        policy, family, delta = keys
        stats = run_stats(sub["run_lengths"])
        groups.append(
            {
                "policy": policy,
                "policy_family": family,
                "delta": delta,
                "services": int(sub["service_id"].nunique()),
                "median_rel_cost": float(sub["rel_cost"].median()),
                "p10_rel_cost": q10(sub["rel_cost"]),
                "p90_rel_cost": q90(sub["rel_cost"]),
                "median_overload_fraction": float(sub["overload_fraction"].median()),
                "p10_overload_fraction": q10(sub["overload_fraction"]),
                "p90_overload_fraction": q90(sub["overload_fraction"]),
                "frac_within_delta": float(sub["within_delta"].mean()),
                "median_margin_mean": float(sub["margin_mean"].median()),
                "median_margin_p95": float(sub["margin_p95"].median()),
                "median_guardrail_rate": float(sub["guardrail_rate"].median()),
                "median_max_overload_run": float(sub["max_overload_run"].median()),
                "median_scaling_churn": float(sub["scaling_churn"].median()),
                **stats,
            }
        )
    return pd.DataFrame(groups)


def summarize_by_stratum(per_service: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for keys, sub in per_service.groupby(["stratum", "policy", "policy_family", "delta"], sort=True, dropna=False):
        stratum, policy, family, delta = keys
        groups.append(
            {
                "stratum": stratum,
                "policy": policy,
                "policy_family": family,
                "delta": delta,
                "services": int(sub["service_id"].nunique()),
                "median_rel_cost": float(sub["rel_cost"].median()),
                "p10_rel_cost": q10(sub["rel_cost"]),
                "p90_rel_cost": q90(sub["rel_cost"]),
                "median_overload_fraction": float(sub["overload_fraction"].median()),
                "p10_overload_fraction": q10(sub["overload_fraction"]),
                "p90_overload_fraction": q90(sub["overload_fraction"]),
                "frac_within_delta": float(sub["within_delta"].mean()),
                "median_max_overload_run": float(sub["max_overload_run"].median()),
            }
        )
    return pd.DataFrame(groups)


def save(fig: plt.Figure, name: str) -> None:
    FIGS_LS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_LS / f"{name}.pdf")
    fig.savefig(FIGS_LS / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {FIGS_LS / (name + '.pdf')}")
    print(f"saved {FIGS_LS / (name + '.png')}")


def plot_baselines(summary_005: pd.DataFrame, hpa_grid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    hpa = hpa_grid.sort_values("hpa_threshold")
    ax.plot(
        hpa["median_overload_fraction"] * 100.0,
        hpa["median_rel_cost"],
        color=PAL["B1"],
        marker="P",
        linewidth=1.6,
        label="B1 tuned HPA grid",
    )
    for _, row in hpa.iterrows():
        ax.annotate(
            f"u={row['hpa_threshold']:.1f}",
            (row["median_overload_fraction"] * 100.0, row["median_rel_cost"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color=PAL["B1"],
        )

    marker_map = {"B2": "x", "B3": "v", "B4": "^", "B6": "o", "B7": "s", "B8": "D", "B9": "*"}
    for _, row in summary_005.iterrows():
        family = row["policy_family"]
        if family == "B1":
            continue
        ax.scatter(
            row["median_overload_fraction"] * 100.0,
            row["median_rel_cost"],
            color=PAL.get(family, "#000000"),
            marker=marker_map.get(family, "o"),
            s=95 if family == "B9" else 70,
            label=f"{family}: {row['policy'].split(': ', 1)[-1]}",
            zorder=4,
        )

    ax.axvline(PRIMARY_DELTA * 100.0, color="black", linestyle=":", linewidth=1.2, label="δ=0.05")
    ax.set_xlabel("Median realised overload (%)")
    ax.set_ylabel("Median relative cost vs observed capacity")
    ax.set_title("Large-scale baseline comparison at δ=0.05")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    save(fig, "ls_fig_baselines_200")


def plot_percentile_vs_conformal(percentile: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    plot_order = [
        ("B6: conformal + guardrail", "B6", "o"),
        ("B7: conformal margin only", "B7", "s"),
        ("B9: empirical percentile + guardrail", "B9", "^"),
    ]
    for policy, family, marker in plot_order:
        sub = percentile[percentile["policy"] == policy].sort_values("delta")
        ax.plot(
            sub["delta"],
            sub["frac_within_delta"],
            color=PAL[family],
            marker=marker,
            linewidth=1.8,
            markersize=6.5,
            label=policy,
        )
    ax.set_xlabel("Nominal risk budget δ")
    ax.set_ylabel("Fraction of services with overload ≤ δ")
    ax.set_title("Conformal rank correction vs empirical percentile")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "ls_fig_percentile_vs_conformal_200")


def assert_clean(*frames: pd.DataFrame) -> None:
    for frame in frames:
        nums = frame.select_dtypes(include=[np.number])
        if not np.isfinite(nums.to_numpy()).all():
            raise ValueError("numeric output contains NaN or infinite values")
        if "frac_within_delta" in frame.columns:
            bad = frame[(frame["frac_within_delta"] < 0) | (frame["frac_within_delta"] > 1)]
            if not bad.empty:
                raise ValueError("frac_within_delta outside [0, 1]")


def main() -> None:
    RESULTS_LS.mkdir(parents=True, exist_ok=True)
    FIGS_LS.mkdir(parents=True, exist_ok=True)

    split_def = load_split_def()
    services = load_all_services_200(split_def)
    rows = []
    total = len(services)

    for idx, (service, history, test) in enumerate(services, start=1):
        stratum = GROUP_MAP_200[service]
        denom = observed_cost(test)
        if denom <= 0:
            raise ValueError(f"{service}: observed denominator is non-positive")

        for threshold in HPA_THRESHOLDS:
            cap = reactive_threshold_capacity(
                history,
                test,
                mu=MU,
                target_utilization=threshold,
                cooldown=3,
            )
            result = metrics_from_capacity(test, cap)
            rows.append(
                row_from_result(
                    service=service,
                    stratum=stratum,
                    policy=f"B1: HPA tuned u={threshold:.1f}",
                    policy_family="B1",
                    delta=PRIMARY_DELTA,
                    result=result,
                    observed_denominator=denom,
                    hpa_threshold=threshold,
                )
            )

        b2 = simulate_policy(
            history,
            test,
            forecast_array=None,
            delta=PRIMARY_DELTA,
            W=W,
            alpha=ALPHA,
            mu=MU,
            rho=RHO,
            guardrail_h=GUARDRAIL_H,
            guardrail_gamma=GUARDRAIL_GAMMA,
            use_margin=False,
            use_guardrail=False,
            **SIM_COST,
        )
        rows.append(
            row_from_result(
                service=service,
                stratum=stratum,
                policy="B2: pure predictive",
                policy_family="B2",
                delta=PRIMARY_DELTA,
                result=b2,
                observed_denominator=denom,
            )
        )

        b3 = simulate_fixed_margin_ls(history, test, PRIMARY_DELTA)
        rows.append(
            row_from_result(
                service=service,
                stratum=stratum,
                policy="B3: fixed margin",
                policy_family="B3",
                delta=PRIMARY_DELTA,
                result=b3,
                observed_denominator=denom,
            )
        )

        b4 = simulate_gaussian_margin_ls(history, test, PRIMARY_DELTA)
        rows.append(
            row_from_result(
                service=service,
                stratum=stratum,
                policy="B4: Gaussian margin",
                policy_family="B4",
                delta=PRIMARY_DELTA,
                result=b4,
                observed_denominator=denom,
            )
        )

        b8 = simulate_policy(
            history,
            test,
            forecast_array=None,
            delta=PRIMARY_DELTA,
            W=W,
            alpha=ALPHA,
            mu=MU,
            rho=RHO,
            guardrail_h=GUARDRAIL_H,
            guardrail_gamma=GUARDRAIL_GAMMA,
            use_margin=False,
            use_guardrail=True,
            **SIM_COST,
        )
        rows.append(
            row_from_result(
                service=service,
                stratum=stratum,
                policy="B8: guardrail only",
                policy_family="B8",
                delta=PRIMARY_DELTA,
                result=b8,
                observed_denominator=denom,
            )
        )

        for delta in DELTAS:
            variants = [
                (
                    "B6: conformal + guardrail",
                    "B6",
                    simulate_policy(
                        history,
                        test,
                        forecast_array=None,
                        delta=delta,
                        W=W,
                        alpha=ALPHA,
                        mu=MU,
                        rho=RHO,
                        guardrail_h=GUARDRAIL_H,
                        guardrail_gamma=GUARDRAIL_GAMMA,
                        use_margin=True,
                        use_guardrail=True,
                        **SIM_COST,
                    ),
                ),
                (
                    "B7: conformal margin only",
                    "B7",
                    simulate_policy(
                        history,
                        test,
                        forecast_array=None,
                        delta=delta,
                        W=W,
                        alpha=ALPHA,
                        mu=MU,
                        rho=RHO,
                        guardrail_h=GUARDRAIL_H,
                        guardrail_gamma=GUARDRAIL_GAMMA,
                        use_margin=True,
                        use_guardrail=False,
                        **SIM_COST,
                    ),
                ),
                (
                    "B9: empirical percentile + guardrail",
                    "B9",
                    simulate_empirical_percentile_margin(history, test, delta),
                ),
            ]
            for policy, family, result in variants:
                rows.append(
                    row_from_result(
                        service=service,
                        stratum=stratum,
                        policy=policy,
                        policy_family=family,
                        delta=delta,
                        result=result,
                        observed_denominator=denom,
                    )
                )

        if idx % 25 == 0 or idx == total:
            print(f"processed {idx}/{total} services")

    per_service = pd.DataFrame(rows)
    summary_all = summarize(per_service)
    summary_005 = summary_all[summary_all["delta"].eq(PRIMARY_DELTA)].copy()
    by_stratum = summarize_by_stratum(per_service)

    hpa_per = per_service[per_service["policy_family"].eq("B1")].copy()
    hpa_grid = summary_005[summary_005["policy_family"].eq("B1")].copy()
    threshold_map = hpa_per.groupby("policy")["hpa_threshold"].first()
    hpa_grid["hpa_threshold"] = hpa_grid["policy"].map(threshold_map).astype(float)
    hpa_grid = hpa_grid.sort_values("hpa_threshold")

    percentile = summary_all[
        summary_all["policy"].isin(
            [
                "B6: conformal + guardrail",
                "B7: conformal margin only",
                "B9: empirical percentile + guardrail",
            ]
        )
    ].copy()

    runlength = summary_all[
        [
            "policy",
            "policy_family",
            "delta",
            "services",
            "run_count",
            "run_length_mean",
            "run_length_median",
            "run_length_p95",
            "run_length_p99",
            "run_length_max",
            "median_max_overload_run",
        ]
    ].copy()

    assert_clean(summary_005, hpa_grid, percentile, runlength)

    summary_005.to_csv(SUMMARY_PATH, index=False)
    hpa_grid.to_csv(HPA_GRID_PATH, index=False)
    percentile.to_csv(PERCENTILE_PATH, index=False)
    runlength.to_csv(RUNLENGTH_PATH, index=False)
    by_stratum.to_csv(RESULTS_LS / "ls_baselines_by_stratum_200.csv", index=False)

    plot_baselines(summary_005, hpa_grid)
    plot_percentile_vs_conformal(percentile)

    print("per_service_rows", len(per_service))
    print("services", per_service["service_id"].nunique())
    print("summary_rows_delta_005", len(summary_005))
    print("rel_cost_definition", "c_total(policy) / c_total(observed_capacity)")
    print("\nSUMMARY_DELTA_005")
    print(summary_005.to_csv(index=False).strip())
    print("\nPERCENTILE_VS_CONFORMAL")
    print(percentile.to_csv(index=False).strip())
    print("\noutputs")
    for path in [
        SUMMARY_PATH,
        HPA_GRID_PATH,
        PERCENTILE_PATH,
        RUNLENGTH_PATH,
        RESULTS_LS / "ls_baselines_by_stratum_200.csv",
        FIGS_LS / "ls_fig_baselines_200.pdf",
        FIGS_LS / "ls_fig_baselines_200.png",
        FIGS_LS / "ls_fig_percentile_vs_conformal_200.pdf",
        FIGS_LS / "ls_fig_percentile_vs_conformal_200.png",
    ]:
        print(path)


if __name__ == "__main__":
    main()
