"""
run_ls_actuation_delay.py
=========================
Large-scale actuation-delay sensitivity.

Capacity decisions computed at step t are applied at t + tau. The first tau
test steps use the starting capacity from the end of the history window.
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


DELTA = 0.05
TAUS = [0, 1, 2, 5]
W = 240
ALPHA = 1.0
RHO = 0.70
GUARDRAIL_H = 2
GUARDRAIL_GAMMA = 1
MU = 1.0

HPA_GRID_PATH = RESULTS_LS / "ls_hpa_grid_200.csv"
OUT_PATH = RESULTS_LS / "ls_actuation_delay_200.csv"

PAL = {
    "B6: conformal + guardrail": "#4CAF50",
    "B1: tuned HPA": "#F44336",
}
MARKERS = {
    "B6: conformal + guardrail": "o",
    "B1: tuned HPA": "P",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
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
        "overload_fraction": float(overload.mean()),
        "c_res": c_res,
        "c_act": c_act,
        "c_vio": c_vio,
        "c_inf": c_inf,
        "c_train": c_train,
        "c_total": c_res + c_act + c_vio + c_inf + c_train,
        "max_overload_run": int(max(runs) if runs else 0),
        "run_lengths": runs,
        "scaling_churn": int(actions.sum()),
    }


def observed_cost(test: pd.DataFrame) -> float:
    return metrics_from_capacity(test, test["replica_count"].to_numpy(dtype=float))["c_total"]


def starting_capacity(history: pd.DataFrame) -> int:
    if "replica_count" in history.columns and not history.empty:
        return max(int(np.ceil(float(history["replica_count"].iloc[-1]))), 1)
    demand = float(history["cpu_sum"].iloc[-1]) if not history.empty else 1.0
    return max(int(np.ceil(demand / MU)), 1)


def apply_actuation_delay(planned_capacity: np.ndarray, tau: int, initial_capacity: int) -> np.ndarray:
    planned = np.maximum(np.ceil(planned_capacity).astype(int), 1)
    if tau == 0:
        return planned.copy()
    if tau < 0:
        raise ValueError("tau must be non-negative")
    applied = np.empty_like(planned)
    fill = min(tau, len(planned))
    applied[:fill] = initial_capacity
    if tau < len(planned):
        applied[tau:] = planned[:-tau]
    return applied


def best_hpa_threshold() -> float:
    if not HPA_GRID_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 3 HPA grid: {HPA_GRID_PATH}")
    grid = pd.read_csv(HPA_GRID_PATH)
    if "hpa_threshold" not in grid.columns:
        raise ValueError(f"{HPA_GRID_PATH} missing hpa_threshold")
    best = grid.sort_values(["median_rel_cost", "median_overload_fraction"]).iloc[0]
    return float(best["hpa_threshold"])


def q10(s: pd.Series) -> float:
    return float(s.quantile(0.10))


def q90(s: pd.Series) -> float:
    return float(s.quantile(0.90))


def summarize(per_service: pd.DataFrame) -> pd.DataFrame:
    grouped = per_service.groupby(["policy", "tau"], sort=False, dropna=False)
    return grouped.agg(
        services=("service_id", "nunique"),
        median_overload_fraction=("overload_fraction", "median"),
        p10_overload_fraction=("overload_fraction", q10),
        p90_overload_fraction=("overload_fraction", q90),
        median_rel_cost=("rel_cost", "median"),
        p10_rel_cost=("rel_cost", q10),
        p90_rel_cost=("rel_cost", q90),
        frac_within_delta=("within_delta", "mean"),
        median_max_overload_run=("max_overload_run", "median"),
        median_scaling_churn=("scaling_churn", "median"),
    ).reset_index()


def assert_clean(summary: pd.DataFrame) -> None:
    nums = summary.select_dtypes(include=[np.number])
    if not np.isfinite(nums.to_numpy()).all():
        raise ValueError("summary contains NaN or infinite values")
    bad = summary[(summary["frac_within_delta"] < 0) | (summary["frac_within_delta"] > 1)]
    if not bad.empty:
        raise ValueError("frac_within_delta outside [0, 1]")


def save(fig: plt.Figure, name: str) -> None:
    FIGS_LS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS_LS / f"{name}.pdf")
    fig.savefig(FIGS_LS / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved {FIGS_LS / (name + '.pdf')}")
    print(f"saved {FIGS_LS / (name + '.png')}")


def plot_delay(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for policy in ["B6: conformal + guardrail", "B1: tuned HPA"]:
        sub = summary[summary["policy"] == policy].sort_values("tau")
        ax.plot(
            sub["tau"],
            sub["median_overload_fraction"] * 100.0,
            color=PAL[policy],
            marker=MARKERS[policy],
            linewidth=1.8,
            markersize=7,
            label=policy,
        )
        ax.fill_between(
            sub["tau"],
            sub["p10_overload_fraction"] * 100.0,
            sub["p90_overload_fraction"] * 100.0,
            color=PAL[policy],
            alpha=0.12,
            linewidth=0,
        )
    ax.axhline(DELTA * 100.0, color="black", linestyle=":", linewidth=1.2, label="δ=0.05")
    ax.set_xlabel("Actuation delay τ (steps)")
    ax.set_ylabel("Realised overload (%)")
    ax.set_title("Actuation-delay sensitivity on 200 services")
    ax.xaxis.set_major_locator(mticker.FixedLocator(TAUS))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "ls_fig_actuation_delay_200")


def main() -> None:
    RESULTS_LS.mkdir(parents=True, exist_ok=True)
    FIGS_LS.mkdir(parents=True, exist_ok=True)

    hpa_u = best_hpa_threshold()
    split_def = load_split_def()
    services = load_all_services_200(split_def)
    rows = []

    for idx, (service, history, test) in enumerate(services, start=1):
        denom = observed_cost(test)
        init_cap = starting_capacity(history)

        b6_base = simulate_policy(
            history,
            test,
            forecast_array=None,
            delta=DELTA,
            W=W,
            alpha=ALPHA,
            mu=MU,
            rho=RHO,
            guardrail_h=GUARDRAIL_H,
            guardrail_gamma=GUARDRAIL_GAMMA,
            use_margin=True,
            use_guardrail=True,
            **SIM_COST,
        )
        b6_planned = b6_base["capacities"]
        b1_planned = reactive_threshold_capacity(
            history,
            test,
            mu=MU,
            target_utilization=hpa_u,
            cooldown=3,
        )

        for policy, planned, threshold in [
            ("B6: conformal + guardrail", b6_planned, np.nan),
            ("B1: tuned HPA", b1_planned, hpa_u),
        ]:
            for tau in TAUS:
                applied = apply_actuation_delay(planned, tau, init_cap)
                metrics = metrics_from_capacity(test, applied)
                rows.append(
                    {
                        "service_id": service,
                        "policy": policy,
                        "tau": tau,
                        "delta": DELTA,
                        "hpa_threshold": threshold,
                        "overload_fraction": metrics["overload_fraction"],
                        "within_delta": bool(metrics["overload_fraction"] <= DELTA),
                        "rel_cost": float(metrics["c_total"] / denom),
                        "observed_c_total": denom,
                        "c_total": metrics["c_total"],
                        "c_res": metrics["c_res"],
                        "c_act": metrics["c_act"],
                        "c_vio": metrics["c_vio"],
                        "c_inf": metrics["c_inf"],
                        "c_train": metrics["c_train"],
                        "max_overload_run": metrics["max_overload_run"],
                        "scaling_churn": metrics["scaling_churn"],
                    }
                )

        if idx % 25 == 0 or idx == len(services):
            print(f"processed {idx}/{len(services)} services")

    per_service = pd.DataFrame(rows)
    summary = summarize(per_service)
    summary["selected_hpa_threshold"] = hpa_u
    assert_clean(summary)
    summary.to_csv(OUT_PATH, index=False)
    plot_delay(summary)

    print("per_service_rows", len(per_service))
    print("services", per_service["service_id"].nunique())
    print("hpa_threshold", hpa_u)
    print("delta", DELTA)
    print("taus", ",".join(str(t) for t in TAUS))
    print("delay_model", "applied[t]=initial_capacity for t<tau else planned[t-tau]")
    print("\nSUMMARY")
    print(summary.to_csv(index=False).strip())
    print("\noutputs")
    for path in [
        OUT_PATH,
        FIGS_LS / "ls_fig_actuation_delay_200.pdf",
        FIGS_LS / "ls_fig_actuation_delay_200.png",
    ]:
        print(path)


if __name__ == "__main__":
    main()
