#!/usr/bin/env python3
"""Audit of the guard-rail mechanism, recovery, and closed-loop delay.

The script uses only pre-test information for the rho sweep diagnostics and
keeps all correction forms fixed before held-out evaluation.  It also replaces
the former shifted-capacity delay calculation with a causal controller whose
overload/soft-state feedback is computed from the capacity actually applied.

Run from any directory with::

    python3 audit/recompute_guardrail_actuation.py
"""

from __future__ import annotations

import json
import platform
import sys
from collections import deque
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
EXP = RESEARCH / "experiments"
DATA = EXP / "data" / "service_timeseries_200"
SELECTION = EXP / "data" / "splits" / "selected_services_200.csv"
SPLIT_PATH = EXP / "data" / "splits" / "split_definition.json"
REACTIVE_PARAMETERS = PAPER / "audit" / "comparative_results" / "reactive_selected_parameters.csv"
OUT = PAPER / "audit" / "guardrail_results"
FIGURES = PAPER / "figures"

DELTA = 0.05
W = 240
H = 2
MU = 1.0
RHOS = (0.70, 0.80, 0.90, 0.95, 0.99)
TAUS = (0, 1, 2, 5)
ALPHAS = (1.0, 1.25, 1.5, 2.0)
SHIFTS = (0.30, 0.50, 1.00)
RECOVERY_WINDOW = 60


def split_frame(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    return frame[
        frame.timestamp.ge(spec["timestamp_start"])
        & frame.timestamp.lt(spec["timestamp_end_exclusive"])
    ].copy()


def load_services() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    selection = pd.read_csv(SELECTION)
    selection["service_id"] = selection.service_id.astype(str)
    if selection.is_focused_20.dtype != bool:
        selection["is_focused_20"] = selection.is_focused_20.astype(str).str.lower().isin({"true", "1"})
    split = json.loads(SPLIT_PATH.read_text())
    specs = {item["name"]: item for item in split["splits"]}
    services: list[dict[str, object]] = []
    for row in selection.itertuples(index=False):
        frame = pd.read_parquet(DATA / f"{row.service_id}.parquet").sort_values("timestamp")
        train = split_frame(frame, specs["train"])
        calibration = split_frame(frame, specs["calibration"])
        test = split_frame(frame, specs["test"])
        if (len(train), len(calibration), len(test)) != (11_520, 2_880, 4_320):
            raise AssertionError(f"unexpected split lengths for {row.service_id}")
        services.append(
            {
                "service_id": str(row.service_id),
                "focused": bool(row.is_focused_20),
                "stratum": str(row.stratum),
                "train": train,
                "calibration": calibration,
                "history": pd.concat([train, calibration], ignore_index=True),
                "test": test,
                "pretest_mean_demand": float(pd.concat([train, calibration]).cpu_sum.mean()),
            }
        )
    scale = pd.DataFrame(
        {"service_id": [s["service_id"] for s in services], "value": [s["pretest_mean_demand"] for s in services]}
    )
    scale["scale_quartile"] = pd.qcut(scale.value.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    scale_map = dict(zip(scale.service_id, scale.scale_quartile.astype(str), strict=True))
    for service in services:
        service["scale_quartile"] = scale_map[str(service["service_id"])]
    return selection, services


def observed_cost(frame: pd.DataFrame) -> float:
    demand = frame.cpu_sum.to_numpy(float)
    cap = np.maximum(np.ceil(frame.replica_count.to_numpy(float)).astype(int), 1)
    return policy_cost(demand, cap)


def policy_cost(demand: np.ndarray, cap: np.ndarray) -> float:
    actions = np.abs(np.diff(cap, prepend=cap[0]))
    return float(cap.sum() + 0.05 * actions.sum() + 10.0 * (demand > cap).sum())


def correction_value(kind: str, nominal: int, trigger: bool) -> int:
    if kind == "none":
        return 0
    if kind == "always_plus1":
        return 1
    if not trigger:
        return 0
    if kind == "fixed_plus1":
        return 1
    if kind == "proportional_5pct":
        return max(1, int(np.ceil(0.05 * nominal)))
    raise ValueError(kind)


def simulate_conformal(
    history: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    rho: float = 0.70,
    correction: str = "fixed_plus1",
    alpha: float = 1.0,
    tau: int = 0,
    demand_override: np.ndarray | None = None,
) -> dict[str, object]:
    history_y = history.cpu_sum.to_numpy(float)
    demand = evaluation.cpu_sum.to_numpy(float) if demand_override is None else np.asarray(demand_override, float)
    previous = float(history_y[-1])
    residuals = np.maximum(np.diff(history_y), 0.0).tolist()
    initial_capacity = max(int(np.ceil(float(history.replica_count.iloc[-1]))), 1)
    pending = deque([initial_capacity] * tau)
    overload_history: list[bool] = []
    soft_history: list[bool] = []
    planned: list[int] = []
    applied: list[int] = []
    nominal_values: list[int] = []
    triggers: list[bool] = []

    for actual in demand:
        window = np.asarray(residuals[-W:], dtype=float)
        level = min(len(window), int(np.ceil((len(window) + 1) * (1.0 - DELTA))))
        margin = alpha * float(np.partition(window, level - 1)[level - 1])
        nominal = max(1, int(np.ceil(previous + margin)))
        trigger = False
        if correction not in {"none", "always_plus1"} and len(overload_history) >= H:
            trigger = all(overload_history[-H:]) or all(soft_history[-H:])
        decision = nominal + correction_value(correction, nominal, trigger)
        planned.append(decision)
        nominal_values.append(nominal)
        triggers.append(trigger)
        if tau:
            pending.append(decision)
            capacity = int(pending.popleft())
        else:
            capacity = decision
        applied.append(capacity)
        overload = bool(actual > MU * capacity)
        soft = bool(actual > rho * MU * capacity)
        overload_history.append(overload)
        soft_history.append(soft)
        residuals.append(max(float(actual - previous), 0.0))
        previous = float(actual)

    cap = np.asarray(applied, dtype=int)
    overload = demand > MU * cap
    return {
        "demand": demand,
        "nominal": np.asarray(nominal_values, dtype=int),
        "planned": np.asarray(planned, dtype=int),
        "capacity": cap,
        "trigger": np.asarray(triggers, dtype=bool),
        "overload": overload,
        "overload_fraction": float(overload.mean()),
        "within_delta": bool(overload.mean() <= DELTA),
        "trigger_rate": float(np.mean(triggers)),
        "relative_cost": policy_cost(demand, cap) / observed_cost(evaluation),
    }


def simulate_reactive_closed_loop(
    history: pd.DataFrame,
    evaluation: pd.DataFrame,
    threshold: float,
    cooldown: int,
    tau: int,
) -> dict[str, object]:
    demand = evaluation.cpu_sum.to_numpy(float)
    previous = float(history.cpu_sum.iloc[-1])
    initial_capacity = max(int(np.ceil(float(history.replica_count.iloc[-1]))), 1)
    commanded = initial_capacity
    last_scale_step = -cooldown
    pending = deque([initial_capacity] * tau)
    applied: list[int] = []
    for step, actual in enumerate(demand):
        desired = max(1, int(np.ceil(previous / threshold)))
        if step - last_scale_step >= cooldown and desired != commanded:
            commanded = desired
            last_scale_step = step
        if tau:
            pending.append(commanded)
            capacity = int(pending.popleft())
        else:
            capacity = commanded
        applied.append(capacity)
        previous = float(actual)
    cap = np.asarray(applied, dtype=int)
    overload = demand > cap
    return {
        "capacity": cap,
        "overload_fraction": float(overload.mean()),
        "within_delta": bool(overload.mean() <= DELTA),
        "relative_cost": policy_cost(demand, cap) / observed_cost(evaluation),
    }


def result_row(service: dict[str, object], period: str, rho: float, correction: str, result: dict[str, object]) -> dict[str, object]:
    return {
        "service_id": service["service_id"],
        "focused": service["focused"],
        "stratum": service["stratum"],
        "scale_quartile": service["scale_quartile"],
        "period": period,
        "rho": rho,
        "correction": correction,
        "overload_fraction": result["overload_fraction"],
        "within_delta": result["within_delta"],
        "relative_cost": result["relative_cost"],
        "trigger_rate": result["trigger_rate"],
        "mean_nominal_capacity": float(np.mean(result["nominal"])),
        "fraction_capacity_one": float(np.mean(result["capacity"] == 1)),
    }


def q10(values: pd.Series) -> float:
    return float(values.quantile(0.10))


def q90(values: pd.Series) -> float:
    return float(values.quantile(0.90))


def summarize_policy(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(groups, as_index=False, dropna=False)
        .agg(
            services=("service_id", "nunique"),
            median_overload_fraction=("overload_fraction", "median"),
            median_relative_cost=("relative_cost", "median"),
            service_compliance=("within_delta", "mean"),
            median_trigger_rate=("trigger_rate", "median"),
            p10_trigger_rate=("trigger_rate", q10),
            p90_trigger_rate=("trigger_rate", q90),
        )
    )


def recovery_metrics(overload: np.ndarray, shift_start: int) -> tuple[int, int]:
    for start in range(shift_start, len(overload) - RECOVERY_WINDOW + 1):
        if float(np.mean(overload[start : start + RECOVERY_WINDOW])) <= DELTA:
            relative_start = start - shift_start
            return relative_start, relative_start + RECOVERY_WINDOW - 1
    horizon = len(overload) - shift_start
    return horizon, horizon


def save_figures(activation: pd.DataFrame, recovery_summary: pd.DataFrame, delay_summary: pd.DataFrame) -> None:
    colors = {"Q1": "#d9eaf7", "Q2": "#8ecae6", "Q3": "#219ebc", "Q4": "#023047"}
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 5.8))
    scale = activation[(activation.period == "test") & (activation.tier == "all200")]
    axes[0].bar(scale.scale_quartile, 100 * scale.median_trigger_rate, color=[colors[q] for q in scale.scale_quartile])
    axes[0].set_ylabel("Median trigger rate (%)")
    axes[0].set_xlabel("Pre-test demand-scale quartile")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Default guard activation")
    variants = ["margin_only", "fixed_guard", "always_plus1", "proportional_5pct"]
    labels = ["Margin only", "Fixed +1", "Always +1", "Proportional 5%"]
    part = recovery_summary[(recovery_summary["shift"] == 1.0) & recovery_summary.variant.isin(variants)].set_index("variant").loc[variants]
    axes[1].errorbar(
        np.arange(len(part)),
        part.median_observable_end,
        yerr=[part.median_observable_end - part.p10_observable_end, part.p90_observable_end - part.median_observable_end],
        fmt="o",
        capsize=4,
        color="#7a1f5c",
    )
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    axes[1].set_ylabel("Observable recovery time (steps)")
    axes[1].set_title("Persistent +100% shift")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.title.set_size(11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_guardrail_characterisation.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    palette = {"conformal_fixed_guard": "#2ca02c", "conformal_margin_only": "#1f77b4", "reactive_pretest_selected": "#d62728"}
    labels = {"conformal_fixed_guard": "Conformal + optional +1", "conformal_margin_only": "Conformal margin only", "reactive_pretest_selected": "Pre-test-selected reactive"}
    for policy in palette:
        part = delay_summary[(delay_summary.alpha == 1.0) & (delay_summary.policy == policy)].sort_values("tau")
        ax.plot(part.tau, 100 * part.service_compliance, marker="o", color=palette[policy], label=labels[policy])
    ax.set_xlabel(r"Actuation delay $\tau$ (steps)")
    ax.set_ylabel(r"Services with overload $\leq\delta$ (%)")
    ax.set_xticks(TAUS)
    ax.set_ylim(80, 101)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"ls_fig_actuation_delay_200.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    selection, services = load_services()

    rho_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []
    for index, service in enumerate(services, 1):
        periods = (
            ("pretest", service["train"], service["calibration"]),
            ("test", service["history"], service["test"]),
        )
        for period, history, evaluation in periods:
            for rho in RHOS:
                result = simulate_conformal(history, evaluation, rho=rho, correction="fixed_plus1")
                rho_rows.append(result_row(service, period, rho, "fixed_plus1", result))
            for correction in ("none", "fixed_plus1", "always_plus1", "proportional_5pct"):
                result = simulate_conformal(history, evaluation, rho=0.70, correction=correction)
                variant_rows.append(result_row(service, period, 0.70, correction, result))
        if index % 50 == 0:
            print(f"guard replay {index}/200", flush=True)

    rho_frame = pd.DataFrame(rho_rows)
    variants = pd.DataFrame(variant_rows)
    for frame in (rho_frame, variants):
        frame["tier"] = np.where(frame.focused, "focused20", "new180")
    rho_all = pd.concat([rho_frame, rho_frame.assign(tier="all200")], ignore_index=True)
    variants_all = pd.concat([variants, variants.assign(tier="all200")], ignore_index=True)
    rho_summary = summarize_policy(rho_all, ["tier", "period", "rho", "correction"])
    variant_summary = summarize_policy(variants_all, ["tier", "period", "rho", "correction"])

    activation = (
        variants_all[(variants_all.correction == "fixed_plus1") & variants_all.tier.isin({"all200", "focused20"})]
        .groupby(["tier", "period", "scale_quartile"], as_index=False, observed=True)
        .agg(
            services=("service_id", "nunique"),
            median_trigger_rate=("trigger_rate", "median"),
            p10_trigger_rate=("trigger_rate", q10),
            p90_trigger_rate=("trigger_rate", q90),
            median_nominal_capacity=("mean_nominal_capacity", "median"),
            median_fraction_capacity_one=("fraction_capacity_one", "median"),
        )
    )

    recovery_rows: list[dict[str, object]] = []
    focused = [service for service in services if service["focused"]]
    variant_map = {
        "margin_only": "none",
        "fixed_guard": "fixed_plus1",
        "always_plus1": "always_plus1",
        "proportional_5pct": "proportional_5pct",
    }
    for service in focused:
        base = service["test"].cpu_sum.to_numpy(float)
        shift_start = len(base) // 2
        for shift in SHIFTS:
            shifted = base.copy()
            shifted[shift_start:] *= 1.0 + shift
            for variant, correction in variant_map.items():
                result = simulate_conformal(service["history"], service["test"], rho=0.70, correction=correction, demand_override=shifted)
                start, observable_end = recovery_metrics(result["overload"], shift_start)
                recovery_rows.append(
                    {
                        "service_id": service["service_id"],
                        "shift": shift,
                        "variant": variant,
                        "retrospective_window_start": start,
                        "observable_window_end": observable_end,
                        "overload_first_120": float(np.mean(result["overload"][shift_start : shift_start + 120])),
                        "trigger_rate_post_shift": float(np.mean(result["trigger"][shift_start:])),
                    }
                )
    recovery = pd.DataFrame(recovery_rows)
    recovery_summary = (
        recovery.groupby(["shift", "variant"], as_index=False)
        .agg(
            services=("service_id", "nunique"),
            p10_retrospective_start=("retrospective_window_start", q10),
            median_retrospective_start=("retrospective_window_start", "median"),
            p90_retrospective_start=("retrospective_window_start", q90),
            max_retrospective_start=("retrospective_window_start", "max"),
            p10_observable_end=("observable_window_end", q10),
            median_observable_end=("observable_window_end", "median"),
            p90_observable_end=("observable_window_end", q90),
            max_observable_end=("observable_window_end", "max"),
            median_overload_first_120=("overload_first_120", "median"),
        )
    )

    parameters = pd.read_csv(REACTIVE_PARAMETERS).set_index("service_id")
    delay_rows: list[dict[str, object]] = []
    for index, service in enumerate(services, 1):
        sid = str(service["service_id"])
        for tau in TAUS:
            reactive = simulate_reactive_closed_loop(
                service["history"], service["test"], float(parameters.loc[sid, "threshold"]), int(parameters.loc[sid, "cooldown"]), tau
            )
            delay_rows.append({"service_id": sid, "policy": "reactive_pretest_selected", "tau": tau, "alpha": 1.0, **reactive})
            for alpha in ALPHAS:
                fixed = simulate_conformal(service["history"], service["test"], correction="fixed_plus1", alpha=alpha, tau=tau)
                delay_rows.append({"service_id": sid, "policy": "conformal_fixed_guard", "tau": tau, "alpha": alpha, **{k: fixed[k] for k in ("overload_fraction", "within_delta", "relative_cost")}})
            margin = simulate_conformal(service["history"], service["test"], correction="none", alpha=1.0, tau=tau)
            delay_rows.append({"service_id": sid, "policy": "conformal_margin_only", "tau": tau, "alpha": 1.0, **{k: margin[k] for k in ("overload_fraction", "within_delta", "relative_cost")}})
        if index % 50 == 0:
            print(f"closed-loop delay {index}/200", flush=True)
    delay = pd.DataFrame(delay_rows)
    delay_summary = (
        delay.groupby(["policy", "tau", "alpha"], as_index=False)
        .agg(
            services=("service_id", "nunique"),
            median_overload_fraction=("overload_fraction", "median"),
            p10_overload_fraction=("overload_fraction", q10),
            p90_overload_fraction=("overload_fraction", q90),
            median_relative_cost=("relative_cost", "median"),
            service_compliance=("within_delta", "mean"),
        )
    )

    rho_frame.to_csv(OUT / "rho_sweep_per_service.csv", index=False)
    rho_summary.to_csv(OUT / "rho_sweep_summary.csv", index=False)
    variants.to_csv(OUT / "guardrail_variants_per_service.csv", index=False)
    variant_summary.to_csv(OUT / "guardrail_variants_summary.csv", index=False)
    activation.to_csv(OUT / "activation_by_scale.csv", index=False)
    recovery.to_csv(OUT / "recovery_per_service.csv", index=False)
    recovery_summary.to_csv(OUT / "recovery_summary.csv", index=False)
    delay.to_csv(OUT / "closed_loop_delay_per_service.csv", index=False)
    delay_summary.to_csv(OUT / "closed_loop_delay_summary.csv", index=False)

    methodology = {
        "input_series": str(DATA.relative_to(EXP)),
        "phase": 4,
        "delta": DELTA,
        "window": W,
        "guard_persistence": H,
        "rho_sweep": list(RHOS),
        "rho_selection": "diagnostic pre-test sweep only; no rho selected from held-out results",
        "corrections": {
            "fixed_plus1": "one unit if trigger is active",
            "proportional_5pct": "max(1, ceil(0.05 * nominal capacity)) if trigger is active; fraction fixed before test",
            "always_plus1": "one unconditional unit; no trigger",
        },
        "scale_definition": "revision-defined quartiles of mean demand over days 0-9 across the finite 200-service suite",
        "recovery": f"start and first causally observable end of the first {RECOVERY_WINDOW}-step window with overload <= delta",
        "delay": "closed-loop queue: decisions are applied after tau steps and guard feedback uses realised overload/soft state under applied capacity",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "services": int(selection.service_id.nunique()),
        "focused_services": int(selection.is_focused_20.sum()),
    }
    (OUT / "methodology.json").write_text(json.dumps(methodology, indent=2) + "\n")
    save_figures(activation, recovery_summary, delay_summary)

    print("\nRHO SWEEP\n", rho_summary.to_string(index=False))
    print("\nVARIANTS\n", variant_summary.to_string(index=False))
    print("\nACTIVATION\n", activation.to_string(index=False))
    print("\nRECOVERY\n", recovery_summary.to_string(index=False))
    print("\nDELAY\n", delay_summary.to_string(index=False))


if __name__ == "__main__":
    if "--verified" in sys.argv:
        DATA = EXP / "data" / "service_timeseries_200_verified"
        REACTIVE_PARAMETERS = PAPER / "audit" / "verified_results" / "comparative" / "reactive_selected_parameters.csv"
        OUT = PAPER / "audit" / "verified_results" / "guardrail"
        FIGURES = PAPER / "figures" / "verified"
    main()
