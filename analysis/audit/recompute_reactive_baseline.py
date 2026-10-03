"""Tune and evaluate the HPA-style reactive comparator without test leakage.

For each of the 200 services, this script selects a utilisation threshold and
cooldown on days 8--9 only.  Among calibration-feasible candidates (temporal
overload <= delta), it chooses the lowest-cost candidate.  If no candidate is
feasible, it chooses the lowest-overload candidate and uses cost as the second
key.  The selected configuration is then evaluated once on held-out days
10--12.  The full fixed-parameter test frontier is retained as a sensitivity
analysis; it is never used for selection.

Run from the paper repository:

    python3 audit/recompute_reactive_baseline.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

import recompute_margin_comparison as margin


PAPER_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PAPER_ROOT.parent / "experiments"
DATA_ROOT = EXPERIMENT_ROOT / "data" / "service_timeseries_200"
SPLIT_PATH = EXPERIMENT_ROOT / "data" / "splits" / "split_definition.json"
SELECTION_PATH = EXPERIMENT_ROOT / "data" / "splits" / "selected_services_200.csv"
OUT_DIR = PAPER_ROOT / "audit" / "comparative_results"
FIG_DIR = PAPER_ROOT / "figures"

DELTA = 0.05
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
COOLDOWNS = (1, 3, 5, 10)
BOOTSTRAP_REPLICATES = 20_000
SEED = 20260920


def split_frames(service_id: str, split: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_name = {item["name"]: item for item in split["splits"]}
    frame = pd.read_parquet(DATA_ROOT / f"{service_id}.parquet").sort_values("timestamp")

    def select(name: str) -> pd.DataFrame:
        spec = by_name[name]
        mask = (
            (frame["timestamp"] >= spec["timestamp_start"])
            & (frame["timestamp"] < spec["timestamp_end_exclusive"])
        )
        selected = frame.loc[mask].copy()
        if len(selected) != spec["expected_rows"]:
            raise ValueError(f"{service_id}: {name} has {len(selected)} rows")
        return selected

    return select("train"), select("calibration"), select("test")


def reactive_capacity(
    history: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    threshold: float,
    cooldown: int,
) -> np.ndarray:
    previous_demand = float(history["cpu_sum"].iloc[-1])
    current_capacity = int(history["replica_count"].iloc[-1])
    last_scale_step = -cooldown
    capacities: list[int] = []

    for step, demand in enumerate(evaluation["cpu_sum"].to_numpy(float)):
        desired = max(int(np.ceil(previous_demand / threshold)), 1)
        if step - last_scale_step >= cooldown and desired != current_capacity:
            current_capacity = desired
            last_scale_step = step
        capacities.append(current_capacity)
        previous_demand = float(demand)
    return np.asarray(capacities, dtype=int)


def metrics(evaluation: pd.DataFrame, capacity: np.ndarray) -> dict:
    demand = evaluation["cpu_sum"].to_numpy(float)
    overload = demand > capacity
    denominator = margin.observed_cost(evaluation)
    raw_cost = margin.total_cost(capacity, demand)
    return {
        "overload_fraction": float(overload.mean()),
        "relative_cost": float(raw_cost / denominator),
        "total_cost": raw_cost,
        "mean_capacity": float(capacity.mean()),
        "scaling_churn": int(np.abs(np.diff(capacity)).sum()),
        "within_delta": bool(overload.mean() <= DELTA),
    }


def evaluate_grid(
    history: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    for cooldown in COOLDOWNS:
        for threshold in THRESHOLDS:
            capacity = reactive_capacity(
                history,
                evaluation,
                threshold=threshold,
                cooldown=cooldown,
            )
            rows.append({"threshold": threshold, "cooldown": cooldown, **metrics(evaluation, capacity)})
    return rows


def choose_calibration_candidate(grid: pd.DataFrame) -> tuple[pd.Series, bool]:
    feasible = grid[grid["within_delta"]]
    if not feasible.empty:
        selected = feasible.sort_values(
            ["relative_cost", "overload_fraction", "threshold", "cooldown"],
            ascending=[True, True, False, True],
        ).iloc[0]
        return selected, True
    selected = grid.sort_values(
        ["overload_fraction", "relative_cost", "threshold", "cooldown"],
        ascending=[True, True, False, True],
    ).iloc[0]
    return selected, False


def summarize_grid(per_service: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for subset in ("focused20", "all200"):
        frame = per_service if subset == "all200" else per_service[per_service["is_focused_20"]]
        for (cooldown, threshold), group in frame.groupby(["cooldown", "threshold"], sort=True):
            rows.append(
                {
                    "subset": subset,
                    "services": len(group),
                    "cooldown": int(cooldown),
                    "threshold": float(threshold),
                    "median_relative_cost": float(group["relative_cost"].median()),
                    "mean_relative_cost": float(group["relative_cost"].mean()),
                    "median_overload": float(group["overload_fraction"].median()),
                    "mean_overload": float(group["overload_fraction"].mean()),
                    "compliance": float(group["within_delta"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for subset in ("focused20", "all200"):
        frame = selected if subset == "all200" else selected[selected["is_focused_20"]]
        rows.append(
            {
                "subset": subset,
                "services": len(frame),
                "calibration_feasible_fraction": float(frame["calibration_feasible"].mean()),
                "median_threshold": float(frame["threshold"].median()),
                "median_cooldown": float(frame["cooldown"].median()),
                "median_relative_cost": float(frame["relative_cost"].median()),
                "mean_relative_cost": float(frame["relative_cost"].mean()),
                "median_overload": float(frame["overload_fraction"].median()),
                "mean_overload": float(frame["overload_fraction"].mean()),
                "compliance": float(frame["within_delta"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_results(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    conformal = pd.read_csv(OUT_DIR / "margin_comparison_per_service.csv")
    conformal = conformal[
        conformal["subset"].eq("all200")
        & conformal["delta"].eq(DELTA)
        & conformal["margin_family"].eq("conformal")
        & conformal["guard"].astype(bool)
    ][["service_id", "overload_fraction", "relative_cost"]].rename(
        columns={
            "overload_fraction": "conformal_overload",
            "relative_cost": "conformal_cost",
        }
    )
    merged = selected.merge(conformal, on="service_id", validate="one_to_one")
    effects: list[dict] = []
    compliance: list[dict] = []
    family_size = 2
    for subset_index, subset in enumerate(("focused20", "all200")):
        frame = merged if subset == "all200" else merged[merged["is_focused_20"]]
        boot_base = pd.DataFrame(
            {
                "stratum": frame["stratum"],
                "conformal_overload": frame["conformal_overload"],
                "reactive_overload": frame["overload_fraction"],
                "conformal_cost": frame["conformal_cost"],
                "reactive_cost": frame["relative_cost"],
            }
        )
        for metric_index, metric in enumerate(("overload", "cost")):
            mean_diff, ci_lo, ci_hi, family_lo, family_hi = margin.bootstrap_mean_difference(
                boot_base,
                f"conformal_{metric}",
                f"reactive_{metric}",
                alpha=0.05 / family_size,
                seed=SEED + 100 + subset_index * 10 + metric_index,
            )
            diff = boot_base[f"conformal_{metric}"] - boot_base[f"reactive_{metric}"]
            p_value = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff).pvalue)
            effects.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "services": len(frame),
                    "mean_difference_conformal_minus_reactive": mean_diff,
                    "pointwise_ci_lo": ci_lo,
                    "pointwise_ci_hi": ci_hi,
                    "familywise_ci_lo": family_lo,
                    "familywise_ci_hi": family_hi,
                    "wilcoxon_two_sided_p": p_value,
                    "family_size": family_size,
                    "family_alpha": 0.05 / family_size,
                }
            )

        conformal_ok = frame["conformal_overload"] <= DELTA
        reactive_ok = frame["overload_fraction"] <= DELTA
        wins = int((conformal_ok & ~reactive_ok).sum())
        losses = int((~conformal_ok & reactive_ok).sum())
        discordant = wins + losses
        compliance.append(
            {
                "subset": subset,
                "conformal_only_compliant": wins,
                "reactive_only_compliant": losses,
                "discordant_pairs": discordant,
                "exact_mcnemar_two_sided_p": (
                    float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
                ),
            }
        )
    return pd.DataFrame(effects), pd.DataFrame(compliance)


def plot_frontier(grid_summary: pd.DataFrame, selected_summary: pd.DataFrame) -> None:
    mpl_dir = PAPER_ROOT / "tmp" / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    conformal = pd.read_csv(OUT_DIR / "margin_comparison_summary.csv")
    colours = {1: "#1565C0", 3: "#00897B", 5: "#F9A825", 10: "#8E24AA"}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), constrained_layout=True)
    for ax, subset in zip(axes, ("focused20", "all200")):
        frame = grid_summary[grid_summary["subset"].eq(subset)]
        for cooldown in COOLDOWNS:
            line = frame[frame["cooldown"].eq(cooldown)].sort_values("threshold")
            ax.plot(
                line["mean_overload"],
                line["mean_relative_cost"],
                marker="o",
                markersize=3.5,
                linewidth=1.1,
                color=colours[cooldown],
                label=f"cooldown {cooldown}",
            )
        tuned = selected_summary[selected_summary["subset"].eq(subset)].iloc[0]
        conf = conformal[
            conformal["subset"].eq(subset)
            & conformal["delta"].eq(DELTA)
            & conformal["margin_family"].eq("conformal")
            & conformal["guard"].astype(bool)
        ].iloc[0]
        ax.scatter(
            tuned["mean_overload"], tuned["mean_relative_cost"],
            marker="*", s=130, color="#D32F2F", edgecolor="black", linewidth=0.5,
            zorder=6, label="pre-test tuned",
        )
        ax.scatter(
            conf["mean_overload"], conf["mean_relative_cost"],
            marker="s", s=55, color="#2E7D32", edgecolor="black", linewidth=0.5,
            zorder=6, label="conformal+guard",
        )
        ax.axvline(DELTA, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_title("Focused 20" if subset == "focused20" else "All 200")
        ax.set_xlabel("Mean held-out overload")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(alpha=0.25, linestyle="--")
        ax.text(
            0.98,
            0.97,
            f"Compliance: tuned {tuned['compliance']:.0%}; conformal {conf['compliance']:.0%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
        )
    axes[0].set_ylabel("Mean relative cost")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7.5)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_reactive_frontier.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_reactive_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # The original combined figure includes historical learned-model curves.
    # Do not generate it in the verified persistence-only replay.
    if DATA_ROOT.name == "service_timeseries_200_verified":
        return

    focused_models = pd.read_csv(EXPERIMENT_ROOT / "results" / "c5_frontier_density.csv")
    palette = {
        "persistence": "#9C27B0",
        "arima": "#FB8C00",
        "xgb": "#43A047",
        "lstm": "#1E88E5",
    }
    labels = {
        "persistence": "Persistence + conformal",
        "arima": "ARIMA + conformal",
        "xgb": "XGBoost + conformal",
        "lstm": "LSTM + conformal",
    }
    focused_tuned = selected_summary[selected_summary["subset"].eq("focused20")].iloc[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for model in ("persistence", "arima", "xgb", "lstm"):
        line = focused_models[focused_models["model"].eq(model)].sort_values("median_overload")
        ax.plot(
            line["median_overload"] * 100,
            line["median_rel_cost"],
            marker="o",
            linewidth=1.8,
            linestyle="--" if model == "persistence" else "-",
            color=palette[model],
            label=labels[model],
        )
    ax.axhline(0.196344, color="black", linestyle=":", linewidth=1.4,
               label="Clairvoyant zero-overload reference (0.196)")
    ax.plot(
        focused_tuned["median_overload"] * 100,
        focused_tuned["median_relative_cost"],
        marker="P",
        color="#D32F2F",
        markersize=12,
        linestyle="none",
        label=(
            "Reactive, pre-test selected "
            f"({focused_tuned['median_relative_cost']:.3f}; "
            f"{focused_tuned['compliance']:.0%} comply)"
        ),
    )
    ax.set_xlabel("Median overload fraction (%)")
    ax.set_ylabel("Median relative cost\n(fraction of observed-capacity cost)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.savefig(FIG_DIR / "fig1_cost_risk_frontier.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig1_cost_risk_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(SELECTION_PATH)
    split = json.loads(SPLIT_PATH.read_text())
    calibration_rows: list[dict] = []
    selection_rows: list[dict] = []
    test_rows: list[dict] = []
    test_grid_rows: list[dict] = []

    for index, service in selection.iterrows():
        service_id = str(service["service_id"])
        train, calibration, test = split_frames(service_id, split)
        calibration_grid = pd.DataFrame(evaluate_grid(train, calibration))
        selected, feasible = choose_calibration_candidate(calibration_grid)
        for row in calibration_grid.to_dict("records"):
            calibration_rows.append(
                {
                    "service_id": service_id,
                    "stratum": service["stratum"],
                    "is_focused_20": bool(service["is_focused_20"]),
                    **row,
                }
            )

        threshold = float(selected["threshold"])
        cooldown = int(selected["cooldown"])
        selection_rows.append(
            {
                "service_id": service_id,
                "stratum": service["stratum"],
                "is_focused_20": bool(service["is_focused_20"]),
                "threshold": threshold,
                "cooldown": cooldown,
                "calibration_feasible": feasible,
                "calibration_overload": float(selected["overload_fraction"]),
                "calibration_relative_cost": float(selected["relative_cost"]),
            }
        )

        pre_test = pd.concat([train, calibration], ignore_index=True)
        selected_capacity = reactive_capacity(
            pre_test, test, threshold=threshold, cooldown=cooldown
        )
        test_rows.append(
            {
                "service_id": service_id,
                "stratum": service["stratum"],
                "is_focused_20": bool(service["is_focused_20"]),
                "threshold": threshold,
                "cooldown": cooldown,
                "calibration_feasible": feasible,
                **metrics(test, selected_capacity),
            }
        )
        for row in evaluate_grid(pre_test, test):
            test_grid_rows.append(
                {
                    "service_id": service_id,
                    "stratum": service["stratum"],
                    "is_focused_20": bool(service["is_focused_20"]),
                    **row,
                }
            )
        if (index + 1) % 25 == 0:
            print(f"processed {index + 1}/200 services", flush=True)

    calibration_frame = pd.DataFrame(calibration_rows)
    parameter_frame = pd.DataFrame(selection_rows)
    test_frame = pd.DataFrame(test_rows)
    grid_frame = pd.DataFrame(test_grid_rows)
    selected_summary = summarize_selected(test_frame)
    grid_summary = summarize_grid(grid_frame)
    effects, compliance = paired_results(test_frame)

    calibration_frame.to_csv(OUT_DIR / "reactive_calibration_grid_per_service.csv", index=False)
    parameter_frame.to_csv(OUT_DIR / "reactive_selected_parameters.csv", index=False)
    test_frame.to_csv(OUT_DIR / "reactive_test_per_service.csv", index=False)
    grid_frame.to_csv(OUT_DIR / "reactive_test_grid_per_service.csv", index=False)
    selected_summary.to_csv(OUT_DIR / "reactive_selected_summary.csv", index=False)
    grid_summary.to_csv(OUT_DIR / "reactive_test_grid_summary.csv", index=False)
    effects.to_csv(OUT_DIR / "reactive_paired_effects.csv", index=False)
    compliance.to_csv(OUT_DIR / "reactive_compliance_tests.csv", index=False)

    methodology = {
        "input_series": str(DATA_ROOT.relative_to(EXPERIMENT_ROOT)),
        "selection": "per service, calibration days 8-9 only",
        "selection_objective": (
            "minimum relative cost among candidates with calibration overload <= delta; "
            "minimum overload then cost if none feasible"
        ),
        "delta_temporal": DELTA,
        "fleet_level_target": None,
        "thresholds": list(THRESHOLDS),
        "cooldowns": list(COOLDOWNS),
        "test_role": "one-shot held-out evaluation plus descriptive full fixed-grid frontier",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
    }
    (OUT_DIR / "reactive_methodology.json").write_text(json.dumps(methodology, indent=2) + "\n")
    plot_frontier(grid_summary, selected_summary)

    print("\nPre-test selected reactive summary")
    print(selected_summary.to_string(index=False))
    print("\nPaired conformal-minus-reactive effects")
    print(effects.to_string(index=False))
    print("\nPaired compliance")
    print(compliance.to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--verified" in sys.argv:
        DATA_ROOT = EXPERIMENT_ROOT / "data" / "service_timeseries_200_verified"
        OUT_DIR = PAPER_ROOT / "audit" / "verified_results" / "comparative"
        FIG_DIR = PAPER_ROOT / "figures" / "verified"
        margin.OUT_DIR = OUT_DIR
        margin.DATA_ROOT = DATA_ROOT
    main()
