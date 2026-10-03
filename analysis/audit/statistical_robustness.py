#!/usr/bin/env python3
"""Dependence-aware statistical robustness analysis for the paper.

The analysis has four parts:

1. Wilson confidence intervals for service-level budget compliance.
2. Stratified bootstrap intervals that preserve the five workload strata.
3. Paired stratified-bootstrap confidence intervals for policy effect sizes,
   alongside one-sided Wilcoxon tests and family-wise Bonferroni correction.
4. A hierarchical moving-block bootstrap that resamples time blocks within
   services and services within strata, thereby preserving local temporal
   dependence instead of treating 4,320 minutes as independent Bernoulli draws.

The script is deterministic and writes only to ``audit/statistical_results``.
It expects the sibling ``experiments`` directory used by this project.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
DEFAULT_EXPERIMENTS = RESEARCH / "experiments"
OUTPUT = PAPER / "audit" / "statistical_results"
DELTA = 0.05
SEED = 20260920


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--temporal-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def quantile_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return tuple(np.quantile(values, [alpha / 2, 1 - alpha / 2]).astype(float))


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - radius, centre + radius


def stratified_indices(strata: np.ndarray, b: int, rng: np.random.Generator) -> np.ndarray:
    groups = [np.flatnonzero(strata == value) for value in sorted(np.unique(strata))]
    return np.concatenate([rng.choice(group, size=(b, len(group)), replace=True) for group in groups], axis=1)


def stratified_bootstrap_stat(
    values: np.ndarray,
    strata: np.ndarray,
    b: int,
    rng: np.random.Generator,
    statistic: str,
) -> np.ndarray:
    indices = stratified_indices(strata, b, rng)
    sampled = values[indices]
    if statistic == "mean":
        return sampled.mean(axis=1)
    if statistic == "median":
        return np.median(sampled, axis=1)
    raise ValueError(statistic)


def paired_effect_rows(
    frame: pd.DataFrame,
    comparisons: list[tuple[str, str, str, str, str]],
    family: str,
    b: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Return Wilcoxon tests and bootstrap CIs for paired A-B effects."""
    rows: list[dict] = []
    family_alpha = 0.05 / len(comparisons)
    for label, policy_a, policy_b, metric, alternative in comparisons:
        subset = frame[frame.policy.isin([policy_a, policy_b])]
        wide = subset.pivot(index=["service_id", "stratum"], columns="policy", values=metric)
        wide = wide[[policy_a, policy_b]].dropna().reset_index()
        x = wide[policy_a].to_numpy(float)
        y = wide[policy_b].to_numpy(float)
        diff = x - y
        strata = wide.stratum.to_numpy(str)
        mean_boot = stratified_bootstrap_stat(diff, strata, b, rng, "mean")
        median_boot = stratified_bootstrap_stat(diff, strata, b, rng, "median")
        mean_lo, mean_hi = quantile_interval(mean_boot)
        med_lo, med_hi = quantile_interval(median_boot)
        simultaneous_lo, simultaneous_hi = quantile_interval(mean_boot, family_alpha)
        statistic, p_value = stats.wilcoxon(
            x,
            y,
            alternative=alternative,
            zero_method="zsplit",
            method="auto",
        )
        rows.append(
            {
                "family": family,
                "comparison": label,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "metric": metric,
                "alternative": alternative,
                "n_pairs": len(diff),
                "n_nonzero": int(np.count_nonzero(np.abs(diff) > 1e-15)),
                "mean_difference": float(diff.mean()),
                "mean_ci95_lo": mean_lo,
                "mean_ci95_hi": mean_hi,
                "mean_familywise_ci_lo": simultaneous_lo,
                "mean_familywise_ci_hi": simultaneous_hi,
                "median_difference": float(np.median(diff)),
                "median_ci95_lo": med_lo,
                "median_ci95_hi": med_hi,
                "wilcoxon_stat": float(statistic),
                "p_value_one_sided": float(p_value),
                "bonferroni_alpha": family_alpha,
                "reject_familywise": bool(p_value < family_alpha),
            }
        )
    return rows


def simulate_fixed_margin(history: pd.DataFrame, test: pd.DataFrame, delta: float = DELTA) -> dict:
    hist_y = history.cpu_sum.to_numpy(float)
    demand = test.cpu_sum.to_numpy(float)
    margin = float(np.quantile(np.maximum(hist_y[1:] - hist_y[:-1], 0), 1 - delta))
    forecast = np.concatenate([[hist_y[-1]], demand[:-1]])
    capacity = np.maximum(np.ceil(forecast + margin), 1).astype(int)
    overload = demand > capacity
    c_total = float(capacity.sum() + 0.05 * np.abs(np.diff(np.r_[capacity[0], capacity])).sum() + 10 * overload.sum())
    return {"overload_arr": overload, "overload_fraction": overload.mean(), "c_total": c_total}


def simulate_gaussian_margin(history: pd.DataFrame, test: pd.DataFrame, delta: float = DELTA, window: int = 240) -> dict:
    z = float(stats.norm.ppf(1 - delta))
    hist_y = history.cpu_sum.to_numpy(float)
    demand = test.cpu_sum.to_numpy(float)
    residuals = list(np.maximum(hist_y[1:] - hist_y[:-1], 0))
    previous = hist_y[-1]
    capacities: list[int] = []
    overload: list[bool] = []
    for actual in demand:
        margin = z * float(np.std(residuals[-window:]))
        capacity = max(int(np.ceil(previous + margin)), 1)
        capacities.append(capacity)
        overload.append(bool(actual > capacity))
        residuals.append(max(float(actual - previous), 0.0))
        previous = actual
    capacity_array = np.asarray(capacities)
    overload_array = np.asarray(overload)
    c_total = float(capacity_array.sum() + 0.05 * np.abs(np.diff(np.r_[capacity_array[0], capacity_array])).sum() + 10 * overload_array.sum())
    return {"overload_arr": overload_array, "overload_fraction": overload_array.mean(), "c_total": c_total}


def load_focused_frame(experiments: Path, exp_core) -> pd.DataFrame:
    results = experiments / "results"
    split = json.loads((experiments / "data" / "splits" / "split_definition.json").read_text())
    services = exp_core.load_all_services(split)
    observed = (
        pd.read_csv(results / "baseline_replay_per_service.csv")
        .query("policy == 'observed_capacity'")
        .set_index("msname").c_total
    )
    rows: list[dict] = []

    risk = pd.read_csv(results / "risk_policy_replay_per_service.csv")
    for source, target in {
        "risk_conformal_guardrail": "B6",
        "risk_conformal_margin_only": "B7",
        "guardrail_only_no_margin": "B8",
    }.items():
        sub = risk[risk.policy.eq(source) & np.isclose(risk.delta, DELTA)]
        for item in sub.itertuples():
            rows.append({"service_id": item.msname, "stratum": exp_core.GROUP_MAP[item.msname].split("_")[0], "policy": target,
                         "overload_fraction": item.overload_fraction, "rel_cost": item.relative_cost_vs_observed})

    baseline = pd.read_csv(results / "baseline_replay_per_service.csv")
    for source, target in {
        "predictive_persistence_no_margin": "B2",
        "reactive_threshold_u70_cooldown3": "Reactive",
    }.items():
        for item in baseline[baseline.policy.eq(source)].itertuples():
            rows.append({"service_id": item.msname, "stratum": exp_core.GROUP_MAP[item.msname].split("_")[0], "policy": target,
                         "overload_fraction": item.overload_fraction, "rel_cost": item.relative_cost_vs_observed})

    advanced = pd.read_csv(results / "advanced_policy_replay_per_service.csv")
    for model in ("arima", "xgb", "lstm"):
        policy = f"{model}_conformal_guardrail_d0.05"
        for item in advanced[advanced.policy.eq(policy)].itertuples():
            rows.append({"service_id": item.msname, "stratum": exp_core.GROUP_MAP[item.msname].split("_")[0], "policy": model.upper(),
                         "overload_fraction": item.overload_fraction, "rel_cost": item.relative_cost_vs_observed})

    for service, history, test in services:
        for policy, result in (("B3", simulate_fixed_margin(history, test)), ("B4", simulate_gaussian_margin(history, test))):
            rows.append({"service_id": service, "stratum": exp_core.GROUP_MAP[service].split("_")[0], "policy": policy,
                         "overload_fraction": float(result["overload_fraction"]), "rel_cost": float(result["c_total"] / observed[service])})
    return pd.DataFrame(rows)


def load_large_frame(experiments: Path) -> pd.DataFrame:
    analysis = experiments / "results" / "large_scale" / "analysis"
    frontier = pd.read_csv(analysis / "ls_frontier_per_service_200.csv")
    frontier = frontier[np.isclose(frontier.delta, DELTA)].copy()
    frontier["policy"] = frontier.model.map({"persistence": "B6", "arima": "ARIMA", "xgb": "XGB", "lstm": "LSTM"})
    base = pd.read_csv(analysis / "ls_baselines_per_service_for_stats_200.csv")
    observed = frontier[frontier.model.eq("persistence")].set_index("service_id").observed_c_total
    base["rel_cost"] = base.apply(lambda item: item.c_total / observed[item.service_id], axis=1)
    base["policy"] = base.policy.map({
        "B2: pure predictive": "B2",
        "B4: Gaussian margin": "B4",
        "B6: conformal + guardrail": "B6",
        "B7: conformal margin only": "B7",
        "B8: guardrail only": "B8",
        "B9: empirical percentile + guardrail": "B9",
    })
    # B6 is already present in the frontier table with its normalised cost.
    base = base[~base.policy.eq("B6")]
    columns = ["service_id", "stratum", "policy", "overload_fraction", "rel_cost"]
    return pd.concat([base[columns], frontier[columns]], ignore_index=True)


def add_large_reactive(frame: pd.DataFrame, exp_core_ls, reactive_threshold_capacity) -> pd.DataFrame:
    rows = []
    for service, history, test in exp_core_ls.load_all_services_200(exp_core_ls.load_split_def()):
        demand = test.cpu_sum.to_numpy(float)
        observed_capacity = np.maximum(np.ceil(test.replica_count.to_numpy(float)), 1).astype(int)
        reactive_capacity = reactive_threshold_capacity(history, test, mu=1.0, target_utilization=0.8, cooldown=3)

        def total_cost(capacity: np.ndarray) -> tuple[float, float]:
            overload = demand > capacity
            cost = float(capacity.sum() + 0.05 * np.abs(np.diff(np.r_[capacity[0], capacity])).sum() + 10 * overload.sum())
            return cost, float(overload.mean())

        observed_cost, _ = total_cost(observed_capacity)
        reactive_cost, reactive_overload = total_cost(reactive_capacity)
        rows.append({
            "service_id": service,
            "stratum": exp_core_ls.GROUP_MAP_200[service],
            "policy": "Reactive",
            "overload_fraction": reactive_overload,
            "rel_cost": reactive_cost / observed_cost,
        })
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def compliance_intervals(frame: pd.DataFrame, b: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for policy in ("B6", "LSTM", "XGB", "ARIMA"):
        sub = frame[frame.policy.eq(policy)].drop_duplicates("service_id").sort_values("service_id")
        success = (sub.overload_fraction.to_numpy(float) <= DELTA).astype(float)
        strata = sub.stratum.to_numpy(str)
        k, n = int(success.sum()), len(success)
        wilson_lo, wilson_hi = wilson_interval(k, n)
        boot = stratified_bootstrap_stat(success, strata, b, rng, "mean")
        boot_lo, boot_hi = quantile_interval(boot)
        rows.append({
            "policy": policy,
            "within_budget": k,
            "services": n,
            "compliance": k / n,
            "wilson95_lo": wilson_lo,
            "wilson95_hi": wilson_hi,
            "stratified_bootstrap95_lo": boot_lo,
            "stratified_bootstrap95_hi": boot_hi,
        })
    return pd.DataFrame(rows)


def draw_block_starts(n: int, block: int, b: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    """Draw non-circular moving blocks that reconstruct an n-step series."""
    full, remainder = divmod(n, block)
    starts = rng.integers(0, n - block + 1, size=(b, full))
    short_starts = None
    if remainder:
        short_starts = rng.integers(0, n - remainder + 1, size=b)
    return starts, short_starts


def block_fractions_from_starts(
    series: np.ndarray,
    block: int,
    starts: np.ndarray,
    short_starts: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(series, dtype=np.int64)
    n = len(values)
    remainder = n % block
    prefix = np.r_[0, np.cumsum(values)]
    totals = (prefix[starts + block] - prefix[starts]).sum(axis=1)
    if remainder:
        if short_starts is None:
            raise ValueError("short block starts are required when n is not divisible by block")
        totals += prefix[short_starts + remainder] - prefix[short_starts]
    return totals / n


def moving_block_fractions(series: np.ndarray, block: int, b: int, rng: np.random.Generator) -> np.ndarray:
    starts, short_starts = draw_block_starts(len(series), block, b, rng)
    return block_fractions_from_starts(series, block, starts, short_starts)


def temporal_fraction_matrix(
    arrays: list[np.ndarray],
    block: int,
    b: int,
    rng: np.random.Generator,
    synchronised: bool,
) -> np.ndarray:
    """Resample each service, optionally using the same clock blocks for all services."""
    if synchronised:
        starts, short_starts = draw_block_starts(len(arrays[0]), block, b, rng)
        return np.vstack(
            [block_fractions_from_starts(values, block, starts, short_starts) for values in arrays]
        )
    return np.vstack([moving_block_fractions(values, block, b, rng) for values in arrays])


def paired_temporal_mean_difference(
    arrays_a: list[np.ndarray],
    arrays_b: list[np.ndarray],
    strata: np.ndarray,
    block: int,
    b: int,
    rng: np.random.Generator,
    synchronised: bool,
) -> np.ndarray:
    """Paired hierarchical bootstrap with identical time/service draws for A and B."""
    differences: list[np.ndarray] = []
    common = draw_block_starts(len(arrays_a[0]), block, b, rng) if synchronised else None
    for values_a, values_b in zip(arrays_a, arrays_b):
        starts, short_starts = common or draw_block_starts(len(values_a), block, b, rng)
        fraction_a = block_fractions_from_starts(values_a, block, starts, short_starts)
        fraction_b = block_fractions_from_starts(values_b, block, starts, short_starts)
        differences.append(fraction_a - fraction_b)
    temporal_difference = np.vstack(differences)
    indices = stratified_indices(strata, b, rng)
    columns = np.arange(b)[:, None]
    return temporal_difference[indices, columns].mean(axis=1)


def conformal_gaussian_overload_pair(
    history: pd.DataFrame,
    test: pd.DataFrame,
    delta: float,
    window: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the corrected signed-residual margins under a shared guardrail."""
    hist_y = history.cpu_sum.to_numpy(float)
    demand = test.cpu_sum.to_numpy(float)
    residuals = list(np.diff(hist_y))
    previous = float(hist_y[-1])
    states = {
        "conformal": {"overload": [], "soft": []},
        "gaussian": {"overload": [], "soft": []},
    }
    z = float(stats.norm.ppf(1.0 - delta))
    for actual in demand:
        forecast = max(previous, 0.0)
        residual_window = np.asarray(residuals[-window:], dtype=float)
        positive = np.maximum(residual_window, 0.0)
        rank = min(len(positive), int(np.ceil((len(positive) + 1) * (1.0 - delta))))
        margins = {
            "conformal": float(np.partition(positive, rank - 1)[rank - 1]),
            "gaussian": max(0.0, float(residual_window.mean()) + z * float(residual_window.std(ddof=0))),
        }
        for family, margin in margins.items():
            state = states[family]
            trigger = len(state["overload"]) >= 2 and (
                all(state["overload"][-2:]) or all(state["soft"][-2:])
            )
            capacity = max(int(np.ceil(forecast + margin)), 1) + int(trigger)
            state["overload"].append(bool(float(actual) > capacity))
            state["soft"].append(bool(float(actual) > 0.70 * capacity))
        residuals.append(float(actual) - forecast)
        previous = float(actual)
    return (
        np.asarray(states["conformal"]["overload"], dtype=bool),
        np.asarray(states["gaussian"]["overload"], dtype=bool),
    )


def temporal_hierarchical_bootstrap(
    experiments: Path,
    exp_core_ls,
    exp_core,
    b: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = exp_core_ls.load_split_def()
    services = exp_core_ls.load_all_services_200(split)
    selection = exp_core_ls.load_selection().set_index("service_id")
    saved = pd.read_csv(experiments / "results" / "large_scale" / "analysis" / "ls_frontier_per_service_200.csv")
    saved = saved[np.isclose(saved.delta, DELTA)]
    models = ("persistence", "arima", "xgb", "lstm")
    arrays: dict[str, list[np.ndarray]] = {model: [] for model in models}
    margin_arrays: dict[float, dict[str, list[np.ndarray]]] = {
        delta: {"conformal": [], "gaussian": []} for delta in (0.05, 0.01)
    }
    service_ids: list[str] = []
    for index, (service, history, test) in enumerate(services, start=1):
        service_ids.append(service)
        for model in models:
            forecast = None if model == "persistence" else exp_core_ls.load_forecast_ls(model, service)
            result = exp_core.simulate_policy(history, test, forecast, delta=DELTA, W=240, alpha=1.0,
                                              mu=1.0, rho=0.70, guardrail_h=2, guardrail_gamma=1,
                                              use_margin=True, use_guardrail=True)
            arrays[model].append(np.asarray(result["overload_arr"], dtype=bool))
            expected = saved[saved.service_id.eq(service) & saved.model.eq(model)].overload_fraction.iloc[0]
            if not np.isclose(result["overload_fraction"], expected):
                raise RuntimeError(f"Replay mismatch for {model}/{service}: {result['overload_fraction']} != {expected}")
        for delta in margin_arrays:
            conformal, gaussian = conformal_gaussian_overload_pair(history, test, delta)
            margin_arrays[delta]["conformal"].append(conformal)
            margin_arrays[delta]["gaussian"].append(gaussian)
        if index % 25 == 0:
            print(f"replayed {index}/200 services", flush=True)

    strata = selection.loc[service_ids].stratum.to_numpy(str)
    rows: list[dict] = []
    for model in models:
        point_fractions = np.asarray([values.mean() for values in arrays[model]])
        point_compliance = float(np.mean(point_fractions <= DELTA))
        for block in (30, 60, 120, 240, 480, 1440):
            for scheme, synchronised in (
                ("independent_within_service", False),
                ("synchronised_across_services", True),
            ):
                temporal = temporal_fraction_matrix(arrays[model], block, b, rng, synchronised)
                temporal_only = np.mean(temporal <= DELTA, axis=0)
                indices = stratified_indices(strata, b, rng)
                columns = np.arange(b)[:, None]
                hierarchical = np.mean(temporal[indices, columns] <= DELTA, axis=1)
                hierarchical_mean_ol = np.mean(temporal[indices, columns], axis=1)
                t_lo, t_hi = quantile_interval(temporal_only)
                h_lo, h_hi = quantile_interval(hierarchical)
                hm_lo, hm_hi = quantile_interval(hierarchical_mean_ol)
                rows.append({
                    "model": model,
                    "resampling_scheme": scheme,
                    "block_minutes": block,
                    "services": len(service_ids),
                    "test_steps": len(arrays[model][0]),
                    "point_compliance": point_compliance,
                    "temporal_only_median_compliance": float(np.median(temporal_only)),
                    "temporal_only_ci95_lo": t_lo,
                    "temporal_only_ci95_hi": t_hi,
                    "hierarchical_median_compliance": float(np.median(hierarchical)),
                    "hierarchical_ci95_lo": h_lo,
                    "hierarchical_ci95_hi": h_hi,
                    "point_mean_overload": float(point_fractions.mean()),
                    "hierarchical_mean_overload_ci95_lo": hm_lo,
                    "hierarchical_mean_overload_ci95_hi": hm_hi,
                    "bootstrap_replicates": b,
                })

    paired_rows: list[dict] = []
    for delta, family_arrays in margin_arrays.items():
        point = float(np.mean([
            conformal.mean() - gaussian.mean()
            for conformal, gaussian in zip(family_arrays["conformal"], family_arrays["gaussian"])
        ]))
        for block in (30, 60, 120, 240, 480, 1440):
            for scheme, synchronised in (
                ("independent_within_service", False),
                ("synchronised_across_services", True),
            ):
                draws = paired_temporal_mean_difference(
                    family_arrays["conformal"],
                    family_arrays["gaussian"],
                    strata,
                    block,
                    b,
                    rng,
                    synchronised,
                )
                lo, hi = quantile_interval(draws)
                paired_rows.append({
                    "delta": delta,
                    "guardrail": True,
                    "contrast": "conformal_minus_corrected_gaussian",
                    "resampling_scheme": scheme,
                    "block_minutes": block,
                    "services": len(service_ids),
                    "test_steps": len(family_arrays["conformal"][0]),
                    "point_mean_overload_difference": point,
                    "bootstrap_median_difference": float(np.median(draws)),
                    "bootstrap_ci95_lo": lo,
                    "bootstrap_ci95_hi": hi,
                    "bootstrap_replicates": b,
                })
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def main() -> int:
    args = parse_args()
    experiments = args.experiments_root.resolve()
    if not experiments.exists():
        raise FileNotFoundError(experiments)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Experiment modules use paths relative to the parent of the experiments directory.
    os.chdir(experiments.parent)
    sys.path.insert(0, str(experiments / "src"))
    import exp_core  # type: ignore
    import exp_core_ls  # type: ignore
    from policies import reactive_threshold_capacity  # type: ignore

    rng = np.random.default_rng(args.seed)
    focused = load_focused_frame(experiments, exp_core)
    large = add_large_reactive(load_large_frame(experiments), exp_core_ls, reactive_threshold_capacity)

    focused_overload = [
        ("B6 vs B7", "B6", "B7", "overload_fraction", "less"),
        ("B6 vs B8", "B6", "B8", "overload_fraction", "less"),
        ("B6 vs B2", "B6", "B2", "overload_fraction", "less"),
        ("B6 vs B3", "B6", "B3", "overload_fraction", "less"),
        ("B6 vs B4", "B6", "B4", "overload_fraction", "less"),
        ("Persistence vs ARIMA", "B6", "ARIMA", "overload_fraction", "less"),
        ("XGBoost vs ARIMA", "XGB", "ARIMA", "overload_fraction", "less"),
        ("LSTM vs ARIMA", "LSTM", "ARIMA", "overload_fraction", "less"),
        ("LSTM vs XGBoost", "LSTM", "XGB", "overload_fraction", "less"),
    ]
    focused_cost = [
        ("LSTM vs reactive cost", "LSTM", "Reactive", "rel_cost", "less"),
        ("XGBoost vs reactive cost", "XGB", "Reactive", "rel_cost", "less"),
    ]
    large_overload = [
        ("B6 vs B7", "B6", "B7", "overload_fraction", "less"),
        ("B6 vs B8", "B6", "B8", "overload_fraction", "less"),
        ("B6 vs B2", "B6", "B2", "overload_fraction", "less"),
        ("B6 vs B9", "B6", "B9", "overload_fraction", "less"),
        ("B6 vs B4", "B6", "B4", "overload_fraction", "less"),
        ("XGBoost vs ARIMA", "XGB", "ARIMA", "overload_fraction", "less"),
        ("LSTM vs ARIMA", "LSTM", "ARIMA", "overload_fraction", "less"),
        ("LSTM vs XGBoost", "LSTM", "XGB", "overload_fraction", "less"),
    ]
    large_cost = [
        ("B6 vs reactive cost", "B6", "Reactive", "rel_cost", "less"),
        ("B6 vs B7 cost", "B6", "B7", "rel_cost", "greater"),
    ]

    effect_rows = []
    effect_rows += paired_effect_rows(focused, focused_overload, "focused_overload", args.bootstrap_replicates, rng)
    effect_rows += paired_effect_rows(focused, focused_cost, "focused_cost", args.bootstrap_replicates, rng)
    effect_rows += paired_effect_rows(large, large_overload, "large_overload", args.bootstrap_replicates, rng)
    effect_rows += paired_effect_rows(large, large_cost, "large_cost", args.bootstrap_replicates, rng)
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(OUTPUT / "paired_effect_intervals.csv", index=False)

    compliance = compliance_intervals(large, args.bootstrap_replicates, rng)
    compliance.to_csv(OUTPUT / "service_compliance_intervals.csv", index=False)

    temporal, paired_temporal = temporal_hierarchical_bootstrap(
        experiments, exp_core_ls, exp_core, args.temporal_replicates, rng
    )
    temporal.to_csv(OUTPUT / "temporal_block_bootstrap.csv", index=False)
    paired_temporal.to_csv(OUTPUT / "paired_temporal_margin_bootstrap.csv", index=False)

    metadata = {
        "seed": args.seed,
        "delta": DELTA,
        "service_bootstrap_replicates": args.bootstrap_replicates,
        "temporal_bootstrap_replicates": args.temporal_replicates,
        "block_lengths_minutes": [30, 60, 120, 240, 480, 1440],
        "service_bootstrap": "resample with replacement within each of five volatility strata",
        "temporal_bootstrap": "moving non-circular blocks within each 4,320-step service series, under independent-within-service and clock-synchronised-across-service schemes",
        "hierarchical_bootstrap": "temporal block resampling followed by stratified service resampling",
        "paired_temporal_contrast": "conformal minus corrected signed-residual Gaussian; identical time blocks and service draws are used for both policies",
        "effect_definition": "paired policy_a minus policy_b",
        "familywise_intervals": "two-sided percentile intervals with alpha=0.05/m within each revision-fixed analysis family",
    }
    (OUTPUT / "methodology.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nSERVICE-LEVEL COMPLIANCE")
    print(compliance.to_string(index=False))
    print("\nPAIRED EFFECTS")
    print(effects[["family", "comparison", "mean_difference", "mean_ci95_lo", "mean_ci95_hi", "mean_familywise_ci_lo", "mean_familywise_ci_hi", "p_value_one_sided", "reject_familywise"]].to_string(index=False))
    print("\nTEMPORAL/HIERARCHICAL BOOTSTRAP")
    print(temporal.to_string(index=False))
    print("\nPAIRED TEMPORAL CONFORMAL-GAUSSIAN CONTRAST")
    print(paired_temporal.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
