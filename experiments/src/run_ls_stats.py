"""
run_ls_stats.py
===============
Large-scale statistical aggregation and significance tests.

Outputs
-------
experiments/results/large_scale/analysis/ls_statistical_tests_200.csv
experiments/results/large_scale/analysis/ls_clopper_pearson_200.csv
experiments/results/large_scale/analysis/ls_by_stratum_summary_200.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta

sys.path.insert(0, str(Path("experiments/src")))

from exp_core_ls import (  # noqa: E402
    GROUP_MAP_200,
    RESULTS_LS,
    SIM_COST,
    load_all_services_200,
    load_split_def,
    simulate_policy,
)
from run_ls_baselines import (  # noqa: E402
    ALPHA,
    GUARDRAIL_GAMMA,
    GUARDRAIL_H,
    MU,
    RHO,
    W,
    metrics_from_capacity,
    simulate_empirical_percentile_margin,
    simulate_gaussian_margin_ls,
)


DELTA = 0.05
DELTAS = [0.10, 0.05, 0.03, 0.01]

FRONTIER_PER_SERVICE = RESULTS_LS / "ls_frontier_per_service_200.csv"
FRONTIER_BY_STRATUM = RESULTS_LS / "ls_frontier_by_stratum_200.csv"
BASELINE_STATS_PER_SERVICE = RESULTS_LS / "ls_baselines_per_service_for_stats_200.csv"

STAT_TESTS_PATH = RESULTS_LS / "ls_statistical_tests_200.csv"
CP_PATH = RESULTS_LS / "ls_clopper_pearson_200.csv"
CP_PER_SERVICE_PATH = RESULTS_LS / "ls_clopper_pearson_per_service_200.csv"
BY_STRATUM_PATH = RESULTS_LS / "ls_by_stratum_summary_200.csv"


def result_to_row(service: str, stratum: str, policy: str, result: dict, n_intervals: int) -> dict:
    overload_arr = result.get("overload_arr")
    if overload_arr is not None:
        overload_count = int(np.asarray(overload_arr, dtype=bool).sum())
    else:
        overload_count = int(round(float(result["overload_fraction"]) * n_intervals))
    return {
        "service_id": service,
        "stratum": stratum,
        "policy": policy,
        "delta": DELTA,
        "n_intervals": n_intervals,
        "overload_count": overload_count,
        "overload_fraction": float(result["overload_fraction"]),
        "c_total": float(result["c_total"]),
        "c_res": float(result["c_res"]),
        "c_act": float(result["c_act"]),
        "c_vio": float(result["c_vio"]),
        "max_overload_run": int(result["max_overload_run"]),
    }


def materialize_baseline_stats_per_service() -> pd.DataFrame:
    """
    Create a per-service baseline table needed for paired Wilcoxon tests.

    run_ls_baselines.py writes only summaries, so the per-service rows are
    computed here and cached in RESULTS_LS before statistical testing.
    """
    if BASELINE_STATS_PER_SERVICE.exists():
        return pd.read_csv(BASELINE_STATS_PER_SERVICE)

    split_def = load_split_def()
    rows = []
    services = load_all_services_200(split_def)
    for idx, (service, history, test) in enumerate(services, start=1):
        stratum = GROUP_MAP_200[service]
        n = int(len(test))

        b2 = simulate_policy(
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
            use_margin=False,
            use_guardrail=False,
            **SIM_COST,
        )
        rows.append(result_to_row(service, stratum, "B2: pure predictive", b2, n))

        b4 = simulate_gaussian_margin_ls(history, test, DELTA)
        rows.append(result_to_row(service, stratum, "B4: Gaussian margin", b4, n))

        b6 = simulate_policy(
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
        rows.append(result_to_row(service, stratum, "B6: conformal + guardrail", b6, n))

        b7 = simulate_policy(
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
            use_guardrail=False,
            **SIM_COST,
        )
        rows.append(result_to_row(service, stratum, "B7: conformal margin only", b7, n))

        b8 = simulate_policy(
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
            use_margin=False,
            use_guardrail=True,
            **SIM_COST,
        )
        rows.append(result_to_row(service, stratum, "B8: guardrail only", b8, n))

        b9 = simulate_empirical_percentile_margin(history, test, DELTA)
        rows.append(result_to_row(service, stratum, "B9: empirical percentile + guardrail", b9, n))

        if idx % 25 == 0 or idx == len(services):
            print(f"materialized baseline stats rows for {idx}/{len(services)} services")

    out = pd.DataFrame(rows)
    out.to_csv(BASELINE_STATS_PER_SERVICE, index=False)
    return out


def load_frontier_stats_per_service() -> pd.DataFrame:
    if not FRONTIER_PER_SERVICE.exists():
        raise FileNotFoundError(f"Missing Phase 2 per-service frontier: {FRONTIER_PER_SERVICE}")
    df = pd.read_csv(FRONTIER_PER_SERVICE)
    df = df[df["delta"].eq(DELTA)].copy()
    df["policy"] = df["model"].map(
        {
            "persistence": "B6: conformal + guardrail",
            "arima": "arima",
            "xgb": "xgb",
            "lstm": "lstm",
        }
    )
    df["n_intervals"] = 4320
    df["overload_count"] = (df["overload_fraction"] * df["n_intervals"]).round().astype(int)
    return df[
        [
            "service_id",
            "stratum",
            "policy",
            "delta",
            "n_intervals",
            "overload_count",
            "overload_fraction",
            "c_total",
            "c_res",
            "c_act",
            "c_vio",
            "max_overload_run",
        ]
    ].copy()


def paired_values(df: pd.DataFrame, policy_a: str, policy_b: str) -> tuple[np.ndarray, np.ndarray]:
    wide = df[df["policy"].isin([policy_a, policy_b])].pivot_table(
        index="service_id",
        columns="policy",
        values="overload_fraction",
        aggfunc="first",
    )
    missing = {policy_a, policy_b} - set(wide.columns)
    if missing:
        raise ValueError(f"Missing policies for paired test: {sorted(missing)}")
    wide = wide[[policy_a, policy_b]].dropna()
    if len(wide) != 200:
        raise ValueError(f"Expected 200 paired services for {policy_a} vs {policy_b}, got {len(wide)}")
    return wide[policy_a].to_numpy(dtype=float), wide[policy_b].to_numpy(dtype=float)


def run_wilcoxon_tests(stats_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("B6: conformal + guardrail", "B7: conformal margin only", "B6 vs B7"),
        ("B6: conformal + guardrail", "B8: guardrail only", "B6 vs B8"),
        ("B6: conformal + guardrail", "B2: pure predictive", "B6 vs B2"),
        ("B6: conformal + guardrail", "B9: empirical percentile + guardrail", "B6 vs B9 percentile"),
        ("B6: conformal + guardrail", "B4: Gaussian margin", "B6 vs B4 Gaussian"),
        ("xgb", "arima", "xgb vs arima"),
        ("lstm", "arima", "lstm vs arima"),
        ("lstm", "xgb", "lstm vs xgb"),
    ]
    alpha_bonferroni = 0.05 / len(comparisons)
    rows = []
    for policy_a, policy_b, label in comparisons:
        x, y = paired_values(stats_df, policy_a, policy_b)
        diff = x - y
        nonzero = int(np.count_nonzero(np.abs(diff) > 1e-15))
        try:
            stat, p_value = stats.wilcoxon(x, y, alternative="less", zero_method="zsplit")
        except ValueError:
            stat, p_value = np.nan, np.nan
        reject = bool(np.isfinite(p_value) and p_value < alpha_bonferroni)
        rows.append(
            {
                "comparison": label,
                "policy_a": policy_a,
                "policy_b": policy_b,
                "alternative": "policy_a overload_fraction < policy_b overload_fraction",
                "n_pairs": int(len(x)),
                "n_nonzero_differences": nonzero,
                "median_overload_a": float(np.median(x)),
                "median_overload_b": float(np.median(y)),
                "median_difference_a_minus_b": float(np.median(diff)),
                "mean_difference_a_minus_b": float(np.mean(diff)),
                "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "alpha_bonferroni": alpha_bonferroni,
                "reject_H0": reject,
            }
        )
    return pd.DataFrame(rows)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if k == 0:
        lo = 0.0
    else:
        lo = float(beta.ppf(alpha / 2.0, k, n - k + 1))
    if k == n:
        hi = 1.0
    else:
        hi = float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def run_clopper_pearson(stats_df: pd.DataFrame) -> pd.DataFrame:
    policies = [
        "B6: conformal + guardrail",
        "B7: conformal margin only",
        "B8: guardrail only",
        "B2: pure predictive",
        "B9: empirical percentile + guardrail",
        "B4: Gaussian margin",
        "arima",
        "xgb",
        "lstm",
    ]
    rows = []
    for policy in policies:
        sub = stats_df[stats_df["policy"].eq(policy)].copy()
        if sub.empty:
            continue
        widths = []
        for _, row in sub.iterrows():
            lo, hi = clopper_pearson(int(row["overload_count"]), int(row["n_intervals"]))
            widths.append(hi - lo)
            rows.append(
                {
                    "policy": policy,
                    "service_id": row["service_id"],
                    "stratum": row["stratum"],
                    "delta": DELTA,
                    "n_intervals": int(row["n_intervals"]),
                    "overload_count": int(row["overload_count"]),
                    "overload_fraction": float(row["overload_fraction"]),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci_width": hi - lo,
                }
            )
    per_service = pd.DataFrame(rows)
    per_service.to_csv(CP_PER_SERVICE_PATH, index=False)
    summary = per_service.groupby("policy", sort=False).agg(
        services=("service_id", "nunique"),
        median_ci_width=("ci_width", "median"),
        p10_ci_width=("ci_width", lambda s: float(s.quantile(0.10))),
        p90_ci_width=("ci_width", lambda s: float(s.quantile(0.90))),
        max_ci_width=("ci_width", "max"),
        median_overload_fraction=("overload_fraction", "median"),
    ).reset_index()
    summary["delta"] = DELTA
    return summary


def build_by_stratum_summary() -> pd.DataFrame:
    if not FRONTIER_BY_STRATUM.exists():
        raise FileNotFoundError(f"Missing Phase 2 by-stratum frontier: {FRONTIER_BY_STRATUM}")
    df = pd.read_csv(FRONTIER_BY_STRATUM)
    out = df[df["model"].eq("persistence") & df["delta"].isin(DELTAS)].copy()
    out = out[
        [
            "stratum",
            "model",
            "delta",
            "services",
            "median_rel_cost",
            "median_overload_fraction",
            "frac_within_delta",
            "p10_rel_cost",
            "p90_rel_cost",
            "p10_overload_fraction",
            "p90_overload_fraction",
        ]
    ].sort_values(["stratum", "delta"], ascending=[True, False])
    out["policy"] = "B6: conformal + guardrail"
    return out[
        [
            "stratum",
            "policy",
            "model",
            "delta",
            "services",
            "median_rel_cost",
            "median_overload_fraction",
            "frac_within_delta",
            "p10_rel_cost",
            "p90_rel_cost",
            "p10_overload_fraction",
            "p90_overload_fraction",
        ]
    ]


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

    baseline_stats = materialize_baseline_stats_per_service()
    frontier_stats = load_frontier_stats_per_service()
    stats_df = pd.concat([baseline_stats, frontier_stats], ignore_index=True, sort=False)

    # Drop duplicate B6 rows, preferring the baseline materialisation.
    stats_df = stats_df.drop_duplicates(["service_id", "policy", "delta"], keep="first")

    tests = run_wilcoxon_tests(stats_df)
    cp = run_clopper_pearson(stats_df)
    by_stratum = build_by_stratum_summary()

    assert_clean(tests, cp, by_stratum)

    tests.to_csv(STAT_TESTS_PATH, index=False)
    cp.to_csv(CP_PATH, index=False)
    by_stratum.to_csv(BY_STRATUM_PATH, index=False)

    print("baseline_stats_rows", len(baseline_stats))
    print("frontier_stats_rows_delta_005", len(frontier_stats))
    print("wilcoxon_tests", len(tests))
    print("cp_rows", len(cp))
    print("by_stratum_rows", len(by_stratum))
    print("\nTESTS")
    print(tests.to_csv(index=False).strip())
    print("\nCP_POLICY_SUMMARY")
    print(cp.to_csv(index=False).strip())
    print("\nBY_STRATUM")
    print(by_stratum.to_csv(index=False).strip())
    print("\noutputs")
    for path in [STAT_TESTS_PATH, CP_PATH, CP_PER_SERVICE_PATH, BY_STRATUM_PATH, BASELINE_STATS_PER_SERVICE]:
        print(path)


if __name__ == "__main__":
    main()
