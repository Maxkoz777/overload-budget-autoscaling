from __future__ import annotations

from pathlib import Path

import pandas as pd

from metrics import CostConfig
from simulator import run_all_services


RESULTS_DIR = Path("experiments/results")
PER_SERVICE_PATH = RESULTS_DIR / "baseline_replay_per_service.csv"
SUMMARY_PATH = RESULTS_DIR / "baseline_replay_summary.csv"


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby("policy", sort=False)
    summary = grouped.agg(
        services=("msname", "nunique"),
        median_relative_cost=("relative_cost_vs_observed", "median"),
        p10_relative_cost=("relative_cost_vs_observed", lambda s: s.quantile(0.10)),
        p90_relative_cost=("relative_cost_vs_observed", lambda s: s.quantile(0.90)),
        median_overload_fraction=("overload_fraction", "median"),
        p10_overload_fraction=("overload_fraction", lambda s: s.quantile(0.10)),
        p90_overload_fraction=("overload_fraction", lambda s: s.quantile(0.90)),
        median_capacity_mean=("capacity_mean", "median"),
        median_scaling_churn=("scaling_churn", "median"),
        median_max_overload_run=("max_overload_run", "median"),
    )
    return summary.reset_index()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cost_config = CostConfig(c_res=1.0, c_act=0.05, c_vio=10.0)
    results = run_all_services(mu=1.0, cost_config=cost_config)

    observed_cost = (
        results[results["policy"] == "observed_capacity"]
        .set_index("msname")["c_total"]
        .rename("observed_c_total")
    )
    results = results.join(observed_cost, on="msname")
    results["relative_cost_vs_observed"] = results["c_total"] / results["observed_c_total"]

    summary = summarize(results)
    results.to_csv(PER_SERVICE_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("per_service_rows", len(results))
    print("services", results["msname"].nunique())
    print("policies", results["policy"].nunique())
    print("mu", 1.0)
    print("cost_config", cost_config)
    print("\nSUMMARY")
    print(summary.to_csv(index=False).strip())
    print("\noutputs")
    print(PER_SERVICE_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
