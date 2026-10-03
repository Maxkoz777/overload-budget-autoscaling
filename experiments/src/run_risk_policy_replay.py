from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metrics import CostConfig, evaluate_capacity
from risk_policy import RiskPolicyConfig, simulate_risk_policy
from simulator import split_frame


TIME_SERIES_DIR = Path("experiments/data/service_timeseries")
SPLIT_DEFINITION_PATH = Path("experiments/data/splits/split_definition.json")
RESULTS_DIR = Path("experiments/results")
PER_SERVICE_PATH = RESULTS_DIR / "risk_policy_replay_per_service.csv"
SUMMARY_PATH = RESULTS_DIR / "risk_policy_replay_summary.csv"
DIAGNOSTICS_PATH = RESULTS_DIR / "risk_policy_diagnostics_sample.csv"

DELTAS = [0.10, 0.05, 0.01]
WINDOW = 240
ALPHA = 1.0
RHO = 0.70
GUARDRAIL_H = 2
GUARDRAIL_GAMMA = 1
MU = 1.0


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["policy", "delta"], sort=False, dropna=False)
    return grouped.agg(
        services=("msname", "nunique"),
        median_relative_cost=("relative_cost_vs_observed", "median"),
        p10_relative_cost=("relative_cost_vs_observed", lambda s: s.quantile(0.10)),
        p90_relative_cost=("relative_cost_vs_observed", lambda s: s.quantile(0.90)),
        median_overload_fraction=("overload_fraction", "median"),
        p10_overload_fraction=("overload_fraction", lambda s: s.quantile(0.10)),
        p90_overload_fraction=("overload_fraction", lambda s: s.quantile(0.90)),
        median_margin_mean=("margin_mean", "median"),
        median_margin_p95=("margin_p95", "median"),
        median_guardrail_activation_rate=("guardrail_activation_rate", "median"),
        median_capacity_mean=("capacity_mean", "median"),
        median_scaling_churn=("scaling_churn", "median"),
        median_max_overload_run=("max_overload_run", "median"),
    ).reset_index()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_definition = json.loads(SPLIT_DEFINITION_PATH.read_text())
    split_map = {split["name"]: split for split in split_definition["splits"]}
    cost_config = CostConfig(c_res=1.0, c_act=0.05, c_vio=10.0)

    rows = []
    diagnostics_rows = []

    for path in sorted(TIME_SERIES_DIR.glob("MS_*.parquet")):
        df = pd.read_parquet(path).sort_values("timestamp")
        service = str(df["msname"].iloc[0])
        train = split_frame(df, split_map["train"])
        calibration = split_frame(df, split_map["calibration"])
        test = split_frame(df, split_map["test"])
        history = pd.concat([train, calibration], ignore_index=True)

        observed = evaluate_capacity(
            service=service,
            policy="observed_capacity",
            timestamps=test["timestamp"],
            demand=test["cpu_sum"].to_numpy(dtype=float),
            capacity=test["replica_count"].to_numpy(dtype=int),
            mu=MU,
            cost_config=cost_config,
        )
        observed_cost = float(observed["c_total"])

        for delta in DELTAS:
            configs = [
                (
                    "risk_conformal_margin_only",
                    RiskPolicyConfig(
                        delta=delta,
                        window=WINDOW,
                        alpha=ALPHA,
                        mu=MU,
                        rho=RHO,
                        guardrail_h=GUARDRAIL_H,
                        guardrail_gamma=GUARDRAIL_GAMMA,
                        use_margin=True,
                        use_guardrail=False,
                    ),
                ),
                (
                    "risk_conformal_guardrail",
                    RiskPolicyConfig(
                        delta=delta,
                        window=WINDOW,
                        alpha=ALPHA,
                        mu=MU,
                        rho=RHO,
                        guardrail_h=GUARDRAIL_H,
                        guardrail_gamma=GUARDRAIL_GAMMA,
                        use_margin=True,
                        use_guardrail=True,
                    ),
                ),
                (
                    "guardrail_only_no_margin",
                    RiskPolicyConfig(
                        delta=delta,
                        window=WINDOW,
                        alpha=ALPHA,
                        mu=MU,
                        rho=RHO,
                        guardrail_h=GUARDRAIL_H,
                        guardrail_gamma=GUARDRAIL_GAMMA,
                        use_margin=False,
                        use_guardrail=True,
                    ),
                ),
            ]

            for policy_name, config in configs:
                capacity, diagnostics = simulate_risk_policy(history, test, config)
                metrics = evaluate_capacity(
                    service=service,
                    policy=policy_name,
                    timestamps=test["timestamp"],
                    demand=test["cpu_sum"].to_numpy(dtype=float),
                    capacity=capacity,
                    mu=MU,
                    cost_config=cost_config,
                )
                metrics.update(
                    {
                        "delta": delta,
                        "window": WINDOW,
                        "alpha": ALPHA,
                        "rho": RHO,
                        "guardrail_h": GUARDRAIL_H,
                        "guardrail_gamma": GUARDRAIL_GAMMA,
                        "margin_mean": float(diagnostics["margin"].mean()),
                        "margin_p50": float(diagnostics["margin"].quantile(0.50)),
                        "margin_p95": float(diagnostics["margin"].quantile(0.95)),
                        "guardrail_activation_rate": float(
                            diagnostics["guardrail_trigger"].mean()
                        ),
                        "relative_cost_vs_observed": float(metrics["c_total"])
                        / observed_cost,
                    }
                )
                rows.append(metrics)

                if service in {"MS_14526", "MS_25320"} and policy_name == "risk_conformal_guardrail":
                    sample = diagnostics.head(120).copy()
                    sample.insert(0, "msname", service)
                    sample.insert(1, "delta", delta)
                    sample.insert(2, "policy", policy_name)
                    diagnostics_rows.append(sample)

    results = pd.DataFrame(rows)
    summary = summarize(results)
    results.to_csv(PER_SERVICE_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    if diagnostics_rows:
        pd.concat(diagnostics_rows, ignore_index=True).to_csv(DIAGNOSTICS_PATH, index=False)

    print("per_service_rows", len(results))
    print("services", results["msname"].nunique())
    print("policies", results["policy"].nunique())
    print("deltas", ",".join(str(d) for d in DELTAS))
    print("window", WINDOW)
    print("alpha", ALPHA)
    print("rho", RHO)
    print("guardrail_h", GUARDRAIL_H)
    print("guardrail_gamma", GUARDRAIL_GAMMA)
    print("\nSUMMARY")
    print(summary.to_csv(index=False).strip())
    print("\noutputs")
    print(PER_SERVICE_PATH)
    print(SUMMARY_PATH)
    if diagnostics_rows:
        print(DIAGNOSTICS_PATH)


if __name__ == "__main__":
    main()
