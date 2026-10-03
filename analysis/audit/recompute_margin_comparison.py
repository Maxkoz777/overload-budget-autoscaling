"""Recompute the symmetric conformal-versus-Gaussian comparison.

The script uses the saved 200-service trace suite and performs a factorial
comparison of margin family (split conformal or Gaussian plug-in) and guard
rail (off or on) at delta in {0.05, 0.01}.  No forecaster is retrained: every
policy uses the same causal one-step persistence forecast and the same rolling
window.  The Gaussian comparator is

    max(0, mean(e) + z_(1-delta) * std(e)),

computed from signed residuals with ddof=0.  It is a transparent parametric
plug-in baseline, not a finite-sample coverage guarantee.

Run from the paper repository:

    python3 audit/recompute_margin_comparison.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, wilcoxon


PAPER_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PAPER_ROOT.parent
EXPERIMENT_ROOT = RESEARCH_ROOT / "experiments"
DATA_ROOT = EXPERIMENT_ROOT / "data" / "service_timeseries_200"
SPLIT_PATH = EXPERIMENT_ROOT / "data" / "splits" / "split_definition.json"
SELECTION_PATH = EXPERIMENT_ROOT / "data" / "splits" / "selected_services_200.csv"
OUT_DIR = PAPER_ROOT / "audit" / "comparative_results"
FIG_DIR = PAPER_ROOT / "figures"

DELTAS = (0.05, 0.01)
WINDOW = 240
MU = 1.0
RHO = 0.70
GUARD_H = 2
GUARD_GAMMA = 1
C_RES = 1.0
C_ACT = 0.05
C_VIO = 10.0
BOOTSTRAP_REPLICATES = 20_000
SEED = 20260920


def split_masks(frame: pd.DataFrame, split: dict) -> tuple[pd.Series, pd.Series]:
    by_name = {item["name"]: item for item in split["splits"]}
    history_end = by_name["calibration"]["timestamp_end_exclusive"]
    test = by_name["test"]
    history_mask = frame["timestamp"] < history_end
    test_mask = (
        (frame["timestamp"] >= test["timestamp_start"])
        & (frame["timestamp"] < test["timestamp_end_exclusive"])
    )
    return history_mask, test_mask


def load_service(service_id: str, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(DATA_ROOT / f"{service_id}.parquet").sort_values("timestamp")
    history_mask, test_mask = split_masks(frame, split)
    return frame.loc[history_mask].copy(), frame.loc[test_mask].copy()


def total_cost(capacity: np.ndarray, demand: np.ndarray) -> float:
    overload = demand > MU * capacity
    churn = np.abs(np.diff(capacity)).sum()
    return float(C_RES * capacity.sum() + C_ACT * churn + C_VIO * overload.sum())


def observed_cost(test: pd.DataFrame) -> float:
    capacity = np.maximum(np.ceil(test["replica_count"].to_numpy(float)), 1).astype(int)
    return total_cost(capacity, test["cpu_sum"].to_numpy(float))


def conformal_margin(window: np.ndarray, delta: float) -> float:
    positive = np.maximum(window, 0.0)
    rank = min(len(positive), int(np.ceil((len(positive) + 1) * (1.0 - delta))))
    return float(np.partition(positive, rank - 1)[rank - 1])


def gaussian_margin(window: np.ndarray, delta: float) -> float:
    z = float(norm.ppf(1.0 - delta))
    return max(0.0, float(window.mean()) + z * float(window.std(ddof=0)))


def simulate(
    history: pd.DataFrame,
    test: pd.DataFrame,
    *,
    delta: float,
    margin_family: str,
    guard: bool,
) -> dict:
    history_y = history["cpu_sum"].to_numpy(float)
    demand = test["cpu_sum"].to_numpy(float)
    residuals = list(np.diff(history_y))
    previous = float(history_y[-1])
    overload_history: list[bool] = []
    soft_history: list[bool] = []
    capacities: list[int] = []
    margins: list[float] = []
    trigger_count = 0

    for realised in demand:
        forecast = max(previous, 0.0)
        window = np.asarray(residuals[-WINDOW:], dtype=float)
        if margin_family == "conformal":
            margin = conformal_margin(window, delta)
        elif margin_family == "gaussian":
            margin = gaussian_margin(window, delta)
        else:
            raise ValueError(f"unknown margin family: {margin_family}")

        nominal = max(int(np.ceil((forecast + margin) / MU)), 1)
        trigger = False
        if guard and len(overload_history) >= GUARD_H:
            trigger = all(overload_history[-GUARD_H:]) or all(soft_history[-GUARD_H:])
        capacity = nominal + (GUARD_GAMMA if trigger else 0)
        overload = float(realised) > MU * capacity
        soft = float(realised) > RHO * MU * capacity

        capacities.append(capacity)
        margins.append(margin)
        trigger_count += int(trigger)
        overload_history.append(bool(overload))
        soft_history.append(bool(soft))
        residuals.append(float(realised) - forecast)
        previous = float(realised)

    capacity_array = np.asarray(capacities, dtype=int)
    overload_array = demand > MU * capacity_array
    return {
        "overload_fraction": float(overload_array.mean()),
        "total_cost": total_cost(capacity_array, demand),
        "mean_capacity": float(capacity_array.mean()),
        "scaling_churn": int(np.abs(np.diff(capacity_array)).sum()),
        "mean_margin": float(np.mean(margins)),
        "guard_rate": float(trigger_count / len(demand)),
    }


def summarize(per_service: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, delta, margin_family, guard), group in per_service.groupby(
        ["subset", "delta", "margin_family", "guard"], sort=True
    ):
        rows.append(
            {
                "subset": subset,
                "services": len(group),
                "delta": delta,
                "margin_family": margin_family,
                "guard": bool(guard),
                "median_overload": float(group["overload_fraction"].median()),
                "mean_overload": float(group["overload_fraction"].mean()),
                "median_relative_cost": float(group["relative_cost"].median()),
                "mean_relative_cost": float(group["relative_cost"].mean()),
                "compliance": float(group["within_delta"].mean()),
                "median_guard_rate": float(group["guard_rate"].median()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean_difference(
    merged: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    *,
    alpha: float,
    seed: int,
) -> tuple[float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    diff = (merged[metric_a] - merged[metric_b]).to_numpy(float)
    strata = merged["stratum"].to_numpy(str)
    unique = sorted(set(strata))
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for b in range(BOOTSTRAP_REPLICATES):
        pieces = []
        for stratum in unique:
            values = diff[strata == stratum]
            pieces.append(rng.choice(values, size=len(values), replace=True))
        draws[b] = np.concatenate(pieces).mean()
    return (
        float(diff.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )


def paired_effects(per_service: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects: list[dict] = []
    compliance_tests: list[dict] = []
    family_size = len(DELTAS) * 2
    family_alpha = 0.05 / family_size
    seed_counter = 0

    for subset in ("focused20", "all200"):
        for delta in DELTAS:
            for guard in (False, True):
                selected = per_service[
                    per_service["subset"].eq(subset)
                    & per_service["delta"].eq(delta)
                    & per_service["guard"].eq(guard)
                ]
                wide = selected.pivot(index="service_id", columns="margin_family")
                base = pd.DataFrame(
                    {
                        "service_id": wide.index,
                        "stratum": wide["stratum"]["conformal"].to_numpy(),
                        "conformal_overload": wide["overload_fraction"]["conformal"].to_numpy(),
                        "gaussian_overload": wide["overload_fraction"]["gaussian"].to_numpy(),
                        "conformal_cost": wide["relative_cost"]["conformal"].to_numpy(),
                        "gaussian_cost": wide["relative_cost"]["gaussian"].to_numpy(),
                    }
                )

                for metric in ("overload", "cost"):
                    seed_counter += 1
                    mean_diff, ci_lo, ci_hi, family_lo, family_hi = bootstrap_mean_difference(
                        base,
                        f"conformal_{metric}",
                        f"gaussian_{metric}",
                        alpha=family_alpha,
                        seed=SEED + seed_counter,
                    )
                    diff = base[f"conformal_{metric}"] - base[f"gaussian_{metric}"]
                    if np.allclose(diff, 0):
                        p_value = 1.0
                    else:
                        p_value = float(wilcoxon(diff, alternative="two-sided").pvalue)
                    effects.append(
                        {
                            "subset": subset,
                            "delta": delta,
                            "guard": guard,
                            "metric": metric,
                            "services": len(base),
                            "mean_difference_conformal_minus_gaussian": mean_diff,
                            "pointwise_ci_lo": ci_lo,
                            "pointwise_ci_hi": ci_hi,
                            "familywise_ci_lo": family_lo,
                            "familywise_ci_hi": family_hi,
                            "wilcoxon_two_sided_p": p_value,
                            "family_size": family_size,
                            "family_alpha": family_alpha,
                        }
                    )

                conformal_ok = base["conformal_overload"] <= delta
                gaussian_ok = base["gaussian_overload"] <= delta
                wins = int((conformal_ok & ~gaussian_ok).sum())
                losses = int((~conformal_ok & gaussian_ok).sum())
                discordant = wins + losses
                compliance_tests.append(
                    {
                        "subset": subset,
                        "delta": delta,
                        "guard": guard,
                        "conformal_only_compliant": wins,
                        "gaussian_only_compliant": losses,
                        "discordant_pairs": discordant,
                        "exact_mcnemar_two_sided_p": (
                            float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
                        ),
                    }
                )
    return pd.DataFrame(effects), pd.DataFrame(compliance_tests)


def plot_factorial(summary: pd.DataFrame) -> None:
    mpl_dir = PAPER_ROOT / "tmp" / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    colours = {"conformal": "#2E7D32", "gaussian": "#AD1457"}
    markers = {False: "o", True: "s"}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), constrained_layout=True)
    for row, subset in enumerate(("focused20", "all200")):
        for col, delta in enumerate(DELTAS):
            ax = axes[row, col]
            frame = summary[summary["subset"].eq(subset) & summary["delta"].eq(delta)]
            for family in ("conformal", "gaussian"):
                path = frame[frame["margin_family"].eq(family)].sort_values("guard")
                ax.plot(
                    path["median_relative_cost"],
                    path["compliance"],
                    color=colours[family],
                    linewidth=1.4,
                    alpha=0.8,
                )
                for item in path.itertuples():
                    ax.scatter(
                        item.median_relative_cost,
                        item.compliance,
                        color=colours[family],
                        marker=markers[bool(item.guard)],
                        s=55,
                        label=f"{family.title()}, {'guard' if item.guard else 'no guard'}",
                    )
            ax.set_title(f"{'Focused 20' if subset == 'focused20' else 'All 200'}, δ={delta:.2f}")
            ax.set_xlabel("Median relative total cost")
            ax.set_ylabel("Service-level compliance")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.grid(alpha=0.25, linestyle="--")
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=7, loc="best")
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig_margin_factorial.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_PATH.read_text())
    selection = pd.read_csv(SELECTION_PATH)
    rows: list[dict] = []

    for index, item in enumerate(selection.itertuples(), start=1):
        history, test = load_service(item.service_id, split)
        denominator = observed_cost(test)
        for delta in DELTAS:
            for margin_family in ("conformal", "gaussian"):
                for guard in (False, True):
                    result = simulate(
                        history,
                        test,
                        delta=delta,
                        margin_family=margin_family,
                        guard=guard,
                    )
                    rows.append(
                        {
                            "service_id": item.service_id,
                            "stratum": item.stratum,
                            "focused": bool(item.is_focused_20),
                            "delta": delta,
                            "margin_family": margin_family,
                            "guard": guard,
                            **result,
                            "relative_cost": result["total_cost"] / denominator,
                            "within_delta": result["overload_fraction"] <= delta,
                        }
                    )
        if index % 25 == 0 or index == len(selection):
            print(f"processed {index}/{len(selection)} services", flush=True)

    all_rows = pd.DataFrame(rows)
    focused = all_rows[all_rows["focused"]].copy()
    focused["subset"] = "focused20"
    all_rows["subset"] = "all200"
    per_service = pd.concat([focused, all_rows], ignore_index=True)
    summary = summarize(per_service)
    effects, compliance = paired_effects(per_service)

    # The historical replay is checked against its saved artefact. Verified
    # inputs are deliberately independent of that historical snapshot.
    saved_rows_verified = 0
    if DATA_ROOT.name == "service_timeseries_200":
        saved = pd.read_csv(
            EXPERIMENT_ROOT / "results" / "large_scale" / "analysis" / "ls_frontier_per_service_200.csv"
        )
        saved = saved[saved["model"].eq("persistence") & saved["delta"].isin(DELTAS)]
        reconstructed = all_rows[
            all_rows["margin_family"].eq("conformal") & all_rows["guard"]
        ]
        check = reconstructed.merge(saved, on=["service_id", "delta"], suffixes=("_new", "_saved"))
        if len(check) != 400:
            raise AssertionError(f"expected 400 saved conformal rows, found {len(check)}")
        if not np.allclose(check["overload_fraction_new"], check["overload_fraction_saved"], atol=1e-12):
            raise AssertionError("conformal overload replay does not match saved results")
        if not np.allclose(check["relative_cost"], check["rel_cost"], atol=1e-12):
            raise AssertionError("conformal cost replay does not match saved results")
        saved_rows_verified = len(check)

    per_service.to_csv(OUT_DIR / "margin_comparison_per_service.csv", index=False)
    summary.to_csv(OUT_DIR / "margin_comparison_summary.csv", index=False)
    effects.to_csv(OUT_DIR / "margin_paired_effects.csv", index=False)
    compliance.to_csv(OUT_DIR / "margin_compliance_tests.csv", index=False)
    metadata = {
        "generated_by": "audit/recompute_margin_comparison.py",
        "services": 200,
        "focused_services": 20,
        "deltas": list(DELTAS),
        "window": WINDOW,
        "gaussian_formula": "max(0, mean(signed residual) + z_(1-delta) * std_ddof0(signed residual))",
        "guard": {"rho": RHO, "h": GUARD_H, "gamma": GUARD_GAMMA},
        "cost": {"resource": C_RES, "churn": C_ACT, "overload": C_VIO},
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": SEED,
        "input_series": str(DATA_ROOT.relative_to(EXPERIMENT_ROOT)),
        "saved_conformal_rows_verified": saved_rows_verified,
    }
    (OUT_DIR / "margin_methodology.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_factorial(summary)

    print("\nMARGIN SUMMARY")
    print(summary.to_string(index=False))
    print("\nPAIRED EFFECTS")
    print(effects.to_string(index=False))
    print("\nCOMPLIANCE TESTS")
    print(compliance.to_string(index=False))


if __name__ == "__main__":
    import sys
    if "--verified" in sys.argv:
        DATA_ROOT = EXPERIMENT_ROOT / "data" / "service_timeseries_200_verified"
        OUT_DIR = PAPER_ROOT / "audit" / "verified_results" / "comparative"
        FIG_DIR = PAPER_ROOT / "figures" / "verified"
    main()
