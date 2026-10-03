#!/usr/bin/env python3
"""Install the MS_25320 past-only refit forecasts and recompute focused outputs.

The historical focused XGBoost/LSTM forecasts for
``MS_25320`` were fitted on a history with eight adjacent-value-filled demand
minutes (days 8--9).  ``audit/refit_focused_causal_fill.py`` repeats the full
search / fit / causal inference pipeline on the past-only series.  This script

1. verifies the refit metadata (service, past-only input hash, 4,320 causal
   predictions, completed status);
2. preserves the historical focused forecast files once under
   ``experiments/results/superseded_2026-09-28/`` and writes new canonical
   ``experiments/results/{xgb,lstm}_forecasts.parquet`` in which only the
   MS_25320 rows are replaced (the other 19 services are checked to be
   semantically identical);
3. reruns every experiment/audit producer that reads the focused learned
   forecasts, splicing focused rows into files that also contain rows from
   inputs outside this package (timeline and statistical-robustness files);
4. checks that replaying MS_25320 on the past-only history gives the same
   held-out policy outcomes as on the historical history for persistence and
   for the new learned forecasts (the eight differing minutes precede the
   residual warm-start window).

Run from the paper directory::

    python3 audit/recompute_focused_learned_downstream.py --write

``--check`` verifies steps 1 and 2 in memory (refit metadata/hashes and the
merge rule); it does not compare with the installed canonical parquet files
and does not repeat step 4 (the replay-invariance check runs only with
``--write``).

Superseded by ``audit/recompute_final_closure_downstream.py``,
whose ``--check`` compares the expected and installed canonical forecasts,
verifies run provenance, trial and fold counts, replays the policies and
repeats the replay-invariance check.  After the final closure the canonical
files no longer equal this script's merge, so its ``--check`` is expected to fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
EXP = RESEARCH / "experiments"
RESULTS = EXP / "results"
SUPERSEDED = RESULTS / "superseded_2026-09-28"
REFIT = PAPER / "audit" / "verified_results" / "focused_causal_refit"
SERVICE = "MS_25320"
PAST_ONLY = EXP / "data" / "service_timeseries_200_verified" / f"{SERVICE}.parquet"
HISTORICAL_INPUT = EXP / "data" / "service_timeseries" / f"{SERVICE}.parquet"
MODELS = ("xgb", "lstm")
TEST_START = 864_000_000
OUT = REFIT / "downstream"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_evidence() -> pd.DataFrame:
    """Record the historical/past-only demand differences for the focused 20."""
    selection = pd.read_csv(EXP / "data/splits/selected_services_200.csv")
    focused = sorted(selection.loc[selection.is_focused_20.astype(bool), "service_id"])
    rows, counts = [], {}
    for service in focused:
        old = pd.read_parquet(EXP / "data/service_timeseries" / f"{service}.parquet").sort_values("timestamp")
        new = pd.read_parquet(EXP / "data/service_timeseries_200_verified" / f"{service}.parquet").sort_values("timestamp")
        assert np.array_equal(old.timestamp.to_numpy(), new.timestamp.to_numpy())
        diff = old.cpu_sum.to_numpy(float) != new.cpu_sum.to_numpy(float)
        counts[service] = int(diff.sum())
        for ts, a, b, miss in zip(old.timestamp.to_numpy()[diff], old.cpu_sum.to_numpy()[diff],
                                  new.cpu_sum.to_numpy()[diff], old.was_missing.to_numpy()[diff]):
            rows.append({"service_id": service, "timestamp": int(ts), "day": int(ts // 86_400_000),
                         "cpu_sum_historical_fill": float(a), "cpu_sum_past_only": float(b),
                         "was_missing": bool(miss)})
    assert {k: v for k, v in counts.items() if v} == {SERVICE: 8}, counts
    frame = pd.DataFrame(rows)
    assert frame.timestamp.lt(TEST_START).all() and frame.was_missing.all()
    return frame


def verify_refit(model: str) -> pd.DataFrame:
    run = REFIT / f"{model}_pastonly"
    meta = json.loads((run / "run_metadata.json").read_text())
    assert meta["status"] == "completed", meta["status"]
    assert meta["service"] == SERVICE and meta["model"] == model
    assert meta["input_sha256"] == sha256(PAST_ONLY), "refit input is not the past-only series"
    for name, digest in meta["artefact_sha256"].items():
        if name.endswith(".log"):
            # stdout/stderr are captured by the calling shell and keep growing
            # after run_metadata.json is written; they are not model artefacts.
            continue
        assert sha256(run / name) == digest, f"hash mismatch: {run / name}"
    frame = pd.read_parquet(run / "forecasts.parquet")
    assert len(frame) == 4320 and frame.timestamp.is_unique
    assert (frame.msname == SERVICE).all() and (frame.model == model).all()
    assert np.isfinite(frame.forecast).all()
    return frame


def canonical_forecasts(model: str, refit: pd.DataFrame, write: bool) -> pd.DataFrame:
    canonical = RESULTS / f"{model}_forecasts.parquet"
    historical = SUPERSEDED / f"{model}_forecasts.parquet"
    if not historical.exists():
        if not write:
            source = canonical
        else:
            SUPERSEDED.mkdir(parents=True, exist_ok=True)
            shutil.copy2(canonical, historical)
            source = historical
    else:
        source = historical
    old = pd.read_parquet(source)
    others = old[old.msname != SERVICE]
    old_service = old[old.msname == SERVICE].sort_values("timestamp")
    assert len(old_service) == 4320
    assert (old_service.timestamp.to_numpy() == refit.sort_values("timestamp").timestamp.to_numpy()).all()
    new_rows = refit[old.columns].astype(old.dtypes.to_dict())
    new = pd.concat([others, new_rows], ignore_index=True)
    new = new.sort_values(["msname", "timestamp"], kind="stable").reset_index(drop=True)
    # Semantic equality of the untouched 19 services.
    lhs = others.sort_values(["msname", "timestamp"]).reset_index(drop=True)
    rhs = new[new.msname != SERVICE].reset_index(drop=True)
    pd.testing.assert_frame_equal(lhs, rhs, check_exact=True)
    assert new.msname.nunique() == 20 and len(new) == 20 * 4320
    if write:
        new.to_parquet(canonical, index=False)
    return new


def run(cmd: list[str], log: Path) -> None:
    with open(log, "w") as handle:
        proc = subprocess.run(cmd, cwd=RESEARCH, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}; see {log}")


def splice_timeline() -> None:
    """Replace focused rows of the timeline files; keep rows from other tiers."""
    target = PAPER / "audit" / "timeline_results"
    saved = {name: pd.read_csv(target / name) for name in (
        "warmstart_sensitivity_per_service.csv", "warmstart_sensitivity_summary.csv")}
    run([sys.executable, str(PAPER / "audit" / "recompute_timeline_sensitivity.py")], OUT / "log_timeline.txt")
    for name, old in saved.items():
        fresh = pd.read_csv(target / name)
        focused = fresh[fresh.tier == "focused20"]
        other = old[old.tier != "focused20"]
        if (fresh.tier != "focused20").any():
            # Full inputs were available: the fresh file is already complete.
            continue
        merged = pd.concat([focused, other], ignore_index=True)[old.columns]
        merged.to_csv(target / name, index=False)


def splice_statistics() -> None:
    """Recompute the focused paired-effect families with the original seed."""
    sys.path.insert(0, str(PAPER / "audit"))
    import statistical_robustness as sr  # noqa: E402
    cwd = os.getcwd()
    os.chdir(RESEARCH)
    sys.path.insert(0, str(EXP / "src"))
    import exp_core  # noqa: E402
    rng = np.random.default_rng(sr.SEED)
    focused = sr.load_focused_frame(EXP, exp_core)
    overload = [
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
    cost = [
        ("LSTM vs reactive cost", "LSTM", "Reactive", "rel_cost", "less"),
        ("XGBoost vs reactive cost", "XGB", "Reactive", "rel_cost", "less"),
    ]
    # Same families, same order and same RNG stream as statistical_robustness.main().
    rows = sr.paired_effect_rows(focused, overload, "focused_overload", 20_000, rng)
    rows += sr.paired_effect_rows(focused, cost, "focused_cost", 20_000, rng)
    os.chdir(cwd)
    path = PAPER / "audit" / "statistical_results" / "paired_effect_intervals.csv"
    old = pd.read_csv(path)
    merged = pd.concat([pd.DataFrame(rows), old[~old.family.str.startswith("focused")]], ignore_index=True)
    merged[old.columns].to_csv(path, index=False)


def replay_input_invariance() -> dict:
    """MS_25320 held-out replays must not depend on the eight pre-test minutes."""
    sys.path.insert(0, str(EXP / "src"))
    cwd = os.getcwd()
    os.chdir(RESEARCH)
    import exp_core  # noqa: E402
    split = json.loads((EXP / "data/splits/split_definition.json").read_text())
    out = {}
    hist_old, test_old = exp_core.load_service_data(HISTORICAL_INPUT, split)
    hist_new, test_new = exp_core.load_service_data(PAST_ONLY, split)
    pd.testing.assert_frame_equal(test_old.reset_index(drop=True), test_new.reset_index(drop=True))
    for model in ("persistence", *MODELS):
        forecast = None
        if model != "persistence":
            frame = pd.read_parquet(RESULTS / f"{model}_forecasts.parquet")
            forecast = frame[frame.msname == SERVICE].sort_values("timestamp").forecast.to_numpy(float)
        for delta in (0.10, 0.05, 0.01):
            a = exp_core.simulate_policy(hist_old, test_old, forecast, delta=delta, W=240,
                                         use_margin=True, use_guardrail=True)
            b = exp_core.simulate_policy(hist_new, test_new, forecast, delta=delta, W=240,
                                         use_margin=True, use_guardrail=True)
            same = all(
                np.array_equal(np.asarray(a[key]), np.asarray(b[key]))
                for key in a if not isinstance(a[key], dict)
            )
            out[f"{model}_d{delta}"] = bool(same)
    # The focused replay script uses its own simulator; check it as well.
    import run_advanced_policy_replay as adv  # noqa: E402
    from risk_policy import RiskPolicyConfig  # noqa: E402
    for model in MODELS:
        frame = pd.read_parquet(RESULTS / f"{model}_forecasts.parquet")
        forecast = frame[frame.msname == SERVICE].sort_values("timestamp").forecast.to_numpy(float)
        for delta in adv.DELTA_VALUES:
            for guard in (False, True):
                config = RiskPolicyConfig(
                    delta=delta, window=adv.WINDOW, alpha=adv.ALPHA, mu=adv.MU, rho=adv.RHO,
                    guardrail_h=adv.GUARDRAIL_H, guardrail_gamma=adv.GUARDRAIL_GAMMA,
                    use_margin=True, use_guardrail=guard,
                )
                a = adv.simulate_policy_with_external_forecast(hist_old, test_old, forecast, config)
                b = adv.simulate_policy_with_external_forecast(hist_new, test_new, forecast, config)
                out[f"advanced_{model}_d{delta}_guard{guard}"] = bool(np.array_equal(a, b))
    os.chdir(cwd)
    assert all(out.values()), out
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    evidence = input_evidence()
    if args.write:
        evidence.to_csv(OUT / "input_differences.csv", index=False)
    refits = {model: verify_refit(model) for model in MODELS}
    for model in MODELS:
        canonical_forecasts(model, refits[model], write=args.write)
    if not args.write:
        print(json.dumps({"refits_verified": True}, indent=2))
        return

    src = EXP / "src"
    py = sys.executable
    run([py, str(src / "run_advanced_policy_replay.py")], OUT / "log_advanced_policy_replay.txt")
    run([py, str(src / "evaluate_advanced_forecasting.py")], OUT / "log_evaluate_advanced_forecasting.txt")
    run([py, str(src / "run_c5_frontier_density.py")], OUT / "log_c5.txt")
    run([py, str(src / "run_c1_adaptive_conformal.py"), "--fresh"], OUT / "log_c1.txt")
    run([py, str(src / "run_supplemental_experiments.py")], OUT / "log_supplemental.txt")
    run([py, str(src / "run_c4_exchangeability.py")], OUT / "log_c4.txt")
    # exp_validator_fixes.fix3 is persistence-only and incompatible with pandas 3.
    # fix6 (exp_overhead_frontier) is a historical output that the
    # manuscript does not use; only fix4 (focused Wilcoxon/Clopper-Pearson) reads
    # the learned forecasts for a reported table and is rerun.
    run([py, "-c", "import sys; sys.path.insert(0, 'experiments/src'); "
         "import exp_validator_fixes as m; m.fix4_statistical_tests()"],
        OUT / "log_validator_fix4.txt")
    splice_timeline()
    splice_statistics()
    audit = PAPER / "audit"
    run([py, str(audit / "recompute_overhead_accounting.py"), "--write"], OUT / "log_overhead.txt")
    run([py, str(audit / "plot_focused_figures.py"), "--write"], OUT / "log_phase6.txt")
    run([py, str(audit / "recompute_arima_causal.py"), "--diagnostic-figure-only"], OUT / "log_c4_figure.txt")
    invariance = replay_input_invariance()

    summary = {
        "service": SERVICE,
        "refit_forecasts_sha256": {m: sha256(REFIT / f"{m}_pastonly" / "forecasts.parquet") for m in MODELS},
        "canonical_forecasts_sha256": {m: sha256(RESULTS / f"{m}_forecasts.parquet") for m in MODELS},
        "superseded_forecasts_sha256": {m: sha256(SUPERSEDED / f"{m}_forecasts.parquet") for m in MODELS},
        "replay_input_invariance": invariance,
        "timing_policy": (
            "experiments/results/{xgb,lstm}_training_log.csv are unchanged historical run logs; "
            "stage timings in the Supplement remain the original training-run profile. "
            "Refit timings are recorded separately in run_metadata.json."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
