#!/usr/bin/env python3
"""Final-closure focused learned runs, executed locally on macOS.

Purpose
-------
1. XGBoost: the historical focused inference placed the feature row of minute
   ``t-1`` against target ``t``.  ``experiments/src/train_xgb.py`` now builds
   the feature row of the target minute (``next_step_features``).  No original
   focused XGBoost checkpoints and no complete best-parameter sets exist (the
   historical training log keeps only eta, max_depth and subsample), so the
   original focused protocol is repeated for all 20 focused services:
   RandomSampler(seed=42), 200 trials, the unchanged expanding-window CV
   (5 configured folds of 2880 validation minutes, of which 4 are feasible on
   the 14,386 usable pre-test rows), final fit, and corrected causal
   inference.
2. LSTM ``MS_25320``: the past-only refit is repeated locally with the
   unchanged CPU protocol (15 TPE trials, seed 42, CPU search space, 2 folds,
   25 epochs, patience 6, scaler on days 0-7).  ``train_lstm.FORCE_DEVICE_MODE = "cpu"`` keeps the CPU mode and
   search space even when MPS is visible.

All 20 services use ``experiments/data/service_timeseries_200_verified``.
For ``MS_25320`` this is the past-only series; for the other 19 services the
preflight stage confirms that timestamps and demand are identical to the
historical focused input ``experiments/data/service_timeseries``.

The environment recorded in every ``run_metadata.json`` is read from the
running process itself.  Canonical runs must execute on macOS (Darwin); the
script refuses otherwise unless ``--smoke`` is given, which writes to a
separate directory, marks the output ``canonical_eligible: false`` and is
intended only for code tests on synthetic data.

The stages are resumable: a service whose ``run_metadata.json`` verifies is
skipped; an interrupted service is rerun from scratch in ``<service>.partial``
and moved into place only after it completes.

Usage (from the paper directory, macOS)::

    PY=python3   # for example /opt/homebrew/bin/python3
    $PY audit/final_closure_local_runs.py --stage preflight
    $PY audit/final_closure_local_runs.py --stage xgb      # ~20 x 20 min on M1
    $PY audit/final_closure_local_runs.py --stage lstm     # MS_25320 only
    $PY audit/final_closure_local_runs.py --stage verify

``reproducibility/run_final_closure_compute.sh`` wraps these steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
RESEARCH = PAPER.parent
EXPERIMENTS = RESEARCH / "experiments"
sys.path.insert(0, str(EXPERIMENTS / "src"))

DEFAULT_OUT = PAPER / "audit" / "verified_results" / "final_closure_2026-09-28"
DEFAULT_INPUT_DIR = EXPERIMENTS / "data" / "service_timeseries_200_verified"
HISTORICAL_INPUT_DIR = EXPERIMENTS / "data" / "service_timeseries"
DEFAULT_SPLIT = EXPERIMENTS / "data" / "splits" / "split_definition.json"
SELECTION = EXPERIMENTS / "data" / "splits" / "selected_services_200.csv"
LSTM_SERVICES = ("MS_25320",)
PAST_ONLY_SERVICE = "MS_25320"
EXPECTED_TEST_ROWS = 4320
TEST_START = 864_000_000
TEST_END_EXCLUSIVE = 1_123_200_000
EXPECTED_XGB_FOLDS = [(2866, 2880), (5746, 2880), (8626, 2880), (11506, 2880)]
RUNNER_VERSION = "final-closure-2026-09-28"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(out: Path, message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    with open(out / "queue.log", "a") as handle:
        handle.write(line + "\n")


def focused_services() -> list[str]:
    selection = pd.read_csv(SELECTION)
    services = sorted(selection.loc[selection.is_focused_20.astype(bool), "service_id"])
    assert len(services) == 20, len(services)
    return services


def environment() -> dict:
    """Describe the interpreter that executes this process (never hand-written)."""
    import optuna
    env = {
        "sys_executable": sys.executable,
        "python": sys.version.split()[0],
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "mac_ver": platform.mac_ver()[0],
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "optuna": optuna.__version__,
        "thread_env": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "PYTORCH_ENABLE_MPS_FALLBACK") if os.environ.get(k) is not None},
    }
    try:
        import xgboost
        env["xgboost"] = xgboost.__version__
        try:
            info = xgboost.build_info()
            env["xgboost_build"] = {k: info.get(k) for k in ("USE_OPENMP", "USE_CUDA", "BUILTIN_PREFETCH_PRESENT") if k in info}
        except Exception:  # pragma: no cover - optional
            pass
    except ImportError:
        pass
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_num_threads"] = torch.get_num_threads()
        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        env["torch_mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except ImportError:
        pass
    return env


def require_macos(args: argparse.Namespace) -> None:
    if args.smoke:
        return
    if platform.system() != "Darwin":
        raise SystemExit(
            f"Refusing canonical run on {platform.system()} {platform.machine()} "
            f"({sys.executable}). The final-closure runs must execute on the author's Mac. "
            "Use --smoke only for code tests on synthetic data."
        )


def capture_studies():
    import optuna
    created = []
    original = optuna.create_study

    def wrapper(*a, **kw):
        study = original(*a, **kw)
        created.append(study)
        return study

    optuna.create_study = wrapper
    return created, original


def trials_frame(created) -> pd.DataFrame:
    rows = []
    for index, study in enumerate(created):
        for trial in study.trials:
            rows.append({"study": index, "number": trial.number, "value": trial.value,
                         "state": str(trial.state),
                         **{f"param_{k}": v for k, v in trial.params.items()}})
    return pd.DataFrame(rows)


def source_hashes() -> dict:
    return {
        "train_xgb.py": sha256(EXPERIMENTS / "src" / "train_xgb.py"),
        "train_lstm.py": sha256(EXPERIMENTS / "src" / "train_lstm.py"),
        "final_closure_local_runs.py": sha256(Path(__file__)),
        "test_xgb_alignment.py": sha256(EXPERIMENTS / "tests" / "test_xgb_alignment.py"),
    }


def preflight(args: argparse.Namespace) -> dict:
    require_macos(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    services = args.services or focused_services()
    inputs = {}
    for service in services:
        path = Path(args.input_dir) / f"{service}.parquet"
        frame = pd.read_parquet(path).sort_values("timestamp")
        record = {"input_path": os.path.relpath(path, RESEARCH), "input_sha256": sha256(path),
                  "rows": int(len(frame))}
        historical = HISTORICAL_INPUT_DIR / f"{service}.parquet"
        if not args.smoke and historical.exists():
            old = pd.read_parquet(historical).sort_values("timestamp")
            same_ts = bool(np.array_equal(old.timestamp.to_numpy(), frame.timestamp.to_numpy()))
            differing = int(np.sum(old.cpu_sum.to_numpy(float) != frame.cpu_sum.to_numpy(float))) if same_ts else -1
            record.update({"historical_input_sha256": sha256(historical),
                           "timestamps_identical_to_historical": same_ts,
                           "demand_minutes_differing_from_historical": differing})
            expected = 8 if service == PAST_ONLY_SERVICE else 0
            if not same_ts or differing != expected:
                raise SystemExit(f"{service}: input differs from the historical focused input "
                                 f"({differing} minutes, expected {expected})")
        inputs[service] = record
    report = {
        "runner_version": RUNNER_VERSION,
        "started": now(),
        "canonical_eligible": not args.smoke,
        "environment": environment(),
        "split_definition_sha256": sha256(Path(args.split_definition)),
        "source_sha256": source_hashes(),
        "inputs": inputs,
    }
    (out / "preflight.json").write_text(json.dumps(report, indent=2, default=str))
    log(out, f"PREFLIGHT ok services={len(services)} system={report['environment']['system']} "
             f"machine={report['environment']['machine']} python={report['environment']['python']}")
    return report


def run_one(model: str, service: str, args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    final = out_root / model / service
    if final.exists() and verify_run(model, service, args, quiet=True):
        log(out_root, f"SKIP {model} {service} (verified)")
        return
    if final.exists():
        raise SystemExit(f"{final} exists but does not verify; inspect or move it before rerunning")
    partial = out_root / model / f"{service}.partial"
    if partial.exists():
        stale = out_root / model / f"{service}.stale-{time.strftime('%Y%m%dT%H%M%S')}"
        partial.rename(stale)
        log(out_root, f"MOVED interrupted run to {stale.name}")
    partial.mkdir(parents=True)
    input_path = Path(args.input_dir) / f"{service}.parquet"
    split_path = Path(args.split_definition)
    split_definition = json.loads(split_path.read_text())

    log(out_root, f"START {model} {service}")
    created, original_create = capture_studies()
    started = now()
    t0 = time.perf_counter()
    try:
        if model == "xgb":
            import train_xgb as trainer
            n_trials = args.n_trials if args.smoke and args.n_trials else trainer.OPTUNA_N_TRIALS
            if not args.smoke:
                assert trainer.OPTUNA_N_TRIALS == 200 and trainer.CV_N_SPLITS == 5
            protocol = {
                "sampler": f"RandomSampler(seed={trainer.RANDOM_SEED})",
                "n_trials": n_trials,
                "cv": "expanding window, ts_cv_splits",
                "cv_configured_n_splits": trainer.CV_N_SPLITS,
                "cv_test_size": trainer.CV_TEST_SIZE,
                "early_stopping_rounds": trainer.EARLY_STOPPING_ROUNDS,
                "final_rounds": 500,
                "features": "14 lags, rolling 5/10, EMA 3/5/10/15, roc, lag diffs, minute-of-day sin/cos",
                "inference_feature_alignment": "target_minute (next_step_features)",
                "seed": trainer.RANDOM_SEED,
            }
            result = trainer.train_service(input_path, split_definition, 1, 1,
                                           n_optuna_trials=n_trials, use_tpe=False,
                                           save_models=True, models_dir=partial / "model")
            meta = json.loads((partial / "model" / f"{service}_meta.json").read_text())
            protocol["cv_effective_n_splits"] = meta["cv_effective_n_splits"]
            protocol["cv_folds"] = meta["cv_folds"]
        else:
            import train_lstm as trainer
            trainer.FORCE_DEVICE_MODE = "cpu"
            trainer.MODELS_DIR = partial / "model"
            protocol = {
                "device_mode": "cpu (FORCE_DEVICE_MODE)",
                "sampler": f"TPESampler(seed={trainer.RANDOM_SEED})",
                "n_trials": trainer.CPU_N_TRIALS,
                "lookbacks": trainer.CPU_LOOKBACKS,
                "units": trainer.CPU_UNITS,
                "layers": trainer.CPU_LAYERS,
                "batch_sizes": trainer.CPU_BATCH_SIZES,
                "max_epochs": trainer.CPU_MAX_EPOCHS,
                "patience": trainer.CPU_PATIENCE,
                "cv_folds": 2,
                "cv_test_size": trainer.CV_TEST_SIZE,
                "scaler": "MinMax(-1,1) fitted on train days 0-7",
                "seed": trainer.RANDOM_SEED,
            }
            if args.smoke and args.n_trials:
                trainer.CPU_N_TRIALS = args.n_trials
                protocol["n_trials"] = args.n_trials
            result = trainer.train_service(input_path, split_definition, 1, 1, save_models=True)
            if result.get("device_mode", "cpu") != "cpu":
                raise RuntimeError(f"LSTM ran in {result.get('device_mode')} mode")
    except BaseException:
        (partial / "error.txt").write_text(traceback.format_exc())
        log(out_root, f"FAIL {model} {service}; see {partial / 'error.txt'}")
        raise
    finally:
        import optuna
        optuna.create_study = original_create
    wall = time.perf_counter() - t0

    forecasts = result.pop("forecasts")
    timestamps = result.pop("timestamps").astype(np.int64)
    fc = pd.DataFrame({"msname": service, "model": model, "timestamp": timestamps, "forecast": forecasts})
    assert len(fc) == EXPECTED_TEST_ROWS and fc.timestamp.is_unique, len(fc)
    assert fc.timestamp.min() == TEST_START and fc.timestamp.max() == TEST_END_EXCLUSIVE - 60_000
    assert np.isfinite(fc.forecast).all()
    fc.to_parquet(partial / "forecasts.parquet", index=False)
    trials = trials_frame(created)
    trials.to_csv(partial / "trials.csv", index=False)
    pd.DataFrame([result]).to_csv(partial / "training_log_row.csv", index=False)
    artefacts = {str(p.relative_to(partial)): sha256(p) for p in sorted(partial.rglob("*")) if p.is_file()}
    metadata = {
        "runner_version": RUNNER_VERSION,
        "service": service,
        "model": model,
        "canonical_eligible": not args.smoke,
        "input_path": os.path.relpath(input_path, RESEARCH),
        "input_sha256": sha256(input_path),
        "split_definition_sha256": sha256(split_path),
        "source_sha256": source_hashes(),
        "protocol": protocol,
        "environment": environment(),
        "started": started,
        "finished": now(),
        "wall_seconds": wall,
        "stage_seconds": {"search": result.get("search_seconds"), "final_fit": result.get("train_seconds"),
                          "inference": result.get("inference_seconds_total")},
        "n_trials_recorded": int(len(trials)),
        "n_trials_complete": int((trials.state == "TrialState.COMPLETE").sum()) if len(trials) else 0,
        "best_cv_mae": result.get("best_mae_cv"),
        "test_metrics": {k: result.get(k) for k in ("mae", "rmse", "underprediction_rate",
                                                     "underprediction_mae", "underprediction_p95", "residual_p95")},
        "artefact_sha256": artefacts,
        "status": "completed",
    }
    (partial / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    partial.rename(final)
    log(out_root, f"END {model} {service} wall={wall:.0f}s mae={metadata['test_metrics']['mae']:.4f} "
                  f"cv={metadata['best_cv_mae']:.4f}")


# Verification (also imported by recompute_final_closure_downstream.py)
def verify_run(model: str, service: str, args: argparse.Namespace | None = None,
               quiet: bool = False, require_current_sources: bool = True) -> bool:
    out_root = Path(args.out_dir) if args else DEFAULT_OUT
    smoke = bool(args and args.smoke)
    run = out_root / model / service
    problems = []
    try:
        meta = json.loads((run / "run_metadata.json").read_text())
        if meta.get("status") != "completed":
            problems.append("status is not completed")
        if meta.get("service") != service or meta.get("model") != model:
            problems.append("service/model mismatch")
        if not smoke:
            if not meta.get("canonical_eligible"):
                problems.append("not canonical-eligible")
            if meta["environment"].get("system") != "Darwin":
                problems.append(f"environment system is {meta['environment'].get('system')}")
        input_path = RESEARCH / meta["input_path"]
        if sha256(input_path) != meta["input_sha256"]:
            problems.append("input hash changed")
        if not smoke and Path(meta["input_path"]).parent.name != "service_timeseries_200_verified":
            problems.append("unexpected input directory")
        split_path = Path(args.split_definition) if args else DEFAULT_SPLIT
        if sha256(split_path) != meta["split_definition_sha256"]:
            problems.append("split definition hash changed")
        if require_current_sources:
            current = source_hashes()
            key = f"train_{model}.py"
            if meta["source_sha256"].get(key) != current[key]:
                problems.append(f"{key} changed since the run")
        for name, digest in meta["artefact_sha256"].items():
            if sha256(run / name) != digest:
                problems.append(f"hash mismatch {name}")
        trials = pd.read_csv(run / "trials.csv")
        expected_trials = meta["protocol"]["n_trials"]
        if not smoke and expected_trials != (200 if model == "xgb" else 15):
            problems.append(f"protocol n_trials={expected_trials}")
        if len(trials) != expected_trials or (trials.state != "TrialState.COMPLETE").any():
            problems.append(f"trials: {len(trials)} recorded, expected {expected_trials} complete")
        if model == "xgb":
            folds = meta["protocol"]["cv_folds"]
            got = [(f["train_rows"], f["val_rows"]) for f in folds]
            if meta["protocol"]["cv_configured_n_splits"] != 5:
                problems.append("configured folds != 5")
            if not smoke and (got != EXPECTED_XGB_FOLDS or meta["protocol"]["cv_effective_n_splits"] != 4):
                problems.append(f"effective folds {got}")
            for f in folds:
                if not (f["train_start"] == 0 and f["train_end"] == f["val_start"] < f["val_end"]):
                    problems.append(f"fold order/overlap {f}")
            if meta["protocol"].get("inference_feature_alignment") != "target_minute (next_step_features)":
                problems.append("inference alignment tag missing")
        else:
            if not str(meta["protocol"].get("device_mode", "")).startswith("cpu"):
                problems.append("LSTM not in CPU mode")
        fc = pd.read_parquet(run / "forecasts.parquet")
        if len(fc) != EXPECTED_TEST_ROWS or not fc.timestamp.is_unique:
            problems.append("forecast rows")
        expected_ts = np.arange(TEST_START, TEST_END_EXCLUSIVE, 60_000, dtype=np.int64)
        if not np.array_equal(np.sort(fc.timestamp.to_numpy(np.int64)), expected_ts):
            problems.append("forecast timestamps differ from the held-out horizon")
        if not np.isfinite(fc.forecast).all() or (fc.forecast < 0).any():
            problems.append("non-finite or negative forecasts")
        if not ((fc.msname == service).all() and (fc.model == model).all()):
            problems.append("forecast labels")
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    if problems and not quiet:
        print(f"VERIFY FAIL {model} {service}: " + "; ".join(problems))
    return not problems


def verify_all(args: argparse.Namespace) -> int:
    services = args.services or focused_services()
    targets = [("xgb", s) for s in services] + [("lstm", s) for s in LSTM_SERVICES if s in services]
    failures = [t for t in targets if not verify_run(*t, args=args)]
    for model, service in targets:
        if (model, service) not in failures:
            print(f"VERIFY OK   {model} {service}")
    print(f"{len(targets) - len(failures)}/{len(targets)} runs verified")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", choices=("preflight", "xgb", "lstm", "verify"), required=True)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--split-definition", default=str(DEFAULT_SPLIT))
    parser.add_argument("--services", nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="non-canonical code test (synthetic data, any OS, separate --out-dir)")
    parser.add_argument("--n-trials", type=int, default=None, help="only with --smoke")
    args = parser.parse_args()
    if args.smoke and Path(args.out_dir).resolve() == DEFAULT_OUT.resolve():
        raise SystemExit("--smoke requires a separate --out-dir")
    if args.n_trials and not args.smoke:
        raise SystemExit("--n-trials is only allowed with --smoke")
    os.chdir(RESEARCH)
    if args.stage == "verify":
        return verify_all(args)
    if args.stage == "preflight":
        preflight(args)
        return 0
    require_macos(args)
    out = Path(args.out_dir)
    if not (out / "preflight.json").exists():
        raise SystemExit("run --stage preflight first")
    services = args.services or focused_services()
    if args.stage == "xgb":
        for service in services:
            run_one("xgb", service, args)
    else:
        for service in LSTM_SERVICES:
            if service in services:
                run_one("lstm", service, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
