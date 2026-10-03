#!/usr/bin/env python3
"""Refit one focused-tier learned forecaster on an explicitly chosen input series.

Purpose
-------
The historical focused-20 input for ``MS_25320`` contained eight pre-test
(days 8--9) CPU-demand minutes produced by an adjacent-value fill.  The
past-only series in ``experiments/data/service_timeseries_200_verified`` fixes
those minutes.  This runner repeats the *complete* focused pre-test pipeline of
the original training scripts for one service and one model:

1. feature / scaler preparation on the chosen history,
2. the original focused hyperparameter search on pre-test folds,
3. the original final fit on pre-test history,
4. causal one-step inference on the unchanged held-out horizon.

It calls ``train_service`` from ``experiments/src/train_xgb.py`` or
``experiments/src/train_lstm.py`` unchanged, with the focused protocol fixed
explicitly (XGBoost: RandomSampler(seed=42), 200 trials, 5 configured folds
-- 4 feasible on 14,386 rows; LSTM: CPU
mode, TPESampler(seed=42), 15 trials, 2 folds, CPU search space).  Outputs are
isolated under ``--out-dir``; nothing is written to ``experiments/models`` or
``experiments/results``.

Usage (from the paper directory)::

    python3 audit/refit_focused_causal_fill.py --model xgb --label pastonly \
        --input ../experiments/data/service_timeseries_200_verified/MS_25320.parquet
    python3 audit/refit_focused_causal_fill.py --model lstm --label pastonly \
        --input ../experiments/data/service_timeseries_200_verified/MS_25320.parquet

A control run with ``--label control_oldfill --input
../experiments/data/service_timeseries/MS_25320.parquet`` repeats the same
procedure on the historical input in the current environment, so that the
imputation effect can be compared within one environment.

Historical runs of this script used the pre-correction XGBoost feature
alignment.  Canonical focused forecasts come from
``audit/final_closure_local_runs.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
EXPERIMENTS = PAPER.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS / "src"))

TEST_START = 864_000_000
TEST_END_EXCLUSIVE = 1_123_200_000
EXPECTED_TEST_ROWS = 4320


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict:
    import optuna
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "optuna": optuna.__version__,
    }
    try:
        import xgboost
        out["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    try:
        import torch
        out["torch"] = torch.__version__
        out["torch_num_threads"] = torch.get_num_threads()
        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        out["torch_mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        pass
    return out


def capture_studies():
    """Keep references to Optuna studies created inside train_service."""
    import optuna
    created = []
    original = optuna.create_study

    def wrapper(*args, **kwargs):
        study = original(*args, **kwargs)
        created.append(study)
        return study

    optuna.create_study = wrapper
    return created, original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=("xgb", "lstm"), required=True)
    parser.add_argument("--input", required=True, help="per-service parquet")
    parser.add_argument("--label", required=True, help="e.g. pastonly or control_oldfill")
    parser.add_argument(
        "--out-dir",
        default=str(PAPER / "audit/verified_results/focused_causal_refit"),
    )
    parser.add_argument(
        "--split-definition",
        default=str(EXPERIMENTS / "data/splits/split_definition.json"),
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    split_path = Path(args.split_definition).resolve()
    out = Path(args.out_dir) / f"{args.model}_{args.label}"
    out.mkdir(parents=True, exist_ok=True)
    split_definition = json.loads(split_path.read_text())
    frame = pd.read_parquet(input_path)
    service = str(frame["msname"].iloc[0])

    created, original_create = capture_studies()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.perf_counter()
    if args.model == "xgb":
        import train_xgb as trainer
        protocol = {
            "sampler": "RandomSampler(seed=42)",
            "n_trials": trainer.OPTUNA_N_TRIALS,
            "cv_n_splits": trainer.CV_N_SPLITS,
            "cv_test_size": trainer.CV_TEST_SIZE,
            "early_stopping_rounds": trainer.EARLY_STOPPING_ROUNDS,
            "final_rounds": "max(500, 200) as in train_final_model",
            "features": "14 lags, rolling 5/10, EMA 3/5/10/15, roc, lag diffs, minute-of-day sin/cos",
            "seed": trainer.RANDOM_SEED,
        }
        assert trainer.OPTUNA_N_TRIALS == 200 and trainer.CV_N_SPLITS == 5
        result = trainer.train_service(
            input_path, split_definition, 1, 1,
            n_optuna_trials=trainer.OPTUNA_N_TRIALS,
            use_tpe=False,
            save_models=True,
            models_dir=out / "model",
        )
    else:
        import torch
        import train_lstm as trainer
        if torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise SystemExit(
                "Accelerator visible: the focused MS_25320 protocol is CPU mode; "
                "refusing to let device selection change the search space."
            )
        trainer.MODELS_DIR = out / "model"
        protocol = {
            "device_mode": "cpu",
            "sampler": "TPESampler(seed=42)",
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
        result = trainer.train_service(input_path, split_definition, 1, 1, save_models=True)
    wall = time.perf_counter() - t0
    import optuna
    optuna.create_study = original_create

    forecasts = result.pop("forecasts")
    timestamps = result.pop("timestamps").astype(np.int64)
    fc = pd.DataFrame({
        "msname": service,
        "model": args.model,
        "timestamp": timestamps,
        "forecast": forecasts,
    })
    # Integrity checks on the held-out horizon.
    assert len(fc) == EXPECTED_TEST_ROWS, len(fc)
    assert fc.timestamp.is_unique
    assert fc.timestamp.min() == TEST_START
    assert fc.timestamp.max() == TEST_END_EXCLUSIVE - 60_000
    assert np.isfinite(fc.forecast).all()
    fc_path = out / "forecasts.parquet"
    fc.to_parquet(fc_path, index=False)

    trials = []
    for study_idx, study in enumerate(created):
        for trial in study.trials:
            trials.append({
                "study": study_idx,
                "number": trial.number,
                "value": trial.value,
                "state": str(trial.state),
                **{f"param_{k}": v for k, v in trial.params.items()},
            })
    pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
    pd.DataFrame([result]).to_csv(out / "training_log_row.csv", index=False)

    artefacts = {
        str(p.relative_to(out)): sha256(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "run_metadata.json"
    }
    metadata = {
        "service": service,
        "model": args.model,
        "label": args.label,
        "input_path": str(input_path.relative_to(PAPER.parent)),
        "input_sha256": sha256(input_path),
        "split_definition_sha256": sha256(split_path),
        "trainer_source_sha256": sha256(EXPERIMENTS / "src" / f"train_{args.model}.py"),
        "runner_source_sha256": sha256(Path(__file__)),
        "protocol": protocol,
        "environment": versions(),
        "started": started,
        "wall_seconds": wall,
        "stage_seconds": {
            "search": result.get("search_seconds"),
            "final_fit": result.get("train_seconds"),
            "inference": result.get("inference_seconds_total"),
        },
        "n_trials_recorded": len(trials),
        "best_cv_mae": result.get("best_mae_cv"),
        "test_metrics": {
            key: result.get(key)
            for key in ("mae", "rmse", "underprediction_rate", "underprediction_mae",
                        "underprediction_p95", "residual_p95")
        },
        "artefact_sha256": artefacts,
        "status": "completed",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(json.dumps({k: metadata[k] for k in ("service", "model", "label", "best_cv_mae",
                                               "test_metrics", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
