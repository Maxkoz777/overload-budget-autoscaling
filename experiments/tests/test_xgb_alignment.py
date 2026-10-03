#!/usr/bin/env python3
"""Behavioural tests for XGBoost feature alignment.

The training row for target minute t (``build_features``/``prepare_xy``)
uses demand up to t-1 and the calendar of t.  The online feature vector used
by ``walk_forward_predict`` must be exactly that row.  These tests need only
NumPy and pandas (XGBoost is replaced by explicit prediction functions).

    python3 experiments/tests/test_xgb_alignment.py      # from the repository root
    python3 -m pytest experiments/tests                  # if pytest exists
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_xgb as tx  # noqa: E402

COLS = list(tx.build_features(np.arange(30.0), np.arange(30) * 60_000).columns)
LAG1 = COLS.index("lag_1")
SIN, COS = COLS.index("sin_minute"), COLS.index("cos_minute")


def _ts(n: int, start_min: int = 0) -> np.ndarray:
    return (start_min + np.arange(n, dtype=np.int64)) * 60_000


def test_lag1_oracle() -> None:
    history = np.arange(1.0, 21.0)
    test = np.array([21.0, 22.0, 23.0])
    ts = _ts(23)
    out = tx.walk_forward_predict(None, history, ts[:20], test, ts[20:],
                                  predict_fn=lambda row: float(row[LAG1]))
    assert out.tolist() == [20.0, 21.0, 22.0], out


def test_target_calendar() -> None:
    history = np.arange(1.0, 21.0)
    ts = _ts(21, start_min=1430)          # target minute crosses midnight
    row = tx.next_step_features(history, ts[:20], int(ts[20]))
    minute = (ts[20] // 60_000) % tx.SEASONAL_PERIOD
    assert np.isclose(row[SIN], np.sin(2 * np.pi * minute / tx.SEASONAL_PERIOD))
    assert np.isclose(row[COS], np.cos(2 * np.pi * minute / tx.SEASONAL_PERIOD))


def test_online_row_equals_training_row() -> None:
    rng = np.random.default_rng(0)
    series = rng.gamma(2.0, 5.0, size=200)
    ts = _ts(200, start_min=777)
    train_rows = tx.build_features(series, ts).to_numpy(dtype=float)
    for t in (15, 16, 57, 120, 199):
        online = tx.next_step_features(series[:t], ts[:t], int(ts[t]))
        assert np.allclose(online, train_rows[t], rtol=0, atol=1e-12, equal_nan=True), t


def test_no_dependence_on_current_or_future_demand() -> None:
    rng = np.random.default_rng(1)
    history = rng.gamma(2.0, 5.0, size=60)
    test = rng.gamma(2.0, 5.0, size=10)
    ts = _ts(70)
    fn = lambda row: float(row @ np.linspace(0.1, 1.0, len(row)))  # noqa: E731
    base = tx.walk_forward_predict(None, history, ts[:60], test, ts[60:], predict_fn=fn)
    for k in range(10):
        perturbed = test.copy()
        perturbed[k:] += 1_000.0              # change demand at and after minute k
        out = tx.walk_forward_predict(None, history, ts[:60], perturbed, ts[60:], predict_fn=fn)
        assert np.array_equal(out[: k + 1], base[: k + 1]), k   # forecasts up to k unchanged
        if k < 9:
            assert not np.array_equal(out[k + 1:], base[k + 1:]), k


def test_first_and_last_test_minute() -> None:
    history = np.arange(1.0, 101.0)
    test = np.arange(101.0, 151.0)
    ts = _ts(150)
    seen = []

    def fn(row):
        seen.append(row.copy())
        return float(row[LAG1])

    out = tx.walk_forward_predict(None, history, ts[:100], test, ts[100:], predict_fn=fn)
    assert len(out) == len(test)
    assert out[0] == history[-1]              # first test minute uses the last history value
    assert out[-1] == test[-2]                # last test minute uses the previous test value
    full = tx.build_features(np.r_[history, test], ts).to_numpy(dtype=float)
    assert np.allclose(seen[0], full[100]) and np.allclose(seen[-1], full[149])


def test_effective_folds_focused_history() -> None:
    folds = tx.effective_cv_folds(14_386)
    assert tx.CV_N_SPLITS == 5 and len(folds) == 4
    assert [f["train_rows"] for f in folds] == [2866, 5746, 8626, 11506]
    assert all(f["val_rows"] == 2880 for f in folds)
    for f in folds:
        assert f["train_start"] == 0 and f["train_end"] == f["val_start"]   # no overlap, ordered
    assert folds[-1]["val_end"] == 14_386


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(tests)}/{len(tests)} tests passed")
