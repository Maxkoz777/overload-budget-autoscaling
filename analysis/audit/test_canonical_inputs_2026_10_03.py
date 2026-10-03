#!/usr/bin/env python3
"""Unit tests for the canonical capacity-unit rule (canonical_inputs_2026_10_03.py).

The rule is a documented relative boundary tolerance, not exact rational arithmetic:
mu is read at full precision and scaled by 1 + 1e-12. These tests pin down its behaviour
at an exact tie, inside the tolerance, and clearly outside it, and check that the ceiling,
the overload count, and the clairvoyant denominator use mu consistently.

Usage:  python3 audit/test_canonical_inputs_2026_10_03.py      (or: pytest audit/...)
"""
from __future__ import annotations

import math
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

ORIGINAL_READ_CSV = pd.read_csv
from canonical_inputs_2026_10_03 import TIE_REL, install  # noqa: E402

install()


def _mu_from_units_csv(value: float) -> tuple[float, float]:
    """Write one unit with the given mu and read it back without and with the rule."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "units.csv"
        pd.DataFrame({"unit_id": ["U1"], "mu_p50": [value]}).to_csv(path, index=False)
        raw = float(ORIGINAL_READ_CSV(path, float_precision="round_trip").mu_p50.iloc[0])
        canonical = float(pd.read_csv(path).mu_p50.iloc[0])
    return raw, canonical


def capacity(demand: float, mu: float) -> int:
    return max(1, math.ceil(demand / mu))


def overloaded(demand: float, k: int, mu: float) -> bool:
    return demand > k * mu


def test_full_precision_parsing():
    raw, _ = _mu_from_units_csv(85 / 3)
    assert raw == 85 / 3                                   # round-trip parsing returns the stored float
    assert 85 / float("28.33333333333333") > 3             # a one-ulp-off parse breaks the tie at 85


def test_exact_tie_resolves_as_in_exact_arithmetic():
    raw, mu = _mu_from_units_csv(13 / 3)
    assert raw == 13 / 3
    assert capacity(65, raw) == 16                         # float artefact: 65 / 4.333333333333333 > 15
    assert capacity(65, mu) == 15                          # exact arithmetic: 65 / (13/3) = 15
    raw, mu = _mu_from_units_csv(49 / 3)
    assert overloaded(245, 15, raw)                        # spurious overload without the rule
    assert not overloaded(245, 15, mu)


def test_value_inside_tolerance_is_treated_as_tie():
    _, mu = _mu_from_units_csv(2.0)
    demand = 6.0 * (1 + 1e-13)                             # strictly above 3 units, within 1e-12
    assert Fraction(demand) / Fraction(2) > 3
    assert capacity(demand, mu) == 3                       # documented behaviour of a tolerance


def test_value_outside_tolerance_is_unchanged():
    _, mu = _mu_from_units_csv(2.0)
    demand = 6.0 * (1 + 1e-9)
    assert capacity(demand, mu) == 4
    assert overloaded(demand, 3, mu)
    assert capacity(5.999, mu) == 3 and not overloaded(5.999, 3, mu)


def test_ceiling_overload_and_denominator_are_consistent():
    rng = np.random.default_rng(20261003)
    for _ in range(2000):
        p, q = int(rng.integers(1, 5000)), int(rng.integers(1, 12))
        _, mu = _mu_from_units_csv(p / q)
        d = float(rng.integers(0, 50_000))
        k = capacity(d, mu)
        assert not overloaded(d, k, mu)                    # clairvoyant capacity never overloads
        if k > 1:
            assert overloaded(d, k - 1, mu)                # and is the smallest such capacity
        exact = max(1, math.ceil(Fraction(int(d)) / Fraction(p, q)))
        assert k == exact                                  # equals exact rational arithmetic here


def test_rule_only_touches_units_csv_capacity_columns():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "other.csv"
        pd.DataFrame({"mu_p50": [2.0]}).to_csv(path, index=False)
        assert float(pd.read_csv(path).mu_p50.iloc[0]) == 2.0
    assert TIE_REL == 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[OK] {t.__name__}")
    print(f"{len(tests)} tests passed")
