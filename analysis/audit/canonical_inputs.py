"""Canonical reading of capacity units.

The external traces store each unit's capacity ``mu`` in ``units.csv``. Two details
decide integer capacity decisions at exact ties, where a demand or protected demand
equals an integer multiple of ``mu`` in exact arithmetic (for example 85 requests with
``mu = 85/3``):

1. Parsing. ``pandas.read_csv`` may parse a 17-digit decimal to a neighbouring float
   depending on the pandas version and ``float_precision``. We read with
   ``float_precision="round_trip"``, which returns exactly the float that was written.
   (A parse one unit in the last place below 85/3, 28.33333333333333, makes 85/mu
   evaluate to 3.0000000000000004.)
2. Ties. Even at full precision, floating-point rounding of ``mu`` resolves some exact
   ties arbitrarily: with ``mu = 13/3`` stored as 4.333333333333333, 65/mu evaluates to
   15.000000000000002, so the ceiling gives 16 replicas instead of 15; with ``mu = 49/3``,
   ``245 > 15*mu`` counts a spurious overload. We therefore apply a documented relative
   boundary tolerance: every ``mu`` is scaled by ``1 + 1e-12`` on reading, so exact ties
   resolve as in exact arithmetic, consistently for every policy, the clairvoyant cost
   denominator, and the overload count. Like any tolerance, it would also treat a strict
   inequality closer than a relative 1e-12 to the boundary as a tie; demands here are
   integer counts and ``mu`` is a ratio of counts, so such near-ties are not expected.
   ``test_canonical_inputs.py`` pins down this behaviour.

The rule acts only on ``units.csv`` files and only on the capacity columns, so the frozen
analysis scripts run unchanged: ``run_canonical.py`` installs it and then executes them.
Alibaba uses ``mu = 1`` and is unaffected.
"""
from __future__ import annotations

TIE_REL = 1e-12
MU_COLUMNS = ("mu_p50", "mu_p90", "mu_K5", "mu_K10", "mu_K20")


def install() -> None:
    import pandas as pd

    if getattr(pd.read_csv, "_canonical_units", False):
        return
    original = pd.read_csv

    def read_csv(path, *args, **kwargs):
        if str(path).endswith("units.csv"):
            kwargs.setdefault("float_precision", "round_trip")
            frame = original(path, *args, **kwargs)
            for col in MU_COLUMNS:
                if col in frame.columns:
                    frame[col] = frame[col].astype(float) * (1.0 + TIE_REL)
            return frame
        return original(path, *args, **kwargs)

    read_csv._canonical_units = True
    pd.read_csv = read_csv
