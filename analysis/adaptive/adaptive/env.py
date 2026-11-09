"""Closed-loop actuation queue shared by all controllers.

Decisions are arrays (U, T): column i is the decision issued at index j = i - tau
(data observed through test minute j - 1) and applied to test minute i. This is the
queue of pac_window_grid.evaluate: for warm traces every test minute uses a decision;
for Alibaba (warm = False) the first tau minutes keep the initial observed capacity.
"""
from __future__ import annotations

import numpy as np

C_ACT, C_VIO = 0.05, 10.0


def applied(D: dict, dec: np.ndarray, tau: int, cap_bound: bool) -> np.ndarray:
    cap = dec.astype(float).copy()
    if not D["warm"] and tau:
        cap[:, :tau] = D["init"][:, None]
    if cap_bound:
        cap = np.minimum(cap, D["M"][:, None])
    return cap


def outcome(D: dict, cap: np.ndarray) -> dict:
    T = D["T"]
    dem = D["y"][:, D["t0"]: D["t0"] + T]
    over = dem > D["mu"][:, None] * cap
    churn = np.abs(np.diff(cap, axis=1)).sum(1)
    res = cap.sum(1)
    total = res + C_ACT * churn + C_VIO * over.sum(1)
    return dict(over=over, overload_fraction=over.mean(1), resource=res / D["denom"],
                churn=C_ACT * churn / D["denom"], penalty=C_VIO * over.sum(1) / D["denom"],
                relative_cost=total / D["denom"])
