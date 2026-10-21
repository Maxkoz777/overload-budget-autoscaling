"""Decision rules of the Kubernetes testbed, one decision per interval.

PACH and ReactiveB1 reproduce the replay rules of the article step by step: after the demand of
interval t is observed, decide() returns the replica count requested for the next interval.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


def pac_rank(W: int, delta: float, eta: float) -> int | None:
    """Smallest l <= W with P(Bin(W, 1 - delta) >= l) <= eta, or None if no such l exists."""
    p = 1.0 - delta
    logs = [math.lgamma(W + 1) - math.lgamma(k + 1) - math.lgamma(W - k + 1) + k * math.log(p) + (W - k) * math.log1p(-p)
            for k in range(W + 1)]
    tail = 0.0
    sf = [0.0] * (W + 2)
    for k in range(W, -1, -1):
        tail += math.exp(logs[k])
        sf[k] = tail
    for l in range(1, W + 1):
        if sf[l] <= eta:
            return l
    return None


def conformal_rank(W: int, delta: float) -> int:
    """Standard split-conformal rank min(W, ceil((W + 1)(1 - delta)))."""
    return min(W, math.ceil((W + 1) * (1 - delta)))


class PACH:
    """PAC-h: persistence forecast, horizon-aligned (tau + 1)-step scores, PAC rank, no offset.

    rank="conformal" replaces the PAC rank by the standard conformal rank (matched-rank ablation).
    """

    def __init__(self, W: int, delta: float, eta: float, tau: int, mu: float, history: list[float], rank: str = "pac"):
        self.W, self.h, self.mu = W, tau + 1, mu
        self.rank = pac_rank(W, delta, eta) if rank == "pac" else conformal_rank(W, delta)
        if self.rank is None or self.rank > W - 1:
            raise ValueError(f"no unsaturated rank for W={W}, delta={delta}, eta={eta}, rank={rank}")
        if len(history) < W + self.h:
            raise ValueError("history shorter than W + tau + 1")
        self.y = deque(history[-(W + self.h):], maxlen=W + self.h)

    def margin(self) -> float:
        y = np.fromiter(self.y, float, len(self.y))
        scores = np.maximum(y[self.h:] - y[:-self.h], 0.0)
        return float(np.partition(scores, self.rank - 1)[self.rank - 1])

    def decide(self, demand: float) -> int:
        self.y.append(float(demand))
        return max(1, math.ceil((self.y[-1] + self.margin()) / self.mu))

    def initial(self) -> int:
        return max(1, math.ceil((self.y[-1] + self.margin()) / self.mu))


class ReactiveB1:
    """B1: k = max(1, ceil(d_prev / (u * mu))) whenever the cooldown has elapsed since the last change."""

    def __init__(self, u: float, cooldown: int, mu: float, last_demand: float):
        self.umu, self.c = u * mu, cooldown
        self.prev = float(last_demand)
        self.cur = max(1.0, math.ceil(last_demand / (0.8 * mu)))
        self.last = -cooldown
        self.s = 0

    def _step(self) -> int:
        desired = max(math.ceil(self.prev / self.umu), 1.0)
        if self.s - self.last >= self.c and desired != self.cur:
            self.cur, self.last = desired, self.s
        return int(self.cur)

    def initial(self) -> int:
        return self._step()

    def decide(self, demand: float) -> int:
        self.prev = float(demand)
        self.s += 1
        return self._step()
