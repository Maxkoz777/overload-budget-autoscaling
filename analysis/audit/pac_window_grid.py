#!/usr/bin/env python3
"""R7 (round-2 review): PAC window/eta grid, one-step ablation, pre-test window selection,
conservativeness, and conformal test martingales on three traces.

Protocol: verified_results/delay_pac_study/protocol.json, key "R7_round2_addendum".
Usage:  python3 audit/pac_window_grid.py [grid] [martingale] [summary]
"""
from __future__ import annotations

import bisect
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delay_and_reactive_grid as rev  # noqa: E402
import external_delay as xd  # noqa: E402
g, ext = rev.g, xd.ext
OUT = rev.OUT

WINDOWS = (240, 480, 960, 1440, 2880)
ETAS = (0.05, 0.10, 0.20)
BUDGETS = (0.01, 0.05)
TAUS = (0, 1, 5)
DATASETS = ("alibaba200", "huawei2023", "azure2019")


def pac_rank(W, delta, eta):
    for l in range(1, W + 1):
        if binom.sf(l - 1, W, 1 - delta) <= eta:
            return l
    return None


def admissible(W, delta, eta):
    r = pac_rank(W, delta, eta)
    return r is not None and r <= W - 1


# ---------------------------------------------------------------- data
def load(ds: str, phase: str) -> dict:
    """Arrays for one evaluation phase: phase 'test' (days 10-12 / test) or 'cal' (days 8-9)."""
    if ds == "alibaba200":
        g.DATA = rev.DATA
        _, S = g.load_services()
        if phase == "test":
            hist = [s["history"] for s in S]; ev = [s["test"] for s in S]
        else:
            hist = [s["train"] for s in S]; ev = [s["calibration"] for s in S]
        y = np.stack([np.r_[h.cpu_sum.to_numpy(float), e.cpu_sum.to_numpy(float)] for h, e in zip(hist, ev)])
        t0 = len(hist[0]); T = len(ev[0])
        init = np.array([max(int(np.ceil(float(h.replica_count.iloc[-1]))), 1) for h in hist], float)
        denom = np.array([g.observed_cost(e) for e in ev])
        return dict(uid=np.array([s["service_id"] for s in S]), y=y, t0=t0, T=T, mu=np.ones(len(S)), init=init,
                    warm=False, denom=denom)
    X, mu, units, split, _ = ext.load_family(ds, "primary")
    X = X.astype(np.float64); mu = mu.astype(float)
    t0, t1 = split["test"] if phase == "test" else split["cal"]
    dem = X[:, t0:t1]
    clair = np.maximum(np.ceil(dem / mu[:, None]), 1.0)
    denom = ext.C_RES * clair.sum(1) + ext.C_ACT * np.abs(np.diff(clair, axis=1)).sum(1) + ext.C_VIO * (dem > mu[:, None] * clair).sum(1)
    return dict(uid=units.unit_id.to_numpy(), y=X[:, :t1], t0=t0, T=t1 - t0, mu=mu, init=None, warm=True, denom=denom)


def margins(D, W, h, ranks, start):
    y, t0, T = D["y"], D["t0"], D["T"]
    pos = np.maximum(y[:, h:] - y[:, :-h], 0.0)
    out = {r: np.empty((y.shape[0], T - start)) for r in ranks}
    kth = sorted({r - 1 for r in ranks})
    for col, i in enumerate(range(start, T)):
        gi = t0 + i
        part = np.partition(pos[:, gi - h - W: gi - h], kth, axis=1)
        for r in ranks:
            out[r][:, col] = part[:, r - 1]
    return out


def evaluate(D, nominal_full, tau, maxtau):
    """nominal_full columns are decision indices -maxtau..T-1; margin-only (no offset) policy."""
    T = D["T"]; dem = D["y"][:, D["t0"]: D["t0"] + T]; mu = D["mu"]
    dec = nominal_full[:, maxtau - tau:]                 # decision indices -tau..T-1
    if D["warm"]:
        cap = dec[:, :T]
    else:
        cap = np.concatenate([np.repeat(D["init"][:, None], tau, axis=1), dec[:, tau:T]], axis=1) if tau else dec[:, :T]
    over = (dem > mu[:, None] * cap).sum(1)
    churn = np.abs(np.diff(cap, axis=1)).sum(1)
    total = cap.sum(1) + 0.05 * churn + 10.0 * over
    return over / T, total / D["denom"]


def run_grid(ds: str, phase: str, taus, scores, etas) -> pd.DataFrame:
    D = load(ds, phase)
    maxtau = max(taus)
    forecast = D["y"][:, D["t0"] - maxtau - 1: D["t0"] + D["T"] - 1]
    rows = []
    for W in WINDOWS:
        if D["t0"] - maxtau - 6 - W < 0:
            continue
        combos = [(d, e, pac_rank(W, d, e)) for d in BUDGETS for e in etas if admissible(W, d, e)]
        if not combos:
            continue
        ranks = tuple(sorted({c[2] for c in combos}))
        hs = sorted({1} | ({t + 1 for t in taus if t} if "h" in scores else set()))
        for h in hs:
            m = margins(D, W, h, ranks, start=-maxtau)
            for d, e, r in combos:
                nom = np.maximum(np.ceil((forecast + m[r]) / D["mu"][:, None]), 1.0)
                for tau in taus:
                    variants = []
                    if h == 1 and "1" in scores:
                        variants.append("one-step" if tau else "tau0")
                    if h == tau + 1 and tau and "h" in scores:
                        variants.append("horizon")
                    for v in variants:
                        frac, rel = evaluate(D, nom, tau, maxtau)
                        for k in range(len(D["uid"])):
                            rows.append((ds, phase, W, d, e, r, tau, v, D["uid"][k], frac[k], rel[k]))
            print(f"  {ds} {phase} W={W} h={h} done", flush=True)
    return pd.DataFrame(rows, columns=["dataset", "phase", "W", "budget", "eta", "rank", "tau", "score", "unit_id", "overload_fraction", "relative_cost"])


# ---------------------------------------------------------------- martingales
EPS = np.linspace(0.005, 0.995, 100)


def martingale_max(p: np.ndarray) -> float:
    """Running maximum of the simple-mixture power martingale (log10 not taken)."""
    lp = np.log(np.clip(p, 1e-300, 1.0))
    logS = np.cumsum(np.log(EPS)[:, None] + (EPS[:, None] - 1.0) * lp[None, :], axis=1)
    mx = logS.max(axis=0)
    mix = mx + np.log(np.exp(logS - mx).mean(axis=0))
    return float(np.exp(min(mix.max(), 700)))


def pvalues(a: np.ndarray, rng, block: int | None) -> np.ndarray:
    p = np.empty(len(a)); past: list[float] = []
    for t, x in enumerate(a):
        if block and t % block == 0:
            past = []
        lo = bisect.bisect_left(past, x); hi = bisect.bisect_right(past, x)
        n = len(past) + 1
        greater = len(past) - hi; equal = hi - lo + 1
        p[t] = (greater + rng.random() * equal) / n
        past.insert(hi, x)
    return p


def run_martingales() -> pd.DataFrame:
    rng = np.random.default_rng(20261002)
    rows = []
    for ds in DATASETS:
        D = load(ds, "test")
        y, t0, T = D["y"], D["t0"], D["T"]
        for h in (1, 2):
            sc = np.maximum(y[:, t0:t0 + T] - y[:, t0 - h:t0 + T - h], 0.0)
            for k in range(sc.shape[0]):
                a = sc[k]
                full = martingale_max(pvalues(a, rng, None))
                pb = pvalues(a, rng, 240)
                blocks = [martingale_max(pb[b:b + 240]) for b in range(0, T - 239, 240)]
                rows.append({"dataset": ds, "h": h, "unit_id": D["uid"][k], "full_max": full,
                             "full_reject_01": full >= 100, "blocks": len(blocks),
                             "block_reject_frac_01": float(np.mean(np.array(blocks) >= 100)),
                             "zero_score_frac": float((a == 0).mean())})
            print(f"  martingales {ds} h={h} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary
def summarise(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = grid.copy()
    grid["within"] = grid.overload_fraction <= grid.budget
    grid["util"] = grid.overload_fraction / grid.budget
    keys = ["dataset", "phase", "W", "budget", "eta", "rank", "tau", "score"]
    s = grid.groupby(keys).agg(n=("unit_id", "size"), compliant=("within", "sum"), mean_relative_cost=("relative_cost", "mean"),
                               mean_overload=("overload_fraction", "mean"), mean_util=("util", "mean"),
                               median_util=("util", "median")).reset_index()
    # pre-test window selection: eta=0.05, tau=0, one-step, calibration phase
    cal = s[(s.phase == "cal") & (s.eta == 0.05) & (s.tau == 0)]
    sel = []
    for (ds, d), grp in cal.groupby(["dataset", "budget"]):
        best = grp.sort_values(["compliant", "mean_relative_cost"], ascending=[False, True]).iloc[0]
        sel.append({"dataset": ds, "budget": d, "selected_W": int(best.W), "cal_compliant": int(best.compliant),
                    "cal_n": int(best.n), "cal_mean_cost": float(best.mean_relative_cost),
                    "candidates": ";".join(f"{int(r.W)}:{int(r.compliant)}/{r.mean_relative_cost:.3f}" for r in grp.itertuples())})
    return s, pd.DataFrame(sel)


def comparator_conservativeness() -> pd.DataFrame:
    rows = []
    r5 = pd.read_csv(OUT / "r5_delay_alibaba_per_service.csv.gz").rename(columns={"service_id": "unit_id"}).assign(dataset="alibaba200")
    r4 = pd.concat([pd.read_csv(OUT / f"r4_delay_{f}_per_unit.csv.gz") for f in ("huawei2023", "azure2019")])
    for df in (r5, r4):
        df = df[df.policy.isin(["B6-h", "reactive_cal_ext_strict", "B6"]) & (df.alpha == 1.0) & df.tau.isin(TAUS)].copy()
        df["util"] = df.overload_fraction / df.budget
        for k, grp in df.groupby(["dataset", "budget", "policy", "tau"]):
            rows.append({**dict(zip(["dataset", "budget", "policy", "tau"], k)), "n": len(grp),
                         "compliant": int((grp.overload_fraction <= grp.budget).sum()),
                         "mean_relative_cost": grp.relative_cost.mean(), "mean_overload": grp.overload_fraction.mean(),
                         "mean_util": grp.util.mean(), "median_util": grp.util.median()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    todo = set(sys.argv[1:]) or {"grid", "martingale", "summary"}
    if "grid" in todo:
        for ds in DATASETS:
            print("grid", ds, flush=True)
            test = run_grid(ds, "test", TAUS, {"1", "h"}, ETAS)
            cal = run_grid(ds, "cal", (0,), {"1"}, (0.05,))
            pd.concat([test, cal]).to_csv(OUT / f"r7_pac_grid_{ds}.csv.gz", index=False)
        # regression against R6 (eta=0.05, horizon-aligned / tau=0, W 480/240 and 1440)
        r6 = pd.concat([pd.read_csv(OUT / f"r6_pac_{ds}_per_unit.csv.gz") for ds in DATASETS])
        r6 = r6[r6.policy.isin(["B7-PAC", "B7-PAC-1440"])]
        mine = pd.concat([pd.read_csv(OUT / f"r7_pac_grid_{ds}.csv.gz") for ds in DATASETS])
        mine = mine[(mine.phase == "test") & (mine.eta == 0.05) & mine.score.isin(["tau0", "horizon"])]
        m = r6.merge(mine, on=["dataset", "unit_id", "budget", "W", "tau"], suffixes=("_r6", ""), validate="one_to_one")
        rev.check("R7 vs R6 rows", len(m), len(r6))
        rev.check("R7 vs R6 max diff", float((m.overload_fraction - m.overload_fraction_r6).abs().max() + (m.relative_cost - m.relative_cost_r6).abs().max()), 0.0, 1e-9)
    if "martingale" in todo:
        run_martingales().to_csv(OUT / "r7_martingales_per_unit.csv", index=False)
    if "summary" in todo:
        grid = pd.concat([pd.read_csv(OUT / f"r7_pac_grid_{ds}.csv.gz") for ds in DATASETS])
        s, sel = summarise(grid)
        s.to_csv(OUT / "r7_pac_grid_summary.csv", index=False)
        sel.to_csv(OUT / "r7_window_selection.csv", index=False)
        comparator_conservativeness().to_csv(OUT / "r7_comparator_conservativeness.csv", index=False)
        if (OUT / "r7_martingales_per_unit.csv").exists():
            mg = pd.read_csv(OUT / "r7_martingales_per_unit.csv")
            mg.groupby(["dataset", "h"]).agg(units=("unit_id", "size"), full_reject_rate=("full_reject_01", "mean"),
                                             mean_block_reject=("block_reject_frac_01", "mean"),
                                             median_block_reject=("block_reject_frac_01", "median")).reset_index().to_csv(OUT / "r7_martingales_summary.csv", index=False)
        print(sel.to_string(index=False))
    (OUT / "methodology_r7.json").write_text(json.dumps({"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}, indent=2) + "\n")
