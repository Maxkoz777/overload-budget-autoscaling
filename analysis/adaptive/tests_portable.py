"""Checks run before the v1 test replay. Exit code != 0 on any failure."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive.data import load, r7, PAC_W
from adaptive.env import applied, outcome
from adaptive import policies as P

FAIL = []


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    if not cond:
        FAIL.append(name)


def synth(y, t0, mu=1.0, warm=True, init=1.0):
    y = np.atleast_2d(np.asarray(y, float)); U = y.shape[0]; T = y.shape[1] - t0
    D = dict(y=y, t0=t0, T=T, mu=np.full(U, mu), init=np.full(U, init), warm=warm, denom=np.ones(U),
             uid=np.arange(U), name="synth")
    D["M"] = np.maximum(np.ceil(2 * y[:, :t0].max(1) / D["mu"]), 1.0)
    return D


def aci_oracle(y, t0, T, mu, M, W, tau, gamma, ast):
    """Scalar reference implementation written independently of policies.aci."""
    h = tau + 1
    alpha = {}
    out_dec, out_err = [], []
    errs = {}
    for i in range(T):
        j = i - tau
        lane = j % h
        if lane not in alpha:
            alpha[lane] = ast
        if (j - h) in errs:
            alpha[lane] = alpha[lane] + gamma * (ast - errs[j - h])
        al = alpha[lane]
        a = t0 + j
        scores = sorted(y[t] - y[t - h] for t in range(a - W, a))
        fc = y[a - 1]
        rank = int(np.ceil((W + 1) * (1 - al)))
        if al <= 0 or rank > W:
            dec, e = M, 0
        elif al >= 1:
            dec, e = 1.0, 1
        else:
            qv = scores[max(rank, 1) - 1]
            dec = max(1.0, float(np.ceil((fc + qv) / mu)))
            e = int(y[t0 + i] > fc + qv)
        errs[j] = e
        out_dec.append(dec); out_err.append(e)
    return np.array(out_dec), np.array(out_err)


def main():
    rng = np.random.default_rng(20261109)
    # 1. PAC-h regression against the frozen evaluator and published counts
    want = {"alibaba200": 200, "huawei2023": 64, "azure2019": 2829}
    cache = {}
    for ds in ("alibaba200", "huawei2023", "azure2019"):
        D = load(ds, "test"); cache[ds] = D
        W = PAC_W[(ds, 0.01)]
        for tau in (0, 1, 5):
            dec = P.pac_h(D, W, 0.01, tau)
            o = outcome(D, applied(D, dec, tau, cap_bound=False))
            r = r7.pac_rank(W, 0.01, 0.05)
            fc = D["y"][:, D["t0"] - tau - 1: D["t0"] + D["T"] - 1]
            m = r7.margins(D, W, tau + 1, (r,), start=-tau)[r]
            nom = np.maximum(np.ceil((fc + m) / D["mu"][:, None]), 1.0)
            f0, c0 = r7.evaluate(D, nom, tau, tau)
            check(f"PAC-h v1 == frozen evaluator {ds} tau={tau}",
                  np.array_equal(o["overload_fraction"], f0) and np.allclose(o["relative_cost"], c0, rtol=0, atol=1e-12))
            if tau == 1:
                n = int((o["overload_fraction"] <= 0.01 + 1e-12).sum())
                check(f"PAC-h published count {ds}", n == want[ds], f"{n}")
    # 2. queue alignment and initial prefix
    D = synth(np.arange(40.0), 20, warm=False, init=7.0)
    dec = np.tile(np.arange(20.0), (1, 1))
    for tau in (0, 1, 5):
        cap = applied(D, dec, tau, cap_bound=False)
        ok = np.array_equal(cap[0, tau:], dec[0, tau:]) and np.all(cap[0, :tau] == 7.0)
        check(f"queue alignment non-warm tau={tau}", ok)
    Dw = synth(np.arange(40.0), 20, warm=True)
    check("queue alignment warm", np.array_equal(applied(Dw, dec, 5, False), dec))
    # 3. scalar ACI oracle on real Huawei units
    Dh = load("huawei2023", "cal")
    sub = {k: (v[:3] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(Dh["uid"]) else v) for k, v in Dh.items()}
    sub["T"] = 600
    for tau in (0, 1, 5):
        for gamma in (0.005, 0.2):
            ast = P.alpha_star(0.05, gamma, tau, 600) if gamma > 0.1 else 0.05
            R = P.aci(sub, 240, tau, [(gamma, ast)])
            ok = True
            for u in range(3):
                d0, e0 = aci_oracle(sub["y"][u], sub["t0"], 600, sub["mu"][u], sub["M"][u], 240, tau, gamma, ast)
                ok &= np.array_equal(d0, R["dec"][0, u]) and np.array_equal(e0, R["err"][0, u].astype(int))
            check(f"ACI vectorised == scalar oracle tau={tau} gamma={gamma}", ok)
    # 4. ACI lane bound and branches on calibration data of every trace
    for ds in ("alibaba200", "huawei2023", "azure2019"):
        Dc = load(ds, "cal"); T = Dc["T"]
        for tau in (0, 1, 5):
            tr = [(g, P.alpha_star(0.05, g, tau, T)) for g in (0.05, 0.2, 0.5)]
            tr = [t for t in tr if t[1] > 0]
            R = P.aci(Dc, PAC_W[(ds, 0.05)], tau, tr)
            ok = True; worst = 0.0
            for k, (g, ast) in enumerate(tr):
                for ln in range(tau + 1):
                    sel = R["lane"] == ln
                    s = R["err"][k][:, sel].sum(1); n = sel.sum()
                    gap = np.abs(s - n * ast).max(); bound = (max(ast, 1 - ast) + g) / g
                    ok &= gap <= bound + 1e-9; worst = max(worst, gap / bound)
                ok &= bool((R["err"][k].mean(1) <= 0.05 + 1e-12).all())
                ok &= bool((R["dec"][k][R["inf"][k]] == np.broadcast_to(Dc["M"][:, None], R["dec"][k].shape)[R["inf"][k]]).all())
                ok &= bool((R["dec"][k][R["emp"][k]] == 1.0).all()) and bool(R["err"][k][R["emp"][k]].all())
                ok &= not bool(R["err"][k][R["inf"][k]].any())
            check(f"ACI-G lane bound, err<=delta, branch conventions {ds} tau={tau}", ok, f"max gap/bound {worst:.3f}")
            cap = applied(Dc, R["dec"][0], tau, cap_bound=True)
            o = outcome(Dc, cap)
            exc = R["inf"][0] | (R["nominal"][0] > Dc["M"][:, None])
            if not Dc["warm"]:
                exc[:, :tau] = True
            viol = o["over"] & ~exc & ~R["err"][0]
            check(f"ACI overload implies formal error outside exceptions {ds} tau={tau}", not viol.any(), f"{int(viol.sum())}")
    # 5. no-lookahead: perturb demand from test minute k on; decisions with data before k must not change
    D = load("huawei2023", "cal")
    sub = {k: (v[:4] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(D["uid"]) else v) for k, v in D.items()}
    for tau in (0, 1, 5):
        k = 700
        y2 = sub["y"].copy(); y2[:, sub["t0"] + k:] *= rng.uniform(0, 5, size=y2[:, sub["t0"] + k:].shape)
        s2 = dict(sub); s2["y"] = y2
        last = k + tau
        a1 = P.aci(sub, 240, tau, [(0.2, P.alpha_star(0.05, 0.2, tau, sub["T"]))])["dec"][0][:, :last + 1]
        a2 = P.aci(s2, 240, tau, [(0.2, P.alpha_star(0.05, 0.2, tau, sub["T"]))])["dec"][0][:, :last + 1]
        b1 = P.bacc(sub, 0.05, tau)["dec"][:, :last + 1]; b2 = P.bacc(s2, 0.05, tau)["dec"][:, :last + 1]
        p1 = P.pac_h(sub, 1440, 0.05, tau)[:, :last + 1]; p2 = P.pac_h(s2, 1440, 0.05, tau)[:, :last + 1]
        check(f"no lookahead ACI/BACC/PAC-h tau={tau}", np.array_equal(a1, a2) and np.array_equal(b1, b2) and np.array_equal(p1, p2))
        a3 = P.aci(s2, 240, tau, [(0.2, P.alpha_star(0.05, 0.2, tau, sub["T"]))])["dec"][0][:, last + 1: last + 50]
        b3 = P.bacc(s2, 0.05, tau)["dec"][:, last + 1: last + 400]
        check(f"perturbation reaches later decisions tau={tau}",
              not np.array_equal(a3, P.aci(sub, 240, tau, [(0.2, P.alpha_star(0.05, 0.2, tau, sub["T"]))])["dec"][0][:, last + 1: last + 50])
              or not np.array_equal(b3, P.bacc(sub, 0.05, tau)["dec"][:, last + 1: last + 400]))
    # 6. synthetic workloads
    t0 = 3000
    const = synth(np.full(t0 + 1000, 10.0), t0, mu=1.0)
    zero = synth(np.zeros(t0 + 1000), t0, mu=1.0)
    burst_y = np.full(t0 + 1000, 5.0); burst_y[t0 + 300: t0 + 330] = 200.0
    burst = synth(burst_y, t0, mu=1.0); burst["M"] = np.array([1000.0])
    for tau in (0, 1, 5):
        for nm, Ds in (("constant", const), ("zero", zero)):
            o_b = outcome(Ds, applied(Ds, P.bacc(Ds, 0.01, tau)["dec"], tau, True))
            o_p = outcome(Ds, applied(Ds, P.pac_h(Ds, 480, 0.01, tau), tau, True))
            R = P.aci(Ds, 480, tau, [(0.005, 0.01)])
            o_a = outcome(Ds, applied(Ds, R["dec"][0], tau, True))
            check(f"{nm} workload: no overload tau={tau}", o_b["over"].sum() == 0 and o_p["over"].sum() == 0 and o_a["over"].sum() == 0)
        rb = P.bacc(burst, 0.05, tau, limit=100)
        check(f"burst: BACC-r trigger respects limit 100 tau={tau}", rb["max_step"] <= 100 + 1e-9, f"max step {rb['max_step']}")
        rb2 = P.bacc(burst, 0.05, tau, limit=10)
        check(f"burst: limit 10 respected tau={tau}", rb2["max_step"] <= 10 + 1e-9, f"{rb2['max_step']}")
    # 7. limit invariant on real data (every issued target)
    Da = load("azure2019", "cal")
    for tau in (0, 1, 5):
        r = P.bacc(Da, 0.01, tau, limit=100)
        check(f"Azure BACC-r |change| <= 100 for every issued target tau={tau}", r["max_step"] <= 100 + 1e-9, f"{r['max_step']}")
        rh = P.bacc(Da, 0.01, tau, limit="hpa")
        up = np.diff(rh["dec"], axis=1); prev = rh["dec"][:, :-1]
        check(f"Azure BACC-r hpa limit up <= max(4, planned) tau={tau}", bool((up <= np.maximum(4, prev) + 1e-9).all()))
        check(f"Azure BACC-r targets within [1, M] tau={tau}", bool((r["dec"] >= 1).all() and (r["dec"] <= Da["M"][:, None]).all()))
    # 8. exploratory v0 reproduction (optional: set OBA_V0_DIR to exploratory_v0 with its loader paths)
    v0dir = os.environ.get("OBA_V0_DIR")
    if v0dir:
        sys.path.insert(0, v0dir)
        import importlib
        import types
        shim = types.ModuleType("common")  # v0 loaded its data through common.py with session paths
        shim.r7, shim.SEL_W, shim.np = r7, PAC_W, np
        shim.pac_h = lambda D, W, delta, tau: None
        sys.modules["common"] = shim
        v0 = importlib.import_module("baselines")
        D = cache["alibaba200"]
        old = pd.read_csv(Path(v0dir) / "out" / "test_per_unit.csv.gz")
        old = old[(old.dataset == "alibaba200") & (old.budget == 0.01) & (old.tau == 1)]
        f, c = r7.evaluate(D, v0.bacc(D, 0.01, 1, 5), 1, 1)
        ref = old[old.policy == "BACC-r"].overload_fraction.to_numpy()
        check("v0 exploratory BACC-r output reproduced", np.allclose(f, ref))
    else:
        print("SKIP v0 exploratory reproduction (OBA_V0_DIR not set)", flush=True)
    if FAIL:
        print("FAILED:", FAIL); sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
