"""Controllers of the adaptive-baseline study (v1): PAC-h, ACI / ACI-G, BACC-r."""
from __future__ import annotations

import math

import numpy as np

from .data import r7

ETA = 0.05


# --------------------------------------------------------------------------- PAC-h
def pac_h(D: dict, W: int, delta: float, tau: int) -> np.ndarray:
    """PAC-h decisions (U, T); identical to pac_window_grid with maxtau = tau."""
    r = r7.pac_rank(W, delta, ETA)
    t0, T = D["t0"], D["T"]
    forecast = D["y"][:, t0 - tau - 1: t0 + T - tau - 1]
    m = r7.margins(D, W, tau + 1, (r,), start=-tau)[r][:, :T]
    return np.maximum(np.ceil((forecast + m) / D["mu"][:, None]), 1.0)


# --------------------------------------------------------------------------- ACI
def alpha_star(delta: float, gamma: float, tau: int, T: int) -> float:
    """Target level such that the interleaved finite-horizon ACI bound equals delta."""
    return delta - (tau + 1) * (1.0 + gamma) / (gamma * T)


def aci(D: dict, W: int, tau: int, tracks: list[tuple[float, float]]) -> dict:
    """Adaptive conformal inference with tau+1 interleaved lanes.

    Decision j (data through j-1) targets minute j+tau. Lane l = j mod (tau+1) is
    updated with the error of its previous decision j-tau-1, whose target j-1 is
    observed. Score: signed (tau+1)-step persistence residual; quantile: order
    statistic of the last W scores at rank ceil((W+1)(1-alpha)). alpha <= 0 or
    rank > W: infinite set (err = 0); alpha >= 1: empty set (err = 1).
    Returns arrays indexed by test target minute 0..T-1.
    """
    y, t0, T, mu, M = D["y"], D["t0"], D["T"], D["mu"], D["M"]
    U = y.shape[0]; h = tau + 1; K = len(tracks)
    g = np.array([k[0] for k in tracks])[:, None]
    ast = np.array([k[1] for k in tracks])[:, None]
    res = np.full_like(y, np.nan)
    res[:, h:] = y[:, h:] - y[:, :-h]
    alpha_lane = np.repeat(ast, U, axis=1)[:, :, None].repeat(h, axis=2)
    dec = np.zeros((K, U, T)); err = np.zeros((K, U, T), bool)
    inf = np.zeros((K, U, T), bool); emp = np.zeros((K, U, T), bool); alph = np.zeros((K, U, T))
    nominal = np.zeros((K, U, T))
    ar = np.arange(U)[None, :]
    for i in range(T):
        j = i - tau; a = t0 + j; lane = j % h
        if i - h >= 0:
            alpha_lane[:, :, lane] += g * (ast - err[:, :, i - h])
        al = alpha_lane[:, :, lane]
        win = np.sort(res[:, a - W: a], axis=1)
        fc = y[:, a - 1]
        rank = np.ceil((W + 1) * (1.0 - al)).astype(np.int64)
        isinf = (al <= 0) | (rank > W)
        isemp = al >= 1
        q = win[ar, np.clip(rank, 1, W) - 1]
        nom = np.maximum(np.ceil((fc[None, :] + q) / mu[None, :]), 1.0)
        d = np.where(isinf, M[None, :], np.where(isemp, 1.0, nom))
        yt = y[:, t0 + i]
        e = np.where(isinf, False, np.where(isemp, True, yt[None, :] > fc[None, :] + q))
        dec[:, :, i] = d; err[:, :, i] = e; inf[:, :, i] = isinf; emp[:, :, i] = isemp; alph[:, :, i] = al
        nominal[:, :, i] = np.where(isinf, np.inf, np.where(isemp, 1.0, nom))
    return dict(dec=dec, err=err, inf=inf, emp=emp, alpha=alph, nominal=nominal, lane=(np.arange(T) - tau) % h)


# --------------------------------------------------------------------------- BACC-r
def bacc(D: dict, delta: float, tau: int, *, h: int = 5, limit=100, c_mult: float = 1.0,
         A_mode: str = "all", trigger: bool = True, replan: str = "floor", cap_bound: bool = True,
         record: bool = False) -> dict:
    """Re-implementation of the BACC controller of arXiv:2606.20575v2 (BACC-r).

    Source mapping (section / equation of v2) is in SOURCE_MAPPING.md.
    Accounting unit: one minute; violation: overload minute; epsilon = delta;
    accounting period: the evaluated period; lambda_safe = mu; startup delay d = tau.
    limit: max change of the issued target per decision (number, np.inf, or 'hpa').
    replan: 'floor' (sizing workload = max(calibrated peak, y_early), Sect. III-E) or
            'mult' (exploratory v0 reading: y_early calibrated with q).
    """
    y, t0, T, mu = D["y"], D["t0"], D["T"], D["mu"]
    U = y.shape[0]; eps = delta; d = tau; c = c_mult * mu
    Mb = D["M"] if cap_bound else np.full(U, np.inf)
    w, hist_len = 1440, 720
    allowance = math.floor(eps * T)

    def lim_up(cur):
        if limit == "hpa":
            return cur + np.maximum(4.0, cur)
        return cur + limit

    def lim_down(cur):
        return np.zeros_like(cur) if limit == "hpa" else cur - limit

    def score(jdec):
        a = t0 + jdec
        s = np.clip((y[:, a + d: a + d + h].max(1) + c) / (y[:, a - 1] + c), 0.05, 20.0)
        return s, jdec + d + h - 1

    pending = []
    jj = -tau - h
    while t0 + jj - 1 >= 0:
        s, comp = score(jj)
        if comp < -tau - hist_len - 1:
            break
        pending.append((comp, s)); jj -= h
    pending.sort(key=lambda z: z[0]); hist = []

    a_pace = np.full(U, eps)
    dec = np.zeros((U, T)); plan_size = np.full((U, T), np.nan); plan_a = np.full((U, T), np.nan)
    capm_all = np.zeros((U, T)); viol = np.zeros((U, T)); cs = np.zeros((U, T + 1))
    K1 = np.zeros(U); U1 = np.zeros(U); K0 = np.zeros(U); U0 = np.zeros(U); A = np.zeros(U)
    cur = None; cur_size = None; cur_a = None; ytil_reg = None; q_cur = np.ones(U)
    held = np.zeros(U); hold_until = np.full(U, -10**9)
    issued_prev = None; max_step = 0.0
    log = [] if record else None
    for i in range(T):
        j = i - tau; a = t0 + j
        m = j - 1
        if 0 <= m < T:
            if not D["warm"] and m < tau:
                capm = np.minimum(D["init"], Mb); ps = np.full(U, np.nan); pa = ps
            else:
                capm = np.minimum(dec[:, m], Mb); ps = plan_size[:, m]; pa = plan_a[:, m]
            capm_all[:, m] = capm
            ov = (y[:, t0 + m] > mu * capm).astype(float)
            viol[:, m] = ov; cs[:, m + 1] = cs[:, m] + ov
            hasp = ~np.isnan(ps)
            exc = hasp & (y[:, t0 + m] > ps)
            K1 += exc * ov; U1 += exc; K0 += (hasp & ~exc) * ov; U0 += hasp & ~exc
            A += np.where(hasp & (exc if A_mode == "missed" else True), pa, 0.0)
        while pending and pending[0][0] <= j - 1:
            hist.append(pending.pop(0))
        el = max(j, 0)
        if el > 0:
            cum = cs[:, el] / el
            lo = max(0, el - w); rol = (cs[:, el] - cs[:, lo]) / (el - lo)
            last = viol[:, el - 1]
        else:
            cum = rol = last = np.zeros(U)
        if (j + tau) % h == 0:
            e = np.maximum(rol, cum) - eps
            a_pace = np.clip(a_pace - np.where(e > 0, 1.0, 0.2) * e, 0.2 * eps, 0.9)
            Nrem = T - el
            if Nrem > 0:
                rho = last * min(d, Nrem)
                u = np.clip((allowance - cs[:, el] - rho) / Nrem, 0, 1)
                p1 = (K1 + 0.5) / (U1 + 1); p0 = (K0 + 0.5) / (U0 + 1)
                pF = np.where((u < eps) & (p1 > p0), np.clip((u - p0) / np.where(p1 > p0, p1 - p0, 1.0), 0, 1), 1.0)
                pF = np.where(u < p0, 0.0, pF)
                kappa = np.clip((U1 + 0.5) / (A + 0.5), 0.1, 10.0)
                ak = np.maximum(0.1 * eps, np.minimum(a_pace, pF / kappa))
            else:
                ak = a_pace.copy()
            hist = [(cp, s) for (cp, s) in hist if cp >= j - 1 - hist_len]
            if hist:
                S = np.sort(np.stack([s for _, s in hist], 1), axis=1); mm = S.shape[1]
                jr = np.minimum(mm, np.ceil((mm + 1) * (1 - ak)).astype(int))
                q = S[np.arange(U), jr - 1]
            else:
                q = np.ones(U)
            q_cur = q
            ytil = np.maximum(0.0, (y[:, a - 1] + c) * q - c)
            x = np.maximum(1.0, np.ceil(ytil / mu))
            if cur is not None:
                x = np.clip(x, lim_down(cur), lim_up(cur))
                x = np.where(j < hold_until, np.maximum(x, held), x)
            x = np.clip(x, 1.0, Mb)
            cur, cur_size, cur_a, ytil_reg = x, ytil, ak, ytil
            if a + d + h <= y.shape[1]:
                s_new, comp_new = score(j)
                pending.append((comp_new, s_new)); pending.sort(key=lambda z: z[0])
        elif trigger:
            ylast = y[:, a - 1]; yprev = y[:, a - 2]
            above = (last > eps) | (rol > eps) | (cum > eps)
            fire = (ylast > cur_size) & above
            if fire.any():
                yearly = ylast + (1 + d) * np.maximum(ylast - yprev, 0.0)
                if replan == "floor":
                    size = np.maximum(ytil_reg, yearly)
                else:
                    size = np.maximum(0.0, (yearly + c) * q_cur - c)
                xe = np.maximum(1.0, np.ceil(size / mu))
                xe = np.clip(np.minimum(xe, lim_up(cur)), 1.0, Mb)
                upd = fire & (xe > cur)
                cur = np.where(upd, xe, cur); cur_size = np.where(upd, size, cur_size)
                held = np.where(upd, cur, held); hold_until = np.where(upd, j + tau, hold_until)
        dec[:, i] = cur; plan_size[:, i] = cur_size; plan_a[:, i] = cur_a
        if issued_prev is not None:
            max_step = max(max_step, float(np.max(np.abs(cur - issued_prev))))
        issued_prev = cur.copy()
    return dict(dec=dec, max_step=max_step, observed_cap=capm_all)
