import math, time, json, sys
import numpy as np
from common import r7, SEL_W, pac_h

GAMMAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)


def alpha_star(delta, gamma, tau, T):
    return delta - (1 + gamma) * (tau + 1) / (gamma * T)


def Mcap(D):
    return np.maximum(np.ceil(2 * D["y"][:, :D["t0"]].max(1) / D["mu"]), 1.0)


def aci_tracks(D, W, tau, tracks):
    """tracks: list of (gamma, astar). Returns dict idx -> (dec[-tau..T-1], saturated count, err fraction)."""
    y, t0, T, mu = D["y"], D["t0"], D["T"], D["mu"]
    U = y.shape[0]; h = tau + 1
    M = Mcap(D)
    res = np.full_like(y, np.nan)
    res[:, h:] = y[:, h:] - y[:, :-h]
    K = len(tracks)
    g = np.array([t[0] for t in tracks])[:, None]
    ast = np.array([t[1] for t in tracks])[:, None]
    alpha = np.repeat(ast[:, :, None], h, axis=2) * np.ones((K, U, h))
    dec = np.zeros((K, U, T + tau))
    sat = np.zeros((K, U)); errs = np.zeros((K, U)); pend = {}
    ar = np.arange(U)
    for col, j in enumerate(range(-tau, T)):
        a = t0 + j
        slot = j % h
        if j - h in pend:
            e = pend.pop(j - h)
            alpha[:, :, slot] = alpha[:, :, slot] + g * (ast - e)
        win = np.sort(res[:, a - W: a], axis=1)
        fc = y[:, a - 1]
        al = alpha[:, :, slot]
        rank = np.ceil((W + 1) * (1 - al)).astype(np.int64)
        inf = (al <= 0) | (rank > W)
        neg = al >= 1
        rk = np.clip(rank, 1, W) - 1
        q = win[ar[None, :], rk]
        cap = np.maximum(np.ceil((fc[None, :] + q) / mu[None, :]), 1.0)
        cap = np.where(inf, M[None, :], np.where(neg, 1.0, cap))
        dec[:, :, col] = cap
        sat += inf & (j >= 0)
        if a + tau < y.shape[1]:
            yt = y[:, a + tau]
            e = np.where(inf, 0.0, np.where(neg, 1.0, (yt[None, :] > fc[None, :] + q).astype(float)))
            pend[j] = e
            if j >= 0:
                errs += e
    return dec, sat, errs / T


def bacc(D, delta, tau, h=5, max_change=100, c_mult=1.0, A_mode='all', trigger=True):
    y, t0, T, mu = D["y"], D["t0"], D["T"], D["mu"]
    U = y.shape[0]; eps = delta; d = tau; c = c_mult * mu
    w = 1440; hist_len = 720
    a_min, a_max = 0.2 * eps, 0.9
    allowance = math.floor(eps * T)
    # score history: list of (completion_rel_minute, scores[U])
    hist = []
    def score(jdec):
        a = t0 + jdec
        lo, hi = a + d, a + d + h
        ymax = y[:, lo:hi].max(1); yhat = y[:, a - 1]
        return np.clip((ymax + c) / (yhat + c), 0.05, 20.0), jdec + d + h - 1
    pending_scores = []
    jj = -tau - h
    while t0 + jj - 1 >= 0:
        s, comp = score(jj)
        if comp < -tau - hist_len - 1:
            break
        pending_scores.append((comp, s)); jj -= h
    pending_scores.sort(key=lambda z: z[0])
    a_pace = np.full(U, eps)
    dec = np.zeros((U, T + tau))
    plan_ytil = np.full((U, T + tau), np.nan); plan_a = np.full((U, T + tau), np.nan)
    viol = np.zeros((U, T)); cs = np.zeros((U, T + 1))
    K1 = np.zeros(U); U1 = np.zeros(U); K0 = np.zeros(U); U0 = np.zeros(U); A = np.zeros(U)
    cur_target = None; cur_ytil = None; cur_a = None; q_cur = np.ones(U)
    init = D["init"]
    for col, j in enumerate(range(-tau, T)):
        a = t0 + j
        # account minute j-1 (test minutes only)
        m = j - 1
        if 0 <= m < T:
            if D["warm"] or m >= tau:
                capm = dec[:, m - tau + tau]  # decision index m - tau -> column m
                yt_plan = plan_ytil[:, m]; a_plan = plan_a[:, m]
            else:
                capm = init; yt_plan = np.full(U, np.nan); a_plan = np.full(U, np.nan)
            ov = (y[:, a - 1] > mu * capm).astype(float)
            viol[:, m] = ov; cs[:, m + 1] = cs[:, m] + ov
            hasp = ~np.isnan(yt_plan)
            exc = hasp & (y[:, a - 1] > yt_plan)
            K1 += exc * ov; U1 += exc; K0 += (hasp & ~exc) * ov; U0 += (hasp & ~exc)
            A += np.where(hasp & (exc if A_mode == 'missed' else True), a_plan, 0.0)
        while pending_scores and pending_scores[0][0] <= j - 1:
            hist.append(pending_scores.pop(0))
        el = max(j, 0)
        regular = ((j + tau) % h == 0)
        if el > 0:
            cum = cs[:, el] / el
            lo = max(0, el - w); rol = (cs[:, el] - cs[:, lo]) / (el - lo)
            last = viol[:, el - 1]
        else:
            cum = rol = last = np.zeros(U)
        if regular:
            e = np.maximum(rol, cum) - eps
            eta = np.where(e > 0, 1.0, 0.2)
            a_pace = np.clip(a_pace - eta * e, a_min, a_max)
            Nrem = T - el
            if Nrem > 0:
                B = cs[:, el]
                rho = last * min(d, Nrem)
                u = np.clip((allowance - B - rho) / Nrem, 0, 1)
                p1 = (K1 + 0.5) / (U1 + 1); p0 = (K0 + 0.5) / (U0 + 1)
                pF = np.ones(U)
                cond = (u < eps) & (p1 > p0)
                pF = np.where(cond, np.clip((u - p0) / np.where(p1 > p0, p1 - p0, 1), 0, 1), pF)
                pF = np.where(u < p0, 0.0, pF)
                kappa = np.clip((U1 + 0.5) / (A + 0.5), 0.1, 10)
                ak = np.maximum(0.1 * eps, np.minimum(a_pace, pF / kappa))
            else:
                ak = a_pace.copy()
            keep = [s for (cp, s) in hist if cp >= j - 1 - hist_len]
            hist = [(cp, s) for (cp, s) in hist if cp >= j - 1 - hist_len]
            if keep:
                S = np.sort(np.stack(keep, 1), axis=1); mm = S.shape[1]
                jr = np.minimum(mm, np.ceil((mm + 1) * (1 - ak)).astype(int))
                q = S[np.arange(U), jr - 1]
            else:
                q = np.ones(U)
            q_cur = q
            yhat = y[:, a - 1]
            ytil = np.maximum(0.0, (yhat + c) * q - c)
            x = np.maximum(1.0, np.ceil(ytil / mu))
            if cur_target is not None:
                if max_change == 'hpa':
                    x = np.minimum(x, cur_target + np.maximum(4, cur_target))
                else:
                    x = np.clip(x, cur_target - max_change, cur_target + max_change)
            cur_target, cur_ytil, cur_a = x, ytil, ak
            if a + d + h <= y.shape[1]:
                s, comp = score(j)
                pending_scores.append((comp, s))
        else:
            ylast = y[:, a - 1]; yprev = y[:, a - 2]
            above = (last > eps) | (rol > eps) | (cum > eps)
            fire = (ylast > cur_ytil) & above & trigger
            if fire.any():
                yearly = ylast + (1 + d) * np.maximum(ylast - yprev, 0)
                ye = np.maximum(0.0, (yearly + c) * q_cur - c)
                xe = np.maximum(1.0, np.ceil(ye / mu))
                upd = fire & (xe > cur_target)
                cur_target = np.where(upd, xe, cur_target)
                cur_ytil = np.where(upd, ye, cur_ytil)
        dec[:, col] = cur_target
        if col < T + tau:
            plan_ytil[:, col] = cur_ytil; plan_a[:, col] = cur_a
    return dec
