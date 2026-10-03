"""
run_synthetic_dkw.py
====================
EXP-3b: Synthetic validation of Theorem VII.2 (DKW finite-sample localization
of the conformal quantile).

Theorem VII.2 holds under an i.i.d. calibration-window model (stronger than the
local-exchangeability assumption of Theorem VII.1). On the real Alibaba trace
(EXP-3) the predicted 1/sqrt(W) rate is masked by local non-stationarity. This
experiment reproduces the assumed i.i.d. regime exactly and checks that the
DKW-predicted rates appear cleanly:

  (R1) E[ sup_x |F_hat_W(x) - F(x)| ]            ~  O(1/sqrt(W))   (DKW directly)
  (R2) E[ |q_hat_{1-delta} - F^{-1}(1-delta)| ]  ~  O(1/sqrt(W))   (quantile loc.)
  (R3) Std[ q_hat_{1-delta} ]                    ~  O(1/sqrt(W))   (var ~ 1/W)
  (R4) the empirical sup-deviation stays within the DKW envelope eps_W(eta)
       at the stated confidence 1-eta (coverage of the inequality itself).

We sweep W over a wide grid and fit log-log slopes; slopes near -0.5 confirm
the O(1/sqrt(W)) rate. Several positive-residual distributions are used,
including a zero-inflated one (atom at 0) to exercise the generalized-quantile
form of the theorem (no continuity assumption).

Outputs:
  experiments/results/exp3b_synthetic_dkw.csv
  experiments/results/exp3b_synthetic_dkw_slopes.csv
  experiments/figures/exp3b_synthetic_dkw.pdf / .png
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260530)

RESULTS = Path("experiments/results")
FIGS    = Path("experiments/figures")
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# Distributions of the positive residual score e^+ (all on R_{>=0}).
# Each entry: name -> (sampler(n), ppf(p), is_continuous)
# Scales chosen so the (1-delta)-quantiles are O(1-50), echoing the trace.
def make_dists():
    d = {}

    # Exponential(scale=10): continuous, smooth, light-ish tail.
    sc = 10.0
    d["Exponential"] = (
        lambda n: RNG.exponential(sc, n),
        lambda p: stats.expon.ppf(p, scale=sc),
        True,
    )

    # Half-normal(sigma=12): continuous.
    sg = 12.0
    d["Half-normal"] = (
        lambda n: np.abs(RNG.normal(0.0, sg, n)),
        lambda p: stats.halfnorm.ppf(p, scale=sg),
        True,
    )

    # Lognormal(s=0.9, scale=8): heavy right tail (realistic for bursty svc).
    s, scl = 0.9, 8.0
    d["Lognormal"] = (
        lambda n: RNG.lognormal(mean=np.log(scl), sigma=s, size=n),
        lambda p: stats.lognorm.ppf(p, s, scale=scl),
        True,
    )

    # Zero-inflated exponential: P(e^+ = 0) = p0 (overpredictions map to 0),
    # else Exponential(scale). Discontinuous -> exercises generalized quantile.
    p0, sc2 = 0.40, 12.0
    def zi_sampler(n):
        x = RNG.exponential(sc2, n)
        mask = RNG.random(n) < p0
        x[mask] = 0.0
        return x
    def zi_ppf(p):
        # F(x) = p0 + (1-p0)*(1-exp(-x/sc2)) for x>0 ; atom p0 at 0.
        if p <= p0:
            return 0.0
        return stats.expon.ppf((p - p0) / (1.0 - p0), scale=sc2)
    d["Zero-infl. exp. (atom@0)"] = (zi_sampler, zi_ppf, False)

    return d


DISTS   = make_dists()
W_GRID  = [30, 60, 120, 240, 480, 960, 1920, 3840]
DELTAS  = [0.10, 0.05, 0.01]
ETA     = 0.05                       # DKW confidence: inequality holds w.p. >= 1-eta
N_TRIAL = 2000                       # calibration draws per (dist, W, delta)


def eps_W(W, eta=ETA):
    return np.sqrt(np.log(2.0 / eta) / (2.0 * W))


def conformal_level(W, delta):
    # ell_W(delta) = ceil((W+1)(1-delta)), clipped to W
    return int(min(W, np.ceil((W + 1) * (1.0 - delta))))


def run():
    rows = []
    for dname, (sampler, ppf, cont) in DISTS.items():
        for delta in DELTAS:
            q_true = float(ppf(1.0 - delta))
            for W in W_GRID:
                if delta < 1.0 / (W + 1):     # enforce non-saturation condition
                    continue
                lvl = conformal_level(W, delta)

                # Vectorized over trials: (N_TRIAL, W) matrix, sort per row once.
                cal = np.sort(sampler(N_TRIAL * W).reshape(N_TRIAL, W), axis=1)
                sup_devs = _supdev(dname, cal, ppf, cont)     # sup_x|F_hat-F| per trial
                qhats    = cal[:, lvl - 1]                    # conformal quantile
                ol_probs = 1.0 - _F_of(dname, qhats, ppf)     # realised overload prob

                rows.append(dict(
                    dist=dname, continuous=cont, delta=delta, W=W,
                    q_true=q_true,
                    mean_sup_dev=float(sup_devs.mean()),
                    p95_sup_dev=float(np.quantile(sup_devs, 0.95)),
                    eps_W=float(eps_W(W)),
                    dkw_envelope_hit=float(np.mean(sup_devs <= eps_W(W))),  # ~>= 1-eta
                    mean_abs_qdev=float(np.mean(np.abs(qhats - q_true))),
                    std_qhat=float(qhats.std(ddof=1)),
                    mean_qhat=float(qhats.mean()),
                    mean_realised_ol=float(ol_probs.mean()),
                    std_realised_ol=float(ol_probs.std(ddof=1)),
                    abs_dev_ol=float(abs(ol_probs.mean() - delta)),
                ))
        print(f"  done: {dname}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exp3b_synthetic_dkw.csv", index=False)

    # Log-log slope fits (expected ~ -0.5 for O(1/sqrt(W)))
    slope_rows = []
    for (dname, delta), g in df.groupby(["dist", "delta"]):
        g = g.sort_values("W")
        lW = np.log(g["W"].to_numpy())
        for metric in ["mean_sup_dev", "mean_abs_qdev", "std_qhat"]:
            y = g[metric].to_numpy()
            ok = y > 0
            slope, intercept = np.polyfit(lW[ok], np.log(y[ok]), 1)
            # R^2
            pred = slope * lW[ok] + intercept
            ss_res = np.sum((np.log(y[ok]) - pred) ** 2)
            ss_tot = np.sum((np.log(y[ok]) - np.log(y[ok]).mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            slope_rows.append(dict(dist=dname, delta=delta, metric=metric,
                                   slope=float(slope), r2=float(r2)))
    sdf = pd.DataFrame(slope_rows)
    sdf.to_csv(RESULTS / "exp3b_synthetic_dkw_slopes.csv", index=False)

    _plot(df, sdf)
    return df, sdf


def _Fleft_of(dname, x):
    """Left-limit of the true CDF, F(x^-). Differs from F only at atoms."""
    if dname.startswith("Zero-infl"):
        p0, sc2 = 0.40, 12.0
        # left limit at x=0 is 0 (atom sits at 0); for x>0 the cont. part is left-continuous.
        out = np.where(x <= 0, 0.0, p0 + (1 - p0) * stats.expon.cdf(x, scale=sc2))
        return np.where(x < 0, 0.0, out)
    return _F_of(dname, x, None)   # continuous: F(x^-) = F(x)


def _supdev(dname, cal, ppf, cont):
    """E-free per-trial sup_x |F_hat_W(x) - F(x)|, tie-aware for atoms.

    For continuous F (no ties a.s.) the standard order-statistic KS formula is
    exact and fully vectorized. For distributions with atoms we account for
    ties by comparing right-continuous ranks to F(v) and left ranks to F(v^-).
    """
    N, W = cal.shape
    Fv = _F_of(dname, cal, ppf)
    if cont:
        i = np.arange(1, W + 1)
        d_plus  = np.max(i / W - Fv, axis=1)
        d_minus = np.max(Fv - (i - 1) / W, axis=1)
        return np.maximum(d_plus, d_minus)
    # tie-aware path (atoms): per-row searchsorted gives right/left ranks.
    Fl = _Fleft_of(dname, cal)
    out = np.empty(N)
    for r in range(N):
        row = cal[r]
        R = np.searchsorted(row, row, side="right")   # #{<= v}
        L = np.searchsorted(row, row, side="left")     # #{<  v}
        dev = np.maximum(np.abs(R / W - Fv[r]), np.abs(L / W - Fl[r]))
        out[r] = dev.max()
    return out


def _F_of(dname, x, ppf):
    """Vectorized true CDF F(x) for each distribution."""
    if dname == "Exponential":
        return stats.expon.cdf(x, scale=10.0)
    if dname == "Half-normal":
        return stats.halfnorm.cdf(x, scale=12.0)
    if dname == "Lognormal":
        return stats.lognorm.cdf(x, 0.9, scale=8.0)
    if dname.startswith("Zero-infl"):
        p0, sc2 = 0.40, 12.0
        out = np.where(x <= 0, p0, p0 + (1 - p0) * stats.expon.cdf(x, scale=sc2))
        out = np.where(x < 0, 0.0, out)
        return out
    raise ValueError(dname)


def _F_scalar(dname, x, ppf):
    return float(_F_of(dname, np.array([x], dtype=float), ppf)[0])


def _plot(df, sdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })
    PAL = {"Exponential": "#2196F3", "Half-normal": "#4CAF50",
           "Lognormal": "#FF9800", "Zero-infl. exp. (atom@0)": "#9C27B0"}

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    d0 = 0.05  # quantile/coverage panels at delta=0.05

    # Panel A: sup-deviation vs W with DKW envelope + 1/sqrt(W) reference.
    sub = df[df["delta"] == d0]
    for dname, g in sub.groupby("dist"):
        g = g.sort_values("W")
        ax[0].loglog(g["W"], g["mean_sup_dev"], "o-", color=PAL[dname],
                     label=dname, ms=4)
    Wg = np.array(W_GRID, dtype=float)
    ax[0].loglog(Wg, eps_W(Wg), "k--", lw=1.6,
                 label=r"DKW envelope $\varepsilon_W(\eta{=}0.05)$")
    ref = sub[sub["dist"] == "Exponential"].sort_values("W")
    c = ref["mean_sup_dev"].iloc[0] * np.sqrt(ref["W"].iloc[0])
    ax[0].loglog(Wg, c / np.sqrt(Wg), ":", color="grey", lw=1.4,
                 label=r"$1/\sqrt{W}$ reference")
    ax[0].set_xlabel("Calibration window $W$ (log)")
    ax[0].set_ylabel(r"$\mathbb{E}\,\sup_x|\hat F_W - F|$ (log)")
    ax[0].set_title("(A) Uniform CDF deviation vs. DKW rate")
    ax[0].legend(frameon=False, loc="lower left")

    # Panel B: |q_hat - q_true| vs W with 1/sqrt(W) reference.
    for dname, g in sub.groupby("dist"):
        g = g.sort_values("W")
        ax[1].loglog(g["W"], g["mean_abs_qdev"], "o-", color=PAL[dname],
                     label=dname, ms=4)
    ref = sub[sub["dist"] == "Exponential"].sort_values("W")
    c = ref["mean_abs_qdev"].iloc[0] * np.sqrt(ref["W"].iloc[0])
    ax[1].loglog(Wg, c / np.sqrt(Wg), ":", color="grey", lw=1.4,
                 label=r"$1/\sqrt{W}$ reference")
    ax[1].set_xlabel("Calibration window $W$ (log)")
    ax[1].set_ylabel(r"$\mathbb{E}\,|\hat q_{1-\delta}-F^{-1}(1-\delta)|$ (log)")
    ax[1].set_title(r"(B) Conformal-quantile deviation ($\delta{=}0.05$)")
    ax[1].legend(frameon=False, loc="lower left")

    # Panel C: realised overload prob vs W (mean +- std), nominal delta line.
    for dname, g in sub.groupby("dist"):
        g = g.sort_values("W")
        ax[2].semilogx(g["W"], g["mean_realised_ol"], "o-", color=PAL[dname],
                       label=dname, ms=4)
        ax[2].fill_between(g["W"],
                           g["mean_realised_ol"] - g["std_realised_ol"],
                           g["mean_realised_ol"] + g["std_realised_ol"],
                           color=PAL[dname], alpha=0.12)
    ax[2].axhline(d0, color="red", ls="--", lw=1.4, label=r"nominal $\delta=0.05$")
    ax[2].set_xlabel("Calibration window $W$ (log)")
    ax[2].set_ylabel(r"Realised overload prob. $P(e_{new}>\hat q)$")
    ax[2].set_title("(C) Coverage stabilises as $W$ grows")
    ax[2].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGS / "exp3b_synthetic_dkw.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "exp3b_synthetic_dkw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Running synthetic DKW validation (EXP-3b)...")
    df, sdf = run()
    print("\n=== log-log slopes (expected ~ -0.5) ===")
    print(sdf.to_string(index=False))
    print("\n=== DKW envelope coverage (should be >= 1-eta = 0.95) ===")
    print(df.groupby("dist")["dkw_envelope_hit"].min().to_string())
