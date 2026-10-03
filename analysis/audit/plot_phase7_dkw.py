#!/usr/bin/env python3
"""Render readable main/supplement DKW figures from frozen experiment CSVs.

No simulation is rerun.  The source experiment is
``experiments/src/run_synthetic_dkw.py``; the two graphics only change its
presentation.  Run with --write, then verify with --check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER.parent / "experiments" / "results"
FIGURES = PAPER / "figures" / "verified"
NAMES = ("fig_dkw_main_phase7", "fig_dkw_full_phase7")
COLORS = {
    "Exponential": "#2763aa",
    "Half-normal": "#23834b",
    "Lognormal": "#d07b16",
    "Zero-infl. exp. (atom@0)": "#8e50a6",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(RESULTS / "exp3b_synthetic_dkw.csv")
    slopes = pd.read_csv(RESULTS / "exp3b_synthetic_dkw_slopes.csv")
    assert set(data.dist) == set(COLORS)
    assert set(np.round(data.delta.unique(), 5)) == {0.01, 0.05, 0.1}
    counts = data.groupby(["dist", "delta"]).W.nunique()
    assert all(counts.loc[(name, 0.01)] == 6 for name in COLORS)
    assert all(counts.loc[(name, level)] == 8 for name in COLORS for level in (0.05, 0.1))
    assert data.groupby(["dist", "delta", "W"]).size().eq(1).all()
    assert data.W.min() == 30 and data.W.max() == 3840
    assert np.allclose(data.eps_W, np.sqrt(np.log(40) / (2 * data.W)))
    assert data.p95_sup_dev.le(data.eps_W + 0.03).all()
    selected = slopes.loc[np.isclose(slopes.delta, 0.05) & slopes.metric.eq("mean_sup_dev")]
    assert len(selected) == 4
    assert selected.slope.between(-0.51, -0.47).all()
    assert selected.r2.gt(0.999).all()
    return data.loc[np.isclose(data.delta, 0.05)].copy(), selected


def draw(data: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(4.05, 3.15), constrained_layout=True)
    for name, color in COLORS.items():
        group = data.loc[data.dist.eq(name)].sort_values("W")
        ax.loglog(group.W, group.mean_sup_dev, "o-", color=color,
                  linewidth=1.6, markersize=3.6, label=name)
    envelope = data.loc[data.dist.eq("Exponential")].sort_values("W")
    ax.loglog(envelope.W, envelope.eps_W, "k--", linewidth=1.3,
              label="DKW 95% envelope")
    ax.set(xlabel="Calibration window $W$", ylabel=r"Mean $\sup_x|\hat F_W(x)-F(x)|$")
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.92)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{NAMES[0]}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(4.45, 7.35), constrained_layout=True)
    for name, color in COLORS.items():
        group = data.loc[data.dist.eq(name)].sort_values("W")
        axes[0].loglog(group.W, group.mean_sup_dev, "o-", color=color,
                       linewidth=1.45, markersize=3, label=name)
        axes[1].loglog(group.W, group.mean_abs_qdev, "o-", color=color,
                       linewidth=1.45, markersize=3)
        axes[2].semilogx(group.W, 100 * group.mean_realised_ol, "o-", color=color,
                         linewidth=1.45, markersize=3)
    axes[0].loglog(envelope.W, envelope.eps_W, "k--", linewidth=1.25,
                   label="DKW 95% envelope")
    axes[0].set(ylabel=r"Mean $\sup_x|\hat F_W-F|$", title="(a) CDF-scale deviation")
    axes[0].legend(fontsize=7.3, loc="upper right", framealpha=0.94)
    axes[1].set(ylabel="Mean absolute quantile error", title="(b) Numerical quantile error")
    axes[2].axhline(5, color="black", linestyle="--", linewidth=1.2)
    axes[2].set(xlabel="Calibration window $W$", ylabel="Mean exceedance (%)",
                title="(c) Held-out score exceedance")
    for ax in axes:
        ax.grid(alpha=0.22, linestyle="--")
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{NAMES[1]}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data, slopes = load()
    if args.write:
        draw(data)
        print(f"Rendered two DKW figures from {len(data)} saved rows; slope range "
              f"{slopes.slope.min():.3f} to {slopes.slope.max():.3f}")
    else:
        assert all((FIGURES / f"{name}.{ext}").exists()
                   for name in NAMES for ext in ("pdf", "png"))
        print("PASS: DKW figure inputs and outputs are complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
