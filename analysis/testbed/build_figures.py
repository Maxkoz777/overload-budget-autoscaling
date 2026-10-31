"""Figures of the Kubernetes validation (PDF for the paper, PNG for review). Run after extract.py and build_tables.py."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import (COLOR, CUSTOM, DATA, FIGURES, LABEL, MARKER, MINUTES, MU, POLICIES, RES, RUNS, TABLES, TB, WARM, jload, meta,
                    per_minute, plan)

PAPER = os.environ.get("PAPER") == "1"
if PAPER:
    LABEL = {**LABEL, "conf": "Standard rank"}
    FIGURES = FIGURES.parent / "figures_paper"
    FIGURES.mkdir(exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CRIT = "#d03b3b"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
W1, W2 = 3.5, 7.2
TABLE_INV = {"PAC-h": "pach", "Conformal": "conf", "B1": "b1", "HPA": "hpa"}
FS = 1.25 if PAPER else 1.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5 * FS, "axes.titlesize": 8 * FS, "axes.labelsize": 7.5 * FS,
                     "xtick.labelsize": 7 * FS, "ytick.labelsize": 7 * FS, "legend.fontsize": 7 * FS, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "lines.linewidth": 1.4, "legend.frameon": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42})
PF = pd.read_csv(DATA / "function_runs.csv")
FN = [f"f{i}" for i in range(1, 13)]
STRATUM = {f["name"]: f["stratum"] for f in plan()["functions"]}


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    print("figure", name, flush=True)


def policy_legend(ax, pols=POLICIES, **kw):
    h = [Line2D([], [], color=COLOR[p], marker=MARKER[p], ls="", ms=5, label=LABEL[p]) for p in pols]
    ax.legend(handles=h, **kw)


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom", ha="right", color=INK)


def ecdf(ax, x, **kw):
    x = np.sort(np.asarray(x, float)[~np.isnan(np.asarray(x, float))])
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kw)


def f01_delays():
    act = jload(TB / "actuator_model.json")
    up = pd.read_csv(DATA / "delays_up_all_runs.csv.gz")
    dn = pd.read_csv(DATA / "delays_down_all_runs.csv.gz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1.6, 1]})
    stages = [("to_started", "started"), ("to_ready", "Ready"), ("endpoint_mid", "ready endpoint"),
              ("to_first_served", "first served")]
    for (c, lab), col in zip(stages, RAMP[1:]):
        ecdf(a, up[c], color=col, label=f"{lab} {up[c].median():.1f} s")
    a.axvline(act["d_up_s"], color=INK2, ls="--", lw=0.9, label=f"model d_up {act['d_up_s']:.1f} s")
    a.set_xlim(0, 28)
    a.set_xlabel("seconds after the scale-up call" if not PAPER else "s after the scale-up call")
    a.set_ylabel("share of pods (ECDF)")
    a.set_title(f"Scale-up: {len(up):,} pods, 3 controllers, 4 runs", loc="left")
    if PAPER:
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, title="medians", title_fontsize=7.5)
    else:
        a.legend(loc="center left", bbox_to_anchor=(0.11, 0.55), title="medians", title_fontsize=6.5)
    panel(a, "a")
    ecdf(b, dn.removed_mid, color=RAMP[2], label=f"endpoints {dn.removed_mid.median():.2f} s")
    ecdf(b, dn.to_deleted, color=RAMP[4], label=f"deleted {dn.to_deleted.median():.1f} s")
    b.axvline(act["d_down_s"], color=INK2, ls="--", lw=0.9, label=f"model d_down {act['d_down_s']:.2f} s")
    b.set_xlim(0, 8)
    b.set_xlabel("seconds after the scale-down call" if not PAPER else "s after the scale-down call")
    b.set_title(f"Scale-down: {len(dn):,} pods", loc="left")
    if PAPER:
        b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, title="medians", title_fontsize=7.5)
    else:
        b.legend(loc="center left", bbox_to_anchor=(0.09, 0.55), title="medians", title_fontsize=6.5)
    panel(b, "b")
    save(fig, "F01_actuation_delays")


def f02_overload_by_function():
    fig, ax = plt.subplots(figsize=(W2, 2.6))
    off = dict(zip(POLICIES, [-0.27, -0.09, 0.09, 0.27]))
    for p in POLICIES:
        g = PF[PF.policy == p].groupby("function")
        lo, hi = g.overload_lo.mean().reindex(FN), g.overload_hi.mean().reindex(FN)
        mn, mx = g.overload_lo.min().reindex(FN), g.overload_hi.max().reindex(FN)
        x = np.arange(12) + off[p]
        ax.vlines(x, mn, mx, color=COLOR[p], lw=0.8, alpha=0.7)
        ax.vlines(x, lo, hi, color=COLOR[p], lw=2.6)
        ax.plot(x, lo, ls="", marker=MARKER[p], color=COLOR[p], ms=4.2, mec="white", mew=0.5)
    ax.axhline(14, color=CRIT, lw=1, ls="--")
    ax.text(11.55, 14.6, "budget: 14 min", color=CRIT, fontsize=6.5 * FS, ha="right")
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{f}\n{STRATUM[f]}" for f in FN])
    ax.set_ylabel("overload minutes per function-run" if not PAPER else "overload min per function-run")
    ax.set_ylim(-1, 40)
    over = PF[(PF.policy == "hpa") & (PF.overload_lo > 40)]
    for _, r in over.iterrows():
        i = FN.index(r.function)
        span = f"{int(r.overload_lo)}" if r.overload_hi == r.overload_lo else f"{int(r.overload_lo)}\u2013{int(r.overload_hi)}"
        ax.text(i + off["hpa"] - 0.08, 38.5, f"HPA run {r.run[-1]}: {span} \u2191", fontsize=6.3 * FS, color=INK2, ha="right", va="top")
    if not PAPER:
        ax.set_title("Overload per function: mean of 4 runs (marker), unknown-minute interval (thick), range over runs (thin)", loc="left")
    policy_legend(ax, loc="upper right", ncol=2)
    save(fig, "F02_overload_by_function")


def f03_cost_vs_overload():
    clair = PF[PF.policy == "pach"].groupby("run").clairvoyant_pod_minutes.sum()
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), sharey=True)
    for ax, day, runs in zip(axes, (10, 11), (("run1", "run3"), ("run2", "run4"))):
        for p in POLICIES:
            pts = []
            for r in runs:
                g = PF[(PF.policy == p) & (PF.run == r)]
                x, y = g.ready_pod_minutes_lo.sum() / 1000, g.overload_lo.sum()
                ax.plot(x, max(y, 0.6), marker=MARKER[p], color=COLOR[p], ms=6, ls="", mec="white", mew=0.6)
                pts.append((x, max(y, 0.6), r[-1]))
            (x1, y1, r1), (x2, y2, r2) = pts
            dxy = (5, -10) if p == "pach" else (5, 3)
            if abs(x1 - x2) < 0.6 and abs(np.log(y1 / y2)) < 0.15:
                ax.annotate(f"{r1}, {r2}", (x1, y1), xytext=dxy, textcoords="offset points", fontsize=6.3, color=INK2)
            else:
                for x, y, rr in pts:
                    ax.annotate(rr, (x, y), xytext=(5, 2), textcoords="offset points", fontsize=6.3, color=INK2)
        ax.axvline(clair[runs[0]] / 1000, color=INK2, ls=":", lw=0.9)
        ax.text(clair[runs[0]] / 1000 + 0.15, 0.62, "zero-overload\nreference", fontsize=6.3, color=INK2, va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 600)
        ax.set_xlim(clair[runs[0]] / 1000 - 0.5, 69)
        ax.set_xlabel("ready pod-minutes per run (thousands)")
        ax.set_title(f"Test day {day} (runs {runs[0][-1]} and {runs[1][-1]}" + (")" if PAPER else ", numbers = run)"), loc="left")
    axes[0].set_ylabel("overload minutes per run")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    policy_legend(axes[1], loc="upper right")
    panel(axes[0], "a")
    panel(axes[1], "b")
    save(fig, "F03_cost_vs_overload")


def f04_h1():
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.4), sharex=True, sharey=True)
    rng = np.random.default_rng(1)
    for ax, p in zip(axes, CUSTOM):
        g = PF[PF.policy == p]
        jx = rng.uniform(-0.18, 0.18, len(g))
        m = g.model_overload.to_numpy(float) + jx
        ax.fill_between([-1, 17], [-3, 15], [1, 19], color=GRID, alpha=0.8, lw=0, label="tolerance +/- 2 min")
        ax.plot([-1, 17], [-1, 17], color=INK2, lw=0.7)
        ax.vlines(m, g.overload_lo, g.overload_hi, color=COLOR[p], lw=1)
        ax.plot(m, g.overload_lo, ls="", marker=MARKER[p], color=COLOR[p], ms=3.6, mec="white", mew=0.4)
        vc = g.h1.value_counts()
        if PAPER:
            u = vc.get("undetermined", 0)
            ax.set_title(f"{LABEL[p]}: {vc.get('within', 0)}/48" + (f" ({u} undet.)" if u else ""), loc="left")
        else:
            ax.set_title(f"{LABEL[p]}\n{vc.get('within', 0)}/48 within, {vc.get('undetermined', 0)} undetermined", loc="left")
        ax.set_xlim(-1, 17)
        ax.set_ylim(-1, 17)
        ax.set_xlabel("model overload minutes")
    axes[0].set_ylabel("observed overload minutes\n(marker: definite, line: + unknown)")
    axes[0].legend(loc="upper left")
    for a, s in zip(axes, "abc"):
        panel(a, s)
    save(fig, "F04_h1_model_vs_observed")


def f05_h3():
    lo = PF.groupby(["function", "policy"]).overload_lo.mean().unstack().reindex(FN)
    hi = PF.groupby(["function", "policy"]).overload_hi.mean().unstack().reindex(FN)
    t = pd.read_csv(TABLES / "T06_h3.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, o, (_, row) in zip(axes, ("conf", "b1"), t.iterrows()):
        dlo, dhi = lo.pach - hi[o], hi.pach - lo[o]
        y = np.arange(12)[::-1]
        ax.axvline(0, color=INK2, lw=0.8)
        ax.hlines(y, dlo, dhi, color=COLOR["pach"], lw=3)
        ax.plot(dlo, y, "|", color=COLOR["pach"], ms=6)
        ax.plot(dhi, y, "|", color=COLOR["pach"], ms=6)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{f} ({STRATUM[f]})" for f in FN])
        ax.set_xlabel(f"PAC-h minus {LABEL[o]} (min per function-run)" if not PAPER else f"PAC-h minus {LABEL[o]} (min)")
        ax.set_title(f"PAC-h vs {LABEL[o]}: {row.conclusion}\nworst-case Holm p = {row.worst_p_holm:.3f}; mean [{row.mean_diff_lo:.2f}, {row.mean_diff_hi:.2f}]\n"
                     f"bootstrap 95% [{row.bootstrap95_lo:.2f}, {row.bootstrap95_hi:.2f}]", loc="left", fontsize=7.2)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(-3.6, 1)
    axes[1].set_xlim(-13, 1)
    panel(axes[0], "a")
    panel(axes[1], "b")
    save(fig, "F05_h3_differences")


def f06_timeseries(run="run1", fn="f7"):
    pm = per_minute(run)
    pm = pm[(pm.function == fn) & (pm.minute >= WARM)]
    hours = (pm.minute.unique() - WARM) / 60
    need = pm[pm.policy == "pach"].requests.to_numpy() / MU
    fig, axes = plt.subplots(4, 1, figsize=(W2, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.42)
    for ax, p in zip(axes, POLICIES):
        g = pm[pm.policy == p].sort_values("minute")
        ax.step(hours, need, where="post", color="#9a9893", lw=0.8, label="demand / mu (pods needed)")
        ax.step(hours, g.ready_lo, where="post", color=COLOR[p], lw=1.1, label="ready endpoints (minute mean)")
        ov = hours[g.overload_status.to_numpy() == 1]
        ax.plot(ov, np.full(len(ov), -0.5), "|", color=CRIT, ms=7, mew=1.2, label="overload minute")
        node = meta(run)["assignment"][p]
        ax.set_title(f"{LABEL[p]} (worker {node}): {len(ov)} overload minutes, {g.ready_lo.sum():,.0f} ready pod-minutes", loc="left")
        ax.set_ylim(-1.2, max(need.max(), pm.ready_lo.max()) + 1)
        ax.set_ylabel("pods")
        top = int(max(need.max(), pm.ready_lo.max())) + 1
        ax.set_yticks(range(0, top + 1, 2 if top > 6 else 1))
    h = [Line2D([], [], color="#9a9893", lw=1, label="demand / mu (pods needed)"), Line2D([], [], color=INK2, lw=1.2, label="ready endpoints, minute mean (colour = policy)"),
         Line2D([], [], color=CRIT, marker="|", ls="", ms=7, mew=1.2, label="overload minute")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.9))
    fig.subplots_adjust(top=0.88)
    axes[-1].set_xlabel(f"hours of test day {meta(run)['day']} ({run}, function {fn})")
    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(0, 25, 3))
    save(fig, f"F06_timeseries_{fn}_{run}")


def f07_node_effect():
    cpu = pd.read_csv(TABLES / "T10_cpu_per_request.csv")
    users = pd.read_csv(TABLES / "X03_node_users.csv")
    inv = TABLE_INV
    cal = pd.read_csv(TABLES / "T10b_cpu_by_node.csv").set_index("node").calibration_phase2
    nodes = ["w1", "w2", "w3", "w4"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.4))
    off = dict(zip(RUNS, [-0.21, -0.07, 0.07, 0.21]))
    for _, r in cpu.iterrows():
        p = inv[r.policy]
        a.plot(nodes.index(r.node) + off[r.run], r.cpu_ms_per_request, marker=MARKER[p], color=COLOR[p], ls="", ms=5, mec="white", mew=0.5)
    a.plot(range(4), [cal[n] for n in nodes], ls="", marker="_", color=INK, ms=14, mew=1.4, label="worker calibration")
    a.legend(loc="upper right")
    a.axhline(3.735, color=INK2, ls="--", lw=0.8)
    a.text(1.5, 3.75, "common calibration 3.735 ms", fontsize=6.3, color=INK2, ha="center", va="bottom")
    a.set_xticks(range(4))
    a.set_xticklabels(nodes)
    a.set_ylabel("CPU time per request (ms)")
    a.set_title("CPU time per request", loc="left")
    for _, r in users.iterrows():
        p = inv[r.policy]
        b.plot(nodes.index(r.node) + off[r.run], r.bad_pct, marker=MARKER[p], color=COLOR[p], ls="", ms=5, mec="white", mew=0.5)
        b.annotate(r.run[-1], (nodes.index(r.node) + off[r.run], r.bad_pct), xytext=(3, 0), textcoords="offset points", fontsize=5.8,
                   color=INK2, va="center")
    b.set_xticks(range(4))
    b.set_xticklabels(nodes)
    b.set_ylabel("bad requests (%)")
    b.set_title("Bad requests (numbers = run)", loc="left")
    h = [Line2D([], [], color=COLOR[p], marker=MARKER[p], ls="", ms=5, label=LABEL[p]) for p in POLICIES]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    panel(a, "a")
    panel(b, "b")
    save(fig, "F07_node_effect")


def run_spans(ax, ymax):
    for r in RUNS:
        t0 = pd.Timestamp(meta(r)["t0"] + WARM * 60, unit="s", tz="UTC").tz_convert("Europe/Moscow")
        t1 = pd.Timestamp(meta(r)["t0"] + MINUTES * 60, unit="s", tz="UTC").tz_convert("Europe/Moscow")
        ax.axvspan(t0, t1, color="#f3f2ef", lw=0, zorder=0)
        ax.text(t0 + (t1 - t0) / 2, ymax, f"run {r[-1]}", ha="center", va="top", fontsize=6.5, color=INK2)


def date_axis(ax):
    if PAPER:
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0, 12], tz="Europe/Moscow"))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d.%m\n%H:%M", tz="Europe/Moscow"))
    else:
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d.%m %H:%M", tz="Europe/Moscow"))


def f08_cpu_hourly():
    h = pd.read_csv(DATA / "hourly_cpu_by_node.csv", parse_dates=["start_utc"])
    h["t"] = h.start_utc.dt.tz_convert("Europe/Moscow")
    fig, axes = plt.subplots(4, 1, figsize=(W2, 4.2), sharex=True, sharey=True)
    for ax, n in zip(axes, ["w1", "w2", "w3", "w4"]):
        run_spans(ax, 5.55)
        for r in RUNS:
            g = h[(h.node == n) & (h.run == r)].sort_values("t")
            p = g.policy.iloc[0]
            ax.plot(g.t, g.ms_per_request, color=COLOR[p], marker=MARKER[p], ms=2.2, lw=1)
        ax.axhline(3.735, color=INK2, ls="--", lw=0.7)
        ax.set_ylabel(f"{n}\nms / request" if not PAPER else n)
        ax.set_ylim(3.0, 5.6)
    if PAPER:
        fig.supylabel("CPU time per request (ms)", fontsize=7.5 * FS)
    else:
        axes[0].set_title("Post hoc: hourly median CPU time per request on each worker (colour = policy on that worker in that run; dashed: common calibration)", loc="left")
    h = [Line2D([], [], color=COLOR[p], marker=MARKER[p], ls="-", ms=4, label=LABEL[p]) for p in POLICIES]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07 if not PAPER else -0.11))
    date_axis(axes[-1])
    axes[-1].set_xlabel("time (Moscow)")
    save(fig, "F08_cpu_hourly_by_worker")


def f09_proxy_users():
    t = pd.read_csv(TABLES / "T14_proxy_vs_users_curve.csv")
    inv = TABLE_INV
    bins = list(dict.fromkeys(t.load_ratio))
    fig, (a, b) = plt.subplots(1, 2, figsize=(W2, 2.4))
    for pl, g in t.groupby("policy", sort=False):
        p = inv[pl]
        g = g.set_index("load_ratio").reindex(bins)
        ok = g.minutes >= 30
        x = np.arange(len(bins))
        a.plot(x[ok], g.bad_pct[ok], color=COLOR[p], marker=MARKER[p], ms=4)
        b.plot(x[ok], 100 * g.share_minutes_bad_gt1pct[ok], color=COLOR[p], marker=MARKER[p], ms=4)
    for ax in (a, b):
        ax.axvspan(4.5, len(bins) - 0.5, color="#fbe9e7", lw=0, zorder=0)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, rotation=30)
        ax.set_xlabel("load ratio = requests / (mu x ready capacity)" if not PAPER else "requests / (mu x ready capacity)")
        ax.text(4.6, ax.get_ylim()[1] * 0.97 if ax is a else 0, "", fontsize=6)
    a.set_ylabel("bad requests (%)")
    a.set_title("Bad-request share (bins with >= 30 minutes)" if not PAPER else "Bad requests", loc="left")
    b.set_ylabel("minutes with > 1% bad requests (%)" if not PAPER else "share of minutes (%)")
    b.set_title("Minutes with > 1% bad (shaded: proxy overload)" if not PAPER else "Minutes with > 1% bad requests", loc="left")
    policy_legend(a, loc="upper left", ncol=2)
    panel(a, "a")
    panel(b, "b")
    save(fig, "F09_proxy_vs_users")


def f10_drift():
    rows = []
    for r in RUNS:
        s = pd.read_csv(DATA / f"seconds_{r}.csv.gz")
        s["hour"] = (s.second - WARM * 60) // 3600
        s["bad"] = s[[f"bad_{p}" for p in POLICIES]].sum(axis=1)
        s["req"] = s[[f"dispatched_{p}" for p in POLICIES]].sum(axis=1)
        s["sync"] = np.where(s.policies_bad >= 3, s.bad, 0)
        g = s.groupby("hour").agg(bad=("bad", "sum"), req=("req", "sum"), sync=("sync", "sum")).reset_index()
        g["t"] = pd.to_datetime(meta(r)["t0"] + WARM * 60 + 3600 * g.hour, unit="s", utc=True).dt.tz_convert("Europe/Moscow")
        g["run"] = r
        rows.append(g)
    g = pd.concat(rows)
    fig, ax = plt.subplots(figsize=(W2, 2.4))
    run_spans(ax, 3.4)
    ax.bar(g.t, 100 * (g.bad - g.sync) / g.req, width=1 / 24 * 0.9, align="edge", color="#9a9893", label="other bad requests")
    ax.bar(g.t, 100 * g.sync / g.req, bottom=100 * (g.bad - g.sync) / g.req, width=1 / 24 * 0.9, align="edge", color=INK,
           label="in seconds with bad requests in >= 3 policies (3 workers) at once")
    ax.set_ylabel("bad requests, all policies (%)")
    ax.set_ylim(0, 3.6)
    date_axis(ax)
    ax.set_xlabel("time (Moscow), hourly bins")
    if not PAPER:
        ax.set_title("Post hoc: bad requests over the campaign, split into seconds with bad requests in >= 3 policies and the rest", loc="left")
    fig.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14 if not PAPER else -0.24))
    save(fig, "F10_drift_synchrony")


def f11_scale_events():
    t = pd.read_csv(TABLES / "X02_scale_events.csv")
    inv = TABLE_INV
    kinds = ["no change", "after scale-up", "after scale-down"]
    fig, ax = plt.subplots(figsize=(W1, 2.2))
    for i, pl in enumerate(["PAC-h", "Conformal", "B1", "HPA"]):
        p = inv[pl]
        g = t[t.policy == pl].set_index("kind").reindex(kinds)
        ax.bar(np.arange(3) + (i - 1.5) * 0.2, g.bad_pct, width=0.18, color=COLOR[p], label=LABEL[p])
    ax.set_xticks(range(3))
    ax.set_xticklabels(kinds)
    ax.set_ylabel("bad requests (%)")
    ax.set_title("Post hoc: bad requests by replica change", loc="left")
    ax.legend(loc="upper left", ncol=2)
    save(fig, "F11_scale_events")


if __name__ == "__main__":
    f06_timeseries(fn="f11")
    for f in (f01_delays, f02_overload_by_function, f03_cost_vs_overload, f04_h1, f05_h3, f06_timeseries, f07_node_effect, f08_cpu_hourly,
              f09_proxy_users, f10_drift, f11_scale_events):
        f()
