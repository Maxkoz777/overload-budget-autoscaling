#!/usr/bin/env python3
"""R10-R12: canonical capacity units, matched-rank ablation, and function-clustered
sensitivity of the confirmatory sign test.

R10  Re-runs every analysis on the Huawei 2023 and Azure 2019 traces with the canonical
     capacity-unit rule of ``canonical_inputs.py`` (full-precision parsing of
     ``units.csv`` and exact-tie resolution). The frozen scripts are imported unchanged;
     their outputs are redirected to ``verified_results_canonical/`` so the frozen
     results stay untouched. Their built-in regression checks compare against earlier
     outputs computed without the rule, so mismatches are recorded instead of stopping
     the run (``r10_check_log.csv``) and every changed summary row is listed in
     ``r10_changed_rows.csv``. Alibaba uses mu = 1 and is copied, not re-run.
R11  Matched-rank ablation (post hoc): PAC-h with the PAC rank replaced by the standard
     conformal rank ceil((W+1)(1-delta)), keeping persistence forecasts, horizon-aligned
     scores, the pre-test-selected window W, and no offset.
R12  Sensitivity of confirmatory hypothesis H1 to the unit of analysis: discordant pairs
     per period, and a sign test with each function as one cluster (all its periods
     together). The frozen test itself is unchanged.

Usage (from the paper/analysis directory):
    python3 audit/canonical_rerun.py [ext r2 r4 r6 r7 r8 r9 r11 r12 compare]
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from canonical_inputs import install  # noqa: E402

install()

import external_replication as ext  # noqa: E402
import external_replication_summary as exs  # noqa: E402
import delay_and_reactive_grid as rev  # noqa: E402
import external_delay as xd  # noqa: E402
import pac_rank_replay as r6  # noqa: E402
import pac_window_grid as r7  # noqa: E402
import confirmatory_replay as r8  # noqa: E402
import pac_stride_sensitivity as r9  # noqa: E402

PAPER = HERE.parent
OLD = HERE / "verified_results"
NEW = HERE / "verified_results_canonical"
EXT_DATA = PAPER.parent / "experiments" / "data" / "external_traces"
FAMILIES = ("huawei2023", "azure2019")
EXT_RUNS = (("huawei2023", "primary"), ("huawei2023", "p90"), ("azure2019", "primary"), ("azure2019", "K5"), ("azure2019", "K20"))
CHECKS: list[dict] = []


# Output redirection
def redirect() -> None:
    if not NEW.exists():
        shutil.copytree(OLD, NEW)
    rev_out = NEW / "delay_pac_study"
    ext_out = NEW / "external_replication_study"
    rev.VER = xd.VER = NEW
    for mod in (rev, xd, r7, r8, r9):
        mod.OUT = rev_out
    ext.OUT = exs.OUT = ext_out

    def record(label, got, want, tol=0.0):
        if isinstance(want, (int, np.integer)) and not isinstance(want, bool):
            ok = got == want
        else:
            ok = abs(float(got) - float(want)) <= max(tol, 0.0)
        CHECKS.append({"label": label, "got": got, "expected": want, "ok": bool(ok)})
        print(f"  [{'OK' if ok else 'DIFF'}] {label}: got {got}, earlier {want}", flush=True)

    rev.check = record
    xd.check = record


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest() -> dict:
    files = sorted(EXT_DATA.rglob("units.csv")) + sorted(EXT_DATA.rglob("series.npz"))
    files += [PAPER.parent / "experiments" / "data" / "splits" / "split_definition.json"]
    return {str(p.relative_to(PAPER.parent)): sha(p) for p in files if p.exists()}


# R10 stages
def stage_ext() -> None:
    for fam, mu in EXT_RUNS:
        ext.run_family(fam, mu)
    exs.summarise()


def stage_r2() -> None:
    rev.run_r2()


def stage_r4() -> None:
    for fam in FAMILIES:
        xd.run(fam).to_csv(xd.OUT / f"r4_delay_{fam}_per_unit.csv.gz", index=False)
    allper = pd.concat([pd.read_csv(xd.OUT / f"r4_delay_{f}_per_unit.csv.gz") for f in FAMILIES], ignore_index=True)
    xd.summarise(allper).to_csv(xd.OUT / "r4_delay_external_summary.csv", index=False)


def stage_r6() -> None:
    for fam in FAMILIES:
        r6.run_external(fam).to_csv(rev.OUT / f"r6_pac_{fam}_per_unit.csv.gz", index=False)
    r6.summarise().to_csv(rev.OUT / "r6_pac_summary.csv", index=False)


def stage_r7() -> None:
    for ds in FAMILIES:
        test = r7.run_grid(ds, "test", r7.TAUS, {"1", "h"}, r7.ETAS)
        cal = r7.run_grid(ds, "cal", (0,), {"1"}, (0.05,))
        pd.concat([test, cal]).to_csv(r7.OUT / f"r7_pac_grid_{ds}.csv.gz", index=False)
    grid = pd.concat([pd.read_csv(r7.OUT / f"r7_pac_grid_{ds}.csv.gz") for ds in r7.DATASETS])
    s, sel = r7.summarise(grid)
    s.to_csv(r7.OUT / "r7_pac_grid_summary.csv", index=False)
    sel.to_csv(r7.OUT / "r7_window_selection.csv", index=False)
    r7.comparator_conservativeness().to_csv(r7.OUT / "r7_comparator_conservativeness.csv", index=False)


def stage_r8() -> None:
    r8.main()


def stage_r9() -> None:
    r9.main()


# R11 matched-rank ablation
def _rank_rows(D, W, delta, rank, horizon, taus=(0, 1, 5)):
    maxtau = max(taus)
    forecast = D["y"][:, D["t0"] - maxtau - 1: D["t0"] + D["T"] - 1]
    out = {}
    for tau in taus:
        h = tau + 1 if horizon else 1
        m = r7.margins(D, W, h, (rank,), start=-maxtau)[rank]
        nom = np.maximum(np.ceil((forecast + m) / D["mu"][:, None]), 1.0)
        out[tau] = r7.evaluate(D, nom, tau, maxtau)
    return out


def stage_r11() -> None:
    sel7 = pd.read_csv(r7.OUT / "r7_window_selection.csv")
    sel8 = pd.read_csv(r8.OUT / "r8_window_selection.csv")
    jobs = [(ds, (lambda ds=ds: r7.load(ds, "test")), {r.budget: int(r.selected_W) for r in sel7[sel7.dataset == ds].itertuples()})
            for ds in r7.DATASETS]
    jobs += [(run, (lambda run=run: r8.phase(run, "test")), {r.budget: int(r.selected_W) for r in sel8[sel8.run == run].itertuples()})
             for run in r8.RUNS]
    rows = []
    for tag, loader, wsel in jobs:
        D = loader()
        for delta, W in sorted(wsel.items()):
            ranks = {"PAC": r7.pac_rank(W, delta, 0.05), "conformal": ext.conf_rank(W, delta)}
            for kind, rank in ranks.items():
                for tau, (frac, rel) in _rank_rows(D, W, delta, rank, True).items():
                    for k, u in enumerate(D["uid"]):
                        rows.append((tag, delta, W, kind, rank, tau, u, float(frac[k]), float(rel[k])))
        print(f"  R11 {tag} done", flush=True)
    per = pd.DataFrame(rows, columns=["dataset", "budget", "W", "rank_kind", "rank", "tau", "unit_id", "overload_fraction", "relative_cost"])
    per["within"] = per.overload_fraction <= per.budget
    per["group"] = np.where(per.dataset.isin(r8.RUNS), "huawei_confirmatory", per.dataset)
    per.to_csv(NEW / "r11_matched_rank_per_unit.csv.gz", index=False)
    summ = per.groupby(["group", "budget", "rank_kind", "tau"]).agg(
        n=("unit_id", "size"), compliant=("within", "sum"), mean_relative_cost=("relative_cost", "mean"),
        mean_overload=("overload_fraction", "mean"), W=("W", lambda w: "/".join(str(x) for x in sorted(set(w)))),
        rank=("rank", lambda r: "/".join(str(x) for x in sorted(set(r))))).reset_index()
    summ.to_csv(NEW / "r11_matched_rank_summary.csv", index=False)
    piv = summ.pivot_table(index=["group", "budget", "tau"], columns="rank_kind", values=["compliant", "mean_relative_cost"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv["pac_cost_premium_pct"] = 100 * (piv.mean_relative_cost_PAC / piv.mean_relative_cost_conformal - 1)
    piv.reset_index().to_csv(NEW / "r11_matched_rank_comparison.csv", index=False)
    print(piv.reset_index().to_string(index=False))


# R12 clustered sign test
def stage_r12() -> None:
    out = []
    for label, folder in (("frozen", OLD / "delay_pac_study"), ("canonical", NEW / "delay_pac_study")):
        per = pd.read_csv(folder / "r8_confirmatory_per_unit.csv.gz")
        for budget in r8.BUDGETS:
            for tau in r8.TAUS:
                a = per[(per.policy == "PAC-h (pre-test W)") & (per.budget == budget) & (per.tau == tau)].set_index(["run", "unit_id"])
                b = per[(per.policy == "Reactive strict") & (per.budget == budget) & (per.tau == tau)].set_index(["run", "unit_id"]).loc[a.index]
                d = (a.within.astype(int) - b.within.astype(int)).rename("d").reset_index()
                byrun = d.groupby("run").d.agg(wins=lambda x: int((x > 0).sum()), losses=lambda x: int((x < 0).sum()))
                fn = d.groupby("unit_id").d.sum()
                fw, fl = int((fn > 0).sum()), int((fn < 0).sum())
                uw = int((d.d > 0).sum()); ul = int((d.d < 0).sum())
                wins_per_fn = d[d.d > 0].groupby("unit_id").size().value_counts().sort_index()
                out.append({"results": label, "budget": budget, "tau": tau, "unit_runs": len(d), "functions": int(d.unit_id.nunique()),
                            "unitrun_wins": uw, "unitrun_losses": ul,
                            "p_unitrun_nominal": binomtest(uw, uw + ul, alternative="greater").pvalue if uw + ul else 1.0,
                            "per_period": ";".join(f"{r}:{int(w)}:{int(l)}" for r, (w, l) in byrun.iterrows()),
                            "function_wins": fw, "function_losses": fl,
                            "p_function_cluster": binomtest(fw, fw + fl, alternative="greater").pvalue if fw + fl else 1.0,
                            "wins_per_winning_function": ";".join(f"{k}x:{v}" for k, v in wins_per_fn.items())})
    res = pd.DataFrame(out)
    res.to_csv(NEW / "r12_confirmatory_clustered.csv", index=False)
    print(res.to_string(index=False))


# Comparison report
SUMMARIES = [
    ("external_replication_study/summary_policies.csv", ["family", "mu", "budget", "policy"]),
    ("delay_pac_study/r2_extended_grid_summary.csv", None),
    ("delay_pac_study/r4_delay_external_summary.csv", None),
    ("delay_pac_study/r6_pac_summary.csv", None),
    ("delay_pac_study/r7_pac_grid_summary.csv", None),
    ("delay_pac_study/r7_window_selection.csv", None),
    ("delay_pac_study/r8_window_selection.csv", None),
    ("delay_pac_study/r8_confirmatory_summary.csv", None),
    ("delay_pac_study/r8_confirmatory_hypotheses.csv", None),
    ("delay_pac_study/r9_stride_summary.csv", None),
    ("delay_pac_study/r9_cost_rule_selection.csv", None),
    ("delay_pac_study/r9_cost_rule_summary.csv", None),
]


def stage_compare() -> None:
    rows = []
    for rel, _ in SUMMARIES:
        a, b = OLD / rel, NEW / rel
        if not (a.exists() and b.exists()):
            rows.append({"file": rel, "row": -1, "column": "missing", "frozen": a.exists(), "canonical": b.exists()})
            continue
        x, y = pd.read_csv(a), pd.read_csv(b)
        if x.shape != y.shape:
            rows.append({"file": rel, "row": -1, "column": "shape", "frozen": str(x.shape), "canonical": str(y.shape)})
            continue
        for c in x.columns:
            if pd.api.types.is_numeric_dtype(x[c]) and x[c].dtype != bool:
                xv, yv = x[c].to_numpy(float), y[c].to_numpy(float)
                diff = ~(np.isclose(xv, yv, rtol=0, atol=1e-9) | (np.isnan(xv) & np.isnan(yv)))
            else:
                diff = (x[c].astype(str) != y[c].astype(str)).to_numpy()
            for i in np.flatnonzero(diff):
                key = {k: x.at[i, k] for k in x.columns if not pd.api.types.is_float_dtype(x[k])}
                rows.append({"file": rel, "row": int(i), "column": c, "frozen": x.at[i, c], "canonical": y.at[i, c],
                             "key": json.dumps({k: str(v) for k, v in key.items()})})
    rep = pd.DataFrame(rows)
    rep.to_csv(NEW / "r10_changed_rows.csv", index=False)
    print(f"{len(rep)} changed cells")
    if len(rep):
        print(rep.groupby(["file", "column"]).size().to_string())


STAGES = {"ext": stage_ext, "r2": stage_r2, "r4": stage_r4, "r6": stage_r6, "r7": stage_r7, "r8": stage_r8,
          "r9": stage_r9, "r11": stage_r11, "r12": stage_r12, "compare": stage_compare}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(STAGES)
    redirect()
    for name in todo:
        print(f"=== {name}", flush=True)
        STAGES[name]()
        if CHECKS:
            pd.DataFrame(CHECKS).to_csv(NEW / "r10_check_log.csv", mode="a", header=not (NEW / "r10_check_log.csv").exists(), index=False)
            CHECKS.clear()
    meta = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "stages": todo, "canonical_rule": "canonical_inputs.py (round_trip parsing; mu *= 1 + 1e-12)",
            "inputs_sha256": manifest()}
    (NEW / f"methodology_r10_{'_'.join(todo)}.json").write_text(json.dumps(meta, indent=2) + "\n")
