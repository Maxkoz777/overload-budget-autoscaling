"""Shared paths, labels and loaders for the testbed analysis (read-only on testbed/)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent
TB = PKG.parents[1] / "testbed"
RES = TB / "results"
OUT = Path(os.environ.get("TESTBED_ANALYSIS_OUT", PKG / "output"))
TABLES, FIGURES, DATA = OUT / "tables", OUT / "figures", OUT / "data"
for _d in (TABLES, FIGURES, DATA):
    _d.mkdir(parents=True, exist_ok=True)
RUNS = ["run1", "run2", "run3", "run4"]
POLICIES = ["pach", "conf", "b1", "hpa"]
CUSTOM = ["pach", "conf", "b1"]
LABEL = {"pach": "PAC-h", "conf": "Conformal", "b1": "B1", "hpa": "HPA"}
COLOR = {"pach": "#2a78d6", "conf": "#eb6834", "b1": "#1baf7a", "hpa": "#eda100"}
MARKER = {"pach": "o", "conf": "s", "b1": "^", "hpa": "D"}
NODE_OF = {"w1": "testbed-workers-4", "w2": "testbed-workers-2", "w3": "testbed-workers-1", "w4": "testbed-workers-3",
           "load": "testbed-workers-0"}
WARM, MINUTES = 30, 1470
TEST_MINUTES = MINUTES - WARM
MU = 240.0
BUDGET = 14.4
MSK = "Europe/Moscow"


def jload(p: Path):
    return json.loads(Path(p).read_text())


def plan() -> dict:
    return jload(TB / "plan" / "plan.json")


def meta(run: str) -> dict:
    return jload(RES / run / "meta.json")


def summary(run: str) -> dict:
    return jload(RES / run / "summary.json")


def validity(run: str) -> dict:
    return jload(RES / run / "validity.json")


def combined() -> dict:
    return jload(RES / "combined.json")


def per_function() -> pd.DataFrame:
    return pd.read_csv(RES / "combined_per_function.csv")


def per_minute(run: str) -> pd.DataFrame:
    pm = pd.read_csv(RES / run / "per_minute.csv.gz")
    pm["run"] = run
    return pm


def delays(run: str) -> pd.DataFrame:
    d = pd.read_csv(RES / run / "delays.csv")
    d["run"] = run
    return d


def node_of(run: str, policy: str) -> str:
    return meta(run)["assignment"].get(policy, "")


def esc(t: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"), ("<=", r"$\le$"), (">=", r"$\ge$"),
                 ("<", r"$<$"), (">", r"$>$"), ("+/-", r"$\pm$")):
        t = t.replace(a, b)
    return t


def to_tex(df: pd.DataFrame, path: Path, caption: str, label: str, fmt: dict | None = None, index: bool = False) -> None:
    """Write a booktabs LaTeX table without external dependencies."""
    fmt = fmt or {}
    d = df.reset_index() if index else df
    cols = list(d.columns)

    def cell(c, v):
        if isinstance(v, float) and np.isnan(v):
            return "--"
        if c in fmt:
            return fmt[c].format(v)
        if isinstance(v, (float, np.floating)):
            if float(v).is_integer():
                return str(int(v))
            return f"{v:.1f}" if abs(v) >= 100 else f"{v:.4g}"
        return esc(str(v))

    align = "l" + "r" * (len(cols) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(esc(str(c)) for c in cols) + r" \\", r"\midrule"]
    for _, r in d.iterrows():
        lines.append(" & ".join(cell(c, r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def save_table(df: pd.DataFrame, name: str, caption: str, fmt: dict | None = None) -> None:
    df.to_csv(TABLES / f"{name}.csv", index=False)
    to_tex(df, TABLES / f"{name}.tex", caption, f"tab:{name}", fmt)
