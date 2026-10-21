#!/usr/bin/env python3
"""Negative and positive tests of capacity.py, analyze.py and combine.py on synthetic data.

Every test works on a temporary copy of testbed/, so the repository is not touched.
Usage (from the repository root):  python3 testbed/tests/run_tests.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

TB = Path(__file__).resolve().parents[1]
SYNTH = TB / "tests" / "synth.py"
FAILED: list[str] = []


def run(*args: str, cwd: Path, ok: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    if ok and p.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("ok    " if cond else "FAIL  ") + name + ("" if cond else f": {detail}"))
    if not cond:
        FAILED.append(name)


def workspace() -> Path:
    root = Path(tempfile.mkdtemp(prefix="testbed-tests-"))
    shutil.copytree(TB, root / "testbed", ignore=shutil.ignore_patterns(
        "plan", "capacity", "runs", "results", "local_stage", "capacity.json", "actuator_model.json", "pod_iter.json",
        "cluster", "__pycache__"))
    return root


def analyze_case(root: Path, mode: str, minutes: int = 90, fit: bool = False) -> tuple[dict, pd.DataFrame, subprocess.CompletedProcess]:
    out = root / "results" / mode
    run(str(SYNTH), "run", str(root), str(out), mode, str(minutes), cwd=root)
    args = ["testbed/analyze.py", str(out)] + (["--fit-actuator"] if fit else [])
    p = run(*args, cwd=root, ok=False)
    v = json.loads((out / "validity.json").read_text()) if (out / "validity.json").exists() else {}
    f = pd.read_csv(out / "per_function.csv") if (out / "per_function.csv").exists() else pd.DataFrame()
    return v, f, p


def test_analyze() -> None:
    root = workspace()
    run(str(SYNTH), "plan", str(root), cwd=root)
    v, f, p = analyze_case(root, "good", fit=True)
    check("fit on a good run", p.returncode == 0 and (root / "testbed" / "actuator_model.json").exists(), p.stderr[-500:])
    model = json.loads((root / "testbed" / "actuator_model.json").read_text())
    check("fitted d_up between 18 and 20.5 s", 18 <= model["d_up_s"] <= 20.5, str(model["d_up_s"]))
    check("fitted d_down between 0 and 1.2 s", 0 <= model["d_down_s"] <= 1.2, str(model["d_down_s"]))
    att = model["attribution"]
    check("every created and removed pod attributed once",
          att["up_attributed"] == att["pods_created_in_test"] and att["down_attributed"] == att["pods_removed_in_test"], str(att))
    v, f, p = analyze_case(root, "good")
    check("good run valid", v.get("valid") is True, str(v.get("problems"))[:300])
    check("good run implementation exact", bool(f[f.policy != "hpa"].implementation_exact.all()))
    check("good run has H1 for every custom pair", set(f[f.policy != "hpa"].h1.dropna()) <= {"within", "outside", "undetermined"}
          and f[f.policy != "hpa"].h1.notna().all())
    expectations = {
        "gap_one_function": "pach-f1: telemetry coverage",
        "missing_series": "hpa-f2: telemetry coverage",
        "dropped": "dispatched differ from planned",
        "lag": "dispatch lag p99",
        "nodecisions": "decision log has",
        "partial": "load.csv incomplete",
        "ready_above_bound": "more ready endpoints than the replica bound",
    }
    for mode, needle in expectations.items():
        v, f, p = analyze_case(root, mode)
        problems = " | ".join(v.get("problems", []))
        check(f"{mode}: invalid without a crash", p.returncode == 0 and v.get("valid") is False and needle in problems,
              f"rc={p.returncode} {problems[:300]} {p.stderr[-300:]}")
    v, f, p = analyze_case(root, "gap_one_function")
    g = f[(f.policy == "pach") & (f.function == "f1")].iloc[0]
    check("gap gives unknown minutes and a wider interval",
          g.unknown_minutes > 0 and g.ready_pod_minutes_hi > g.ready_pod_minutes_lo and g.overload_hi >= g.overload_lo, g.to_dict())
    others = f[~((f.policy == "pach") & (f.function == "f1"))]
    check("gap affects only its own series", bool((others.ready_pod_minutes_hi == others.ready_pod_minutes_lo).all()))
    shutil.rmtree(root)


def write_run(d: Path, run: str, day: int, assignment: dict, rows: list[dict], valid: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "per_function.csv", index=False)
    (d / "summary.json").write_text(json.dumps({"run": run, "day": day, "assignment": assignment}))
    (d / "validity.json").write_text(json.dumps({"valid": valid, "problems": [] if valid else ["synthetic"]}))


def test_combine() -> None:
    root = workspace()
    sys.path.insert(0, str(root / "testbed"))
    from render import RUNS, assignment
    functions = [f"f{i}" for i in range(1, 13)]
    (root / "testbed" / "plan").mkdir()
    (root / "testbed" / "plan" / "plan.json").write_text(json.dumps({"functions": [{"name": f} for f in functions]}))

    def campaign(base: Path, intervals: dict, valid_all: bool = True, runs=RUNS) -> list[str]:
        dirs = []
        for i in runs:
            rows = [{"function": fn, "policy": pol, "overload_lo": intervals[pol][0], "overload_hi": intervals[pol][1],
                     "h2": "undetermined", "ready_pod_minutes_lo": 100.0, "ready_pod_minutes_hi": 100.0}
                    for fn in functions for pol in ("pach", "conf", "b1", "hpa")]
            d = base / f"run{i}"
            write_run(d, f"run{i}", RUNS[i]["day"], assignment(i), rows, valid=valid_all or i != 2)
            dirs.append(str(d))
        return dirs

    dirs = campaign(root / "counter", {"pach": (0, 10), "conf": (1, 11), "b1": (1, 11), "hpa": (0, 0)})
    p = run("testbed/combine.py", *dirs, cwd=root, ok=False)
    out = json.loads((root / "counter" / "combined.json").read_text()) if p.returncode == 0 else {}
    c = out.get("h3", {}).get("pach_vs_conf", {})
    check("H3 interval counterexample is not supported",
          c.get("conclusion") == "undetermined" and c.get("sign_test_worst_case", {}).get("p_holm", 0) >= 0.05
          and c.get("mean_difference_interval") == [-11.0, 9.0], json.dumps(c)[:400] + p.stderr[-300:])
    dirs = campaign(root / "clear", {"pach": (0, 0), "conf": (3, 3), "b1": (5, 6), "hpa": (0, 0)})
    out = json.loads(run("testbed/combine.py", *dirs, cwd=root) and (root / "clear" / "combined.json").read_text())
    check("H3 with known intervals and a clear difference is supported",
          out["h3"]["pach_vs_conf"]["conclusion"] == "supported" and out["confirmatory"] is True, json.dumps(out["h3"])[:300])
    dirs = campaign(root / "invalid", {"pach": (0, 0), "conf": (3, 3), "b1": (5, 6), "hpa": (0, 0)}, valid_all=False)
    p = run("testbed/combine.py", *dirs, cwd=root, ok=False)
    check("combine refuses a campaign with an invalid run", p.returncode != 0 and "run2 is invalid" in (p.stdout + p.stderr))
    p = run("testbed/combine.py", *dirs, "--descriptive", cwd=root, ok=False)
    check("combine --descriptive marks the output", p.returncode == 0 and (root / "invalid" / "combined_descriptive.json").exists()
          and not (root / "invalid" / "combined.json").exists())
    dirs = campaign(root / "three", {"pach": (0, 0), "conf": (3, 3), "b1": (5, 6), "hpa": (0, 0)}, runs={k: RUNS[k] for k in (1, 2, 3)})
    p = run("testbed/combine.py", *dirs, cwd=root, ok=False)
    check("combine refuses three runs", p.returncode != 0 and "instead of" in (p.stdout + p.stderr))
    shutil.rmtree(root)


def test_capacity() -> None:
    root = workspace()
    tb = root / "testbed"

    def cap() -> dict:
        return json.loads((tb / "capacity.json").read_text())

    def synth_phase(phase: str, mode: str) -> Path:
        out = root / "results" / f"{phase}-{mode}"
        shutil.rmtree(out, ignore_errors=True)
        run(str(SYNTH), "capacity", str(tb / "capacity" / phase), str(out), mode, cwd=root)
        return out

    run("testbed/capacity.py", "plan1", cwd=root)
    p = run("testbed/capacity.py", "analyze1", "--results", str(synth_phase("phase1", "slowcpu")), cwd=root, ok=False)
    it = json.loads((tb / "pod_iter.json").read_text()) if (tb / "pod_iter.json").exists() else {}
    check("capacity: 7 ms per request halves ITER and asks for phase 1 again", p.returncode == 3 and 12500 <= it.get("iter", 0) <= 13300, p.stdout + str(it))
    p = run("testbed/capacity.py", "analyze1", "--results", str(synth_phase("phase1", "good")), cwd=root, ok=False)
    check("capacity: phase 1 planned with the old ITER is refused", p.returncode != 0 and "another ITER" in p.stderr)
    run("testbed/capacity.py", "plan1", cwd=root)
    p = run("testbed/capacity.py", "analyze1", "--results", str(synth_phase("phase1", "missing_second")), cwd=root, ok=False)
    check("capacity: missing seconds rejected", p.returncode != 0 and "incomplete" in p.stderr)
    p = run("testbed/capacity.py", "analyze1", "--results", str(synth_phase("phase1", "lost")), cwd=root, ok=False)
    check("capacity: undelivered requests rejected", p.returncode != 0 and "incomplete" in p.stderr)
    run("testbed/capacity.py", "analyze1", "--results", str(synth_phase("phase1", "good")), cwd=root)
    check("capacity: ITER kept when one request costs 3.5 ms", json.loads((tb / "pod_iter.json").read_text())["iter"] == it.get("iter"))
    r_star = json.loads((tb / "capacity" / "phase1_result.json").read_text())["r_star"]
    check("capacity: r* = 30 on a perfect phase 1", r_star == 30, str(r_star))
    run("testbed/capacity.py", "plan2", cwd=root)
    run("testbed/capacity.py", "analyze2", "--results", str(synth_phase("phase2", "good")), cwd=root)
    check("capacity: candidate is not verified", cap()["verified"] is False and cap()["rate_per_second"] == r_star + 1)
    candidate = (tb / "capacity.json").read_text()
    (tb / "plan").mkdir(exist_ok=True)

    def sizing() -> None:
        (tb / "plan" / "sizing.json").write_text(json.dumps({"mu_phys_per_minute": cap()["mu_phys_per_minute"],
                                                             "policies": {"pach": {"peak_pods": 66}, "b1": {"peak_pods": 60}}}))
    sizing()
    run("testbed/capacity.py", "plan3", cwd=root)
    meta = json.loads((tb / "capacity" / "phase3" / "meta.json").read_text())
    check("capacity: phase 3 loads the peak cohort", meta["peak_pods"] == 66 and sum(meta["deployments"].values()) == 66)
    run("testbed/capacity.py", "analyze3", "--results", str(synth_phase("phase3", "good")), cwd=root)
    check("capacity: phase 3 verifies mu", cap()["verified"] is True and cap()["rate_per_second"] == r_star + 1)
    (tb / "capacity.json").write_text(candidate)
    (tb / "plan" / "sizing.json").write_text(json.dumps({"mu_phys_per_minute": 1, "policies": {"pach": {"peak_pods": 66}}}))
    p = run("testbed/capacity.py", "plan3", cwd=root, ok=False)
    check("capacity: phase 3 refuses sizing for another mu", p.returncode != 0)
    lowered = []
    notes = []
    for _ in range(6):
        sizing()
        run("testbed/capacity.py", "plan3", cwd=root)
        p = run("testbed/capacity.py", "analyze3", "--results", str(synth_phase("phase3", "cohort_fail")), cwd=root, ok=False)
        lowered.append(cap()["rate_per_second"])
        notes.append(cap()["note"])
        if p.returncode != 0:
            break
    check("capacity: failures lower mu by 1 down to r*-2, repeat r*-2 once (A1), then stop",
          lowered == [r_star, r_star - 1, r_star - 2, r_star - 2, r_star - 2] and p.returncode != 0
          and "amendment A1" in notes[3] and cap()["verified"] is False, str(lowered) + str(notes))
    shutil.rmtree(root)


def test_cpu_snapshot() -> None:
    root = workspace()
    sys.path.insert(0, str(root / "testbed"))
    import importlib
    analyze = importlib.import_module("analyze")
    t0 = 1_800_000_000.0
    per_minute = pd.DataFrame({"policy": ["pach"], "function": ["f1"], "minute": [0], "requests": [600.0]})
    for name, cols in (("old", ["t", "namespace", "pod", "function", "window_s", "millicores"]),
                       ("new", ["t", "snapshot", "namespace", "pod", "function", "metrics_timestamp", "window_s", "millicores"])):
        d = root / name
        d.mkdir()
        rows = []
        for k in range(4):
            for j, pod in enumerate(("a", "b")):
                t = t0 + 15 * k + 1.0 + 0.001 * j
                r = {"t": t if name == "old" else t0 + 15 * k + 1.0, "snapshot": k, "namespace": "pach", "pod": pod,
                     "function": "f1", "metrics_timestamp": "", "window_s": "15s", "millicores": 10.0}
                rows.append({c: r[c] for c in cols})
        pd.DataFrame(rows).to_csv(d / "cpu.csv", index=False)
        got = analyze.cpu_per_request(d, per_minute, t0, slice(0, 1)).get("pach")
        check(f"CPU of two pods in one snapshot is summed ({name} format)", got is not None and abs(got - 2.0) < 1e-9, str(got))
    shutil.rmtree(root)


def test_render_pull_secret() -> None:
    import os
    root = workspace()
    run("testbed/capacity.py", "plan1", cwd=root)

    def pods(secret: str) -> list[dict]:
        env = {k: v for k, v in os.environ.items() if k != "TESTBED_PULL_SECRET"}
        if secret:
            env["TESTBED_PULL_SECRET"] = secret
        subprocess.run([sys.executable, "testbed/render.py", "--registry", "cr.example/r", "--start", "0", "--capacity", "phase1"],
                       cwd=root, env=env, check=True, capture_output=True)
        items = json.loads((root / "testbed" / "runs" / "capacity_phase1" / "manifests.yaml").read_text())["items"]
        return [d["spec"]["template"]["spec"] if d["kind"] == "Deployment" else d["spec"] for d in items if d["kind"] in ("Deployment", "Pod")]
    with_secret, without = pods("testbed-pull"), pods("")
    check("render: every pod names the pull secret of a private registry",
          len(with_secret) > 4 and all(s.get("imagePullSecrets") == [{"name": "testbed-pull"}] for s in with_secret))
    check("render: no pull secret for a public registry", all("imagePullSecrets" not in s for s in without))
    shutil.rmtree(root)


def test_freeze_verify() -> None:
    root = workspace()
    tb = root / "testbed"
    (tb / "results").mkdir()
    sys.path.insert(0, str(tb))
    import importlib
    freeze = importlib.import_module("freeze")
    (tb / "FREEZE.json").write_text(json.dumps({"sha256": freeze.inputs()}))
    p = run("testbed/freeze.py", "--verify", cwd=root, ok=False)
    check("freeze --verify accepts unchanged inputs", p.returncode == 0, p.stdout + p.stderr)
    (tb / "results" / "x.csv").write_text("1")
    p = run("testbed/freeze.py", "--verify", cwd=root, ok=False)
    check("freeze --verify ignores results/", p.returncode == 0, p.stdout + p.stderr)
    with open(tb / "policies.py", "a") as f:
        f.write("\n")
    p = run("testbed/freeze.py", "--verify", cwd=root, ok=False)
    check("freeze --verify detects a changed input", p.returncode != 0 and "changed: testbed/policies.py" in p.stdout, p.stdout)
    (tb / "new_input.py").write_text("")
    p = run("testbed/freeze.py", "--verify", cwd=root, ok=False)
    check("freeze --verify detects a new input", p.returncode != 0 and "new: testbed/new_input.py" in p.stdout, p.stdout)
    shutil.rmtree(root)


if __name__ == "__main__":
    test_freeze_verify()
    test_render_pull_secret()
    test_cpu_snapshot()
    test_capacity()
    test_combine()
    test_analyze()
    print(f"\n{len(FAILED)} failed" + (": " + ", ".join(FAILED) if FAILED else ""))
    sys.exit(1 if FAILED else 0)
