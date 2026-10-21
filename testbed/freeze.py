#!/usr/bin/env python3
"""Record the SHA-256 of every testbed input in FREEZE.json before the main runs.

Requires, besides the committed code: plan/ (from prepare.py with the cloud capacity),
capacity.json, actuator_model.json, and cluster/ with images.json (image digests as pushed),
cluster.json (Kubernetes and metrics-server versions, nodes and their testbed-node labels), and
toolchain.txt (versions of Go, Python, NumPy, kubectl). capacity.json must be verified by phase 3
and plan/ prepared with its mu. Generated run folders, results and the local-stage archive are
not part of the freeze.

Admission to the main runs needs only a verified capacity and a technically valid cloud pilot
(results/pilot/validity.json); no result of H1-H3 is a condition.

Usage (from the repository root, after committing testbed/):
  python3 testbed/freeze.py            write FREEZE.json
  python3 testbed/freeze.py --verify   compare every input with FREEZE.json (before each main run)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "testbed"
REQUIRED = ("capacity.json", "actuator_model.json", "plan/plan.json", "plan/sizing.json",
            "cluster/images.json", "cluster/cluster.json", "cluster/toolchain.txt")
EXCLUDED_DIRS = {"results", "runs", "local_stage", "__pycache__"}


def inputs() -> dict[str, str]:
    files = sorted(p for p in TB.rglob("*") if p.is_file() and p.name != "FREEZE.json"
                   and not EXCLUDED_DIRS.intersection(p.relative_to(TB).parts))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def verify() -> None:
    frozen = json.loads((TB / "FREEZE.json").read_text())["sha256"]
    now = inputs()
    changed = sorted(k for k in frozen if k in now and now[k] != frozen[k])
    missing = sorted(set(frozen) - set(now))
    extra = sorted(set(now) - set(frozen))
    for label, keys in (("changed", changed), ("missing", missing), ("new", extra)):
        for k in keys:
            print(f"{label}: {k}")
    if changed or missing or extra:
        raise SystemExit("inputs differ from FREEZE.json")
    print(f"{len(now)} inputs identical to FREEZE.json")


def main() -> None:
    if sys.argv[1:] == ["--verify"]:
        verify()
        return
    missing = [p for p in REQUIRED if not (TB / p).exists()]
    if missing:
        raise SystemExit(f"missing before freeze: {missing}")
    cap = json.loads((TB / "capacity.json").read_text())
    plan = json.loads((TB / "plan" / "plan.json").read_text())
    sizing = json.loads((TB / "plan" / "sizing.json").read_text())
    if not cap.get("verified"):
        raise SystemExit("capacity.json is not verified by phase 3")
    if not cap["mu_phys_per_minute"] == plan["mu_phys_per_minute"] == sizing["mu_phys_per_minute"]:
        raise SystemExit("plan/ was not prepared with the verified capacity; rerun prepare.py")
    pilot = TB / "results" / "pilot" / "validity.json"
    if not pilot.exists() or not json.loads(pilot.read_text()).get("valid"):
        raise SystemExit("no technically valid cloud pilot in results/pilot")
    dirty = subprocess.run(["git", "status", "--porcelain", "testbed"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("commit testbed/ first:\n" + dirty)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = inputs()
    out = {"commit": commit, "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "images": json.loads((TB / "cluster" / "images.json").read_text()), "sha256": sha}
    (TB / "FREEZE.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{len(sha)} files at commit {commit}; commit FREEZE.json next")


if __name__ == "__main__":
    main()
