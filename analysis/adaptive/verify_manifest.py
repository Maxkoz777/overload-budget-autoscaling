"""Preflight: compare the files of this folder and the data inputs with the frozen v1 manifest.

Usage: OBA_REPO=<repository root> python3 verify_manifest.py
Exit code 0 when every frozen code file, protocol file, data input, repository module and
selected-parameter file matches the manifest, which is also what run.py requires before a test run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from adaptive.data import REPO

HERE = Path(__file__).resolve().parent


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    m = json.loads((HERE / "manifest_v1.json").read_text())
    bad = []
    for name, h in m["protocol"].items():
        if sha(HERE / name) != h:
            bad.append(("protocol", name))
    for name, h in m["selected_parameters"].items():
        if sha(HERE / "out_v1" / name) != h:
            bad.append(("selected parameters", name))
    data = REPO / "experiments" / "data"
    for name, h in m["data_inputs"].items():
        p = data / name
        if not p.exists() or sha(p) != h:
            bad.append(("data input", name))
    audit = REPO / "analysis" / "audit"
    for name, h in m["repository_audit_modules"].items():
        if sha(audit / name) != h:
            bad.append(("repository module", name))
    changed = [c for c, h in m["code"].items() if sha(HERE / c) != h]
    print(f"protocol {len(m['protocol'])}, parameters {len(m['selected_parameters'])}, data inputs {len(m['data_inputs'])}, "
          f"repository modules {len(m['repository_audit_modules'])}, code {len(m['code'])}")
    for kind, name in bad:
        print("MISMATCH", kind, name)
    for c in changed:
        print("MISMATCH code", c)
    ok = not bad and not changed
    print("PREFLIGHT", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
