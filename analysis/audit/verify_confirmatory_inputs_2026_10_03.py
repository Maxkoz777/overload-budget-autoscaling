#!/usr/bin/env python3
"""Check the prepared confirmatory inputs against the values used in the article.

The R8 protocol stores the SHA-256 of each period's ``series.npz`` file. A ``.npz`` file is a
zip archive, and its container bytes need not be identical across environments even when
every array is: in our clean-folder rebuild all entries had the same CRC-32, compressed
data, and 1980-01-01 timestamp, and the files differed only in local-header fields (flags
and size fields) written by the zip library. This script therefore checks
(1) the frozen code hashes, (2) a content hash of the arrays in each ``series.npz``, and
(3) the SHA-256 of each ``units.csv``, which is written as plain text and is reproducible.
The expected values below were computed from the inputs of the article's confirmatory
replay; a rebuild from the raw Huawei trace in a clean folder reproduces all of them.

Usage (from the analysis directory, after prep_huawei_confirmatory_2026_10_02.py):
    python3 audit/verify_confirmatory_inputs_2026_10_03.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "experiments" / "data" / "external_traces" / "huawei2023_confirmatory"
PROTOCOL = HERE / "verified_results" / "revision_2026-10-02" / "protocol.json"

ARRAY_CONTENT_SHA256 = {
    "A_days000_018": "85469641510329dacfdc20c4493161f7bf46ea11e3a09206cea98dee260be62a",
    "B_days147_165": "32878fabbb51c3c157dd71684fb3f7ae871eafd0cae398df067ae053c0e18303",
    "C_days168_184": "b7a6829749941693e32f2b70c04133faa193424e2de1fabb41a57a711f5bf66b",
}
UNITS_CSV_SHA256 = {
    "A_days000_018": "cabc85b0786cece7fc9b22ebe75f5cc10539be30c26ce0fa81d4cfb1e9598718",
    "B_days147_165": "9a86a3122335472b7b2d1d2b27337fa73a318966d41339e703eb5d7a2a8616fd",
    "C_days168_184": "aa2dbde4ece08099a27362f6f7015def3a62f0d5a1a46528b36638cf581e30f0",
}


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def content_sha(p: Path) -> str:
    z = np.load(p, allow_pickle=True)
    h = hashlib.sha256()
    for k in sorted(z.files):
        a = z[k]
        if a.dtype == object:
            a = a.astype(str)
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main() -> int:
    ok = True
    code = json.loads(PROTOCOL.read_text())["R8_confirmatory"]["code_sha256"]
    for name, want in code.items():
        good = file_sha(HERE / name) == want
        ok &= good
        print(f"[{'OK' if good else 'MISMATCH'}] code  {name}")
    for run in ARRAY_CONTENT_SHA256:
        g1 = content_sha(DATA / run / "series.npz") == ARRAY_CONTENT_SHA256[run]
        g2 = file_sha(DATA / run / "units.csv") == UNITS_CSV_SHA256[run]
        ok &= g1 and g2
        print(f"[{'OK' if g1 else 'MISMATCH'}] arrays {run}/series.npz")
        print(f"[{'OK' if g2 else 'MISMATCH'}] file   {run}/units.csv")
    print("all checks passed" if ok else "some checks failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
