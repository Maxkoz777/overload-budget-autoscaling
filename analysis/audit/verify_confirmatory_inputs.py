#!/usr/bin/env python3
"""Check the prepared confirmatory inputs against the values used in the article.

Frozen code. The protocol stores the SHA-256 of the six scripts of the confirmatory replay
under the file names they had when it was frozen (release v1.0.0,
https://doi.org/10.5281/zenodo.23125235, preserves them byte for byte). Later releases gave
these scripts and the protocol folders descriptive names; the only edits to the scripts are
the module, file, and folder names in their own text. The code check therefore maps the
current names back to the frozen ones (``RENAMED`` and ``FOLDERS`` below) and hashes the reconstructed text, which must equal the stored hash.

Data. The R8 protocol stores the SHA-256 of each period's ``series.npz`` file. A ``.npz`` file is a
zip archive, and its container bytes need not be identical across environments even when
every array is: in our clean-folder rebuild all entries had the same CRC-32, compressed
data, and 1980-01-01 timestamp, and the files differed only in local-header fields (flags
and size fields) written by the zip library. This script therefore checks
(1) the frozen code hashes, (2) a content hash of the arrays in each ``series.npz``, and
(3) the SHA-256 of each ``units.csv``, which is written as plain text and is reproducible.
The expected values below were computed from the inputs of the article's confirmatory
replay; a rebuild from the raw Huawei trace in a clean folder reproduces all of them.

Usage (from the analysis directory, after prepare_confirmatory_periods.py):
    python3 audit/verify_confirmatory_inputs.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "experiments" / "data" / "external_traces" / "huawei2023_confirmatory"
PROTOCOL = HERE / "verified_results" / "delay_pac_study" / "protocol.json"

RENAMED = {
    "canonical_inputs_2026_10_03": "canonical_inputs",
    "exact_envelope_2026_09_29": "resource_overload_envelope",
    "external_replication_2026_09_30": "external_replication",
    "external_replication_summary_2026_09_30": "external_replication_summary",
    "final_extension_2026_09_29": "resource_axis_extension",
    "phase8_focused_fill_sensitivity": "focused_fill_sensitivity",
    "plot_cost_compliance_tau1_2026_10_02": "plot_cost_vs_compliance",
    "plot_delay_three_traces_2026_10_02": "plot_delay_compliance",
    "plot_external_replication_2026_09_30": "plot_external_replication",
    "plot_final_extension_2026_09_29": "plot_resource_axis",
    "plot_phase6_figures": "plot_focused_figures",
    "plot_phase7_dkw": "plot_dkw",
    "plot_verified_phase2": "plot_verified_suite",
    "prep_external_traces_2026_09_30": "prepare_external_traces",
    "prep_huawei_confirmatory_2026_10_02": "prepare_confirmatory_periods",
    "revision_2026_10_02": "delay_and_reactive_grid",
    "revision_alibaba_delay_2026_10_02": "alibaba_delay",
    "revision_canonical_2026_10_03": "canonical_rerun",
    "revision_confirmatory_2026_10_02": "confirmatory_replay",
    "revision_external_delay_2026_10_02": "external_delay",
    "revision_pac_rank_2026_10_02": "pac_rank_replay",
    "revision_round2_2026_10_02": "pac_window_grid",
    "revision_round3_2026_10_02": "pac_stride_sensitivity",
    "revision_subset_2026_10_02": "pretest_subset",
    "tables_final_extension_2026_09_29": "tables_resource_axis",
    "test_canonical_inputs_2026_10_03": "test_canonical_inputs",
    "verify_confirmatory_inputs_2026_10_03": "verify_confirmatory_inputs",
    "verify_phase8_upstream": "verify_series_upstream",
    "readiness_checks_2026_09_23": "strict_budget_crosscheck",
}
FOLDERS = {
    "external_replication_2026-09-30": "external_replication_study",
    "final_extension_2026-09-29": "resource_axis_study",
    "revision_2026-10-02": "delay_pac_study",
}
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


def frozen_text_sha(frozen_name: str) -> str:
    stem = frozen_name[:-3]
    text = (HERE / f"{RENAMED.get(stem, stem)}.py").read_text()
    names = {**RENAMED, **FOLDERS}
    for old in sorted(names, key=lambda k: len(names[k]), reverse=True):
        text = re.sub(r"(?<![\w-])" + re.escape(names[old]) + r"(?![\w-])", old, text)
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    ok = True
    code = json.loads(PROTOCOL.read_text())["R8_confirmatory"]["code_sha256"]
    for name, want in code.items():
        good = frozen_text_sha(name) == want
        ok &= good
        print(f"[{'OK' if good else 'MISMATCH'}] code  {name} (now {RENAMED.get(name[:-3], name[:-3])}.py)")
    if not DATA.exists():
        print("prepared confirmatory data not found: run prepare_confirmatory_periods.py for the data checks")
        return 0 if ok else 1
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
