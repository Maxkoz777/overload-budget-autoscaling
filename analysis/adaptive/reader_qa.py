"""Release QA: the fallback parquet reader (adaptive/pq.py) against pyarrow on all 200 Alibaba files.

Usage: OBA_REPO=<repository root> python3 reader_qa.py   (requires pyarrow)
"""
from __future__ import annotations

import sys

import numpy as np
import pyarrow.parquet as papq

from adaptive import pq
from adaptive.data import REPO

COLS = ["timestamp", "replica_count", "cpu_sum", "was_missing"]
files = sorted((REPO / "experiments" / "data" / "service_timeseries_200_verified").glob("*.parquet"))
bad = 0
for f in files:
    a = papq.read_table(f, columns=COLS).to_pandas()
    b = pq.read(str(f), COLS)
    for c in COLS:
        if not np.array_equal(a[c].to_numpy(), b[c]):
            bad += 1
            print("MISMATCH", f.name, c)
print(f"{len(files)} files x {len(COLS)} columns, mismatches: {bad}")
sys.exit(1 if bad or len(files) != 200 else 0)
