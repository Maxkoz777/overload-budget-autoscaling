#!/usr/bin/env python3
"""Run an analysis script with the canonical capacity-unit rule installed.

Usage (from the paper/analysis directory):
    python3 audit/run_canonical.py audit/<script>.py [arguments ...]

See canonical_inputs_2026_10_03.py. The executed script itself is not modified.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical_inputs_2026_10_03 import install  # noqa: E402

install()
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
