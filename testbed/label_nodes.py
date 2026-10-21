#!/usr/bin/env python3
"""Label the four workers testbed-node=w1..w4 by a seeded permutation and the driver node load.

Usage (from the repository root):  python3 testbed/label_nodes.py WORKER1 WORKER2 WORKER3 WORKER4 --driver NODE
"""
from __future__ import annotations

import argparse
import subprocess

import numpy as np

SEED = 20261017


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workers", nargs=4)
    ap.add_argument("--driver", required=True)
    a = ap.parse_args()
    nodes = sorted(a.workers)
    perm = np.random.RandomState(SEED).permutation(4)
    for label, i in zip(("w1", "w2", "w3", "w4"), perm):
        subprocess.run(["kubectl", "label", "node", nodes[i], f"testbed-node={label}", "--overwrite"], check=True)
        print(f"{label} = {nodes[i]}")
    subprocess.run(["kubectl", "label", "node", a.driver, "testbed-node=load", "--overwrite"], check=True)
    print(f"load = {a.driver}")


if __name__ == "__main__":
    main()
