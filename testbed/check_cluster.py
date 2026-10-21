#!/usr/bin/env python3
"""Check a cluster before calibration: node labels, sizing rule, metrics API, default storage class.

Sizing rule (PROTOCOL.md, Sect. 3): on every worker, P x 40m x 1.25 plus the CPU requests of the
pods already running there (DaemonSets, system pods) must fit into the allocatable CPU, where P is
the largest peak of concurrent pods of one custom policy (plan/sizing.json, else 67). The driver
node must fit the driver pod's requests. Writes cluster/check.json; exit code 1 if a check fails.

Usage (from the repository root):  python3 testbed/check_cluster.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TB = Path(__file__).resolve().parent
POD_MILLICORES = 40
MARGIN = 1.25
DRIVER_MILLICORES = 2000 + 5 * 200


def kubectl(*args: str) -> dict:
    return json.loads(subprocess.run(["kubectl", *args, "-o", "json"], check=True, capture_output=True, text=True).stdout)


def millicores(q: str) -> float:
    return float(q[:-1]) if q.endswith("m") else float(q) * 1000


def main() -> None:
    sizing = TB / "plan" / "sizing.json"
    peak = max(v["peak_pods"] for v in json.loads(sizing.read_text())["policies"].values()) if sizing.exists() else 67
    nodes = kubectl("get", "nodes")["items"]
    pods = kubectl("get", "pods", "-A")["items"]
    used: dict[str, float] = {}
    for p in pods:
        if p.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            continue
        node = p["spec"].get("nodeName")
        req = sum(millicores(c.get("resources", {}).get("requests", {}).get("cpu", "0")) for c in p["spec"]["containers"])
        used[node] = used.get(node, 0.0) + req
    out, ok = {"peak_pods": peak, "nodes": []}, True
    labels = sorted(n["metadata"]["labels"].get("testbed-node", "") for n in nodes)
    for n in nodes:
        name, label = n["metadata"]["name"], n["metadata"]["labels"].get("testbed-node")
        alloc = millicores(n["status"]["allocatable"]["cpu"])
        need = peak * POD_MILLICORES * MARGIN if label in ("w1", "w2", "w3", "w4") else (DRIVER_MILLICORES if label == "load" else 0)
        fits = alloc - used.get(name, 0.0) >= need
        ok &= fits or label is None
        out["nodes"].append({"name": name, "label": label, "allocatable_m": alloc, "requested_by_others_m": used.get(name, 0.0),
                             "needed_m": need, "fits": fits})
        print(f"{name:40s} {str(label):5s} allocatable {alloc:6.0f}m, others {used.get(name, 0.0):5.0f}m, needed {need:5.0f}m: "
              f"{'ok' if fits else 'TOO SMALL'}")
    if [x for x in labels if x] != ["load", "w1", "w2", "w3", "w4"]:
        print("labels missing: expected load, w1..w4"); ok = False
    metrics = subprocess.run(["kubectl", "get", "--raw", "/apis/metrics.k8s.io/v1beta1/nodes"], capture_output=True, text=True)
    out["metrics_api"] = metrics.returncode == 0
    print("metrics API:", "ok" if out["metrics_api"] else "MISSING (install metrics-server)")
    scs = kubectl("get", "storageclass")["items"]
    default = [s["metadata"]["name"] for s in scs
               if s["metadata"].get("annotations", {}).get("storageclass.kubernetes.io/is-default-class") == "true"]
    out["default_storage_class"] = default
    print("default storage class:", default or "MISSING")
    ok &= out["metrics_api"] and bool(default)
    out["ok"] = bool(ok)
    (TB / "cluster").mkdir(exist_ok=True)
    (TB / "cluster" / "check.json").write_text(json.dumps(out, indent=1) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
