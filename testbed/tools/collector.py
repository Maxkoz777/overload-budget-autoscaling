#!/usr/bin/env python3
"""Record the state of the testbed about once per second.

counts.csv   per observation of a namespace (with its own timestamp): for every function the
             replicas requested in the Deployment, the pods that exist and are not terminating,
             the terminating pods, and the pods that are ready endpoints of the Service
pods.csv     one row per pod: UID, ReplicaSet, node, creation, scheduling, container start,
             Ready condition, first and last time as a ready endpoint, deletionTimestamp
             (call + grace period), first time the pod was no longer listed
hpa.csv      the HPA status of every function
cpu.csv      CPU usage of every pod from metrics-server, every 15 s; all rows of one response share the
             snapshot number and its time t; metrics_timestamp is the end of the metric's window
cluster.json Kubernetes version, nodes (labels, capacity, allocatable, kubelet), effective HPA specs

Usage: collector.py --namespaces pach,conf,b1,hpa --until UNIX_TIME --outdir DIR [--t0 UNIX_TIME]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kube import Kube  # noqa: E402

POD_COLS = ["namespace", "pod", "uid", "function", "replicaset", "node", "created", "scheduled", "started",
            "ready", "endpoint_first", "endpoint_last", "deleted", "gone"]


def ts(s: str | None) -> float | None:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def millicores(q: str) -> float:
    if q.endswith("n"):
        return float(q[:-1]) / 1e6
    if q.endswith("u"):
        return float(q[:-1]) / 1e3
    if q.endswith("m"):
        return float(q[:-1])
    return float(q) * 1000


def snapshot_cluster(kube: Kube, nss: list[str], out: Path) -> None:
    info: dict = {"collected_at": time.time()}
    for key, path in (("version", "/version"), ("nodes", "/api/v1/nodes")):
        try:
            info[key] = kube.get(path)
        except OSError as e:
            info[key] = {"error": str(e)}
    if isinstance(info.get("nodes"), dict) and "items" in info["nodes"]:
        info["nodes"] = [{"name": n["metadata"]["name"], "labels": n["metadata"].get("labels", {}),
                          "capacity": n["status"].get("capacity"), "allocatable": n["status"].get("allocatable"),
                          "nodeInfo": n["status"].get("nodeInfo")} for n in info["nodes"]["items"]]
    if "hpa" in nss:
        try:
            info["hpa_specs"] = [{"name": h["metadata"]["name"], "spec": h.get("spec")}
                                 for h in kube.get("/apis/autoscaling/v2/namespaces/hpa/horizontalpodautoscalers")["items"]]
        except (OSError, KeyError) as e:
            info["hpa_specs"] = {"error": str(e)}
    (out / "cluster.json").write_text(json.dumps(info, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespaces", required=True)
    ap.add_argument("--until", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--t0", type=float, default=None)
    a = ap.parse_args()
    kube = Kube(a.proxy)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    nss = a.namespaces.split(",")
    if a.t0 is not None and not (out / "meta.json").exists():
        (out / "meta.json").write_text(json.dumps({"t0": a.t0, "interval_seconds": 60}) + "\n")
    snapshot_cluster(kube, nss, out)
    f_counts = open(out / "counts.csv", "w", newline="")
    counts = csv.writer(f_counts)
    counts.writerow(["t", "namespace", "function", "spec_replicas", "pods_existing", "pods_terminating", "endpoints_ready"])
    f_hpa = open(out / "hpa.csv", "w", newline="")
    hpa_w = csv.writer(f_hpa)
    hpa_w.writerow(["t", "namespace", "function", "current", "desired", "cpu_utilisation", "last_scale_time"])
    f_cpu = open(out / "cpu.csv", "w", newline="")
    cpu_w = csv.writer(f_cpu)
    cpu_w.writerow(["t", "snapshot", "namespace", "pod", "function", "metrics_timestamp", "window_s", "millicores"])
    f_err = open(out / "collector_errors.csv", "w", newline="")
    err_w = csv.writer(f_err)
    err_w.writerow(["t", "namespace", "what", "error"])
    pods: dict[str, dict] = {}
    tick = 0
    while time.time() < a.until:
        loop_start = time.time()
        for ns in nss:
            try:
                deps = kube.get(f"/apis/apps/v1/namespaces/{ns}/deployments")["items"]
                plist = kube.get(f"/api/v1/namespaces/{ns}/pods")["items"]
                eps = kube.get(f"/apis/discovery.k8s.io/v1/namespaces/{ns}/endpointslices")["items"]
                t = time.time()
            except OSError as e:
                err_w.writerow([round(time.time(), 3), ns, "state", str(e)])
                continue
            ready_pods: dict[str, set] = {}
            for s in eps:
                fn = s["metadata"].get("labels", {}).get("kubernetes.io/service-name")
                for e in s.get("endpoints") or []:
                    if (e.get("conditions") or {}).get("ready") and e.get("targetRef"):
                        ready_pods.setdefault(fn, set()).add(e["targetRef"]["name"])
            existing: dict[str, int] = {}
            terminating: dict[str, int] = {}
            for p in plist:
                md, st = p["metadata"], p.get("status", {})
                fn = md.get("labels", {}).get("app")
                if md.get("deletionTimestamp"):
                    terminating[fn] = terminating.get(fn, 0) + 1
                elif st.get("phase") not in ("Succeeded", "Failed"):
                    existing[fn] = existing.get(fn, 0) + 1
                conds = {c["type"]: c for c in st.get("conditions", [])}
                cs = (st.get("containerStatuses") or [{}])[0]
                owner = next((o["name"] for o in md.get("ownerReferences", []) if o.get("kind") == "ReplicaSet"), None)
                rec = pods.setdefault(f"{ns}/{md['name']}", {
                    "namespace": ns, "pod": md["name"], "uid": md.get("uid"), "function": fn, "replicaset": owner,
                    "node": None, "created": ts(md["creationTimestamp"]), "scheduled": None, "started": None,
                    "ready": None, "endpoint_first": None, "endpoint_last": None, "deleted": None, "gone": None})
                rec["node"] = p.get("spec", {}).get("nodeName") or rec["node"]
                if conds.get("PodScheduled", {}).get("status") == "True":
                    rec["scheduled"] = rec["scheduled"] or ts(conds["PodScheduled"].get("lastTransitionTime"))
                rec["started"] = rec["started"] or ts((cs.get("state", {}).get("running") or {}).get("startedAt"))
                if conds.get("Ready", {}).get("status") == "True":
                    rec["ready"] = rec["ready"] or ts(conds["Ready"].get("lastTransitionTime"))
                if md.get("deletionTimestamp"):
                    rec["deleted"] = rec["deleted"] or ts(md["deletionTimestamp"])
                if md["name"] in ready_pods.get(fn, set()):
                    rec["endpoint_first"] = rec["endpoint_first"] or t
                    rec["endpoint_last"] = t
            listed = {f"{ns}/{p['metadata']['name']}" for p in plist}
            for key, rec in pods.items():
                if rec["namespace"] == ns and rec["gone"] is None and key not in listed:
                    rec["gone"] = t
            for d in deps:
                fn = d["metadata"]["name"]
                counts.writerow([round(t, 3), ns, fn, d["spec"].get("replicas"), existing.get(fn, 0),
                                 terminating.get(fn, 0), len(ready_pods.get(fn, ()))])
            if ns == "hpa":
                try:
                    for h in kube.get(f"/apis/autoscaling/v2/namespaces/{ns}/horizontalpodautoscalers")["items"]:
                        st = h.get("status", {})
                        cpu = next((m["resource"]["current"].get("averageUtilization") for m in st.get("currentMetrics") or []
                                    if m.get("type") == "Resource"), None)
                        hpa_w.writerow([round(time.time(), 3), ns, h["metadata"]["name"], st.get("currentReplicas"),
                                        st.get("desiredReplicas"), cpu, st.get("lastScaleTime")])
                except OSError as e:
                    err_w.writerow([round(time.time(), 3), ns, "hpa", str(e)])
            if tick % 15 == 0:
                try:
                    items = kube.get(f"/apis/metrics.k8s.io/v1beta1/namespaces/{ns}/pods")["items"]
                    snap_t = round(time.time(), 3)
                    for m in items:
                        fn = m["metadata"].get("labels", {}).get("app")
                        cpu_w.writerow([snap_t, tick, ns, m["metadata"]["name"], fn, m.get("timestamp"), m.get("window"),
                                        round(sum(millicores(c["usage"]["cpu"]) for c in m["containers"]), 3)])
                except (OSError, KeyError, ValueError) as e:
                    err_w.writerow([round(time.time(), 3), ns, "metrics", str(e)])
        for f in (f_counts, f_hpa, f_cpu, f_err):
            f.flush()
        tick += 1
        if tick % 60 == 0:
            write_pods(out, pods)
        time.sleep(max(0.0, 1.0 - (time.time() - loop_start)))
    write_pods(out, pods)
    snapshot_cluster(kube, nss, out)


def write_pods(out: Path, pods: dict) -> None:
    tmp = out / "pods.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POD_COLS)
        w.writeheader()
        w.writerows(pods.values())
    tmp.replace(out / "pods.csv")


if __name__ == "__main__":
    main()
