#!/usr/bin/env python3
"""Write the Kubernetes manifests of a testbed run or of a capacity phase.

Workers carry the node label testbed-node=w1..w4 and the driver node testbed-node=load. In run r
(1-4) policy i of (pach, conf, b1, hpa) runs on worker w((i + r - 1) mod 4 + 1), so that over
the four runs every policy runs once on every worker (PROTOCOL.md, Sect. 4). The driver pod on
the node "load" runs the load generator, the collector, the three custom controllers, and a
container from which the results are copied.

Usage (from the repository root):
  python3 testbed/render.py --registry REG --start UNIX_TIME --run 1|2|3|4
  python3 testbed/render.py --registry REG --start UNIX_TIME --pilot
  python3 testbed/render.py --registry REG --start UNIX_TIME --capacity phase1|phase2|phase3
The manifests are written to testbed/runs/<name>/manifests.yaml together with the effective plan.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "testbed"
PLAN = TB / "plan"
POLICIES = ("pach", "conf", "b1", "hpa")
WORKERS = ("w1", "w2", "w3", "w4")
RUNS = {1: {"day": 10, "seed": 1}, 2: {"day": 11, "seed": 1}, 3: {"day": 10, "seed": 1}, 4: {"day": 11, "seed": 1}}
MAX_REPLICAS = 40
POD = {"cpu": "40m", "memory_request": "32Mi", "memory_limit": "64Mi", "iter": 26400, "warmup_seconds": 15, "max_inflight": 8}
VERSION = "1.1"
IMAGES = TB / "cluster" / "images.json"
POD_ITER = TB / "pod_iter.json"
PULL_SECRET = os.environ.get("TESTBED_PULL_SECRET", "")


def pod_iter() -> int:
    """Work per request (SHA-256 rounds): set on the cluster by capacity.py phase 1, else the default."""
    return int(json.loads(POD_ITER.read_text())["iter"]) if POD_ITER.exists() else POD["iter"]


def image(registry: str, name: str) -> str:
    """Image reference, pinned by digest once cluster/images.json exists."""
    if IMAGES.exists():
        digest = json.loads(IMAGES.read_text()).get(name)
        if digest:
            return f"{registry}/{name}@{digest}"
    return f"{registry}/{name}:{VERSION}"


def pull(spec: dict) -> dict:
    """Name the registry pull secret when the registry is private (TESTBED_PULL_SECRET)."""
    if PULL_SECRET:
        spec["imagePullSecrets"] = [{"name": PULL_SECRET}]
    return spec


def assignment(run: int) -> dict[str, str]:
    return {pol: WORKERS[(i + run - 1) % 4] for i, pol in enumerate(POLICIES)}


def doc(d: dict) -> str:
    return json.dumps(d, indent=1)


def worker(ns: str, name: str, node: str, registry: str, replicas: int, hpa_target: int | None = None) -> list[dict]:
    labels = {"app": name, "testbed": "worker"}
    dep = {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": name, "namespace": ns, "labels": labels},
           "spec": {"replicas": replicas, "selector": {"matchLabels": {"app": name}},
                    "template": {"metadata": {"labels": labels},
                                 "spec": pull({"nodeSelector": {"testbed-node": node}, "terminationGracePeriodSeconds": 5,
                                          "containers": [{"name": "worker", "image": image(registry, "testbed-worker"),
                                                          "imagePullPolicy": "IfNotPresent",
                                                          "env": [{"name": "ITER", "value": str(pod_iter())},
                                                                  {"name": "WARMUP_SECONDS", "value": str(POD["warmup_seconds"])},
                                                                  {"name": "MAX_INFLIGHT", "value": str(POD["max_inflight"])}],
                                                          "ports": [{"containerPort": 8080}],
                                                          "resources": {"requests": {"cpu": POD["cpu"], "memory": POD["memory_request"]},
                                                                        "limits": {"cpu": POD["cpu"], "memory": POD["memory_limit"]}},
                                                          "readinessProbe": {"httpGet": {"path": "/healthz", "port": 8080},
                                                                             "periodSeconds": 2, "timeoutSeconds": 2,
                                                                             "failureThreshold": 3, "successThreshold": 1}}]})}}}
    svc = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": name, "namespace": ns},
           "spec": {"selector": {"app": name}, "ports": [{"port": 80, "targetPort": 8080}]}}
    out = [dep, svc]
    if hpa_target is not None:
        out.append({"apiVersion": "autoscaling/v2", "kind": "HorizontalPodAutoscaler", "metadata": {"name": name, "namespace": ns},
                    "spec": {"scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": name},
                             "minReplicas": 1, "maxReplicas": MAX_REPLICAS,
                             "metrics": [{"type": "Resource", "resource": {"name": "cpu", "target": {
                                 "type": "Utilization", "averageUtilization": hpa_target}}}]}})
    return out


def rbac(namespaces: list[str]) -> list[dict]:
    rules = [{"apiGroups": ["apps"], "resources": ["deployments", "deployments/scale"], "verbs": ["get", "list", "patch"]},
             {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"]},
             {"apiGroups": ["discovery.k8s.io"], "resources": ["endpointslices"], "verbs": ["list"]},
             {"apiGroups": ["autoscaling"], "resources": ["horizontalpodautoscalers"], "verbs": ["list"]},
             {"apiGroups": ["metrics.k8s.io"], "resources": ["pods"], "verbs": ["list"]}]
    docs = [{"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": "driver", "namespace": "testbed"}},
            {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRole", "metadata": {"name": "testbed-driver"}, "rules": rules},
            {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRole", "metadata": {"name": "testbed-nodes"},
             "rules": [{"apiGroups": [""], "resources": ["nodes"], "verbs": ["get", "list"]}]},
            {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRoleBinding", "metadata": {"name": "testbed-nodes"},
             "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "testbed-nodes"},
             "subjects": [{"kind": "ServiceAccount", "name": "driver", "namespace": "testbed"}]}]
    for ns in namespaces:
        docs.append({"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding", "metadata": {"name": "testbed-driver", "namespace": ns},
                     "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "testbed-driver"},
                     "subjects": [{"kind": "ServiceAccount", "name": "driver", "namespace": "testbed"}]})
    return docs


def driver(registry: str, start: int, until: int, namespaces: list[str], controllers: list[str], files: dict) -> list[dict]:
    tools = image(registry, "testbed-tools")
    py = ["python3"]
    containers = [
        {"name": "loadgen", "image": image(registry, "testbed-loadgen"),
         "args": ["-config", "/plan/loadgen.json", "-start", str(start), "-out", "/results/load.csv", "-served", "/results/served.csv"],
         "resources": {"requests": {"cpu": "2", "memory": "1Gi"}}},
        {"name": "collector", "image": tools, "command": py + ["/tools/collector.py", "--namespaces", ",".join(namespaces),
                                                               "--until", str(until), "--outdir", "/results", "--t0", str(start)]},
    ]
    for pol in controllers:
        containers.append({"name": f"controller-{pol}", "image": tools,
                           "command": py + ["/tools/controller.py", "--plan", "/plan/plan.json", "--policy", pol,
                                            "--namespace", pol, "--out", f"/results/decisions_{pol}.csv"]})
    containers.append({"name": "results", "image": tools, "command": ["sleep", "infinity"]})
    for c in containers:
        c["imagePullPolicy"] = "IfNotPresent"
        c["volumeMounts"] = [{"name": "plan", "mountPath": "/plan"}, {"name": "results", "mountPath": "/results"}]
        c.setdefault("resources", {"requests": {"cpu": "200m", "memory": "128Mi"}})
    return [{"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "plan", "namespace": "testbed"}, "data": files},
            {"apiVersion": "v1", "kind": "PersistentVolumeClaim", "metadata": {"name": "results", "namespace": "testbed"},
             "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": "5Gi"}}}},
            {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "driver", "namespace": "testbed"},
             "spec": pull({"serviceAccountName": "driver", "restartPolicy": "Never", "nodeSelector": {"testbed-node": "load"},
                      "containers": containers,
                      "volumes": [{"name": "plan", "configMap": {"name": "plan"}},
                                  {"name": "results", "persistentVolumeClaim": {"claimName": "results"}}]})}]


def run_manifests(a) -> tuple[str, list[dict], dict]:
    plan = json.loads((PLAN / "plan.json").read_text())
    fns = plan["functions"]
    if a.pilot:
        name, day, seed, assign = "pilot", "pilot", plan["seed"], assignment(1)
        plan["minutes"] = 120
        plan["warmup_minutes"] = 0
    else:
        name, day, seed, assign = f"run{a.run}", RUNS[a.run]["day"], RUNS[a.run]["seed"], assignment(a.run)
    plan.update({"run": name, "day": day, "seed": seed, "assignment": assign})
    docs: list[dict] = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": n}} for n in ("testbed", *POLICIES)]
    for pol in POLICIES:
        for fn in fns:
            init = fn["initial_replicas_reactive_pilot"] if a.pilot else fn["initial_replicas_reactive"][str(day)]
            docs += worker(pol, fn["name"], assign[pol], a.registry, init, fn["hpa_target_cpu_percent"] if pol == "hpa" else None)
    docs += rbac(list(POLICIES))
    files = {"plan.json": json.dumps(plan, indent=1)}
    for fn in fns:
        hist = f"{fn['name']}_pilot_history.csv" if a.pilot else f"{fn['name']}_history_d{day}.csv"
        run = f"{fn['name']}_pilot.csv" if a.pilot else f"{fn['name']}_run_d{day}.csv"
        files[f"{fn['name']}_history.csv"] = (PLAN / hist).read_text()
        files[f"{fn['name']}_run.csv"] = (PLAN / run).read_text()
    files["loadgen.json"] = json.dumps({
        "schedules": {fn["name"]: f"/plan/{fn['name']}_run.csv" for fn in fns},
        "targets": [{"name": f"{pol}-{fn['name']}", "function": fn["name"], "url": f"http://{fn['name']}.{pol}.svc.cluster.local/work"}
                    for pol in POLICIES for fn in fns],
        "seed": seed, "minutes": plan["minutes"], "interval_seconds": 60}, indent=1)
    until = int(a.start + plan["minutes"] * 60 + 300)
    docs += driver(a.registry, a.start, until, list(POLICIES), ["pach", "conf", "b1"], files)
    return name, docs, plan


def capacity_manifests(a) -> tuple[str, list[dict], dict]:
    phase = TB / "capacity" / a.capacity
    meta = json.loads((phase / "meta.json").read_text())
    cfg = json.loads((phase / "loadgen.json").read_text())
    namespaces = [f"cap-{w}" for w in WORKERS]
    docs: list[dict] = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": n}} for n in ("testbed", *namespaces)]
    for w in WORKERS:
        for dep, replicas in meta["deployments"].items():
            docs += worker(f"cap-{w}", dep, w, a.registry, replicas)
    docs += rbac(namespaces)
    files = {"loadgen.json": json.dumps(cfg, indent=1)}
    for fn in cfg["schedules"]:
        files[f"{fn}.csv"] = (phase / f"{fn}.csv").read_text()
    until = int(a.start + cfg["minutes"] * 60 + 300)
    docs += driver(a.registry, a.start, until, namespaces, [], files)
    return f"capacity_{a.capacity}", docs, {"phase": a.capacity, "start": a.start}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--run", type=int, choices=sorted(RUNS))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--capacity", choices=("phase1", "phase2", "phase3"))
    a = ap.parse_args()
    if sum(map(bool, (a.run, a.pilot, a.capacity))) != 1:
        raise SystemExit("choose exactly one of --run, --pilot, --capacity")
    name, docs, plan = capacity_manifests(a) if a.capacity else run_manifests(a)
    out = TB / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifests.yaml").write_text(doc({"apiVersion": "v1", "kind": "List", "items": docs}) + "\n")
    (out / "effective_plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(f"wrote {out / 'manifests.yaml'}: {len(docs)} objects, minute 0 at {a.start}")
    if "assignment" in plan:
        print("policy -> worker:", plan["assignment"])


if __name__ == "__main__":
    main()
