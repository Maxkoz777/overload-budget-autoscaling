#!/usr/bin/env bash
# Descriptive snapshot of the five nodes before the cluster is deleted (after run 4 and combine only).
# Not part of the protocol or the freeze: CPU model, steal time since boot, and a short single-thread benchmark per node.
#   caffeinate -dimsu bash analysis/testbed/ops/node_snapshot.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
[ -f "$HOME/.testbed-cloud.env" ] && source "$HOME/.testbed-cloud.env"
PY="${TESTBED_VENV:-$HOME/.testbed-venv}/bin/python3"
OUT="testbed/results/ops/node_snapshot_$(date '+%Y%m%d_%H%M%S')"
NS=nodesnap
fail() { echo "FAILED: $*"; exit 1; }

for ns in testbed pach conf b1 hpa; do
  kubectl get ns "$ns" >/dev/null 2>&1 && fail "namespace $ns exists: a run is still in the cluster, the snapshot would disturb it"
done
mkdir -p "$OUT"
kubectl get nodes -o json > "$OUT/nodes.json" || fail "kubectl get nodes"
kubectl describe nodes > "$OUT/nodes_describe.txt" 2>&1
kubectl get nodes -L testbed-node -o wide > "$OUT/nodes_wide.txt" 2>&1

kubectl delete ns $NS --ignore-not-found --wait=true >/dev/null 2>&1
kubectl create ns $NS >/dev/null || fail "create namespace"
kubectl -n $NS create secret docker-registry testbed-pull --docker-server="${TESTBED_REGISTRY%%/*}" \
  --docker-username="$TESTBED_PULL_USER" --docker-password="$TESTBED_PULL_PASSWORD" >/dev/null 2>&1 || fail "pull secret"

"$PY" - "$TESTBED_REGISTRY" > "$OUT/pods.json" <<'PY' || fail "manifest"
import json, subprocess, sys
reg = sys.argv[1]
digest = json.load(open("testbed/cluster/images.json"))["testbed-tools"]
code = r'''
import hashlib, json, os, socket, time
def stat():
    f = open("/proc/stat").readline().split()[1:]
    return dict(zip(["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"], map(int, f[:8])))
info = open("/proc/cpuinfo").read().split("\n\n")
first = dict(l.split(":", 1) for l in info[0].splitlines() if ":" in l)
first = {k.strip(): v.strip() for k, v in first.items()}
mhz = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
def bench(sec=10.0):
    b, n, t = b"x" * 64, 0, time.perf_counter()
    while time.perf_counter() - t < sec:
        for _ in range(2000):
            b = hashlib.sha256(b).digest()
        n += 2000
    return n / (time.perf_counter() - t)
s0 = stat()
runs = [bench() for _ in range(5)]
s1 = stat()
d = {k: s1[k] - s0[k] for k in s1}
print(json.dumps({
    "node": os.environ.get("NODE"), "hostname": socket.gethostname(),
    "model_name": first.get("model name"), "microcode": first.get("microcode"), "cache_size": first.get("cache size"),
    "cpus": len(mhz), "cpu_mhz": mhz, "hypervisor_flag": "hypervisor" in first.get("flags", "").split(),
    "uptime_s": float(open("/proc/uptime").read().split()[0]),
    "stat_since_boot": s0,
    "steal_fraction_since_boot": s0["steal"] / max(1, sum(s0.values())),
    "steal_fraction_during_bench": d["steal"] / max(1, sum(d.values())),
    "sha256_per_s_single_thread": runs,
    "loadavg": open("/proc/loadavg").read().split()[:3],
}))
'''
nodes = json.loads(subprocess.run(["kubectl", "get", "nodes", "-o", "json"], capture_output=True, text=True, check=True).stdout)["items"]
items = []
for n in nodes:
    name = n["metadata"]["name"]
    if "node-role.kubernetes.io/control-plane" in n["metadata"].get("labels", {}) or "node-role.kubernetes.io/master" in n["metadata"].get("labels", {}):
        continue
    items.append({"apiVersion": "v1", "kind": "Pod",
                  "metadata": {"name": "snap-" + name, "namespace": "nodesnap",
                               "labels": {"testbed-node": n["metadata"].get("labels", {}).get("testbed-node", "")}},
                  "spec": {"nodeName": name, "restartPolicy": "Never", "tolerations": [{"operator": "Exists"}],
                           "imagePullSecrets": [{"name": "testbed-pull"}],
                           "containers": [{"name": "snap", "image": f"{reg}/testbed-tools@{digest}",
                                           "imagePullPolicy": "IfNotPresent",
                                           "env": [{"name": "NODE", "value": name}],
                                           "command": ["python3", "-c", code],
                                           "resources": {"requests": {"cpu": "1", "memory": "64Mi"}, "limits": {"memory": "256Mi"}}}]}})
print(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}))
PY
kubectl apply -f "$OUT/pods.json" >/dev/null || fail "apply"
n=0
while true; do
  st=$(kubectl -n $NS get pods --no-headers 2>/dev/null | awk '{print $3}' | sort | uniq -c | tr '\n' ' ')
  echo "$(date '+%T') $st"
  kubectl -n $NS get pods --no-headers 2>/dev/null | awk '$3!="Completed" && $3!="Error"' | grep -q . || break
  n=$((n+1)); [ $n -ge 60 ] && { kubectl -n $NS get pods -o wide; echo "some pods did not finish in 30 minutes; partial snapshot kept"; break; }
  sleep 30
done
for p in $(kubectl -n $NS get pods -o name); do
  kubectl -n $NS logs "$p" >> "$OUT/snapshot.jsonl" 2>/dev/null
done
kubectl -n $NS get pods -o wide > "$OUT/snap_pods.txt" 2>&1
kubectl delete ns $NS --wait=true >/dev/null 2>&1
"$PY" - "$OUT/snapshot.jsonl" "$OUT/nodes.json" <<'PY'
import json, statistics, sys
lab = {n["metadata"]["name"]: n["metadata"].get("labels", {}).get("testbed-node", "") for n in json.load(open(sys.argv[2]))["items"]}
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    print(f'{lab.get(d["node"], ""):5} {d["node"]:20} {d["model_name"]} | mhz {statistics.mean(d["cpu_mhz"]):.0f} | '
          f'steal boot {100*d["steal_fraction_since_boot"]:.2f}% bench {100*d["steal_fraction_during_bench"]:.2f}% | '
          f'sha256/s median {statistics.median(d["sha256_per_s_single_thread"]):,.0f}')
PY
echo "snapshot in $OUT"
