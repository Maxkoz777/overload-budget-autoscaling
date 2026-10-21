#!/usr/bin/env bash
# Operator script of the Kubernetes testbed on the cloud cluster (PROTOCOL.md, Sect. 8, steps 2-4).
# It is part of testbed/ and therefore of the freeze.
#
# Before: kubectl points at the cloud cluster, docker is logged in to the registry, and
#   export TESTBED_REGISTRY=<registry host>/<registry name>
# and, for a private registry, the read credentials of the nodes:
#   export TESTBED_PULL_USER=... TESTBED_PULL_PASSWORD=...
#
#   bash testbed/ops/cloud_stage.sh info                      environment and cluster report
#   caffeinate -dimsu bash testbed/ops/cloud_stage.sh         calibration: tests, labels, images, checks,
#                                                             phases 1-3, pilot (about 6 h)
#   caffeinate -dimsu bash testbed/ops/cloud_stage.sh from pilot   resume the calibration at a step
#   bash testbed/ops/cloud_stage.sh freeze                    local branch testbed-cloud, commit, FREEZE.json (no push)
#   caffeinate -dimsu bash testbed/ops/cloud_stage.sh run 1   main run 1 (about 25 h); then 2, 3, 4
#   caffeinate -dimsu bash testbed/ops/cloud_stage.sh attach 1   wait for and fetch run 1 already in the cluster
#   caffeinate -dimsu bash testbed/ops/cloud_stage.sh attach pilot   the same for the pilot
#   bash testbed/ops/cloud_stage.sh combine                   the four runs together
#
# No attempt is lost: before a namespace is deleted, the driver's results, manifests, logs and the
# reason are copied to testbed/results/<name>_<attempt>; earlier result folders are never overwritten.
# testbed/results/attempts.tsv lists every attempt; testbed/results/<name> is the current one.
set -uo pipefail
TB="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$TB/.." && pwd)"
RES="$TB/results"
OUT="$RES/ops"
mkdir -p "$OUT"
export TESTBED_DATA="${TESTBED_DATA:-$(cd "$REPO/.." && pwd)/experiments/data/external_traces}"
LOG="$OUT/cloud_stage.log"
STATUS="$OUT/status.txt"
exec > >(tee -a "$LOG") 2>&1
VENV="${TESTBED_VENV:-$HOME/.testbed-venv}"
PY="$VENV/bin/python3"
REG="${TESTBED_REGISTRY:-}"
if [ -n "${TESTBED_PULL_USER:-}" ]; then export TESTBED_PULL_SECRET=testbed-pull; else unset TESTBED_PULL_SECRET; fi
STEPS=(info tests labels images prepull check phase1 phase2 prepare phase3 pilot)
CAP_NS="cap-w1 cap-w2 cap-w3 cap-w4"
POL_NS="pach conf b1 hpa"

stamp() { date '+%Y%m%d_%H%M%S'; }
step() { echo; echo "=== $(date '+%F %T') $*"; echo "$(date '+%F %T') running: $*" > "$STATUS"; }
fail() { echo "FAILED: $*"; echo "$(date '+%F %T') FAILED: $*" > "$STATUS"; exit 1; }
cd "$REPO" || fail "repository not found at $REPO"
export PATH="$VENV/bin:$PATH"

do_info() {
  step "info"
  {
    echo "date: $(date '+%F %T %z')"
    for t in docker kubectl python3 git; do printf '%-8s %s\n' "$t" "$(command -v $t || echo MISSING)"; done
    docker version --format 'docker client {{.Client.Version}}, server {{.Server.Version}}' 2>&1 | head -2
    docker buildx version 2>&1 | head -1
    echo "registry: ${REG:-NOT SET (export TESTBED_REGISTRY=...)}"
    echo "pull secret: ${TESTBED_PULL_SECRET:-none (public registry)}"
    echo "kubectl context: $(kubectl config current-context 2>&1)"
    kubectl version 2>&1 | head -3
    kubectl get nodes -o wide -L testbed-node 2>&1
    kubectl get storageclass 2>&1
    kubectl top nodes 2>&1 | head -8
    echo "repository HEAD: $(git log --oneline -1), branch $(git branch --show-current)"
  } | tee "$OUT/environment.txt"
  [ -n "$REG" ] || fail "TESTBED_REGISTRY is not set"
  kubectl get nodes >/dev/null 2>&1 || fail "kubectl cannot reach the cluster"
  case "$(kubectl config current-context)" in *k3d*) fail "kubectl points at the local k3d cluster";; esac
}

do_tests() {
  step "tests"
  if [ ! -x "$PY" ]; then
    python3 -m venv "$VENV" && "$VENV/bin/pip" install -q "numpy>=2.3,<2.4" "pandas>=2.3,<3" scipy || fail "python environment"
  fi
  "$PY" testbed/tests/run_tests.py > "$OUT/tests.txt" 2>&1 || fail "tests (see $OUT/tests.txt)"
  tail -1 "$OUT/tests.txt"
  "$PY" testbed/check_equivalence.py > "$OUT/equivalence.txt" 2>&1
  grep -q "all decisions identical" "$OUT/equivalence.txt" || fail "equivalence (see $OUT/equivalence.txt)"
  tail -1 "$OUT/equivalence.txt"
}

do_labels() {
  step "labels"
  local nodes n driver workers
  nodes=$(kubectl get nodes --field-selector spec.unschedulable=false -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | sort)
  n=$(echo "$nodes" | grep -c .)
  [ "$n" -eq 5 ] || fail "expected 5 schedulable nodes, found $n"
  driver=$(echo "$nodes" | head -1)
  workers=$(echo "$nodes" | tail -4 | tr '\n' ' ')
  "$PY" testbed/label_nodes.py $workers --driver "$driver" || fail "labels"
  kubectl get nodes -L testbed-node
}

do_images() {
  step "images (linux/amd64, pushed to $REG)"
  local arch meta digest
  arch=$(kubectl get nodes -o jsonpath='{.items[0].status.nodeInfo.architecture}')
  [ "$arch" = "amd64" ] || fail "nodes are $arch; adjust --platform"
  meta="$OUT/build_$(stamp)"; mkdir -p "$meta"
  docker buildx build --platform linux/amd64 --provenance=false --push --metadata-file "$meta/testbed-worker.json" \
    -t "$REG/testbed-worker:1.1" testbed/app || fail "worker image"
  docker buildx build --platform linux/amd64 --provenance=false --push --metadata-file "$meta/testbed-loadgen.json" \
    -t "$REG/testbed-loadgen:1.1" testbed/loadgen || fail "loadgen image"
  docker buildx build --platform linux/amd64 --provenance=false --push --metadata-file "$meta/testbed-tools.json" \
    -t "$REG/testbed-tools:1.1" -f testbed/tools/Dockerfile testbed || fail "tools image"
  bash testbed/record_cluster.sh "$REG" "$meta" || fail "record_cluster"
  cat testbed/cluster/images.json
  cleanup pulltest
  kubectl create ns pulltest >/dev/null || fail "namespace pulltest"
  pull_secret pulltest
  digest=$("$PY" -c "import json; print(json.load(open('testbed/cluster/images.json'))['testbed-tools'])")
  ov=()
  [ -n "${TESTBED_PULL_SECRET:-}" ] && ov=(--overrides="{\"apiVersion\":\"v1\",\"spec\":{\"imagePullSecrets\":[{\"name\":\"$TESTBED_PULL_SECRET\"}]}}")
  kubectl -n pulltest run pull --image="$REG/testbed-tools@$digest" --restart=Never ${ov[@]+"${ov[@]}"} --command -- \
    python3 -c "import numpy; print('pulled, numpy', numpy.__version__)" >/dev/null
  kubectl -n pulltest wait --for=jsonpath='{.status.phase}'=Succeeded pod/pull --timeout=300s >/dev/null \
    || { kubectl -n pulltest describe pod pull | tail -15; fail "nodes cannot pull from $REG (registry integration or read access)"; }
  kubectl -n pulltest logs pull
  cleanup pulltest
}

do_prepull() {   # pull each node's images once (load: loadgen + tools; w1..w4: worker), so pods start from the node cache
  local n=0 state
  step "pre-pull of the images (load node: loadgen, tools; workers: worker), retried until done"
  kubectl -n prepull get ds >/dev/null 2>&1 || { kubectl create ns prepull >/dev/null || fail "namespace prepull"; }
  pull_secret prepull
  "$PY" - "$REG" > "$OUT/prepull.json" <<'PY' || fail "prepull manifest"
import json, os, sys
reg = sys.argv[1]
images = json.load(open("testbed/cluster/images.json"))
def ds(name, roles, names):
    spec = {"terminationGracePeriodSeconds": 0,
            "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
                {"matchExpressions": [{"key": "testbed-node", "operator": "In", "values": roles}]}]}}},
            "containers": [{"name": n.replace("testbed-", ""), "image": f"{reg}/{n}@{images[n]}", "imagePullPolicy": "IfNotPresent",
                            "resources": {"requests": {"cpu": "1m", "memory": "8Mi"}}} for n in names]}
    for c in spec["containers"]:
        if c["name"] == "tools":
            c["command"] = ["/bin/sh", "-c", "sleep infinity"]
    if os.environ.get("TESTBED_PULL_SECRET"):
        spec["imagePullSecrets"] = [{"name": os.environ["TESTBED_PULL_SECRET"]}]
    return {"apiVersion": "apps/v1", "kind": "DaemonSet", "metadata": {"name": name, "namespace": "prepull"},
            "spec": {"selector": {"matchLabels": {"app": name}}, "template": {"metadata": {"labels": {"app": name}}, "spec": spec}}}
print(json.dumps({"apiVersion": "v1", "kind": "List", "items": [
    ds("prepull-load", ["load"], ["testbed-loadgen", "testbed-tools"]),
    ds("prepull-workers", ["w1", "w2", "w3", "w4"], ["testbed-worker"])]}))
PY
  kubectl -n prepull delete ds prepull --ignore-not-found >/dev/null 2>&1
  kubectl apply -f "$OUT/prepull.json" >/dev/null || fail "apply prepull"
  while true; do
    state=$(kubectl -n prepull get pods -o json 2>/dev/null | "$PY" -c '
import json, sys
pods = [p for p in json.load(sys.stdin).get("items", []) if p["metadata"].get("labels", {}).get("app") in ("prepull-load", "prepull-workers")]
done = [p for p in pods if len(p.get("status", {}).get("containerStatuses", [])) == len(p["spec"]["containers"])
        and all(c.get("imageID") for c in p["status"]["containerStatuses"])]
print(f"{len(done)}/5", "ok" if len(pods) == 5 and len(done) == 5 else "wait")')
    echo "$(date '+%T') pre-pull: nodes with their images $state"
    case "$state" in *ok) break ;; esac
    n=$((n+1)); [ $n -ge 120 ] && { kubectl -n prepull get pods -o wide; kubectl get events -n prepull --sort-by=.lastTimestamp | tail -15; fail "pre-pull did not finish in 60 minutes (rerun: from images keeps the pulled layers)"; }
    sleep 30
  done
  cleanup prepull
}

clear_leftovers() {   # namespaces of an earlier failed step hold node capacity: save them, then delete them
  if kubectl get ns testbed >/dev/null 2>&1; then
    case "$(driver_state)" in
      Running|Pending) [ "${FORCE_NEW:-0}" = "1" ] || fail "a driver is still running in the cluster: use 'attach', or FORCE_NEW=1 to save it and start anew" ;;
    esac
    save_attempt previous left_in_cluster
  fi
  cleanup testbed $CAP_NS $POL_NS prepull pulltest
}

do_check() {
  step "cluster check (sizing, metrics API, storage)"
  clear_leftovers
  "$PY" testbed/check_cluster.py || fail "cluster check (see testbed/cluster/check.json)"
}

start() { echo $(( $(date +%s) / 60 * 60 + ${1:-300} )); }

pull_secret() {   # namespaces: create them with the registry pull secret (private registry only)
  local ns
  [ -n "${TESTBED_PULL_SECRET:-}" ] || return 0
  [ -n "${TESTBED_PULL_PASSWORD:-}" ] || fail "TESTBED_PULL_PASSWORD is not set"
  for ns in "$@"; do
    kubectl get ns "$ns" >/dev/null 2>&1 || kubectl create ns "$ns" >/dev/null || fail "namespace $ns"
    kubectl -n "$ns" create secret docker-registry "$TESTBED_PULL_SECRET" --docker-server="${REG%%/*}" \
      --docker-username="$TESTBED_PULL_USER" --docker-password="$TESTBED_PULL_PASSWORD" \
      --dry-run=client -o yaml | kubectl apply -f - >/dev/null || fail "pull secret in $ns"
  done
}

cleanup() {   # delete namespaces and verify that they are gone
  local ns
  kubectl delete ns "$@" --ignore-not-found --wait=true --timeout=900s >/dev/null 2>&1
  for ns in "$@"; do
    kubectl get ns "$ns" >/dev/null 2>&1 && fail "namespace $ns still exists after deletion"
  done
  true
}

driver_state() {
  kubectl -n testbed get pod driver -o json 2>/dev/null | "$PY" -c '
import json, sys
try:
    p = json.load(sys.stdin)
except Exception:
    print("missing"); sys.exit()
st = {c["name"]: c.get("state", {}) for c in p.get("status", {}).get("containerStatuses", [])}
bad = [n for n, s in st.items() if "terminated" in s and s["terminated"].get("exitCode", 0) != 0]
waiting = [n + ":" + str(s["waiting"].get("reason")) for n, s in st.items() if "waiting" in s and s["waiting"].get("reason") not in (None, "ContainerCreating")]
if bad: print("crashed " + " ".join(bad))
elif waiting: print("waiting " + " ".join(waiting))
elif "terminated" in st.get("collector", {}): print("done")
else: print(p.get("status", {}).get("phase", "unknown"))'
}

save_attempt() {   # name, reason: copy whatever the driver holds before anything is deleted
  local name=$1 reason=$2 dest c
  dest="$RES/${name}_$(stamp)_$reason"
  mkdir -p "$dest"
  kubectl -n testbed cp driver:/results "$dest/raw" -c results >/dev/null 2>&1 || echo "no raw results could be copied" > "$dest/raw_missing.txt"
  for c in loadgen collector controller-pach controller-conf controller-b1; do
    kubectl -n testbed logs driver -c $c > "$dest/log_$c.txt" 2>/dev/null; done
  kubectl -n testbed describe pod driver > "$dest/driver_describe.txt" 2>&1
  kubectl -n testbed get configmap plan -o json > "$dest/plan_configmap.json" 2>/dev/null
  kubectl get events -A --sort-by=.lastTimestamp > "$dest/events.txt" 2>&1
  [ -d "testbed/runs/$name" ] && cp -R "testbed/runs/$name" "$dest/run_folder"
  echo "$reason" > "$dest/reason.txt"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$(basename "$dest")" "$reason" >> "$RES/attempts.tsv"
  echo "saved attempt in $dest"
}

wait_driver() {   # name; API errors are retried for 30 minutes before giving up (nothing is deleted)
  local name=$1 n=0 lost=0 s
  while true; do
    s=$(driver_state)
    case "$s" in
      done) break ;;
      missing|unknown) lost=$((lost+1))
               [ $lost -ge 60 ] && fail "$name: driver not visible for 30 minutes; resume with 'attach' when the API is back" ;;
      crashed*|waiting*) save_attempt "$name" failed
               fail "$name: driver $s (attempt saved; namespaces left in place for inspection)" ;;
      *) lost=0 ;;
    esac
    if [ $((n % 20)) -eq 0 ]; then
      echo "$(date '+%T') $name: driver $s"
      { echo "== $(date '+%F %T') $name"; kubectl top nodes 2>&1; kubectl get pods -A --no-headers 2>/dev/null | awk '$4!="Running" && $4!="Completed"' | head -10; } >> "$OUT/monitor.log"
    fi
    n=$((n+1)); sleep 30
  done
}

launch() {   # name, render args, namespaces
  local name=$1 args=$2 S; shift 2
  if kubectl get ns testbed >/dev/null 2>&1; then
    case "$(driver_state)" in
      Running|Pending) [ "${FORCE_NEW:-0}" = "1" ] || fail "a driver is still running in the cluster: use 'attach', or FORCE_NEW=1 to save it and start anew" ;;
    esac
    save_attempt previous left_in_cluster
  fi
  cleanup testbed $CAP_NS $POL_NS
  pull_secret testbed "$@"
  S=$(start "${LEAD:-300}")
  "$PY" testbed/render.py --registry "$REG" --start $S $args || fail "render $name"
  kubectl apply --server-side -f testbed/runs/$name/manifests.yaml >/dev/null || fail "apply $name"
  echo "minute 0 at $(date -r $S '+%F %T')"
}

check_images() {   # name: the running pods use exactly the recorded digests
  kubectl get pods -A -o json | "$PY" -c '
import json, sys
want = set(json.load(open("testbed/cluster/images.json")).values())
ids = {c.get("imageID", "") for p in json.load(sys.stdin)["items"] if p["metadata"]["namespace"] in sys.argv[1:]
       for c in p.get("status", {}).get("containerStatuses", [])}
got = {i.split("@")[-1] for i in ids if "@" in i}
bad = got - want
print(json.dumps({"ok": not bad, "running_digests": sorted(got), "unexpected": sorted(bad)}))
sys.exit(1 if bad else 0)' testbed $CAP_NS $POL_NS > "$RES/$1/images_check.json" \
    || echo "WARNING: $1 ran images other than those in cluster/images.json (see images_check.json)"
}

fetch() {   # name: wait, copy, keep earlier folders, then delete the namespaces
  local name=$1 c
  wait_driver "$name"
  [ -e "$RES/$name" ] && mv "$RES/$name" "$RES/${name}_superseded_$(stamp)"
  mkdir -p "$RES"
  kubectl -n testbed cp driver:/results "$RES/$name" -c results >/dev/null || { save_attempt "$name" fetch_failed; fail "fetch $name"; }
  [ -f "$RES/$name/load.csv" ] || { save_attempt "$name" no_load_csv; fail "fetch $name: load.csv missing"; }
  for c in loadgen collector controller-pach controller-conf controller-b1; do
    kubectl -n testbed logs driver -c $c > "$RES/$name/log_$c.txt" 2>/dev/null; done
  kubectl get events -A --sort-by=.lastTimestamp > "$RES/$name/events.txt" 2>&1
  kubectl -n testbed describe pod driver > "$RES/$name/driver_describe.txt" 2>&1
  cp -R "testbed/runs/$name" "$RES/$name/run_folder"
  check_images "$name"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$name" "fetched" >> "$RES/attempts.tsv"
  cleanup testbed $CAP_NS $POL_NS
}

archive_result() {   # name, reason: keep a scored folder under a new name before the step is repeated
  mv "$RES/$1" "$RES/${1}_$(stamp)_$2"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$(ls -td "$RES/${1}_"*"_$2" | head -1 | xargs basename)" "$2" >> "$RES/attempts.tsv"
}

do_phase1() {
  local attempt rc
  for attempt in 1 2 3 4; do
    step "capacity phase 1, attempt $attempt (about 55 minutes)"
    "$PY" testbed/capacity.py plan1 || fail "plan1"
    launch capacity_phase1 "--capacity phase1" $CAP_NS; fetch capacity_phase1
    "$PY" testbed/capacity.py analyze1 --results "$RES/capacity_phase1"; rc=$?
    [ $rc -eq 0 ] && return
    [ $rc -eq 3 ] || fail "analyze1"
    archive_result capacity_phase1 iter_changed
  done
  fail "phase 1: ITER did not settle"
}

do_phase2() {
  step "capacity phase 2 (about 45 minutes)"
  "$PY" testbed/capacity.py plan2 || fail "plan2"
  launch capacity_phase2 "--capacity phase2" $CAP_NS; fetch capacity_phase2
  "$PY" testbed/capacity.py analyze2 --results "$RES/capacity_phase2" || fail "analyze2"
}

do_prepare() {
  step "prepare"
  "$PY" testbed/prepare.py || fail "prepare"
  clear_leftovers
  "$PY" testbed/check_cluster.py || fail "sizing check with the plan (see testbed/cluster/check.json)"
}

verified() { "$PY" -c "import json,sys; sys.exit(0 if json.load(open('testbed/capacity.json'))['verified'] else 1)"; }

do_phase3() {
  local attempt
  for attempt in 1 2 3 4; do
    step "capacity phase 3, attempt $attempt (about 30 minutes)"
    "$PY" testbed/capacity.py plan3 || fail "plan3"
    launch capacity_phase3 "--capacity phase3" $CAP_NS; fetch capacity_phase3
    "$PY" testbed/capacity.py analyze3 --results "$RES/capacity_phase3" || fail "calibration failed (see capacity.json)"
    verified && return
    archive_result capacity_phase3 "$("$PY" -c "import json; print('repeat_at_floor' if 'amendment A1' in json.load(open('testbed/capacity.json'))['note'] else 'mu_lowered')")"
    "$PY" testbed/prepare.py || fail "prepare"
  done
  fail "phase 3 not verified"
}

do_pilot() {
  step "pilot, 12 functions, 4 policies (about 2 h 15 min)"
  "$PY" testbed/prepare.py || fail "prepare"
  LEAD=600 launch pilot "--pilot" $POL_NS; fetch pilot
  analyze_pilot
}

analyze_pilot() {
  "$PY" testbed/analyze.py "$RES/pilot" --fit-actuator > "$RES/pilot/analyze_fit.log" 2>&1 || fail "fit-actuator (see analyze_fit.log)"
  "$PY" testbed/analyze.py "$RES/pilot" > "$RES/pilot/analyze.log" 2>&1 || fail "analyze pilot"
  head -1 "$RES/pilot/analyze_fit.log"
  "$PY" -c "
import json; s = json.load(open('$RES/pilot/summary.json')); v = json.load(open('$RES/pilot/validity.json'))
print('valid', v['valid'], v['problems'][:5]); print('cpu ms per request', s['cpu_ms_per_request'], 'calibration', s.get('cpu_ms_per_request_at_calibration'))
for p, d in s['policies'].items(): print(p, d['h2'], d.get('h1'), d.get('implementation_exact'))"
  echo "Calibration and pilot done. Review the pilot before the freeze."
}

do_freeze() {
  step "freeze (local branch testbed-cloud, no push)"
  local f
  for f in capacity.json plan/plan.json actuator_model.json cluster/images.json results/pilot/validity.json; do
    [ -e testbed/$f ] || fail "testbed/$f missing"; done
  verified || fail "capacity.json not verified"
  if git rev-parse --verify testbed-cloud >/dev/null 2>&1; then git checkout testbed-cloud || fail "checkout"; else git checkout -b testbed-cloud || fail "branch"; fi
  git add testbed && git commit -q -m "Testbed: cloud calibration and pilot before the freeze" || true
  "$PY" testbed/freeze.py || fail "freeze.py"
  git add testbed/FREEZE.json && git commit -q -m "Testbed: record input hashes (freeze)" || fail "commit FREEZE.json"
  git log --oneline -3
  echo "Frozen on the local branch testbed-cloud. Nothing was pushed."
}

check_frozen() {
  [ -f testbed/FREEZE.json ] || fail "not frozen: run the freeze step first"
  "$PY" testbed/freeze.py --verify || fail "inputs differ from FREEZE.json"
}

analyze_run() {
  "$PY" testbed/analyze.py "$RES/run$1" > "$RES/run$1/analyze.log" 2>&1 || fail "analyze run$1"
  "$PY" -c "
import json; v = json.load(open('$RES/run$1/validity.json')); print('run$1 valid', v['valid'], v['problems'][:5])"
}

do_run() {
  local i=$1
  check_frozen
  step "run $i (about 25 h)"
  LEAD=600 launch run$i "--run $i" $POL_NS
  fetch run$i
  analyze_run $i
}

do_attach() {
  local i=$1
  step "attach to $i"
  if [ "$i" = "pilot" ]; then fetch pilot; analyze_pilot; return; fi
  check_frozen
  fetch run$i
  analyze_run $i
}

do_combine() {
  step "combine"
  "$PY" testbed/combine.py "$RES/run1" "$RES/run2" "$RES/run3" "$RES/run4" > "$RES/combine.log" 2>&1 \
    || { tail -20 "$RES/combine.log"; fail "combine"; }
  head -40 "$RES/combine.log"
}

case "${1:-all}" in
  info) do_info ;;
  freeze) do_freeze ;;
  run) do_info >/dev/null; do_run "${2:?run number 1-4}" ;;
  attach) do_attach "${2:?run number 1-4 or pilot}" ;;
  combine) do_combine ;;
  all|from)
    first=info; [ "${1:-all}" = "from" ] && first="${2:?step name}"
    started=0
    for s in "${STEPS[@]}"; do
      [ "$s" = "$first" ] && started=1
      [ $started -eq 1 ] && "do_$s"
    done
    [ $started -eq 1 ] || fail "unknown step $first" ;;
  *) fail "unknown command $1" ;;
esac
step "DONE ${1:-all} ${2:-}"
echo "$(date '+%F %T') DONE ${1:-all} ${2:-}" > "$STATUS"
