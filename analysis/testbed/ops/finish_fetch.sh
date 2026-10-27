#!/usr/bin/env bash
# Recovery of a main run whose fetch failed (kubectl cp: unexpected EOF). Outside testbed/, so not part of the freeze.
# Every file is checked against sha256 computed inside the driver; namespaces are deleted only after all files match.
#   source ~/.testbed-cloud.env && caffeinate -dimsu bash analysis/testbed/ops/finish_fetch.sh 1
set -uo pipefail
N=${1:-}
case "$N" in 1|2|3|4) ;; *) echo "usage: bash finish_fetch.sh 1|2|3|4"; exit 2 ;; esac
REPO="$(cd "$(dirname "$0")/../../.." && pwd)" || exit 1
cd "$REPO" || exit 1
[ -f "$HOME/.testbed-cloud.env" ] && source "$HOME/.testbed-cloud.env"
PY="${TESTBED_VENV:-$HOME/.testbed-venv}/bin/python3"
RES=testbed/results
name=run$N
CAP_NS="cap-w1 cap-w2 cap-w3 cap-w4"
POL_NS="pach conf b1 hpa"
mkdir -p "$RES/ops"
exec > >(tee -a "$RES/ops/finish_fetch.log") 2>&1
stamp() { date '+%Y%m%d_%H%M%S'; }
fail() { echo "FAILED: $*"; echo "$(date '+%F %T') FAILED: finish_fetch $name: $*" > "$RES/ops/status.txt"; exit 1; }
sum_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

echo "=== $(date '+%F %T') finish_fetch $name"
"$PY" testbed/freeze.py --verify || fail "inputs differ from FREEZE.json"
kubectl -n testbed get pod driver >/dev/null 2>&1 || fail "driver pod not found"
ec=$(kubectl -n testbed get pod driver -o jsonpath='{.status.containerStatuses[?(@.name=="collector")].state.terminated.exitCode}')
[ "$ec" = "0" ] || fail "collector has not finished with exit code 0 (got '$ec')"
grep -q "\"run\": \"$name\"" <(kubectl -n testbed exec driver -c results -- cat /results/meta.json </dev/null) || fail "the driver does not hold $name"

TMP="$RES/${name}_recovering_$(stamp)"
mkdir -p "$TMP"
kubectl -n testbed exec driver -c results -- sh -c 'cd /results && find . -maxdepth 1 -type f | sed "s|^\./||" | sort | xargs sha256sum' </dev/null > "$TMP/sha256_driver.txt" \
  || fail "sha256 inside the driver failed"
grep -q ' load.csv$' "$TMP/sha256_driver.txt" || fail "load.csv not listed in the driver"
echo "driver holds $(wc -l < "$TMP/sha256_driver.txt" | tr -d ' ') files"
if kubectl -n testbed exec driver -c results -- sh -c 'command -v gzip' </dev/null >/dev/null 2>&1; then ZIP=1; else ZIP=0; fi

SRC=$(ls -d "$RES/${name}"_*_fetch_failed/raw 2>/dev/null | sort | tail -1)
while read -r sum f; do
  ok=0
  for cand in ${SRC:+"$SRC/$f"} "$RES/$name/$f"; do
    if [ -f "$cand" ] && [ "$(sum_of "$cand")" = "$sum" ]; then
      cp -p "$cand" "$TMP/$f"; ok=1; echo "$f: matches the driver, taken from $cand"; break
    fi
  done
  try=1
  while [ $ok = 0 ] && [ $try -le 6 ]; do
    if [ $ZIP = 1 ]; then
      kubectl -n testbed exec driver -c results -- gzip -1 -c "/results/$f" </dev/null 2>/dev/null | gunzip -c > "$TMP/$f.part" 2>/dev/null
    else
      kubectl -n testbed exec driver -c results -- cat "/results/$f" </dev/null 2>/dev/null > "$TMP/$f.part"
    fi
    if [ "$(sum_of "$TMP/$f.part")" = "$sum" ]; then
      mv "$TMP/$f.part" "$TMP/$f"; ok=1; echo "$f: copied from the driver (attempt $try)"
    else
      echo "$f: attempt $try did not match, retrying"; try=$((try+1)); sleep 15
    fi
  done
  [ $ok = 1 ] || fail "$f could not be copied intact; nothing deleted, partial copy in $TMP"
done < "$TMP/sha256_driver.txt"
rm -f "$TMP"/*.part
(cd "$TMP" && shasum -a 256 -c sha256_driver.txt) || fail "final sha256 check failed; nothing deleted"

[ -e "$RES/$name" ] && mv "$RES/$name" "$RES/${name}_superseded_$(stamp)"
mv "$TMP" "$RES/$name"
for c in loadgen collector controller-pach controller-conf controller-b1; do
  kubectl -n testbed logs driver -c $c > "$RES/$name/log_$c.txt" 2>/dev/null; done
kubectl get events -A --sort-by=.lastTimestamp > "$RES/$name/events.txt" 2>&1
kubectl -n testbed describe pod driver > "$RES/$name/driver_describe.txt" 2>&1
cp -R "testbed/runs/$name" "$RES/$name/run_folder"
kubectl get pods -A -o json | "$PY" -c '
import json, sys
want = set(json.load(open("testbed/cluster/images.json")).values())
ids = {c.get("imageID", "") for p in json.load(sys.stdin)["items"] if p["metadata"]["namespace"] in sys.argv[1:]
       for c in p.get("status", {}).get("containerStatuses", [])}
got = {i.split("@")[-1] for i in ids if "@" in i}
bad = got - want
print(json.dumps({"ok": not bad, "running_digests": sorted(got), "unexpected": sorted(bad)}))
sys.exit(1 if bad else 0)' testbed $CAP_NS $POL_NS > "$RES/$name/images_check.json" \
  || echo "WARNING: $name ran images other than those in cluster/images.json (see images_check.json)"
printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$name" "fetched_recovered" >> "$RES/attempts.tsv"
echo "$(date '+%F %T') $name fetched and verified"

kubectl delete ns testbed $CAP_NS $POL_NS --ignore-not-found --wait=true --timeout=900s >/dev/null 2>&1
for ns in testbed $CAP_NS $POL_NS; do kubectl get ns "$ns" >/dev/null 2>&1 && fail "namespace $ns still exists after deletion"; done
echo "namespaces deleted"

"$PY" testbed/analyze.py "$RES/$name" > "$RES/$name/analyze.log" 2>&1 || fail "analyze $name"
"$PY" -c "import json; v = json.load(open('$RES/$name/validity.json')); print('$name valid', v['valid'], v['problems'][:5])"
echo "$(date '+%F %T') done: $name recovered and analyzed" > "$RES/ops/status.txt"
