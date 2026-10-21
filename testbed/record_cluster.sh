#!/usr/bin/env bash
# Record the provenance the freeze needs: image digests, cluster and node versions, toolchain.
# Usage (from the repository root, after pushing the images):
#   testbed/record_cluster.sh REGISTRY_FOR_PUSH [BUILD_METADATA_DIR]
# With BUILD_METADATA_DIR (files <image>.json from `docker buildx build --metadata-file`), the digests
# are those of the new build, checked against the registry; otherwise they come from docker inspect.
set -euo pipefail
REG="$1"
META="${2:-}"
OUT=testbed/cluster
mkdir -p "$OUT"
{
  echo "{"
  first=1
  for name in testbed-worker testbed-loadgen testbed-tools; do
    if [ -n "$META" ]; then
      digest=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['containerimage.digest'])" "$META/$name.json")
      remote=$(docker buildx imagetools inspect "$REG/$name:1.1" | awk '/^Digest:/ {print $2; exit}')
      [ "$digest" = "$remote" ] || { echo "registry has $remote for $name:1.1, the build produced $digest" >&2; exit 1; }
    else
      digest=$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$REG/$name:1.1" 2>/dev/null | grep "^$REG/$name@" | head -1 | sed 's/.*@//' || true)
    fi
    [ -n "$digest" ] || { echo "no digest for $name" >&2; exit 1; }
    [ $first -eq 1 ] || echo ","
    printf '  "%s": "%s"' "$name" "$digest"
    first=0
  done
  echo
  echo "}"
} > "$OUT/images.json"
python3 - "$OUT/cluster.json" <<'PY'
import json, subprocess, sys
run = lambda *a: json.loads(subprocess.run(["kubectl", *a, "-o", "json"], check=True, capture_output=True, text=True).stdout)
nodes = run("get", "nodes")["items"]
ms = subprocess.run(["kubectl", "-n", "kube-system", "get", "deploy", "metrics-server", "-o", "jsonpath={.spec.template.spec.containers[0].image}"],
                    capture_output=True, text=True).stdout
out = {"version": json.loads(subprocess.run(["kubectl", "version", "-o", "json"], capture_output=True, text=True).stdout or "{}"),
       "metrics_server_image": ms,
       "nodes": [{"name": n["metadata"]["name"], "testbed-node": n["metadata"]["labels"].get("testbed-node"),
                  "instance_type": n["metadata"]["labels"].get("node.kubernetes.io/instance-type"),
                  "capacity": n["status"]["capacity"], "allocatable": n["status"]["allocatable"],
                  "nodeInfo": n["status"]["nodeInfo"]} for n in nodes]}
open(sys.argv[1], "w").write(json.dumps(out, indent=1) + "\n")
PY
{
  go version 2>/dev/null || echo "go: not installed (images are built in Docker)"
  python3 -c "import sys, numpy, pandas, scipy; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__)"
  kubectl version --client 2>/dev/null | head -1
  docker version --format 'docker {{.Server.Version}}' 2>/dev/null || true
  docker buildx version 2>/dev/null || true
} > "$OUT/toolchain.txt"
echo "wrote $OUT/images.json, $OUT/cluster.json, $OUT/toolchain.txt"
