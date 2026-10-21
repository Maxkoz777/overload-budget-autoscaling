# Kubernetes testbed

Runs PAC-h, its standard-rank ablation, the reactive controller B1, and the Kubernetes HPA side
by side on a Kubernetes cluster, replaying selected functions of the confirmatory Huawei periods
in real time, and compares the live control loop with the article's replay assumptions. The
hypotheses, fixed choices, and validity rules are in [PROTOCOL.md](PROTOCOL.md).

| File | Purpose |
|---|---|
| `policies.py` | PAC-h (PAC or standard rank) and B1 as step-by-step controllers |
| `check_equivalence.py` | `policies.py` against the article's replay code, minute by minute |
| `capacity.py` | Calibration of the common pod capacity mu and CPU utilisation c (phases 1-3) |
| `prepare.py` | Function selection, plan, causal histories, sizing |
| `render.py` | Manifests of a run (`--run 1..4`), the pilot (`--pilot`), or a capacity phase |
| `label_nodes.py` | Seeded labels `testbed-node=w1..w4` and `load` |
| `ops/cloud_stage.sh` | Operator script for the cloud: calibration, pilot, freeze, runs; keeps every attempt |
| `check_cluster.py` | Labels, sizing rule against allocatable CPU, metrics API, default storage class |
| `record_cluster.sh` | Image digests, cluster and toolchain versions for the freeze |
| `freeze.py` | Hashes of all inputs in `FREEZE.json` |
| `analyze.py` | One run: validity, overload intervals, actuator model (H1), H2, users, delays |
| `combine.py` | Complete campaign of four valid runs: H1 criteria, H2, H3 on overload intervals, repeatability |
| `tests/` | Synthetic negative and positive tests: `python3 testbed/tests/run_tests.py` |
| `LOCAL_STAGE.md`, `local_stage/` | Local engineering stage of 18 October 2026 and its raw data |
| `app/`, `loadgen/`, `tools/` | Worker service, load generator, controller and collector images |

The trace folder defaults to `experiments/data/external_traces`; set `TESTBED_DATA` to the folder
that holds `huawei2023_confirmatory/` if it lives elsewhere.

## Images

```bash
REG_PUSH=...   # registry address used for docker push
REG=...        # the same registry as seen from the cluster
docker build -t $REG_PUSH/testbed-worker:1.1 testbed/app && docker push $REG_PUSH/testbed-worker:1.1
docker build -t $REG_PUSH/testbed-loadgen:1.1 testbed/loadgen && docker push $REG_PUSH/testbed-loadgen:1.1
docker build -t $REG_PUSH/testbed-tools:1.1 -f testbed/tools/Dockerfile testbed && docker push $REG_PUSH/testbed-tools:1.1
```

## Local engineering check (k3d)

```bash
k3d registry create testbed-registry --port 5050
k3d cluster create testbed --agents 5 --registry-use k3d-testbed-registry:5050
python3 testbed/label_nodes.py k3d-testbed-agent-0 k3d-testbed-agent-1 k3d-testbed-agent-2 k3d-testbed-agent-3 --driver k3d-testbed-agent-4
REG_PUSH=localhost:5050; REG=k3d-testbed-registry:5050   # then build and push the images
python3 testbed/check_equivalence.py
python3 testbed/tests/run_tests.py
```

All k3d nodes share one machine, so local numbers are engineering checks only. The stage of
18 October 2026 ran on single-node k3s instead (`LOCAL_STAGE.md`).

## Common steps (local check and cloud)

```bash
start() { echo $(( $(date +%s) / 60 * 60 + ${1:-300} )); }
apply_and_fetch() {   # $1 = folder name under testbed/runs and testbed/results; waits for the driver
  kubectl apply --server-side -f testbed/runs/$1/manifests.yaml
  echo "wait until the collector container of testbed/driver has terminated, then:"
  echo "mkdir -p testbed/results/$1 && kubectl -n testbed cp driver:/results testbed/results/$1 -c results"
}

python3 testbed/capacity.py plan1                                   # capacity, phase 1 (45 min)
S=$(start); python3 testbed/render.py --registry $REG --start $S --capacity phase1; apply_and_fetch capacity_phase1
python3 testbed/capacity.py analyze1 --results testbed/results/capacity_phase1
kubectl delete ns testbed cap-w1 cap-w2 cap-w3 cap-w4
python3 testbed/capacity.py plan2                                   # capacity, phase 2 (36 min)
S=$(start); python3 testbed/render.py --registry $REG --start $S --capacity phase2; apply_and_fetch capacity_phase2
python3 testbed/capacity.py analyze2 --results testbed/results/capacity_phase2   # candidate capacity.json
kubectl delete ns testbed cap-w1 cap-w2 cap-w3 cap-w4

python3 testbed/prepare.py                                          # plan and sizing with the candidate mu
python3 testbed/capacity.py plan3                                   # capacity, phase 3 (20 min)
S=$(start); python3 testbed/render.py --registry $REG --start $S --capacity phase3; apply_and_fetch capacity_phase3
python3 testbed/capacity.py analyze3 --results testbed/results/capacity_phase3   # verified, or mu - 1
kubectl delete ns testbed cap-w1 cap-w2 cap-w3 cap-w4
# if analyze3 lowered mu: move testbed/results/capacity_phase3 aside, then prepare.py, plan3 and phase 3 again

S=$(start 600); python3 testbed/render.py --registry $REG --start $S --pilot; apply_and_fetch pilot   # 2 h, day 9
python3 testbed/analyze.py testbed/results/pilot --fit-actuator     # writes actuator_model.json
kubectl delete ns testbed pach conf b1 hpa
```

## Cloud only: freeze and main runs

```bash
testbed/record_cluster.sh $REG_PUSH
git add testbed && git commit -m "Freeze the Kubernetes testbed protocol"
python3 testbed/freeze.py && git add testbed/FREEZE.json && git commit -m "Record testbed input hashes"

for RUN in 1 2 3 4; do                                              # about 25 hours each
  S=$(start 600); python3 testbed/render.py --registry $REG --start $S --run $RUN; apply_and_fetch run$RUN
  python3 testbed/analyze.py testbed/results/run$RUN
  kubectl delete ns testbed pach conf b1 hpa
done
python3 testbed/combine.py testbed/results/run1 testbed/results/run2 testbed/results/run3 testbed/results/run4
```
