# Local engineering stage, 17–18 October 2026: raw data

**Kind:** live runs on a real Kubernetes cluster, sandbox only. These runs are not synthetic tests and not cloud runs. They are engineering checks and are not evidence for H1–H3.

## Environment

- Single-node k3s v1.31.4+k3s1 on a 2-vCPU, 7 GiB Linux VM. Details are in `sandbox/environment.txt`.
- Worker image `testbed-worker:1.1`, built from `app/main.go` and imported into containerd.
- Loadgen, collector and controllers ran as host processes through `kubectl proxy`. They were started by `sandbox/run_phase.sh`.
- Before applying, `sandbox/harness.py` rewrote the manifests:
  - dropped the driver pod, PVC and ConfigMap;
  - removed nodeSelector;
  - set CPU request to 10m (limit stayed 40m);
  - multiplied the HPA target by 4 so that it refers to 40m.
- `sandbox/lgconfig.py` pointed the load generator at the Services' ClusterIPs.
- `sandbox/code_as_run/` is the code at the end of the stage:
  - `WORKERS = ("w1",)` in `capacity.py` and `render.py`;
  - `% len(WORKERS)` in `render.assignment`;
  - the controller and collector are the post-stage versions. During the pilot, the collector had no `gone` column and the controller had no `called_at` column.
- The pilot used f1, f5 and f9 of `plan.json` (μ = 420/min). The subset plan is in `pilot/manifests/plan/`.

## Contents

| Folder | Content |
|---|---|
| `capacity_phase1/` | Phase 1, 45 min. `raw/` holds the files written by loadgen and collector; `manifests/` holds the rendered and local manifests and the effective plan. |
| `capacity_phase2/` | Phase 2 as run under the old rule: 36 min single pod, then 2/4/8 pods at 5 rps per pod. |
| `capacity/` | Phase plans, `phase1_result.json`, and the resulting sandbox `capacity.json` (μ = 7 rps, c = 0.645; not verified under the new phase-3 rule). |
| `pilot/` | 120-minute pilot. Inside it: `raw/` (load, counts, pods, served, hpa, cpu, decisions, logs), `manifests/`, `analysis_as_run/` (analysis of 18 Oct, 04:50) and `analysis_v3/` (same raw data, analysed with the code after the review). |
| `pilot_attempt1_aborted/` | First pilot start. It was stopped after 7 minutes because HPA utilisation was computed against the 10m request (HPA scaled to 20 pods), and restarted after the harness fix. |

## Commands as run (sandbox)

```
python3 testbed/capacity.py plan1; LEAD=120 run_phase.sh capacity_phase1 "--capacity phase1" cap-w1 -
python3 testbed/capacity.py analyze1 --results testbed/results/capacity_phase1
python3 testbed/capacity.py plan2; LEAD=120 run_phase.sh capacity_phase2 "--capacity phase2" cap-w1 -
python3 testbed/capacity.py analyze2 --results testbed/results/capacity_phase2
python3 testbed/prepare.py      # then plan.json reduced to f1, f5, f9 for the pilot
LEAD=180 run_phase.sh pilot "--pilot" pach,conf,b1,hpa pach,conf,b1
python3 testbed/analyze.py testbed/results/pilot --fit-actuator; python3 testbed/analyze.py testbed/results/pilot
```

## `mac_k3d_2026-10-18.tar.gz`

This is the second rehearsal, run on a Mac with k3d using the final manifests (in-cluster driver, 12 functions, 4 policies). The archive holds:

- `local_stage.log` and `monitor.log`;
- the environment report;
- the test and equivalence outputs;
- `repo_artifacts_*/`, with everything the rehearsal wrote into `testbed/`: capacity phases including the ITER = 26400 attempt, plan, `pod_iter.json`, `actuator_model.json`, cluster records, manifests and pilot results.

A summary is in `../LOCAL_STAGE.md`. Checksums are in `SHA256SUMS`.
