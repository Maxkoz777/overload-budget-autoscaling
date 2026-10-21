# Local engineering stage (2026-10-18)

The stage ran on single-node k3s v1.31.4+k3s1 in a 2-vCPU, 7 GiB Linux sandbox. These are engineering checks of the pipeline only. They are not used for the cloud calibration, the freeze, or the paper, and they are not evidence for H1–H3. The raw data, manifests, sandbox scripts and the code as run are in `local_stage/`.

## Deviations from the cloud setup

- **One node.** `WORKERS = ("w1",)` is set in `capacity.py` and `render.py`.
- **Host-side driver.** Loadgen, collector and controllers ran as host processes through `kubectl proxy`. The in-cluster driver pod, service account/RBAC, DNS names, PVC and result copy were not exercised.
- **Pod CPU.** The request was 10m and the limit 40m. The HPA `averageUtilization` was multiplied by 4 so that it still refers to 40m. Neither change applies in the cloud, where request = limit = 40m.
- **Pilot scope.** The pilot used 3 of the 12 functions (f1, f5, f9: one per stratum) during night hours, with at most 5 pods per function. Peak-scale behaviour (60+ pods per policy) was not tested.

## Capacity (old two-phase rule)

| Phase | Result |
|---|---|
| 1 | r* = 6 rps. Bad fraction is 0 at 2–6 rps, 2.7% at 8 rps and 15% at 10 rps. |
| 2 | μ = 7 rps (420/min) and c = 0.645. |

μ is the top of the phase-2 grid (r*+1).

The multi-pod steps (2/4/8 pods) ran at 5 rps per pod, not at μ = 7. So linearity at μ was **not** shown. The new phase 3 (PROTOCOL Sect. 4) checks k·μ and the peak cohort before μ is accepted.

## Two-hour pilot (calibration day 9; pach, conf, b1, hpa)

### Validity

The run is technically valid:

- decision logs are complete and identical to the offline rule;
- 0 failed scale calls;
- dispatch lag p99 is 10.7 ms;
- telemetry coverage is 100% for every function-policy series;
- no Pending pods.

### Actuator delays (re-analysed with the post-review code)

Each delay is the midpoint of the interval between the last observation before and the first observation after the change. For this run the call time is `decided_at`.

| Quantity | Median midpoint | Median interval |
|---|---|---|
| d_up | 18.67 s | [18.15, 19.17] s |
| d_down | 0.31 s | [0, 0.61] s |

The ~1 s observation interval limits precision; the narrow spread of d_up does not show sub-second accuracy.

### Attribution

| Change | Requested | Observed | Attributed | Attributed twice |
|---|---|---|---|---|
| Pods created | 87 | 87 | 87 | 0 |
| Pods removed | 96 | 88 | 88 | 0 |

Eight removed pods exited between two polls and were not recorded, because the collector had no `gone` column at the time.

### H1 (in sample)

9/9 function-policy pairs are within the criteria. The model was fitted on the same run, so this checks the mechanics only.

### H2, pod-time and request quality

| Policy | H2 compliant | Ready pod-minutes | Bad requests |
|---|---|---|---|
| pach | 3/3 | 937 | 0.03% |
| conf | 3/3 | 891 | 0.03% |
| hpa | 3/3 | 876 | 0.03% |
| b1 | 2/3 | 708 | 0.15% |

The b1 failure is f9: 2 overload minutes against a budget of 1.2.

These numbers describe this pilot only. They do not show that PAC-h keeps a guarantee on live load.

HPA scaled f9 between 2 and 5 replicas (13 scale events). The first pilot start was aborted after 7 minutes: with the 10m request, HPA utilisation was computed against the request and HPA scaled to 20 pods. The harness was fixed and the pilot restarted (`local_stage/`).

## Fixes made during the stage

1. `render.py` writes a single `List` object, which kubectl accepts.
2. `tools/controller.py` finds `policies.py` outside the image.
3. `tools/collector.py`:
   - tolerates HPA objects without `spec`;
   - adds a `gone` column for pods that disappear between polls.
4. `analyze.py`:
   - a partial run is reported as invalid instead of crashing;
   - `--fit-actuator` stops when it has no records;
   - scale-up windows account for the 1 s creation-time resolution.

## Changes after the review of 18 October (`reviews/jocc-k8s-local-2026-10-18`)

1. **Capacity** (`capacity.py`):
   - phase 2 measures single-pod rates only;
   - new phase 3 verifies the candidate μ at 2/4/8 pods × μ and with the peak cohort P (from `plan/sizing.json`) on all workers at once;
   - pre-registered fallback: lower μ by 1 down to r*−2, otherwise the calibration fails;
   - each phase is scored only if delivered exactly as scheduled and with enough CPU samples;
   - `freeze.py` requires a verified `capacity.json` and `plan/` prepared with it.
2. **H3** (`combine.py`):
   - differences use the cross bounds [pach_lo − other_hi, pach_hi − other_lo];
   - the sign test is computed at both ends, and a comparison is "supported" only if the worst case is significant after Holm, "undetermined" if only the best case is;
   - `combine.py` refuses an incomplete campaign (four valid runs with the protocol's assignments and every function-policy row) unless `--descriptive` is given, in which case the output is marked as such.
3. **Coverage and unknown time** (`analyze.py`):
   - coverage is checked for every function-policy series, and a missing series fails;
   - the upper bound on ready capacity in unknown time comes from the decision log: the largest acknowledged replica count in effect from 10 s before the minute to its end, or maxReplicas = 40 for the HPA;
   - every observation above this bound makes the run invalid.
4. **Delays** (`analyze.py`, `tools/controller.py`):
   - the controller logs `called_at`, the start of the successful API request;
   - delays are intervals between observations, and d_up and d_down are fitted on the midpoints;
   - per policy, the analysis reports requested, observed, attributed and censored pod changes, and the fit stops if a pod is attributed twice;
   - fixed a 1 ms rounding mismatch between `counts.csv` and `pods.csv` that had produced negative down delays.
5. **Tests.** The negative and positive tests are executable: `python3 testbed/tests/run_tests.py`, 29 checks, all passing.

## Rehearsal with the final manifests on k3d (Mac mini, 18 October 2026)

`k8s-local-run/local_stage.sh` ran the rehearsal. The archive is `local_stage/mac_k3d_2026-10-18.tar.gz`.

### Setup

- Docker Desktop 27.5.1 with 8 CPUs and 7.7 GiB, arm64.
- k3d 5.9.0 with k3s v1.31.4: 5 agents plus a registry.
- The final manifests ran unchanged: images by digest, the in-cluster driver pod, RBAC, PVC, 12 functions and 4 policies.
- Phase 3 was skipped, because a laptop cannot carry 4 × 66 pods.

### Results

| Check | Result |
|---|---|
| Tests | 32/32 at the time (38 after the changes below) |
| Decision equivalence | all identical |
| Images | built, pushed and pulled by digest |
| Results | fetched from the PVC with `kubectl cp` |
| Driver | ran for 2 × 45 min, 36 min and 120 min |

### Work per request

With ITER = 26400 one request cost 7.8 ms of CPU, more than the 4 ms quota of the 40m pod per 100 ms. Every worker then failed the 1% criterion already at 2 rps.

This led to the ITER rule in phase 1 (PROTOCOL Sect. 4). ITER was set to 11900, giving 3.9 ms per request. Calibration then gave r* = 4, μ = 4 rps and c = 0.39; w1 (the node that also runs system pods) was the slowest.

### Pilot (calibration day 9, 120 min, run-1 assignment)

**Validity and actuation**

- Valid; all 36 custom pairs implementation-exact; 0 failed scale calls; dispatch lag p99 7.5 ms.
- In minute 92 a gap between observations (largest 6.3 s within one series) left every series below 95% coverage in 1 of 120 minutes, within the 1% rule.
- Attribution: 160/160 created and 190/190 removed pods attributed, none twice.
- d_up = 18.5 s with interval [18.0, 19.0] s; d_down ≤ 0.63 s.

**Results per policy**

H1 was checked in sample: 36/36 pairs within.

| Policy | H2 compliant | Ready pod-minutes |
|---|---|---|
| pach | 12/12 | 3123 |
| conf | 12/12 | 3043 |
| b1 | 10/12 | 2584 |
| hpa | 4/12 | 2221 |

**Proxy overload versus users**

| Policy | Overload function-minutes (proxy) | Function-minutes with >1% bad requests | Bad requests |
|---|---|---|---|
| pach | [1, 1] | 21 | 0.11% |
| conf | [3, 3] | 18 | 0.12% |
| b1 | [6, 6] | 27 | 0.16% |
| hpa | [315, 317] | 20 | 0.14% |

Of 1,440 function-minutes per policy. Only 4 of the HPA's 315 definite overload minutes had more than 1% bad requests. The pilot therefore shows how the policies keep the capacity margin defined by μ, not an advantage of PAC-h in service quality. The proxy and the user metrics are reported together (PROTOCOL Sect. 6, H4). The HPA rule is kept.

**CPU time per request**

- Calibration (single pods): 3.9 ms; 3.5–4.2 ms over the 2–10 rps levels.
- Pilot: 2.4–3.0 ms, independent of the per-pod rate (2.4 ms at 4–5 rps per pod), with every ready pod present in the metrics.
- The review found that `analyze.py` grouped the CPU of a function's pods by identical timestamps, which can split one metrics response. The collector now writes a snapshot number per response and the analyzer sums within snapshots (for old files, snapshots are reconstructed from gaps between rows).
- The Mac pilot value is unchanged by the fix (2.5 ms), so the cause of the gap is not established; it may lie in the host (Docker VM on heterogeneous Apple cores). On the cloud, a gap above 20% is investigated before the freeze (PROTOCOL Sect. 8).

## Review of the Mac rehearsal (`reviews/jocc-k8s-mac-2026-10-18`) and changes

The reviewer reproduced the results from the raw CSV files and agreed to proceed to the cloud.

1. **CPU per snapshot.** The collector writes `snapshot` and `metrics_timestamp`; the analyzer sums within snapshots; a test covers the reviewer's counterexample.
2. **No attempt is lost.** `ops/cloud_stage.sh` is now inside `testbed/` and therefore frozen. It:
   - copies raw files, manifests, logs and the reason before deleting a namespace;
   - never overwrites a result folder and lists every attempt in `results/attempts.tsv`;
   - verifies that deleted namespaces are gone;
   - refuses to start over a running driver, retries the API for 30 minutes and resumes with `attach`.
3. **Freeze.**
   - `freeze.py --verify` compares every input with `FREEZE.json` before each main run.
   - Admission requires a verified capacity and a valid pilot only.
   - Image digests come from the build metadata, are checked against the registry, and are recorded per run against the running pods.
