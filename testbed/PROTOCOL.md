# Kubernetes testbed validation of PAC-h: protocol

Status: **draft v3** (after the review of the local engineering stage, 18 October 2026). It is frozen by a commit before the main
runs; `freeze.py` then records the commit, the image digests, and the SHA-256 of every input in
`FREEZE.json`. Nothing in this file changes after the freeze; deviations are listed in
`DEVIATIONS.md` with the reason. All runs use one cloud provider and one cluster.

## 1. Question

How does PAC-h behave when it controls a real Kubernetes cluster, and how well do the article's
replay assumptions describe it? The testbed replaces the replay's idealised actuation (a fixed
FIFO delay of whole minutes) by the real path: scale API, scheduling, container start, application
warm-up, readiness, and Service endpoints. The days replayed here (relative days 10 and 11 of the
confirmatory Huawei periods) were already replayed in the article, so the testbed checks the
transfer of the mechanism to a live system; it is not a new out-of-sample test of PAC-h. Claims
are limited to this service, these profiles, and this cluster.

## 2. Fixed inputs

- **Data.** The three confirmatory Huawei 2023 periods of the article (Sect. 6.3). For test day D
  (10 or 11) of a selected function: the 3,000 minutes before the warm-up are the controllers' prior
  data, the 30 minutes before day D are a live warm-up, and the 1,440 minutes of day D are the test.
- **Selection** (`prepare.py`). Eligible: peak demand on days 0-9 needs at most 10 capacity units and
  the median on calibration days 8-9 at least 2. Eligible unit-runs are split into terciles of
  burstiness (peak-to-mean demand on days 0-9); with seed 20261017, four per tercile are drawn
  without repeating a function. Result: 12 functions (periods C 7, A 3, B 2); no test-day data used.
- **Scaling.** Per-minute request counts are scaled so that one pod serves `mu` requests per second
  (Sect. 4) and rounded to integers; replica counts equal the replay's up to this rounding.
- **Policies**, one namespace each:

  | Policy | Definition |
  |---|---|
  | `pach` | PAC-h (Algorithm 1): persistence forecast, (tau+1)-step scores with tau = 1, PAC rank, delta = 1%, eta = 0.05, W of the function's period in the article (A, B: 1440; C: 2880), no offset |
  | `conf` | as `pach` with the standard conformal rank (matched-rank ablation, Sect. 6.2) |
  | `b1` | reactive B1; threshold u and cooldown selected on days 8-9 at delta = 1% by the article's rule (cheapest calibration-feasible pair, u in {0.5, ..., 0.99}, cooldown in {1, 3, 5, 10}) |
  | `hpa` | Kubernetes HPA, default behaviour, target CPU utilisation round(100 u c), u of B1, c from Sect. 4; minReplicas 1, maxReplicas 40 |

  The custom controllers decide 0.5 s after each minute ends, once the load generator has
  dispatched every request of that minute, and call the scale API.
- **Pod.** CPU request = limit = 40m, memory 32/64 MiB, work per request fixed in CPU time: ITER
  SHA-256 rounds such that one request costs 3.0-4.0 ms of CPU on the cluster's workers (default
  ITER = 26400, set in calibration phase 1, Sect. 4), application warm-up 15 s, readiness probe every 2 s (3 failures), load shedding above 8
  concurrent requests. The 40m pod is calibrated anew (Sect. 4); its dynamics are not assumed to
  equal those of the 50m pilot. Images are pinned by digest.
- **Load.** Open loop. In minute m each function receives exactly the scheduled number of requests
  at uniformly random times within the minute, identical across policies; new connection per
  request; timeout 3 s. A request is bad if it is rejected (503), times out, fails, or takes longer
  than 500 ms (the testbed's SLO, not the SLO of the Huawei functions).
- **Time.** Decision interval 60 s of wall-clock time, no compression.

## 3. Cluster and runs

Four identical workers (4 vCPU, 8 GiB, non-preemptible, guaranteed CPU share 100%, node autoscaling
off) and one driver node of the same size, all in one node group. At cluster creation the five
nodes are sorted by name; the first is the driver node (label `load`), and the other four receive
the labels `testbed-node=w1..w4` in the order of a permutation drawn with seed 20261017. The mapping is recorded in `cluster/cluster.json`.

| Run | Test day | Load seed | pach | conf | b1 | hpa |
|---|---|---|---|---|---|---|
| 1 | 10 | 1 | w1 | w2 | w3 | w4 |
| 2 | 11 | 1 | w2 | w3 | w4 | w1 |
| 3 | 10 | 1 | w3 | w4 | w1 | w2 |
| 4 | 11 | 1 | w4 | w1 | w2 | w3 |

Every policy runs once on every worker, and every day twice. Runs 1/3 and 2/4 replay identical
requests; their differences combine worker and time effects ("platform variation"), not load noise.
The 48 function-runs per policy are repeated measurements of 12 functions.

**Sizing rule.** For every custom policy, the peak of the concurrent sum of replicas of the 12
functions on days 3-9 (`plan/sizing.json`) times 40m times 1.25, plus the CPU requests of the
DaemonSets measured on the cluster, must not exceed a worker's allocatable CPU. On days 3-9 the
peak is 66-67 pods, that is 3.35 vCPU before DaemonSets. If the rule fails on 4-vCPU workers, all
workers are replaced by 6-vCPU workers before the freeze. The 12 functions are kept. Pending or
unschedulable pods in the main runs are outcomes and are reported, not corrected.

## 4. Calibration on the cloud cluster (before the freeze, days 8-9 data or synthetic load only)

**Capacity mu and CPU utilisation c** (`capacity.py`), on every worker in parallel, with the final
manifests (CPU request = limit = 40m):
- phase 1: one pod; 3 minutes at 2 requests/s, then 4, 6, ..., 30 requests/s, 3 minutes each.
  First, the CPU time of one request is the mean over workers of the pod CPU in the last two
  minutes of the 2 and 4 requests/s levels divided by the rate. If it lies outside [3.0, 4.0] ms,
  ITER is rescaled to round(ITER x 3.5 / measured, -2) (`pod_iter.json`) and phase 1 is repeated.
  This keeps the work per request comparable across clusters (on the 40m pod, a request that
  costs more than the 4 ms CFS quota of one 100 ms period cannot be served within a period).
  Then r1_w = highest level whose last two minutes have a bad fraction below 1%; r* = min_w r1_w.
- phase 2: one pod at r*-2, r*-1, r*, r*+1 requests/s, three repeats each in a seeded random order,
  3 minutes each. mu_w = highest rate at which all three repeats have a bad fraction below 1% in
  their last two minutes; the candidate **mu = min_w mu_w** (one common scale for all policies).
  c = mean pod CPU at rate mu (last two minutes, all repeats and workers) / 40m.
- `prepare.py` with the candidate mu (selection, schedules, B1 pairs, HPA targets, sizing).
- phase 3 (verification of the candidate, all workers at the same time): 2, 4, and 8 pods at
  k x mu requests/s, 5 minutes each; then the **peak cohort**: P pods at P x mu requests/s for 5
  minutes, P = the largest peak of concurrent pods of one custom policy in `plan/sizing.json`
  (the 2-, 4- and 8-pod Deployments plus P - 14 further pods, all loaded at mu per pod). Every step
  must have a bad fraction below 1% in its last two minutes on every worker. If a step fails, mu is
  lowered by 1 request/s, `prepare.py` is rerun, and phase 3 is repeated at the new mu; mu is never
  lowered below r*-2 (the lowest rate whose c is measured). If phase 3 fails at r*-2, it is
  repeated once at r*-2 with the same criterion (amendment A1, Sect. 10: a change of the acceptance
  rule); if the repeat fails, this calibration procedure stops and is redesigned before the freeze. Only a verified mu is
  called the testbed's capacity; it is an operational capacity at the 1% criterion, not a proven
  maximum, and the grid is not extended after the main runs.
- A phase is scored only if it was delivered exactly as planned (every second of every target,
  planned = schedule, dispatched = planned, an outcome for every request) and every CPU estimate
  has at least four samples; otherwise it is repeated. All curves, failed verifications and
  repetitions are kept in `capacity.json` and the raw result folders.

**Actuator model** (`analyze.py --fit-actuator`, cloud pilot of two hours of calibration day 9,
all four policies on the run-1 assignment, controllers with causal history):
- target quantity: the number of pods that are **ready endpoints** of the Service;
- a call's time is the start of its successful API request (`called_at`);
- pods are attributed to the scale call in whose window they were created (scale-up) or removed
  (scale-down); creation times have 1 s resolution, so a scale-up window ends 1 s before the next
  call. The analysis reports, per policy, the requested and observed pod changes, attributed,
  unattributed and censored pods, and refuses the fit if a pod is attributed twice;
- the change of a pod's endpoint status lies between two consecutive observations (about 1 s
  apart); d_up and d_down are the medians of the midpoints of these intervals, from the call to
  the pod first being a ready endpoint and to the removed pod leaving the endpoints. The medians of
  both interval ends are reported; a d_down below the observation interval is reported as such;
- rule: at a scale call to k replicas, if k exceeds the live count (ready + pending - scheduled
  removals), the difference becomes pending pods that turn ready at call + d_up; if k is lower,
  pending pods are cancelled first (latest first) and the remaining difference leaves at call +
  d_down; the model starts from the first observed ready count of the run.
- first-served time (first successful response from the pod) is reported separately.
The model and its fit are fixed in `actuator_model.json` before the freeze.

## 5. Measurements

For every function, policy, and test minute m:
- **ready capacity**: time integral of the number of ready endpoints, observed about once per second
  with its own timestamp; an observation holds until the next one if they are at most 2.5 s apart,
  otherwise the remaining time is unknown;
- **overload**: requests in m > mu x 60 x mean ready capacity in m. With unknown time, the capacity
  lies between the integral over known time (unknown = 0) and that integral plus the unknown time
  times an upper bound on ready endpoints: for a custom policy the largest replica count
  acknowledged by the API and in effect at any time from 10 s before the minute to its end (from
  the decision log, not from telemetry; every observation exceeding this bound is reported and
  makes the run invalid), for the HPA its maxReplicas (40). The minute is overloaded, not
  overloaded, or unknown. Function-run overload is reported as an interval [definite, definite +
  unknown]; unknown minutes never count as successes;
- **pod-time**: ready pod-minutes, existing (not terminating) pod-minutes, and terminating pod-minutes,
  each separately; these are resource proxies, not cloud cost (the workers are paid for in any case);
- **users**: bad-request fraction; shares of 503, timeouts, and errors; p95/p99 latency of successful
  requests (histogram buckets), always shown with the failure shares;
- **actuation**: for every scale call of a custom controller, the times to pod creation, scheduling,
  container start, Ready, ready endpoint, and first successful response; for scale-downs, the times
  to deletion and to leaving the endpoints; HPA pods relative to their creation;
- **FIFO reference**: overload under the article's FIFO actuation with the logged decisions shifted by
  0 or 1 minute (descriptive, decisions fixed at h = 2).

## 6. Hypotheses and criteria

- **A. Implementation (condition for validity).** For every custom policy, function, and minute,
  the logged decision equals the offline rule's decision on the logged request counts with the same
  history and initial state (full arrays; an empty or incomplete log fails).
- **H1 (primary, actuator model).** For each of the 48 function-runs of `pach`, `conf`, and `b1`, the
  model's ready capacity is computed from the logged scale calls and compared with the observed one:
  (i) |model overload minutes - observed overload minutes| <= 2 for every value in the observed
  interval, and (ii) observed / model ready pod-minutes in [0.95, 1.05]. A function-run is
  "within" if both hold, "outside" if either fails for every admissible value, otherwise
  "undetermined". Criterion per policy: at least 90% of its 48 function-runs within (undetermined
  counts as not within) and, for every run, the aggregate ready pod-minutes ratio in [0.95, 1.05].
  Also reported: the signed and absolute overload gaps, false-positive and false-negative minutes,
  F1 over minutes with known status (1 if neither has an overload minute), the mean absolute error
  of the ready capacity in pods, and the FIFO references. These tolerances are engineering choices
  for this one model, not consequences of the theory.
- **H2 (practical target).** `pach` keeps at most 14 overload minutes of 1,440 on every function in
  every run: "compliant" if the upper end of the interval is at most 14, "violated" if the lower end
  exceeds 14, otherwise "undetermined". This is not a consequence of the coverage theorem (even with
  independent overload of exactly 1% per minute, P(at most 14 of 1,440) = 0.53); a violation is
  neither a code error nor a refutation of the theorem. Reported for every function-run with
  overload, budget excess, pod-time, and user metrics.
- **H3 (exploratory).** Function as unit (overload minutes averaged over the four runs): one-sided
  sign tests pach < conf and pach < b1, Holm-adjusted over the two tests; bootstrap over functions
  (10,000 resamples, 12 clusters) of the mean difference. With overload intervals [lo, hi], the
  difference pach - other of a function lies in [pach_lo - other_hi, pach_hi - other_lo]. The sign
  test is computed at the upper ends of these intervals (largest admissible p-value) and at the
  lower ends (smallest); "supported" if the largest Holm-adjusted p is below 0.05, "not supported"
  if the smallest is not, otherwise "undetermined". The bootstrap interval is reported for both
  ends. H1-H3 are computed only for a complete campaign of four valid runs with every function and
  policy (`combine.py` refuses otherwise; `--descriptive` output is marked as such).
- **H4 (descriptive).** HPA against B1 (no universal ordering is claimed: the HPA observes CPU every
  15 s, the custom controllers observe requests every minute); delay distributions; user metrics.
  For the HPA, overload (a proxy defined by mu) is always shown together with the user metrics
  (bad-request share, 503s, timeouts, latency) and the proxy-versus-users table, because the HPA
  can run pods above mu while users see few bad requests (observed in the local pilot). The CPU
  time per request in each run (`analyze.py`, `cpu_ms_per_request`) is reported next to the value
  implied by the calibration (c x 40m / mu); the HPA target rule is not changed after the pilot.
- **Proxy and users (descriptive).** Per minute, overload status against a bad-request share above 1%
  (2 x 2 table) and the bad-request share as a function of requests / (mu x ready capacity).
- **Repeatability (descriptive).** Differences between runs 1 and 3 and between runs 2 and 4.

## 7. Validity of a run

A run is **invalid** only when measurements are lost or the protocol was not executed: a missing
or crashed driver container; an incomplete or mismatching decision log (Hypothesis A); any
target-minute with planned requests different from the schedule, dispatched different from planned,
or requests without outcome after the drain; telemetry coverage below 95% in more than 1% of the
test minutes of any function-policy series (a missing series fails); more ready endpoints observed
than the replica bound of Sect. 5; or a 99th percentile of the per-second maximum dispatch lag
above 1 s. Technical validity and the scientific criteria are separate: in a valid run, unknown
time can still make H1 or H2 undetermined. An invalid run is kept, reported with its reason, and repeated with the same day, seed, and
assignment. Overload, 503s, readiness failures, Pending pods, failed scale calls, and API delays
are outcomes, not reasons for invalidity.

## 8. Order of work

1. Local engineering checks (done 18 October 2026 on single-node k3s, `LOCAL_STAGE.md`):
   equivalence (`check_equivalence.py`), `tests/run_tests.py`, capacity phases and a two-hour pilot
   with three functions and host-side driver processes; then a rehearsal with the final manifests on
   k3d (in-cluster driver, 12 functions, 4 policies). These data are engineering checks only and are
   archived in `local_stage/`.
2. Cloud cluster, with the final manifests and images (in-cluster driver pod, service account and
   RBAC, PVC, all four policies and 12 functions): provenance (`record_cluster.sh`), capacity
   phases 1-3 with `prepare.py` between 2 and 3, sizing check against allocatable CPU and
   DaemonSets, two-hour pilot on calibration day 9 with `--fit-actuator`. The pilot also checks
   CPU throttling, Pending pods, the driver's stability and the storage needed for a full day. If the
   pilot's CPU time per request differs from the calibration's by more than 20%, the cause is
   investigated and the calibration may be repeated before the freeze (calibration data only).
   Every engineering and main run, including failed, invalid and repeated attempts, is kept with
   its raw files, manifests, logs and the reason (`ops/cloud_stage.sh` copies them before any
   namespace is deleted and lists them in `results/attempts.tsv`). Image digests are taken from the
   metadata of the build that was pushed and checked against the registry and the running pods.
3. Fixes from step 2 are committed (local branch, not published until the study is complete);
   `prepare.py` with the verified capacity; `freeze.py`; commit `FREEZE.json`. The freeze covers all
   of `testbed/`, including the operator script `ops/cloud_stage.sh`. Admission to the main runs
   requires only a verified capacity and a technically valid pilot; no result of H1-H3 is a
   condition. Before each main run, `freeze.py --verify` compares every input with `FREEZE.json`.
4. Runs 1-4 as in Sect. 3; `analyze.py` after each; `combine.py` at the end; back up all data;
   delete the cluster.

## 9. Reporting

All results of the four runs are reported whatever their direction, including failed criteria,
undetermined outcomes, and invalid runs with their repetition. A comparison in which the HPA meets
the budget with fewer pod-minutes than PAC-h is reported as it is; it is not a reason to change
the setup. Results are described as measurements on this testbed: they do not establish the
exchangeability assumed by the coverage theorem for live load, and a measured actuation delay does
not make the FIFO model exact or conservative.

## 10. Amendments before the freeze

**A1 (2026-10-21, VK Cloud cluster, after the first failure at mu = r*-2, before the pilot and any
main-run data).**

*Observations.* Phase 3 failed at mu = 5 (w1 peak cohort: 476 of 40,200 requests bad, 1.18%) and
at mu = r*-2 = 4 in one of the 16 checks (w1, 2 pods: 20 of 960, 2.08%: 19 successful responses
slower than 500 ms and one timeout, no 503). The other 15 checks at mu = 4 passed, among them the
w1 peak cohort (132 of 31,680, 0.42%). In the failed evaluation window (seconds [180, 300)) the bad
requests fall in seconds 273-276 (4, 8, 4, 4); at mu = 5 the same step had bad requests in seconds
206, 269, 280 and 287 (1, 1, 5, 4). Short latency bursts also occur on several targets in the same
second (for example seconds 420-453 of the mu = 5 attempt, outside any evaluation window). All pods
were Ready several minutes before t0 and none was replaced; the cluster events contain no warnings
during the load. The cause of the bursts is not established: the shared load generator and network
path, CPU scheduling on the nodes and CPU throttling at the 40m limit are possible explanations, and
no throttling telemetry was recorded. The failure at mu = 4 remains a result of the calibration.

*Change of the acceptance rule.* At r*-2 a failed phase 3 is repeated **once**, as a full repeat
of all 16 checks with the same mu, load, steps, evaluation windows, 1% criterion and nodes. mu is
accepted only if every check of the repeat passes. Granting a second attempt raises the probability
of acceptance (with independent attempts that pass with probability q, from q to 2q - q^2;
independence is not established here). A passed repeat therefore means "mu = 4 is accepted as the
working scale of the testbed under amendment A1; the first attempt at this mu failed", not that the
true bad-request share is shown to be below 1%. All attempts (mu = 5, both attempts at mu = 4) are
reported and are not pooled under another criterion. If the repeat fails, this calibration
procedure stops; the calibration or the infrastructure is redesigned and documented as a new
amendment, without further repeats. The repeat is counted from `capacity.json`, which is not reset.

*Clarifications made together with A1.*
- CPU in phase 3 is descriptive; the four-sample requirement applies to the CPU estimates of phases
  1-2 (cost per request, c). Per-worker c at mu = 4 (about 0.415, 0.345, 0.389, 0.346 for w1-w4)
  is reported next to the common c = 0.3735; the HPA target rule is unchanged. The resulting HPA
  targets (19-37% of the 40m request) describe a specifically calibrated HPA, not a typical
  60-80% configuration, and are reported as such.
- r* is an operational estimate on the rate grid under the user criterion, not a proven maximum;
  mu = 4 is "one third below r*", not a guaranteed physical headroom.
- The nodes are kept as they are (no replacement of w1 before or after the repeat); node identities
  and the per-node cost per request are recorded. The rotation balances a constant additive node
  effect; it does not remove policy x node or day x node interactions or changes of the hosts over
  time (each policy sees each trace day on two nodes). Results are shown per run next to the
  aggregates; H3 stays exploratory, and the 48 function-runs share nodes and the driver and are not
  48 independent cloud replications.
- User metrics are computed over all requests, including latency episodes of unknown or presumed
  infrastructure origin; no seconds are removed. Episodes may be marked for diagnosis; an analysis
  without them is at most supplementary, under one rule fixed before the main runs and identical
  for all policies (none is planned). A bad SLO does not invalidate a correctly executed run. A
  result on the mu x Ready proxy (H1-H3) does not establish a user-level SLO, and the infrastructure
  can also affect Ready times, scaling delays and the HPA through CPU.
- The pilot that fits the actuator model is not an independent test of H1.
