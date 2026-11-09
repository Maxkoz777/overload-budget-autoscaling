# BACC-r: mapping to arXiv:2606.20575v2 (13 October 2026)

BACC-r is our re-implementation of the controller described in v2. It is not a reproduction of the published system: forecasts, the replica profile and the actuation queue are those of our replay. No public code existed when this mapping was written.

| v2 element (section / equation) | BACC-r (adaptive/policies.py, `bacc`) | Adaptation |
|---|---|---|
| Decision interval h = 5 min, startup delay d (Sect. III-C1, Table II) | regular decision when (j + tau) mod 5 = 0; d = tau | d is the replay delay tau |
| Forecast of h + d minutes, AutoARIMA (Sect. IV-A(e)) | persistence: every forecast equals the last observed demand | same forecast as PAC-h; AutoARIMA not used |
| Service window W_k = {t_k+d+1, ..., t_k+d+h}, peak target, Eq. (3) | `score`: max of demand over minutes a+d ... a+d+h-1 of decision j (data through a-1) | none |
| Ratio score with c, clipped to [1/20, 20], Eq. (4) | `score`; c = c_mult * mu | c not given in v2; c = mu primary, 0.1 mu and 10 mu as sensitivities |
| Calibration history of 12 h, oldest dropped (Sect. IV-A(e)) | scores whose window completed within the last 720 minutes; pre-filled from the 720 minutes before the period | none |
| Rank j_k = min{m, ceil((m+1)(1-a_k))}, q = 1 without scores, Eq. (5) | `jr`, `q` | none |
| Calibrated demand, Eq. (6) | `ytil` | none |
| Profile f, lambda_safe, Eqs. (7)-(8) | x = max(1, ceil(ytil / mu)) | overload proxy is a step at mu, so lambda_safe = mu |
| Replica bounds and maximum change per decision, measured from the planned count (Sect. III-C3b, Table II: 100) | every issued target, regular or early, is limited relative to the last issued target, then bounded to [1, M_u] | M_u is the common resource bound of all controllers; 'no limit' and an HPA-style limit are sensitivities |
| Pacing, Eq. (9): rolling (w = 1440) and cumulative rates, eta 1.0 / 0.2, bounds [0.2 eps, 0.9] | `a_pace` | one accounting unit per minute |
| Remaining-budget ceiling, Eqs. (10)-(12), Jeffreys 1/2, kappa in [0.1, 10], a_cal,min = 0.1 eps | `u`, `pF`, `kappa`, `ak` | A_k: sum of the plan's a over all minutes with plans (canonical); 'missed minutes only' is a deliberate ablation |
| Budget-aware trigger and Replan (Sect. III-E, Algorithm 1 lines 13-15) | non-regular minutes: if demand exceeds the sizing workload of the plan in force and any violation rate exceeds eps, the sizing workload becomes max(calibrated peak of the latest regular decision, y_early), y_early = y + (1+d) max(y - y_prev, 0); increase only; held until Ready (j + tau) | 'minimum sizing workload' read literally; the exploratory v0 reading (y_early multiplied by q) is a sensitivity |
| Accounting period T, budget eps (Eqs. (1)-(2)) | T = evaluated period of the unit; eps = delta; minute-based allowance floor(eps T) | none |
| Scale-down effective in the next minute (Sect. III-C3b) | scale-down passes through the same tau-minute queue as scale-up | common actuation queue of the replay |
