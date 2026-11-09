# Finite-horizon bound of the implemented ACI-G

Setting (adaptive/policies.py, `aci`). Test target minutes i = 0..T-1; the decision for minute i is issued at j = i - tau with data through j - 1. Lane l = j mod (tau+1). In lane l the decisions form a subsequence j_1 < j_2 < ... < j_{n_l}, consecutive elements tau+1 apart, with sum_l n_l = T. The first decision of each lane uses alpha = alpha*; afterwards alpha_{j} = alpha_{j-tau-1} + gamma (alpha* - err_{j-tau-1}), and err_{j-tau-1} (target minute j-1) is observed before decision j.

Set convention. alpha <= 0, or rank ceil((W+1)(1-alpha)) > W: infinite set, err = 0. alpha >= 1: empty set, err = 1. Otherwise err = 1{y_i > forecast + q}.

Lemma (Gibbs and Candes 2021, Lemma 4.1 and Prop. 4.1, applied to one lane). Under this convention alpha stays in [-gamma, 1+gamma], and for every lane
|sum_{k<=n_l} err_{j_k} - n_l alpha*| = |alpha_{j_{n_l}+tau+1} - alpha*| / gamma <= (max(alpha*, 1-alpha*) + gamma) / gamma.

Summing over the tau+1 lanes,
(1/T) sum_i err_i <= alpha* + (tau+1)(max(alpha*, 1-alpha*) + gamma)/(gamma T) <= alpha* + (tau+1)(1+gamma)/(gamma T).
With alpha* = delta - (tau+1)(1+gamma)/(gamma T) > 0, the formal miscoverage of every unit is at most delta over the test period. This is deterministic and needs no assumption on the data.

From formal errors to overload. The applied capacity is cap_i = min(M_u, D_i), with D_i = max(1, ceil((forecast+q)/mu)) for a finite set, M_u for an infinite set and 1 for an empty set. If the set is finite and D_i <= M_u, then cap_i * mu >= forecast + q, so overload at i implies err_i = 1. Overload without a formal error can therefore occur only in exceptional minutes: (a) infinite set, where capacity is M_u rather than unbounded; (b) finite set with D_i > M_u (resource bound binding); (c) for Alibaba, the first tau minutes, which keep the observed initial capacity. Hence

overload fraction <= delta + (exceptional overload minutes)/T,

and the budget is guaranteed only when no overload occurs in exceptional minutes. The scripts report formal errors, actual overload and the three kinds of exceptional overload separately.

Status. The ACI update and its bound are those of Gibbs and Candes; the interleaving for delayed feedback, the finite-horizon target alpha* and the resource bound are our adaptation (ACI-G). Plain ACI (alpha* = delta, gamma = 0.005) has the same per-sequence bound, which at our horizons is (tau+1)(1-delta+gamma)/(gamma T), e.g. about 0.07 for tau = 1 and T = 5760.
