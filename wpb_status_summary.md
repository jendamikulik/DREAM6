# Status report: toward $\mathcal B_n = \Theta(\log n)$ for an explicit quartic zero-repair pair

*Prepared for a general mathematically-trained reader. Written by an AI assistant (Claude) summarizing a joint numerical/exploratory investigation. Nothing below should be read as a peer-reviewed result; see the epistemic tags.*

## 0. How to read this document

Every claim below is tagged:

- **[PROVEN]** — a short, elementary, rigorous argument exists and has been checked by hand; no computation is required to trust it.
- **[NUMERICAL]** — established by explicit computation on finitely many concrete instances ($N = 65, 129, 257, 513$), typically cross-checked by two independently written code paths converging to the same figures to 6–13 significant digits. This is strong *evidence*, not a theorem.
- **[OPEN]** — stated but not established in either sense.
- **[REFUTED]** — a specific hypothesis that the numerics ruled out.

Where two figures for the same quantity are reported, they come from two independently implemented pipelines (different code, sometimes different specific coupling realizations), run by two different parties in the collaboration and cross-checked against each other. This is a meaningful form of replication, but it is not formal verification and not peer review.

## 1. The target statement

Let $\mathcal B_n$ be defined via a *causal zero-repair tree*: increment laws $\mu_P,\mu_Q$ on a finite alphabet, and for each history $h$ of length $\le n$ a buffer law $B_h$, subject to the exact recursion
$$\mu_Q * B_h = \sum_i p_i\,\delta_{x_i} * B_{hi}\qquad(\text{no repair permitted after a mismatch is observed}).$$
$\mathcal B_n$ is the infimum of $\mathbb E_{B_\varnothing} J$ over feasible such trees of depth $n$ — informally, the minimal expected buffer that lets a fixed causal (non-anticipating) coupling of the two increment processes run for $n$ steps.

**[PROVEN, taken as given from earlier in the program]** $\mathcal B_n = \Omega(\log n)$.

**[OPEN, the actual target]** $\mathcal B_n = O(\log n)$, hence $\mathcal B_n = \Theta(\log n)$.

The reduction strategy: it suffices to prove a scale-uniform doubling estimate
$$\mathcal B_{2n} \le \mathcal B_n + C,$$
which in turn would follow from a **Safe-Anchor Lemma**: existence, at every dyadic scale $m$, of an admissible parent state with a *prefix-compatible* continuation whose descendant remains inside a fixed tube around a critical fixed point, at a bounded per-step "inventory" cost $\Delta\mathcal B \le C_{\rm bill}$.

This entire report concerns the search for that lemma. **No progress on $\mathcal B_n$ itself is claimed.** What follows is entirely about the *geometry and dynamics of the reduced model* that the lemma would need to control.

## 2. The critical instance

The explicit extremal pair is
$$P = \left(\tfrac18,\tfrac14,\tfrac14,\tfrac14,\tfrac18\right),\qquad Q = \left(\tfrac{7}{64},\tfrac{5}{16},\tfrac{5}{32},\tfrac{5}{16},\tfrac{7}{64}\right),$$
with probability generating functions satisfying $G_P(z) - G_Q(z) = \tfrac{1}{64}(1-z)^4$: $P,Q$ agree in their 0th–3rd moments and first differ at 4th order. This quartic-order agreement is what forces the $\log n$ scale in the lower bound and is the source of the critical geometry below.

**[PROVEN]** At the associated reduced fixed point, with $r = 1-m$, $R = (2-\mu_2)/r$:
$$T = (m,\mu_2,\mu_4) = \left(\tfrac79,\tfrac43,6\right),\qquad r = \tfrac29,\ R = 3,\ rR^2 = 2,$$
so the normalized quartic defect $\chi'/\chi = 2/(rR^2) = 1$ exactly (marginal, neither growing nor decaying to leading order).

## 3. The formal framework (rigorous core)

For two probability laws $A, B$ on a finite grid at scale $m$, let $\mathcal P_m = \Pi(A,B)$ be the transportation polytope of couplings, $F_m:\mathcal P_m \to \mathbb R^3$ the map $\pi \mapsto (\text{mass}, \mu_2, \mu_4)$ of a matched/residual construction, and $\mathrm{Child}_m$ the renormalization map sending a coupling to a new pair of laws at scale $\sim 2m$ (via a residual self-convolution and an affine re-normalization step).

Define the **Farkas margin**
$$\delta_m(y) = \inf_{\|z\|_1 = 1}\big[h_m(z) - z\cdot y\big],\qquad h_m(z) = \sup_{\pi\in\mathcal P_m} z\cdot F_m\pi.$$
By convexity of $F_m(\mathcal P_m)$ (a linear image of a polytope), $\delta_m(y)\ge\rho$ iff $y+\rho[-1,1]^3\subseteq F_m(\mathcal P_m)$, and — because a cube is the convex hull of its $2^3$ vertices — this containment can be *certified exactly* by checking only the $8$ cube vertices, an important simplification used throughout.

**[PROVEN]** Reduced repair/Hoffman-type estimate: if $y+\delta B_\infty^3 \subseteq F_m(\mathcal P_m)$ and $\|F_m\pi - y\|_\infty = \varepsilon$, then there is $\pi'$ with $F_m\pi' = y$ and
$$\|\pi'-\pi\|_1 \le \frac{2\varepsilon}{\delta+\varepsilon} \le \frac{2}{\delta}\varepsilon.$$
(Elementary convex-combination argument; independent of $m$.)

## 4. What the numerics established this round

### 4.1 Infrastructure

**[NUMERICAL]** The full renormalization pipeline (bootstrap from a fixed seed, exact target-fitting at $N=65$, iterated $\mathrm{Child}$ map) is deterministic and was reproduced **bit-for-bit** from source by an independent execution, including at $N=129$.

**[NUMERICAL]** A standalone reduced-implementation of the profile map contained a normalization bug ($\sqrt{\mathrm{Var}A+\mathrm{Var}B}$ instead of $\sqrt{(\mathrm{Var}A+\mathrm{Var}B)/2}$); after the one-line fix it reproduces the reference implementation's $F_m$ values to machine precision. Two independent, unrelated numerical pathologies were also identified and diagnosed in the shared solver stack: (i) HiGHS false-infeasibility at large scale from mismatched absolute feasibility tolerances across marginals spanning $10^{2}$–$10^{-28}$ in magnitude; (ii) the same effect at the level of individual decision-variable bounds when couplings are parametrized additively from a base point.

### 4.2 The controller basis and its limits

**[NUMERICAL]** A 3-dimensional "circulation" basis for locally perturbing $F_m\pi$ within its fiber was found; a specific triple (dubbed `core075×cos1`) gives a marginally better worst-case-over-scales conditioning ($\sigma_{\min}\approx 0.00404$ vs. $0.00394$ for the originally reported triple, both at $N=65,129,257$).

**[REFUTED]** That this local-conditioning quality (a property of $\sigma_{\min}$ of the controller Jacobian) predicts anything about descendant safety. On a fixed $N=129$ parent, the `core075×cos1` direction gave a **worse** safe-deficit at the child scale (see §4.4) than the naive baseline, despite better raw target-margin, despite identical active-facet geometry.

### 4.3 Failure of one-step contraction

Define the **safe-deficit** $d_m^{\rm safe}$ as the $L^\infty$ distance (in scaled coordinates) that a target $T$ must be moved before a center with certified reduced margin $\ge \rho_0 = 5\times10^{-4}$ exists in the child's own reachable region.

**[NUMERICAL, two independent branches]**

| branch | $d_{129}^{\rm safe}$ | $d_{257}^{\rm safe}$ | $d_{513}^{\rm safe}$ | ratio $257\!\to\!513$ | ratio $129\!\to\!513$ |
|---|---|---|---|---|---|
| A (naïve nearest-center lineage) | 0.012713 | 0.003798 | 0.010947 | **2.88** | 0.861 |
| B (independently-selected lineage) | 0.010266 | 0.006144 | 0.007228 | **1.18** | 0.704 |

**[REFUTED]** The hypothesis $d_{2m}^{\rm safe} \le \lambda\, d_m^{\rm safe}$ for a universal $\lambda<1$: both independently-computed branches show *growth* at the $257\to513$ step. The two-step ($129\to513$) ratio is $<1$ on both branches, leaving a weaker "even-step" or aggregate boundedness hypothesis open (untested beyond one doubling as of this report).

### 4.4 Greedy current-optimization is the wrong objective

**[NUMERICAL]** On a *single fixed* $N=129$ parent state, interpolating $\pi_\theta = (1-\theta)\pi_A + \theta\pi_B$ between two admissible interior anchors ($\rho\approx5\times10^{-4}$ and $\rho\approx10^{-3}$) gave, over $\theta\in\{0,0.25,\dots,1\}$, a *monotonically worsening* current-target distance and a *monotonically improving* descendant safe-deficit (linear fit $R^2\approx0.9996$; quadratic $R^2\approx0.99994$). The implied local exchange rate is
$$\frac{-\Delta d^{\rm safe}_{\rm child}}{\Delta d_{\rm parent}} \approx 2.04.$$

**[NUMERICAL]** Independent replication of this experiment on a second, genealogically distinct $N=129$ parent reproduced the *sign* of the effect but at roughly $10$–$20\times$ smaller magnitude and without full monotonicity (a small reversal appears between $\theta=0.5$ and $1$). **Conclusion:** the control signal is real but its *strength is state-dependent*; it should not yet be treated as a universal constant.

### 4.5 Non-Markovianity of the reduced profile

This is, in the collaboration's own assessment, the most structurally important finding of the session.

**[NUMERICAL, confirmed by two independent mechanisms]** Two couplings can satisfy the same reduced profile constraint to solver precision, $F_m\pi_1 \approx F_m\pi_2$ (agreeing to $\sim10^{-9}$–$10^{-13}$), while their descendants differ measurably: in one instance the resulting safe-deficits were $0.003798$ vs. $0.004146$ (a $9.2\%$ relative difference), with $\|\Delta A'\|_1\approx\|\Delta B'\|_1\approx0.0085$ between the two children. A second, independently-run comparison (a "reconstructed" vs. an "exact" realization of the same profile) showed a consistent effect of comparable order.

**Consequence:** $y=(m,\mu_2,\mu_4)$ alone is **not a sufficient (Markov) state** for the renormalization dynamics; a hidden "phase"/fiber coordinate genuinely affects future viability. Any Bellman-style value function for the safe-anchor problem must be a function $V_m(y,\xi)$ of some such extra coordinate, not $V_m(y)$ alone. This also retroactively explains §4.2 and §4.4: local controllability, raw target-margin, and descendant viability are three *different* quantities, and none of the first two determines the third.

### 4.6 A genuine numerical obstruction, and its resolution

**[NUMERICAL]** A first attempt to estimate the sensitivity of the child map to this phase coordinate, by finite-differencing a coupling reconstructed via a linear program with an *identically zero* objective, **did not converge**: the estimated Jacobian changed substantially (including sign flips in some entries) when the step size was shrunk by a factor of $8$. Diagnosis: a zero-objective LP has no canonical selection rule over its (typically high-dimensional) optimal face, so the returned vertex is a discontinuous function of the target — the finite difference was measuring solver path-dependence, not a derivative.

**[NUMERICAL]** Replacing this with the **maximum (relative) entropy section**
$$S_{\rm ME}(y) = \arg\max\Big\{-\sum_{ij}\pi_{ij}\log\tfrac{\pi_{ij}}{A_iB_j} : \pi\in\Pi(A,B),\ F\pi=y\Big\},$$
which is the *unique* maximizer of a strictly concave program (Gibbs form $\pi_{ij}\propto A_iB_j\exp(\lambda^\top F_{ij})$), resolved this: after correcting a scaled-vs-raw units mismatch and choosing a genuinely-interior (not exactly-critical) base point $y_0$, the finite-difference Jacobian of $h\circ\mathrm{Child}\circ S_{\rm ME}$ converged cleanly across two step sizes $\Delta = 2^{-14}, 2^{-17}$:
$$\varepsilon_J := \frac{\|J(\Delta_1)-J(\Delta_2)\|_F}{\|J(\Delta_2)\|_F} = 1.6\times10^{-3}.$$
(An auxiliary second-difference "curvature" diagnostic was found to be dominated by solver-precision noise at the smaller step and is not itself informative here; this is a benign artifact, distinct from the convergence of $J$ itself, and is consistent with elementary Taylor-expansion bookkeeping.)

### 4.7 A first quantitative "capacity" estimate

**[NUMERICAL, single state, not yet replicated]** Sampling $8$ dual (Farkas) support directions, orthogonalizing them into a whitened $3$-dimensional embedding $Q$ ($Q^\top Q = I_3$), and forming the reduced $3\times3$ operator $J_{\rm eff} = Q^\top J_{\rm raw}$ gives singular values
$$\sigma(J_{\rm eff}) \approx (1.972,\ 1.008,\ 0.249),$$
i.e. one clearly expanding direction, one nearly rate-neutral direction, and one contracting direction. The associated (as yet *unwhitened*, i.e. metric-dependent) "positive-rate" proxy
$$R_{\rm bits} = \sum_{k=1}^3 \log_2^+\sigma_k \approx 0.991\ \text{bit}$$
is a loose numerical analogue of the classical data-rate/topological-entropy bound for stabilizing an unstable linear system over a rate-limited channel (Wong–Brockett 1999; Nair–Evans and others) — **an analogy, not an applicable theorem**, since the underlying map is not linear and this is not (yet) shown to be scale-uniform. A residual $\|(I-QQ^\top)J_{\rm raw}\|_F/\|J_{\rm raw}\|_F \approx 0.235$ indicates that a non-trivial (though minority) part of the child's response is a genuine deformation of the reachable region's shape, not a rigid $3$-parameter translation.

## 5. Open problems, in priority order (as currently assessed)

1. **State-uniformity of §4.7.** Does a second, genealogically independent parent state reproduce $\sigma \approx (2,1,\tfrac14)$ (or at least the same *order of magnitude* and dominant right singular direction)? Not yet tested. This is the immediate next check before any further theoretical weight is placed on the "$\approx 1$ bit" figure.
2. **Whitening.** $R_{\rm bits}$ above uses no metric normalization by the actual parent quantization-cell size or child slack budget; the theorem-relevant quantity requires $\widetilde J = W_{\rm out}J_{\rm eff}W_{\rm in}^{-1}$.
3. **Existence, not just local sensitivity.** Nothing above proves existence of a scale-uniform admissible anchor; it characterizes the local geometry *around* specific numerically-found points.
4. **The inventory bridge.** The relation "bounded phase state + safe transition $\Rightarrow \Delta\mathcal B \le C_{\rm bill}$" is a *hypothesis*, not derived from anything above. This is flagged in the collaboration's own record as a control condition, not a definition of the actual accounted quantity, and remains completely open.
5. **A precise replacement for the refuted one-step contraction hypothesis** (§4.3) — plausibly a two-step or aggregate-boundedness statement, currently untested beyond one doubling.

## 6. One-paragraph summary

Numerical work this round (i) refuted the simplest candidate contraction law for the safe-anchor construction, (ii) showed that greedy short-term optimization is a provably wrong control policy on at least one concrete instance, (iii) established — and this is the structurally load-bearing result — that the natural 3-dimensional reduced description of the problem is not a sufficient statistic for its own dynamics, and (iv) built and cross-validated (to $\varepsilon_J = 1.6\times10^{-3}$ across independent step sizes) the first numerically stable estimate of a "hidden-phase sensitivity" operator, whose spectrum suggests — on one instance so far — an effective control budget on the order of one bit per dyadic scale. None of this proves the target upper bound; it substantially sharpens what a correct proof would need to formalize.
