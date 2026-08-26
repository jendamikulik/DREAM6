# The Bracket, Not the Tube
### A Certification Mechanism Shared by Two Independent Estimation Problems

*Notes toward a general principle, extracted from two unrelated research threads pursued in parallel.*

## 0. What this note is, and is not

This note isolates a single recurring pattern — call it the **witness/converse bracket** — that appeared, independently, in two structurally unrelated problems worked on during the same period: a safe-anchor/renormalization program (here "WPB") and a causal-Shannon-floor resource-accounting program (here "Bill"). The pattern itself is not new; it is the ordinary shape of weak duality, stated abstractly enough to make an otherwise-easy-to-miss cross-instantiation visible. What is being claimed here is narrow and checkable: that WPB's compressed Farkas margin bracket and Bill's synchronous-rate bracket are literally the same schema with different data substituted in, and that this shared schema is a *tool*, not a shared *object* — the two problems do not become the same problem by sharing it.

## 1. The abstract mechanism

Let $\mathcal A$ be a feasible set, $f:\mathcal A\to\mathbb R$ an objective, and

$$V^\star \;:=\; \operatorname*{ext}_{\pi\in\mathcal A} f(\pi)$$

an unknown extremal quantity (ext = inf or sup), not directly computable because $\mathcal A$ or $f$ is too large or too implicit to optimize over exhaustively.

A bracket $[L,U]\ni V^\star$ is built from two logically distinct kinds of certificate, and it matters which kind is used on which side.

**Species 1 — converse (relaxation) bounds.** A bound $g$ is a converse if it is established by an argument valid *simultaneously* for every $\pi\in\mathcal A$, without exhibiting any particular one — typically by relaxing $f$ or $\mathcal A$ coordinatewise, by weak LP duality, or by an information-theoretic inequality (a transport/coupling converse, a counting bound). Because the argument covers the whole feasible set at once, it bounds $V^\star$ for free.

**Species 2 — constructive witnesses.** A bound is a witness if it is established by exhibiting *one* explicit $\pi_0\in\mathcal A$, verifying admissibility directly (normalization, marginals, positivity, exactness of the induced law), and evaluating $f(\pi_0)$. This gives $V^\star \le f(\pi_0)$ (or $\ge$, for a sup) trivially, but only because that one object was actually checked.

A bracket is then

$$L \;\le\; V^\star \;\le\; U,$$

where each of $L,U$ may be built from either species — both ends can be converses (as in §3), or one end a converse and the other a witness (as in §4). The quantity $\Delta:=U-L\ge 0$ is not a hedge. It is an exact, computable statement of present ignorance, and it is falsifiable: any tightening of either side is a checkable claim, not a change of opinion.

This is nothing more than weak duality plus feasibility. Its only content here is that stating it at this level of abstraction makes two otherwise unrelated computations visibly the same move.

## 2. Two instances from the same period

### 2.1 Instance A — the compressed Farkas margin (WPB)

Here $V^\star=\delta_m(y)$, the reduced safety margin of a target point relative to a renormalization-step reachable region $F_m(\mathcal P_m)\subset\mathbb R^3$, defined via the support-function infimum $\delta_m(y)=\inf_{\|z\|_1=1}[h_m(z)-z\cdot y]$.

Direct computation of $h_m(z)=\sup_{\pi\in\mathcal P_m} z\cdot F_m\pi$ becomes expensive as the transport polytope's support grows. The bracket is obtained by replacing the fine-grained per-cell coefficients of the objective with their **coordinatewise minimum** (resp. **maximum**) over coarsened index groups:

$$h_m^{\rm lower}(z) \;\le\; h_m(z) \;\le\; h_m^{\rm upper}(z),$$

because $\sup$ of a pointwise-smaller (resp. larger) function over the *same* feasible set is itself smaller (resp. larger). Both ends here are **Species 1**: no explicit coupling is exhibited on either side; both bounds come from the same relaxation device run in opposite directions. Propagated through the margin formula this gives the certified bracket $\delta_{\rm lower}\le\delta_{\rm true}\le\delta_{\rm upper}$, observed in practice to close to a width of order $10^{-11}$ on audited states — tight enough that the relaxation, though structurally approximate, is numerically decisive.

### 2.2 Instance B — the synchronous rate (Bill)

Here $V^\star=C^{\rm syn}_\infty(P,Q)$, the asymptotic per-symbol entropy required by a synchronous stable-event realization contract for a two-action controlled channel.

The **lower** bound is Species 1: an information-spectrum/optimal-transport converse. Any synchronous table, evaluated along any admissible history, must dominate a quantile-transport functional of the two actions' surprisal laws; averaging gives $C^{\rm syn}_\infty \ge \tfrac12\big(H(P)+H(Q)+W_1(\mu_P,\mu_Q)\big)$, valid for *every* feasible table without exhibiting one.

The **upper** bound is Species 2: an explicit 23-atom joint law on four coordinates, verified to (i) sum to one, (ii) reproduce the required product marginal under all four action pairs, and (iii) have computable entropy $H(\tilde G)$. This is a genuine constructive witness, not a relaxation — a single concrete object, checked directly.

The resulting bracket, $1.7285\lesssim C^{\rm syn}_\infty \lesssim 1.9391$ for the equal-entropy crash-test pair, mixes both species by construction: converse below, witness above. The gap is explicitly posed as an open problem in the source material, not smoothed over.

## 3. The degenerate case: when the bracket closes

The same period produced one instance where $L$ and $U$ were shown to coincide **exactly**, not just numerically close: the causal-tree rate $C^{\rm causal}_\infty(P,Q)=\max\{H(P),H(Q)\}$.

The lower bound is immediate Species 1 (any policy that always plays the entropy-maximizing action already forces this rate on the realized trajectory). The upper bound is Species 2, but of an *asymptotic-family* kind rather than a single fixed witness: an explicit sequence of finite constructions (block-coupled, anti-diagonally reversed) whose cost provably converges to the same value as the block length grows. When a converse and a constructed achieving sequence meet, the bracket does not merely narrow — it stops being a bracket and becomes a theorem. This is the same event, structurally, as a linear program's primal and dual optima coinciding: strong duality is the special case of a bracket of width zero.

It is worth recording explicitly that closing a bracket to zero is not guaranteed by having a converse and a witness in hand; both instances in §2 remain open precisely because no such matching construction has yet been found for them.

## 4. What does *not* transfer

The shared mechanism should not be mistaken for a shared object, and the disanalogy is exact enough to state precisely.

Instance B's bracket certifies **one fixed number**, attached to one channel, at one (asymptotic) horizon. There is no second scale at which $C^{\rm syn}_\infty$ must be re-established; nothing about the problem requires the bracket to reproduce itself.

Instance A's bracket certifies **one term of a sequence indexed by the renormalization scale** $m$, and the open problem is not the value of any single $\delta_m$ — several individual $\delta_m$ have in fact been bracketed tightly — but whether an admissible lift can be chosen at every $m$ so that the *next* scale's bracket is again satisfiable, indefinitely: $\mathfrak K_m\to\mathfrak K_{2m}$ for all $m$, not for one $m$. This is a viability/invariance question superimposed on top of the bracketing tool, and nothing in §1–§3 addresses it. A bracket can be computed at every scale individually and the sequence of brackets can still fail to propagate; conversely, propagation would not by itself make individual brackets tighter.

In the vocabulary of §1: both problems use the same *method for pricing an unknown extremal quantity*. Only one of them additionally requires that quantity to be re-priced, favorably, forever.

## 5. A working checklist

For an unresolved bracket in either program, or a new one, the pattern above suggests a fixed order of questions, matching the order in which both instances above were actually built:

1. What is $\mathcal A$ and $f$, precisely — is the extremal quantity even well-defined as stated?
2. Is there a Species-1 converse available cheaply (a relaxation, a duality argument, an information-theoretic inequality) that holds for the whole feasible set at once?
3. Is there a Species-2 witness available — one explicit, fully verified feasible object, or an explicit *sequence* of such objects with a provable limiting cost?
4. Report $L$, $U$, and $\Delta=U-L$ together, never $U$ or $L$ alone, and never a point estimate presented as if it were either.
5. Ask separately, as a distinct question, whether anything about the underlying problem forces this certification to be repeated at a growing family of scales or instances — and if so, treat that as a second, independent obligation, not as something the bracket itself discharges.

## 6. On priority

Nothing in §1 is new mathematics; it is weak duality together with the standard achievability/converse pairing used throughout information theory and combinatorial optimization, named here only to make an implicit repetition explicit. The contribution of this note, such as it is, is narrow: an explicit check that two concretely different computations, produced independently on the same night, are literally the same schema with different data, and an equally explicit statement — §4 — of where that shared schema stops mattering.

## 7. One sentence

Two problems shared a way of pricing what they did not yet know; they did not, on that account, share what they were trying to price.
