# Objectivity Without Fundamentality  
## Whole-Brain Emulation and the Limits of Anti-Skeptical Arguments

### Abstract

Classical skepticism asks whether an apparently objective world might nevertheless fail to be fundamental: perhaps experience is a dream, an illusion, or the product of an external mechanism. A prominent anti-skeptical response argues that sufficiently global doubt is conceptually defective. Concepts such as *dream*, *illusion*, *error*, *reality*, and *object* acquire their meaning only within a shared and normally reliable world; therefore the hypothesis that *everything* is unreal may seem to destroy the contrast on which its own formulation depends.

Whole-brain emulation changes the structure of the problem.

Suppose that a conscious agent can be instantiated by a computational process and coupled to a sufficiently rich simulated environment. The agent may then inhabit a stable, public, causally structured world containing persistent objects and other agents. Russellian object persistence, Wittgensteinian linguistic practices, and Davidsonian triangulation can all hold inside that world. Nevertheless, the world may be implemented by a substrate inaccessible to its inhabitants.

The central claim of this note is therefore not that we are simulated. It is the logically weaker and much more robust statement

\[
\boxed{
\text{observer-independent reality does not imply substrate-level fundamentality}.
}
\]

This distinction can be formalized. We prove an observational-indistinguishability theorem showing that if two implementations induce exactly the same complete interaction histories for an embedded observer, then no internal experiment can distinguish them. We then show that persistence of objects and intersubjective triangulation are invariant under implementation. Consequently, these phenomena may establish an objective world relative to its inhabitants while remaining silent about the ontological character of the substrate implementing that world.

Finally, we show that a precise simulation hypothesis is not equivalent to the semantically unstable statement that “everything is unreal.” It can instead be expressed as a relation between two dynamical systems and therefore possess definite truth conditions. Whole-brain emulation would not establish that our world has such a higher-level implementation, but it would demonstrate that a conscious subject can in principle be correct about ordinary reality while being radically mistaken about the substrate on which that reality depends.

---

# 1. The conceptual distinction

There are at least three logically distinct propositions that are frequently conflated:

\[
\begin{aligned}
R_1 &: \text{There exists a world independent of my present act of perception},\\[2mm]
R_2 &: \text{That world is shared with other observers},\\[2mm]
R_3 &: \text{That world is ontologically fundamental}.
\end{aligned}
\]

Ordinary realism primarily concerns \(R_1\) and \(R_2\).

The simulation question concerns \(R_3\).

The crucial logical observation is

\[
(R_1\land R_2)\not\Rightarrow R_3.
\]

A simulated world can contain persistent objects, causal regularities, other observers, public measurements, scientific institutions, and indefinitely reproducible experiments. Its inhabitants may therefore possess an objective world in every ordinary epistemic sense while remaining ignorant of its implementation.

The error to be avoided is

\[
\boxed{
\text{objectivity}
\quad\Longrightarrow\quad
\text{fundamentality}.
}
\]

The implication does not hold without an additional premise.

---

# 2. Formal framework

Let a world be represented by a dynamical system

\[
\mathcal W=(W,A,O,F,G),
\]

where

- \(W\) is the space of world states;
- \(A\) is the space of actions available to an embedded agent;
- \(O\) is the space of observations;
- \(F\) is the world transition rule;
- \(G\) is the observation map.

For simplicity, begin with deterministic dynamics:

\[
w_{t+1}=F(w_t,a_t),
\]

and

\[
o_t=G(w_t).
\]

An embedded agent is itself represented by an internal state \(m_t\in M\), an update rule \(U\), and an action policy \(\Pi\):

\[
m_{t+1}=U(m_t,o_t),
\]

\[
a_t=\Pi(m_t).
\]

The complete accessible history of the agent up to time \(n\) is

\[
T_n=
(o_0,a_0,o_1,a_1,\ldots,o_n).
\]

Everything the agent can infer experimentally must ultimately be a function of such accessible data.

Thus an internal decision procedure has the form

\[
D_n=\Phi(T_n),
\]

possibly together with internally generated randomness.

---

# 3. What it means for one system to implement another

Calling one physical process a simulation of another must not amount to an arbitrary relabelling of states. We therefore require preservation of dynamics.

Let

\[
\mathcal H=(H,\widetilde F)
\]

be a host dynamical system.

An implementation of \(\mathcal W\) in \(\mathcal H\) consists of an injective encoding

\[
\iota:W\longrightarrow H
\]

and an encoding of admissible actions such that, for an appropriate host evolution time \(\tau\),

\[
\boxed{
\iota(F(w,a))
=
\widetilde F^{\,\tau}
\bigl(\iota(w),\widetilde a\bigr).
}
\]

Equivalently, the following diagram commutes:

\[
\begin{array}{ccc}
W & \xrightarrow{F_a} & W\\
\downarrow\scriptstyle{\iota} && \downarrow\scriptstyle{\iota}\\
H & \xrightarrow{\widetilde F_{\widetilde a}^{\,\tau}} & H .
\end{array}
\]

The requirement is counterfactual: it must preserve not merely one actual trajectory but the relevant family of trajectories generated by admissible interventions.

This eliminates the trivial objection that any sufficiently complicated physical system can be retrospectively mapped onto any finite sequence whatsoever.

---

# 4. Internal equivalence

Consider two implementations

\[
I_1,\qquad I_2
\]

of the same experienced world.

They are **internally observationally equivalent** for an agent if every possible admissible interaction generates the same observable history.

In the stochastic case this becomes equality of probability laws:

\[
\boxed{
\Pr_{I_1}(T_n\in B)
=
\Pr_{I_2}(T_n\in B)
}
\]

for every time \(n\) and every measurable set \(B\) of possible transcripts.

We write

\[
I_1\equiv_{\mathrm{obs}} I_2.
\]

This definition already contains the essential epistemic structure of the problem.

---

# 5. The substrate-underdetermination theorem

### Theorem 1 — Internal indistinguishability

Let \(I_1\) and \(I_2\) be two implementations such that

\[
I_1\equiv_{\mathrm{obs}}I_2.
\]

Then no decision procedure available to the embedded agent can distinguish \(I_1\) from \(I_2\) on the basis of internal evidence.

More precisely, for every measurable decision rule

\[
D=\Phi(T_n),
\]

we have

\[
\Pr_{I_1}(D=d)
=
\Pr_{I_2}(D=d)
\]

for every possible output \(d\).

### Proof

Since

\[
I_1\equiv_{\mathrm{obs}}I_2,
\]

the random transcript \(T_n\) has the same probability distribution under both implementations.

Let

\[
D=\Phi(T_n).
\]

For any measurable set \(C\) of decision outputs,

\[
\Pr_{I_j}(D\in C)
=
\Pr_{I_j}
\left(
T_n\in\Phi^{-1}(C)
\right).
\]

But the distribution of \(T_n\) is identical under \(I_1\) and \(I_2\). Hence

\[
\Pr_{I_1}
\left(
T_n\in\Phi^{-1}(C)
\right)
=
\Pr_{I_2}
\left(
T_n\in\Phi^{-1}(C)
\right).
\]

Therefore

\[
\Pr_{I_1}(D\in C)
=
\Pr_{I_2}(D\in C).
\]

Thus no internal decision rule possesses distinguishing power between the two implementations. \(\square\)

---

## Corollary 1 — Experiment cannot overcome exact equivalence

No amount of additional experimentation helps if every possible experiment is already included in the equivalence relation.

If

\[
I_1\equiv_{\mathrm{obs}}I_2
\]

for arbitrarily long admissible interaction histories, then extending the experiment from \(n\) to \(n+k\) cannot break the equivalence.

### Proof

Apply Theorem 1 to \(T_{n+k}\). \(\square\)

---

## Corollary 2 — Equal priors remain equal

Suppose the embedded observer assigns prior probabilities

\[
P(I_1)=P(I_2)=\frac12.
\]

If

\[
I_1\equiv_{\mathrm{obs}}I_2,
\]

then after observing any transcript \(T\) having nonzero probability,

\[
P(I_1\mid T)
=
P(I_2\mid T)
=
\frac12.
\]

### Proof

Bayes' theorem gives

\[
P(I_j\mid T)
=
\frac{P(T\mid I_j)P(I_j)}
{P(T)}.
\]

Observational equivalence implies

\[
P(T\mid I_1)=P(T\mid I_2).
\]

Equal priors therefore produce equal posteriors. \(\square\)

The theorem does not assert that every simulation must be undetectable.

A poorly constructed simulation may leak information.

The conclusion is conditional:

\[
\boxed{
\text{if the implementations are observationally equivalent,
internal evidence cannot distinguish them.}
}
\]

This is a mathematical statement, not a metaphysical intuition.

---

# 6. Objectivity is weaker than fundamentality

We now formalize the notion of a persistent object.

Let

\[
X:W\rightarrow \mathcal X
\]

extract the state of some object \(x\) from the complete world state.

Call \(x\) **observer-independent within \(\mathcal W\)** if its state and dynamics are defined independently of whether the observer currently receives information about it.

For example, during a period in which the observation map hides \(x\),

\[
G(w_t)\not\supset X(w_t),
\]

the object may nevertheless continue evolving according to

\[
X(w_{t+1})
=
f_X(w_t,a_t).
\]

Thus absence from perception is not absence from the world state.

---

### Proposition 2 — Simulated objectivity

Observer-independent persistence is compatible with simulation.

### Proof

Take any world \(\mathcal W\) containing an observer-independent object \(x\), and let

\[
\iota:W\rightarrow H
\]

be an implementation satisfying

\[
\iota(F(w,a))
=
\widetilde F^{\,\tau}
(\iota(w),\widetilde a).
\]

The encoded state

\[
\iota(w_t)
\]

continues to contain the information determining \(X(w_t)\) even during intervals in which the observer does not perceive \(x\).

Hence \(x\) remains observer-independent relative to the simulated world although the complete state of the world is implemented in \(H\).

Therefore

\[
\text{observer-independent}(x)
\]

does not entail

\[
\text{fundamental}(x).
\]

\(\square\)

---

# 7. Russell's cat

Consider the familiar argument.

An observer sees a cat at time \(t_1\), turns away, and sees the cat again at \(t_3\).

A natural explanation is that the cat persisted through the unobserved interval:

\[
x(t_1)
\longrightarrow
x(t_2)
\longrightarrow
x(t_3).
\]

This is normally more economical than assuming that the cat ceased to exist and was recreated whenever observed.

That inference can be perfectly valid.

But what exactly has been established?

At most:

\[
\boxed{
x(t_2)
\text{ existed independently of the observer's perception at }t_2.
}
\]

It does not follow that

\[
x(t_2)
\]

was implemented in the fundamental substrate rather than in some higher-level representation.

Indeed, construct two cases.

### Case \(P\)

The cat is implemented directly by ordinary physical degrees of freedom.

### Case \(S\)

The cat is a persistent state variable in a simulated environment.

Suppose

\[
P\equiv_{\mathrm{obs}}S.
\]

Then every observation supporting persistence in \(P\) equally supports persistence in \(S\).

Hence the inference

\[
\text{re-identification}
\Rightarrow
\text{persistence}
\]

may be sound, while the stronger inference

\[
\text{persistence}
\Rightarrow
\text{fundamental physical realization}
\]

is invalid.

This yields the following result.

---

### Theorem 3 — Persistence does not identify substrate

No argument whose evidence consists only of observer-independent persistence can, by itself, distinguish a fundamental realization from an observationally equivalent simulated realization.

### Proof

By Proposition 2, observer-independent persistence is present in both realizations.

By Theorem 1, observationally equivalent realizations cannot be distinguished using their internal transcripts.

Therefore persistence evidence cannot select one substrate rather than the other. \(\square\)

Russellian realism and simulation are therefore not contradictory.

A simulated cat need not be an illusion.

It may be a perfectly real cat **at the level of the world inhabited by the observer**.

---

# 8. Two meanings of “real”

The apparent paradox disappears once two predicates are distinguished.

Define

\[
R_W(x)
\]

to mean:

> \(x\) is a causally effective, observer-independent object of world \(W\).

Define

\[
F(x)
\]

to mean:

> \(x\) belongs directly to the fundamental substrate.

Then it is consistent that

\[
R_W(x)=1
\]

while

\[
F(x)=0.
\]

Indeed, in an implemented world,

\[
x
\stackrel{\iota}{\longmapsto}
\iota(x)
\]

may have perfectly objective dynamics while being ontologically dependent upon the host.

Thus

\[
\boxed{
R_W(x)\not\Rightarrow F(x).
}
\]

A large fraction of apparent disputes about simulation arise from treating \(R_W\) and \(F\) as though they were the same predicate.

They are not.

---

# 9. Davidsonian triangulation survives simulation

Consider two agents \(A\) and \(B\) interacting with a shared world \(W\).

A simplified Davidsonian structure may be represented as

\[
\mathfrak D
=
(A,B,W,R_{AB},R_{AW},R_{BW}),
\]

where

- \(R_{AB}\) describes communication and mutual interpretation;
- \(R_{AW}\) relates \(A\) to objects and events;
- \(R_{BW}\) relates \(B\) to those same objects and events.

The epistemic triangle is

\[
A
\longleftrightarrow
B
\]

with both simultaneously related to

\[
W.
\]

Suppose now that the entire structure is implemented in a host system \(H\).

Let

\[
\iota:\mathfrak D\rightarrow\mathfrak D'
\]

be an isomorphism preserving all internally relevant relations.

Then

\[
R_{AB}(a,b)
\iff
R'_{AB}(\iota(a),\iota(b)),
\]

and similarly for relations to the world.

---

### Theorem 4 — Triangulation invariance

Any property of the agent-agent-world structure that is invariant under relational isomorphism is preserved when that structure is implemented on a different substrate.

### Proof

Let \(P\) be a property definable solely in terms of the internal relational structure of \(\mathfrak D\).

If

\[
\iota:\mathfrak D\cong\mathfrak D'
\]

is an isomorphism, then by definition it preserves all relevant objects, relations, and relation instances.

Any structurally defined property true in \(\mathfrak D\) is therefore true in \(\mathfrak D'\).

Hence

\[
P(\mathfrak D)
\iff
P(\mathfrak D').
\]

\(\square\)

---

### Corollary 4.1

Triangulation may establish that an agent is not alone in a private hallucination.

It does not establish that the shared world is ontologically fundamental.

Indeed,

\[
A\leftrightarrow W\leftrightarrow B
\]

may be completely genuine while

\[
W
\]

itself is implemented by another system.

The correct conclusion is therefore

\[
\boxed{
\text{intersubjective objectivity}
\not\Rightarrow
\text{substrate fundamentality}.
}
\]

---

# 10. Wittgenstein and the contrast-class objection

A stronger anti-skeptical argument concerns meaning.

The words

\[
\text{dream},\quad
\text{illusion},\quad
\text{error}
\]

ordinarily function by contrast with

\[
\text{waking},\quad
\text{reality},\quad
\text{truth}.
\]

If someone declares that absolutely everything is a dream, the distinction that gives the word *dream* its ordinary use appears to collapse.

This is a serious objection to the unrestricted proposition

\[
\boxed{\text{Everything is unreal.}}
\]

But a simulation hypothesis need not have this form.

Consider instead the proposition

\[
\mathsf{SIM}(W):
\]

\[
\exists H,\iota,\tau
\quad
\forall w,a:
\quad
\iota(F_W(w,a))
=
F_H^\tau(\iota(w),\widetilde a).
\]

This does not say

\[
W\text{ does not exist}.
\]

It says that the dynamics of \(W\) possess a realization in another dynamical system \(H\).

The statement therefore distinguishes two relations:

\[
\text{existence within }W
\]

and

\[
\text{implementation of }W.
\]

The ordinary contrast between truth and error inside \(W\) remains intact.

A statement such as

\[
\text{“There is a cat in the room.”}
\]

may be true in \(W\).

A statement such as

\[
\text{“The state space }W\text{ is implemented by }H\text{.”}
\]

is a proposition about the relation between two levels.

There is no semantic requirement that the first proposition become false merely because the second is true.

---

### Proposition 5 — The simulation proposition has nontrivial truth conditions

The formal hypothesis \(\mathsf{SIM}(W)\) is neither logically equivalent to the statement “nothing is real” nor automatically true for every world.

### Proof

The condition requires the existence of a host \(H\), an encoding \(\iota\), and transition preservation:

\[
\iota\circ F_W
=
F_H^\tau\circ\iota.
\]

A candidate host lacking such a dynamically preserving embedding fails the condition.

Therefore the proposition can be false.

If the condition holds, objects of \(W\) may nevertheless retain their internal causal relations and observer-independent existence.

Therefore \(\mathsf{SIM}(W)\) does not entail that nothing in \(W\) is real in the internal sense.

Hence \(\mathsf{SIM}(W)\) has nontrivial truth conditions distinct from global unreality. \(\square\)

This does not by itself settle every philosophical question about natural-language meaning.

It establishes the narrower and sufficient point that the simulation question can be given a precise relational formulation that does not collapse into the sentence

\[
\text{“Everything is an illusion.”}
\]

---

# 11. Whole-brain emulation as the decisive construction

We now introduce the crucial conditional assumption.

### WBE assumption

There exists a physical process \(H\) capable of implementing a dynamical system \(M\) whose relevant causal organization is sufficient for the mental life of a particular human subject.

Schematically,

\[
\text{biological brain}
\quad\longrightarrow\quad
\text{emulated brain}.
\]

A stronger version assumes preservation of consciousness:

\[
C(B)=1
\quad\Longrightarrow\quad
C(\iota(B))=1.
\]

This implication is currently a philosophical and scientific assumption, not a theorem of the present analysis.

Nothing below disguises that fact.

Now connect the emulated brain to a simulated environment \(W_S\) reproducing the sensory and motor interface of an ordinary world.

The emulated subject receives

\[
o_t=G_S(w_t)
\]

and generates actions

\[
a_t=\Pi(m_t).
\]

Suppose these interactions reproduce the corresponding interaction structure of a biological subject in world \(W_P\):

\[
W_P\equiv_{\mathrm{obs}}W_S.
\]

Then the subject may possess the same evidence in the two cases.

---

### Theorem 6 — Veridical local realism with false substrate belief

Assume:

1. a conscious whole-brain emulation is possible;
2. a persistent simulated environment can provide the emulation with ordinary sensorimotor interaction;
3. the simulated world contains stable objects and other agents;
4. the emulated subject falsely believes that its experienced world is directly implemented by the fundamental substrate.

Then the subject can simultaneously have:

\[
\text{true ordinary beliefs about its world}
\]

and

\[
\text{a false belief about the world's substrate}.
\]

### Proof

Let \(x\) be a persistent simulated object.

By Proposition 2,

\[
R_{W_S}(x)=1.
\]

Hence propositions such as

\[
\text{“}x\text{ persists when I look away”}
\]

and

\[
\text{“other observers can interact with }x\text{”}
\]

can be true.

Yet by assumption the subject believes

\[
F(x)=1,
\]

whereas the object is implemented within a host and therefore, in the stipulated sense,

\[
F(x)=0.
\]

Thus the subject's ordinary world-directed beliefs may be veridical even while its belief about the substrate is false. \(\square\)

This theorem is the central philosophical consequence of whole-brain emulation.

A subject need not be hallucinating.

Its science need not be fraudulent.

Its language need not be incoherent.

Its friends need not be philosophical zombies.

Its tables need not disappear when unobserved.

And nevertheless the subject may be wrong about what realizes the entire system.

---

# 12. The decisive counterexample

Suppose an emulated physicist named Alice lives in a simulated world.

She measures a gravitational constant,

\[
G_S,
\]

predicts planetary motions,

\[
\ddot{\mathbf r}
=
-\frac{G_SM}{r^3}\mathbf r,
\]

performs interferometry, constructs particle accelerators, and exchanges reproducible measurements with thousands of colleagues.

Inside her world, those theories may be genuinely predictive.

Suppose she writes:

> The external world exists independently of my mind. My laboratory remains there when I sleep. Other people observe the same instruments. The hypothesis that reality is merely my private dream is therefore untenable.

She may be completely correct.

Then she adds:

> Therefore the physical system described by our deepest physics cannot itself be implemented by another physical system.

That conclusion does not follow.

Her evidence established

\[
R_1\land R_2.
\]

She inferred

\[
R_3.
\]

But Theorems 1–4 show that the inference is invalid.

From the host's point of view,

\[
\boxed{
R_1=1,\qquad
R_2=1,\qquad
R_3=0.
}
\]

This is a concrete countermodel to the implication

\[
(R_1\land R_2)\Rightarrow R_3.
\]

One countermodel is sufficient to refute a universal implication.

---

# 13. The anti-skeptical arguments after WBE

The results can now be stated precisely.

## 13.1 What survives

The following may remain entirely correct:

\[
\text{There is an objective world.}
\]

\[
\text{Objects persist independently of perception.}
\]

\[
\text{Other minds share that world.}
\]

\[
\text{Error is meaningful because truth is meaningful.}
\]

\[
\text{Science discovers stable regularities.}
\]

\[
\text{A purely private Cartesian hallucination is not the best explanation.}
\]

None of these claims is threatened merely by the possibility of simulation.

---

## 13.2 What does not follow

None of them alone entails

\[
\boxed{
\text{The experienced world is the bottom level of physical reality.}
}
\]

That is an additional metaphysical claim.

The gap is

\[
\boxed{
\text{epistemic objectivity}
\neq
\text{ontological fundamentality}.
}
\]

Whole-brain emulation makes the distinction vivid because it supplies a possible constructive counterexample rather than merely a verbal skeptical scenario.

---

# 14. Why this differs from dreaming

The dream analogy is structurally imperfect.

Ordinary dreams usually lack:

- long-term causal stability,
- reliable public reproducibility,
- independent agents,
- persistent external memory,
- indefinitely repeatable experiments,
- coherent laws across arbitrarily long periods.

A sufficiently rich simulated world need lack none of these.

Thus:

\[
\text{simulation}\neq\text{ordinary dream}.
\]

More accurately,

\[
\boxed{
\text{simulation}
=
\text{objective world}
+
\text{nonfundamental implementation}.
}
\]

That is why the simulation question survives arguments that may successfully dissolve unrestricted dream skepticism.

---

# 15. An impossibility theorem for perfect simulation detection

Suppose a civilization inside \(W\) devises some ultimate experiment

\[
E^\star
\]

intended to determine whether its world is simulated.

Let the answer be computed by

\[
D^\star=\Phi^\star(T).
\]

If a fundamental implementation \(I_F\) and simulated implementation \(I_S\) satisfy

\[
I_F\equiv_{\mathrm{obs}}I_S,
\]

then by Theorem 1,

\[
\Pr(D^\star=d\mid I_F)
=
\Pr(D^\star=d\mid I_S).
\]

Hence:

\[
\boxed{
\text{There exists no universally valid internal detector
of simulation under exact observational equivalence.}
}
\]

This is not because the simulation hypothesis is meaningless.

It is because two distinct ontological models may be empirically underdetermined by all accessible evidence.

The distinction between

\[
\text{meaninglessness}
\]

and

\[
\text{underdetermination}
\]

is essential.

A proposition may have a perfectly definite truth value even when an embedded observer cannot determine it.

---

# 16. No inference to “we are simulated”

Nothing proved above establishes

\[
P(\mathsf{SIM}(W_{\mathrm{ours}}))>\frac12
\]

or indeed any particular numerical probability.

Such an inference would require additional premises concerning, for example,

- the physical feasibility of conscious emulation;
- computational resources;
- motivations of advanced civilizations;
- numbers of simulated observers;
- anthropic selection principles;
- priors over possible worlds.

The present argument is therefore deliberately weaker.

It establishes only:

\[
\boxed{
\text{If conscious simulated worlds are possible,
ordinary objectivity cannot by itself exclude them.}
}
\]

This conclusion is independent of any particular anthropic probability argument.

---

# 17. A secondary consequence: branching personal identity

Whole-brain emulation creates another mathematically clean problem.

Suppose a person \(P\) is emulated twice:

\[
P
\longrightarrow
P_1,
\]

\[
P
\longrightarrow
P_2.
\]

Assume both copies initially preserve all psychologically relevant information in \(P\).

Let

\[
C(P,P_i)
\]

denote psychological continuity.

Then

\[
C(P,P_1)=1
\]

and

\[
C(P,P_2)=1.
\]

After their creation, different inputs produce

\[
P_1\neq P_2.
\]

---

### Theorem 7 — Branching continuity cannot equal numerical identity

If a continuity relation \(C\) permits branching,

\[
C(P,P_1)=1,
\qquad
C(P,P_2)=1,
\qquad
P_1\neq P_2,
\]

then \(C\) cannot be identical to numerical identity.

### Proof

Suppose for contradiction that

\[
C(x,y)\iff x=y.
\]

Then

\[
C(P,P_1)
\]

implies

\[
P=P_1,
\]

and

\[
C(P,P_2)
\]

implies

\[
P=P_2.
\]

By transitivity of identity,

\[
P_1=P_2.
\]

But by hypothesis,

\[
P_1\neq P_2.
\]

Contradiction.

Therefore branching continuity cannot be numerical identity. \(\square\)

Thus sufficiently exact copying would force a distinction between

\[
\text{psychological continuity}
\]

and

\[
\text{numerical identity}.
\]

This consequence is logically separate from the simulation argument, but it arises from the same technological premise.

---

# 18. The deepest consequence

The deepest question generated by whole-brain emulation is therefore not

\[
\text{“Is reality real?”}
\]

That formulation is too coarse.

The sharper questions are:

\[
\text{At what level is a given object real?}
\]

\[
\text{Which dynamical system implements the world accessible to us?}
\]

\[
\text{Which properties are invariant under changes of implementation?}
\]

\[
\text{Which properties depend on the substrate itself?}
\]

and, if conscious emulation exists,

\[
\boxed{
\text{When does a physical computation constitute a subject of experience?}
}
\]

The first four questions are questions about realization and epistemic accessibility.

The last is a question about consciousness.

They must not be conflated.

---

# 19. Final theorem

The preceding results may be condensed into a single statement.

### Theorem 8 — Objectivity without fundamentality

Let \(S\) be an embedded conscious agent inhabiting a world \(\mathcal W\). Assume that:

1. \(\mathcal W\) contains observer-independent persistent objects;
2. multiple agents interact with the same world structure;
3. internal linguistic and scientific practices distinguish truth from error;
4. \(\mathcal W\) admits two implementations \(I_F\) and \(I_S\);
5. \(I_F\) and \(I_S\) are observationally equivalent for all inhabitants;
6. \(I_F\) is fundamental while \(I_S\) is implemented by a distinct host system.

Then:

\[
\begin{aligned}
&\text{(i) the inhabitants may possess objective knowledge of }\mathcal W,\\
&\text{(ii) their ordinary realism may be correct,}\\
&\text{(iii) no internal test can distinguish }I_F\text{ from }I_S,\\
&\text{(iv) therefore ordinary objectivity does not entail fundamentality.}
\end{aligned}
\]

### Proof

By assumptions 1–3, the inhabitants possess the structural conditions required for ordinary objectivity.

By assumption 5 and Theorem 1, no internal decision procedure distinguishes \(I_F\) from \(I_S\).

By Proposition 2 and Theorem 4, object persistence and intersubjective structure survive a change of implementation.

Yet by assumption 6,

\[
I_F\neq I_S
\]

with respect to substrate fundamentality.

Therefore identical internal objectivity is compatible with distinct ontological implementations.

Hence

\[
\boxed{
\text{ordinary objectivity}
\not\Rightarrow
\text{ontological fundamentality}.
}
\]

\(\square\)

---

# 20. Conclusion

Classical skepticism is often formulated too strongly:

\[
\text{Perhaps everything is unreal.}
\]

That sentence is vulnerable to the objection that it destroys the semantic contrast needed to express itself.

Whole-brain emulation suggests a much more precise hypothesis:

\[
\boxed{
\text{Everything ordinarily encountered may be real,
while the world containing it is implemented by another system.}
}
\]

This hypothesis does not abolish reality.

It stratifies it.

A simulated world can contain genuine persistence, genuine causation, genuine disagreement, genuine discovery, genuine other minds, and genuine objective knowledge.

What its inhabitants may lack is knowledge of the realization relation

\[
\mathcal W
\stackrel{\iota}{\longrightarrow}
\mathcal H.
\]

Consequently, the existence of a stable public world defeats neither simulation nor substrate uncertainty.

It defeats only a much cruder hypothesis: that experience is nothing more than an unstructured private illusion.

The decisive distinction is therefore

\[
\boxed{
\text{real}
\neq
\text{fundamental}.
}
\]

Or, more precisely,

\[
\boxed{
\text{observer-independent reality}
\;\centernot\Rightarrow\;
\text{substrate-level fundamentality}.
}
\]

Whole-brain emulation would not prove that humanity inhabits a simulation.

It would establish something conceptually prior and, in one respect, more important:

\[
\boxed{
\text{a conscious observer can inhabit an objective world
without thereby having access to the ontological level
on which that world is implemented.}
}
\]

At that point the Cartesian question would no longer be supported merely by an imagined demon.

Human beings would have built the counterexample themselves.