#!/usr/bin/env python3
"""Reproducibility checks for *Who Pays the Bill?* Version 4.

The script checks the concrete three-outcome crash test and small finite
instances of the new anti-diagonal causal construction.

It is deliberately split into two kinds of checks:

  (A) manuscript-internal algebra/numerics
      * equal-entropy root q;
      * one-shot MEC witness;
      * the 23-atom synchronous two-step witness;
      * synchronous W1 lower bound;
      * k=0 reciprocal-scale identity;
      * one-step spectral-rigidity inequalities;

  (B) causal Shannon-floor construction
      * largest-residual greedy couplings of product blocks;
      * product-block surprisal W1 and the O(sqrt(L)) bound;
      * numerical check of the two-marginal profile/greedy upper bound;
      * explicit enumeration of a small anti-diagonal causal source;
      * exact path-law checks for every action word in that small instance;
      * source entropy equals the independent-component entropy formula.

The script does NOT prove the external Compton--Katz--Qi--Greenewald--
Kocaoglu profile theorem.  It uses its stated additive constant
    c0 = log2(e)/e
as an external theorem and numerically checks the resulting inequality on
several product-block instances.

Dependencies: numpy, scipy, sympy.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import sympy as sp
from scipy.optimize import brentq


TOL = 2e-11


def H(vals: Iterable[float]) -> float:
    """Shannon entropy in bits."""
    a = np.asarray(list(vals), dtype=float)
    a = a[a > 0.0]
    return float(-(a * np.log2(a)).sum())


def hq(x: float) -> float:
    return -2.0 * x * math.log2(x) - (1.0 - 2.0 * x) * math.log2(1.0 - 2.0 * x)


def varentropy(probs: Sequence[float]) -> float:
    p = np.asarray(probs, dtype=float)
    info = -np.log2(p)
    h = float((p * info).sum())
    return float((p * (info - h) ** 2).sum())


def product_distribution(probs: Sequence[float], L: int) -> Dict[Tuple[int, ...], float]:
    """L-fold product law, retaining outcome labels."""
    p = list(map(float, probs))
    out: Dict[Tuple[int, ...], float] = {}
    for state in itertools.product(range(len(p)), repeat=L):
        mass = 1.0
        for x in state:
            mass *= p[x]
        out[state] = mass
    return out


def entropy_of_dist(d: Mapping[Hashable, float]) -> float:
    return H(d.values())


def largest_residual_greedy(
    A: Mapping[Hashable, float],
    B: Mapping[Hashable, float],
    tol: float = 1e-15,
) -> Dict[Tuple[Hashable, Hashable], float]:
    """Greedy coupling: repeatedly match the largest remaining atoms.

    This is the standard largest-residual formulation: after every match,
    any leftover mass is returned to the heap before the next match.
    """
    heap_a: List[Tuple[float, int, Hashable]] = []
    heap_b: List[Tuple[float, int, Hashable]] = []

    counter = itertools.count()
    for state, mass in A.items():
        if mass > tol:
            heapq.heappush(heap_a, (-float(mass), next(counter), state))
    for state, mass in B.items():
        if mass > tol:
            heapq.heappush(heap_b, (-float(mass), next(counter), state))

    J: Dict[Tuple[Hashable, Hashable], float] = defaultdict(float)

    while heap_a and heap_b:
        na, _, xa = heapq.heappop(heap_a)
        nb, _, xb = heapq.heappop(heap_b)
        a = -na
        b = -nb
        m = min(a, b)
        J[(xa, xb)] += m
        a -= m
        b -= m
        if a > tol:
            heapq.heappush(heap_a, (-a, next(counter), xa))
        if b > tol:
            heapq.heappush(heap_b, (-b, next(counter), xb))

    assert not heap_a and not heap_b

    # Marginal audit.
    ma: Dict[Hashable, float] = defaultdict(float)
    mb: Dict[Hashable, float] = defaultdict(float)
    for (xa, xb), mass in J.items():
        ma[xa] += mass
        mb[xb] += mass
    assert max(abs(ma[x] - float(v)) for x, v in A.items()) < 5e-12
    assert max(abs(mb[x] - float(v)) for x, v in B.items()) < 5e-12
    assert abs(sum(J.values()) - 1.0) < 5e-12
    return dict(J)


def surprisal_law(d: Mapping[Hashable, float]) -> List[Tuple[float, float]]:
    """Return (surprisal, probability-mass) atoms."""
    return sorted((-math.log2(float(p)), float(p)) for p in d.values() if p > 0.0)


def w1_1d_atoms(
    law_a: Sequence[Tuple[float, float]],
    law_b: Sequence[Tuple[float, float]],
) -> float:
    """1-Wasserstein distance by quantile matching of finite 1D laws."""
    a = [[float(x), float(p)] for x, p in sorted(law_a)]
    b = [[float(x), float(p)] for x, p in sorted(law_b)]
    i = j = 0
    ra = a[0][1]
    rb = b[0][1]
    out = 0.0
    while i < len(a) and j < len(b):
        m = min(ra, rb)
        out += m * abs(a[i][0] - b[j][0])
        ra -= m
        rb -= m
        if ra < 1e-14:
            i += 1
            if i < len(a):
                ra = a[i][1]
        if rb < 1e-14:
            j += 1
            if j < len(b):
                rb = b[j][1]
    return out


def target_path_law(action_word: Tuple[int, ...], P: Sequence[float], Q: Sequence[float]):
    """Exact product output law for a fixed action word (0=P, 1=Q)."""
    laws = [P if a == 0 else Q for a in action_word]
    out: Dict[Tuple[int, ...], float] = {}
    for ys in itertools.product(range(3), repeat=len(action_word)):
        p = 1.0
        for t, y in enumerate(ys):
            p *= laws[t][y]
        out[ys] = p
    return out


def enumerate_antidiagonal_source(
    P: Sequence[float],
    Q: Sequence[float],
    m: int,
    L: int,
):
    """Enumerate a small anti-diagonal source for verification.

    Independent components:
      * X^(1) ~ P^L
      * W^(m) ~ Q^L
      * (W^(j), X^(j+1)) ~ J_L, j=1,...,m-1

    J_L is chosen as the largest-residual greedy coupling of Q^L and P^L.
    """
    PL = product_distribution(P, L)
    QL = product_distribution(Q, L)
    JL = largest_residual_greedy(QL, PL)  # first=W block, second=X block

    components = []
    components.append([((x,), px) for x, px in PL.items()])
    for _ in range(m - 1):
        components.append([((w, x), p) for (w, x), p in JL.items()])
    components.append([((w,), pw) for w, pw in QL.items()])

    source = []
    for choice in itertools.product(*components):
        states = [c[0] for c in choice]
        prob = math.prod(c[1] for c in choice)

        Xblocks: List[Tuple[int, ...]] = [states[0][0]]
        Wblocks: List[Tuple[int, ...]] = []
        for j in range(m - 1):
            wj, xnext = states[1 + j]
            Wblocks.append(wj)
            Xblocks.append(xnext)
        Wblocks.append(states[-1][0])

        Xstream = tuple(itertools.chain.from_iterable(Xblocks))
        Wstream = tuple(itertools.chain.from_iterable(Wblocks))
        Zstream = tuple(reversed(Wstream))
        source.append((Xstream, Zstream, prob))

    return source, PL, QL, JL


def realized_output(X: Tuple[int, ...], Z: Tuple[int, ...], action_word: Tuple[int, ...]):
    ip = iq = 0
    y = []
    for a in action_word:
        if a == 0:
            y.append(X[ip])
            ip += 1
        else:
            y.append(Z[iq])
            iq += 1
    return tuple(y)


# -----------------------------------------------------------------------------
# 1. Crash-test channel and equal-entropy root.
# -----------------------------------------------------------------------------

qv = brentq(lambda x: hq(x) - 1.5, 1.0 / 3.0 + 1e-12, 0.5 - 1e-12)
P = np.array([0.5, 0.25, 0.25], dtype=float)
Q = np.array([qv, qv, 1.0 - 2.0 * qv], dtype=float)

assert abs(H(P) - 1.5) < 2e-12
assert abs(H(Q) - 1.5) < 2e-12
assert 3.0 / 8.0 < qv < 5.0 / 12.0

# One-shot MEC witness from the manuscript.
mec_atoms = np.array([0.5 - qv, qv, 2.0 * qv - 0.75, 1.0 - 2.0 * qv, 0.25])
C1 = H(mec_atoms)
E1 = C1 - 1.5


# -----------------------------------------------------------------------------
# 2. Version-3 23-atom synchronous witness: symbolic audit.
# -----------------------------------------------------------------------------

q = sp.symbols("q", positive=True)
entries = {
    (0, 0, 0, 0): (2 * q - 1) * (12 * q - 5) / 8,
    (0, 0, 0, 2): -q * (2 * q - 1),
    (0, 0, 2, 1): q / 4,
    (0, 1, 0, 0): q**2,
    (0, 1, 0, 2): -(2 * q - 1) * (8 * q - 3) / 8,
    (0, 1, 1, 1): q / 4,
    (0, 2, 1, 0): -(2 * q - 1) / 8,
    (0, 2, 2, 0): -(2 * q - 1) * (8 * q - 3) / 4,
    (0, 2, 2, 2): (2 * q - 1) * (16 * q - 7) / 8,
    (1, 0, 0, 0): q / 4,
    (1, 0, 1, 1): (4 * q - 1) / 16,
    (1, 1, 0, 2): -(2 * q - 1) / 8,
    (1, 1, 2, 1): sp.Rational(1, 16),
    (1, 2, 1, 2): -(2 * q - 1) / 8,
    (2, 0, 0, 0): -(32 * q**2 - 40 * q + 11) / 16,
    (2, 0, 0, 1): (4 * q - 1) ** 2 / 16,
    (2, 0, 1, 0): sp.Rational(1, 16),
    (2, 1, 0, 1): (16 * q**2 - 4 * q - 1) / 16,
    (2, 1, 0, 2): -(12 * q - 5) / 16,
    (2, 1, 2, 2): (4 * q - 1) / 16,
    (2, 2, 0, 0): (2 * q - 1) ** 2 / 2,
    (2, 2, 0, 1): -q * (2 * q - 1),
    (2, 2, 2, 0): -(2 * q - 1) / 8,
}

assert sp.simplify(sum(entries.values()) - 1) == 0
Ds = [
    [sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 4)],
    [q, q, 1 - 2 * q],
]
for a1, a2 in itertools.product([0, 1], repeat=2):
    for y1, y2 in itertools.product(range(3), repeat=2):
        lhs = sum(
            v
            for atom, v in entries.items()
            if (atom[0] if a1 == 0 else atom[1]) == y1
            and (atom[2] if a2 == 0 else atom[3]) == y2
        )
        rhs = Ds[a1][y1] * Ds[a2][y2]
        assert sp.simplify(lhs - rhs) == 0

atoms4 = list(itertools.product(range(3), repeat=4))
probs23 = np.array(
    [
        float(sp.N(entries.get(a, 0).subs(q, qv), 40))
        if hasattr(entries.get(a, 0), "subs")
        else float(entries.get(a, 0))
        for a in atoms4
    ]
)
positive23 = probs23[probs23 > 1e-14]
assert len(positive23) == 23
assert positive23.min() > 1e-3
H23 = H(probs23)


# -----------------------------------------------------------------------------
# 3. Synchronous information-spectrum W1 lower bound.
# -----------------------------------------------------------------------------

muP_base = [(1.0, 0.5), (2.0, 0.5)]
muQ_base = [(-math.log2(qv), 2.0 * qv), (-math.log2(1.0 - 2.0 * qv), 1.0 - 2.0 * qv)]
W1_base = w1_1d_atoms(muP_base, muQ_base)
LB_sync = 1.5 + 0.5 * W1_base
UB_sync = H23 / 2.0

assert LB_sync > 1.5
assert UB_sync < C1
assert LB_sync < UB_sync


# -----------------------------------------------------------------------------
# 4. REAL k=0 reciprocal-scale identity.
# -----------------------------------------------------------------------------

p, r = sp.symbols("p r", positive=True)
# p >= r  => max(1/p,1/r)=1/r, so 2F0=r=min(p,r)
assert sp.simplify(2 * (sp.Rational(1, 2) / (1 / r)) - r) == 0
# r >= p  => max(1/p,1/r)=1/p, so 2F0=p=min(p,r)
assert sp.simplify(2 * (sp.Rational(1, 2) / (1 / p)) - p) == 0


# -----------------------------------------------------------------------------
# 5. One-step spectral-rigidity numerical audit.
# -----------------------------------------------------------------------------

lam_star = np.array([qv, 0.25, 1.0 - 2.0 * qv, 0.5 - qv, 2.0 * qv - 0.75])
lam_A = np.array([0.25, 0.25, 1.0 - 2.0 * qv, qv - 0.25, qv - 0.25])
lam_B = np.array([0.75 - qv, 0.25, 1.0 - 2.0 * qv, qv - 0.25, 2.0 * qv - 0.75])
lam_C = np.array([2.0 * qv - 0.5, 0.25, 1.0 - 2.0 * qv, qv - 0.25, 0.5 - qv])
lam_D = np.array([qv, 0.25, qv - 0.25, 0.5 - qv, 0.5 - qv])

# Compare sorted cumulative sums; lambda_star must majorize the listed competitors.
star_sorted = np.sort(lam_star)[::-1]
for other in [lam_A, lam_B, lam_C, lam_D]:
    o = np.sort(other)[::-1]
    cs_diff = np.cumsum(star_sorted) - np.cumsum(o)
    assert cs_diff[:-1].min() > -2e-12
    assert abs(cs_diff[-1]) < 2e-12
    assert H(lam_star) < H(other) - 1e-10


# -----------------------------------------------------------------------------
# 6. Product-block coupling numerics for the causal theorem.
# -----------------------------------------------------------------------------

VP = varentropy(P)
VQ = varentropy(Q)
Aconst = 0.5 * (math.sqrt(VP) + math.sqrt(VQ))
c0 = math.log2(math.e) / math.e

block_rows = []
for L in range(1, 7):
    PL = product_distribution(P, L)
    QL = product_distribution(Q, L)
    JL = largest_residual_greedy(PL, QL)

    HPL = entropy_of_dist(PL)
    HQL = entropy_of_dist(QL)
    HJ = entropy_of_dist(JL)

    w1 = w1_1d_atoms(surprisal_law(PL), surprisal_law(QL))
    moment_w1_upper = L * abs(H(P) - H(Q)) + math.sqrt(L * VP) + math.sqrt(L * VQ)
    profile_value = 0.5 * (HPL + HQL + w1)
    analytic_upper = L * max(H(P), H(Q)) + Aconst * math.sqrt(L) + c0

    assert w1 <= moment_w1_upper + 2e-11
    # External theorem check on this numerical instance.
    assert HJ <= profile_value + c0 + 3e-10
    # Consequence used in the manuscript.
    assert HJ <= analytic_upper + 3e-10

    block_rows.append((L, HJ, w1, profile_value, analytic_upper))


# -----------------------------------------------------------------------------
# 7. Explicit small anti-diagonal causal construction.
# -----------------------------------------------------------------------------

# Small enough to enumerate every source atom and every action word.
M_TEST = 3
L_TEST = 1
N_TEST = M_TEST * L_TEST
source, PL1, QL1, JL1 = enumerate_antidiagonal_source(P, Q, M_TEST, L_TEST)

source_mass = sum(prob for _, _, prob in source)
assert abs(source_mass - 1.0) < 2e-12

# Verify every length-n action word has exactly the required product output law.
max_path_error = 0.0
for action_word in itertools.product([0, 1], repeat=N_TEST):
    observed: Dict[Tuple[int, ...], float] = defaultdict(float)
    for Xstream, Zstream, prob in source:
        observed[realized_output(Xstream, Zstream, action_word)] += prob

    target = target_path_law(action_word, P, Q)
    keys = set(observed) | set(target)
    err = max(abs(observed.get(k, 0.0) - target.get(k, 0.0)) for k in keys)
    max_path_error = max(max_path_error, err)
    assert err < 3e-11, (action_word, err)

# Entropy agrees with the independent-component formula.
H_source_enum = H(prob for _, _, prob in source)
H_source_formula = (
    L_TEST * H(P)
    + L_TEST * H(Q)
    + (M_TEST - 1) * entropy_of_dist(JL1)
)
assert abs(H_source_enum - H_source_formula) < 3e-11

# Combinatorial anti-diagonal inequality: any correlated pair W^(j), X^(j+1)
# would require n+2 actions before both sides become visible.
for j in range(1, M_TEST):  # manuscript indexing j=1,...,m-1
    need_P = j * L_TEST + 1
    need_Q = N_TEST - j * L_TEST + 1
    assert need_P + need_Q == N_TEST + 2


# -----------------------------------------------------------------------------
# 8. Rate-bound illustrations from the analytic theorem.
# -----------------------------------------------------------------------------

def causal_rate_upper(m: int, L: int) -> float:
    hstar = max(H(P), H(Q))
    hmin = min(H(P), H(Q))
    return hstar + hmin / m + Aconst / math.sqrt(L) + c0 / L

rate_examples = [(8, 16), (16, 64), (32, 256), (64, 1024)]
rate_values = [(m, L, causal_rate_upper(m, L)) for m, L in rate_examples]
assert rate_values[-1][2] < rate_values[0][2]
assert rate_values[-1][2] > 1.5


# -----------------------------------------------------------------------------
# Report.
# -----------------------------------------------------------------------------

print("WHO PAYS THE BILL? v4 -- REPRODUCIBILITY AUDIT")
print("=" * 68)
print(f"q                              = {qv:.17f}")
print(f"H(P), H(Q)                     = {H(P):.15f}, {H(Q):.15f}")
print(f"C1 = MEC witness               = {C1:.15f}")
print(f"E1 = C1 - 1.5                  = {E1:.15f}")
print()
print("SYNCHRONOUS CHECKS")
print(f"23-atom witness entropy        = {H23:.15f}")
print(f"23-atom witness rate           = {UB_sync:.15f}")
print(f"W1(base surprisal laws)        = {W1_base:.15f}")
print(f"synchronous lower bound        = {LB_sync:.15f}")
print(f"certified sync interval        = [{LB_sync:.15f}, {UB_sync:.15f}]")
print()
print("CAUSAL BLOCK INGREDIENTS")
print(f"V_P                            = {VP:.15f}")
print(f"V_Q                            = {VQ:.15f}")
print(f"A=(sqrt(VP)+sqrt(VQ))/2        = {Aconst:.15f}")
print(f"c0=log2(e)/e                   = {c0:.15f}")
print()
print("L   H(greedy J_L)      W1(product surprisals)   profile value      analytic upper")
for L, HJ, w1, profile_value, analytic_upper in block_rows:
    print(f"{L:<2d}  {HJ:>16.12f}   {w1:>20.12f}   {profile_value:>14.12f}   {analytic_upper:>14.12f}")
print()
print("ANTI-DIAGONAL ENUMERATION")
print(f"test m,L,n                     = {M_TEST},{L_TEST},{N_TEST}")
print(f"enumerated source atoms        = {len(source)}")
print(f"max path-law absolute error    = {max_path_error:.3e}")
print(f"H(source), formula             = {H_source_enum:.15f}, {H_source_formula:.15f}")
print()
print("ANALYTIC CAUSAL RATE UPPER BOUNDS")
for m, L, rate in rate_values:
    print(f"m={m:<3d} L={L:<5d} n={m*L:<7d} rate <= {rate:.12f}")
print()
print("CRASH-TEST ASYMPTOTIC LEDGER")
print(f"C_inf^causal                   = 1.500000000000000  (analytic theorem)")
print(f"C_inf^syn lower                = {LB_sync:.15f}")
print(f"strict asymptotic gap >=       = {LB_sync - 1.5:.15f}")
print()
print("ALL CHECKS PASSED")
