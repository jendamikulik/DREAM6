#!/usr/bin/env python3
"""
WHO PAYS THE BILL? v5 -- REPRODUCIBILITY / PROOF-INGREDIENT AUDIT
================================================================

Purpose
-------
This script is a companion audit for Version 5. It does NOT replace the
analytic proofs in the manuscript. Instead it separates and checks:

(A) EXACT SYMBOLIC IDENTITIES
    - 23-atom synchronous witness normalization and all four transversal laws
    - k=0 reciprocal-scale min/max identity
    - parity-ladder moment matching through arbitrary tested order
    - one-step spectral-rigidity inequalities (symbolic sign structure)

(B) FINITE EXHAUSTIVE / CONSTRUCTIVE CHECKS
    - one-step MEC witness value
    - explicit small anti-diagonal causal source, enumerating every latent atom
      and every action word
    - vector-state / projection representation of the same synchronous law

(C) NUMERICAL ANALYTIC-ID CHECKS
    - surprisal operator moments and characteristic function
    - Kantorovich dual LP = primal W1
    - REAL Mellin spectral density normalization
    - Fourier transform of the Mellin density = scale-orbit correlation
    - dilation-energy second moment formula
    - reciprocal-scale overlap relation
    - hard-edge / smooth-edge local coefficients
    - large-k harmonic-mean / sech limits
    - spectral-probe inequality D_eta <= sigma_eta W1

(D) ASYMPTOTIC NUMERICAL ILLUSTRATIONS
    - crash-pair block W1 / sqrt(L) -> Gaussian coefficient
    - parity arithmetic barrier under convolution
    - analytic causal rate upper bounds from the anti-diagonal theorem

The script intentionally labels claims that remain analytic theorems rather
than pretending that finite computation proves an asymptotic statement.

Dependencies
------------
numpy
scipy
sympy

Tested with Python 3.11+.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import sympy as sp

from scipy.integrate import quad
from scipy.optimize import brentq, linprog
from scipy.special import gammaln
from scipy.stats import binom


# =============================================================================
# Global tolerances
# =============================================================================

TOL = 2e-11
FT_TOL = 2e-8
QUAD_EPS = 2e-11


# =============================================================================
# Basic information-theoretic helpers
# =============================================================================

def H(vals: Iterable[float]) -> float:
    """Shannon entropy in bits."""
    a = np.asarray(list(vals), dtype=float)
    a = a[a > 0.0]
    return float(-(a * np.log2(a)).sum())


def hq(x: float) -> float:
    """Entropy of Q=(x,x,1-2x)."""
    return -2.0 * x * math.log2(x) - (1.0 - 2.0 * x) * math.log2(1.0 - 2.0 * x)


def varentropy(probs: Sequence[float]) -> float:
    p = np.asarray(probs, dtype=float)
    info = -np.log2(p)
    h = float((p * info).sum())
    return float((p * (info - h) ** 2).sum())


def aggregate_surprisal_law(probs: Sequence[float], ndigits: int = 14) -> List[Tuple[float, float]]:
    """Aggregate equal surprisal values."""
    d: Dict[float, float] = defaultdict(float)
    for p in probs:
        p = float(p)
        u = -math.log2(p)
        # Rounding only groups values known to be mathematically equal
        # (e.g. duplicate probabilities q,q or 1/4,1/4).
        d[round(u, ndigits)] += p
    return sorted((float(u), float(m)) for u, m in d.items())


def entropy_of_dist(d: Mapping[Hashable, float]) -> float:
    return H(d.values())


def product_distribution(probs: Sequence[float], L: int) -> Dict[Tuple[int, ...], float]:
    """Explicit L-fold product law retaining outcome labels."""
    p = list(map(float, probs))
    out: Dict[Tuple[int, ...], float] = {}
    for state in itertools.product(range(len(p)), repeat=L):
        mass = 1.0
        for x in state:
            mass *= p[x]
        out[state] = mass
    return out


def surprisal_law_of_dist(d: Mapping[Hashable, float]) -> List[Tuple[float, float]]:
    """Return (surprisal, probability mass) atoms, aggregated by value."""
    agg: Dict[float, float] = defaultdict(float)
    for p in d.values():
        if p > 0.0:
            u = -math.log2(float(p))
            agg[round(u, 13)] += float(p)
    return sorted((float(u), float(m)) for u, m in agg.items())


def w1_1d_atoms(
    law_a: Sequence[Tuple[float, float]],
    law_b: Sequence[Tuple[float, float]],
) -> float:
    """Exact quantile matching algorithm for finite 1D laws (floating masses)."""
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
    return float(out)


def w1_dual_lp(
    law_a: Sequence[Tuple[float, float]],
    law_b: Sequence[Tuple[float, float]],
) -> float:
    """Independent Kantorovich--Rubinstein dual LP check.

    On a sorted finite support, adjacent Lipschitz constraints are sufficient.
    We fix f(x_0)=0 to remove the irrelevant additive constant.
    """
    xs = sorted(set([float(x) for x, _ in law_a] + [float(x) for x, _ in law_b]))
    idx = {x: i for i, x in enumerate(xs)}
    wa = np.zeros(len(xs))
    wb = np.zeros(len(xs))
    for x, p in law_a:
        wa[idx[float(x)]] += float(p)
    for x, p in law_b:
        wb[idx[float(x)]] += float(p)
    diff = wa - wb

    A_ub = []
    b_ub = []
    for i in range(len(xs) - 1):
        gap = xs[i + 1] - xs[i]
        row = np.zeros(len(xs))
        row[i + 1] = 1.0
        row[i] = -1.0
        A_ub.append(row)
        b_ub.append(gap)
        A_ub.append(-row)
        b_ub.append(gap)

    bounds = [(0.0, 0.0)] + [(None, None)] * (len(xs) - 1)
    res = linprog(
        -diff,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )
    assert res.success, res.message
    return float(-res.fun)


# =============================================================================
# Largest-residual greedy coupling
# =============================================================================

def largest_residual_greedy(
    A: Mapping[Hashable, float],
    B: Mapping[Hashable, float],
    tol: float = 1e-15,
) -> Dict[Tuple[Hashable, Hashable], float]:
    """Repeatedly match the currently largest residual atoms."""
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

    ma: Dict[Hashable, float] = defaultdict(float)
    mb: Dict[Hashable, float] = defaultdict(float)
    for (xa, xb), mass in J.items():
        ma[xa] += mass
        mb[xb] += mass

    assert max(abs(ma[x] - float(v)) for x, v in A.items()) < 5e-12
    assert max(abs(mb[x] - float(v)) for x, v in B.items()) < 5e-12
    assert abs(sum(J.values()) - 1.0) < 5e-12
    return dict(J)


# =============================================================================
# Anti-diagonal causal construction
# =============================================================================

def target_path_law(
    action_word: Tuple[int, ...],
    P: Sequence[float],
    Q: Sequence[float],
) -> Dict[Tuple[int, ...], float]:
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


def realized_output(
    X: Tuple[int, ...],
    Z: Tuple[int, ...],
    action_word: Tuple[int, ...],
) -> Tuple[int, ...]:
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


# =============================================================================
# Block surprisal laws for two-point surprisal distributions
# =============================================================================

def two_point_convolution_law(
    v1: float,
    w1: float,
    v2: float,
    w2: float,
    L: int,
) -> List[Tuple[float, float]]:
    """Law of sum of L iid {v1,v2} variables."""
    assert abs(w1 + w2 - 1.0) < 1e-12
    ks = np.arange(L + 1)
    vals = ks * v1 + (L - ks) * v2
    probs = binom.pmf(ks, L, w1)
    idx = np.argsort(vals)
    return [
        (float(vals[i]), float(probs[i]))
        for i in idx
        if probs[i] > 0.0
    ]


# =============================================================================
# REAL / Mellin kernel helpers
# =============================================================================

def Hk_ratio(k: int, r: float) -> float:
    """H_k(r)/H_k(1) from the normalized integral representation."""
    assert 0.0 <= r <= 1.0

    def integrand(u: float) -> float:
        return (1.0 - u) ** k * (1.0 - r * u) ** k

    val, _ = quad(
        integrand,
        0.0,
        1.0,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=300,
    )
    # H_k(r)/H_k(1) = (2k+1) * integral
    return float((2 * k + 1) * val)


def Hk_value(k: int, r: float) -> float:
    H1 = (k + 1) ** 2 / (2.0 * (2 * k + 1))
    return H1 * Hk_ratio(k, r)


def Rk_analytic(k: int, t: float) -> float:
    """Normalized dilation correlation from H_k."""
    tau = abs(float(t))
    r = math.exp(-tau)
    return math.exp(-tau / 2.0) * Hk_ratio(k, r)


def wk_density(xi: float, k: int) -> float:
    """Explicit normalized Mellin spectral density w_k."""
    xi = float(xi)
    js = np.arange(k + 1, dtype=float) + 0.5
    logw = (
        math.log(2 * k + 1)
        - math.log(2.0 * math.pi)
        + 2.0 * gammaln(k + 1)
        - float(np.log(xi * xi + js * js).sum())
    )
    return math.exp(logw)


def wk_fourier(k: int, t: float) -> float:
    """Fourier transform of even w_k using oscillatory quadrature."""
    val, err = quad(
        lambda x: wk_density(x, k),
        0.0,
        np.inf,
        weight="cos",
        wvar=float(t),
        epsabs=2e-12,
        epsrel=2e-12,
        limit=600,
    )
    return float(2.0 * val)


def phi_surprisal(probs: Sequence[float], t: float) -> complex:
    p = np.asarray(probs, dtype=float)
    u = -np.log2(p)
    return complex(np.sum(p * np.exp(1j * float(t) * u)))


def power_sum_vertical(probs: Sequence[float], t: float) -> complex:
    """Z_P(1-it/ln2)."""
    p = np.asarray(probs, dtype=float)
    s = 1.0 - 1j * float(t) / math.log(2.0)
    return complex(np.sum(p ** s))


def A_k(k: int, p: float, q: float) -> float:
    """Normalized reciprocal-scale overlap A_k(p,q)."""
    m = min(p, q)
    M = max(p, q)
    return m * Hk_ratio(k, m / M)


def A_k_orbit(k: int, p: float, q: float) -> float:
    return math.sqrt(p * q) * Rk_analytic(k, abs(math.log(p / q)))


def c_k(k: int, u: float, v: float) -> float:
    p = 2.0 ** (-u)
    q = 2.0 ** (-v)
    return -math.log2(A_k(k, p, q))


def D_eta_REAL(P: Sequence[float], Q: Sequence[float], k: int) -> float:
    """D_eta where eta has density w_k."""
    assert k >= 1

    def integrand(x: float) -> float:
        d = phi_surprisal(P, x) - phi_surprisal(Q, x)
        return (d.real * d.real + d.imag * d.imag) * wk_density(x, k)

    val, _ = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=1e-10,
        epsrel=2e-9,
        limit=600,
    )
    return math.sqrt(2.0 * val)


# =============================================================================
# Parity-ladder helpers
# =============================================================================

def parity_law_exact(R: int, even: bool) -> Dict[int, sp.Rational]:
    d: Dict[int, sp.Rational] = {}
    for j in range(R + 2):
        if (j % 2 == 0) == even:
            d[R + j] = sp.Rational(math.comb(R + 1, j), 2 ** R)
    return d


def parity_law_float(R: int, even: bool) -> Dict[int, float]:
    return {u: float(m) for u, m in parity_law_exact(R, even).items()}


def convolve_integer_law(d: Mapping[int, float], L: int) -> Dict[int, float]:
    out: Dict[int, float] = {0: 1.0}
    for _ in range(L):
        nxt: Dict[int, float] = defaultdict(float)
        for x, p in out.items():
            for y, q in d.items():
                nxt[x + y] += p * q
        out = dict(nxt)
    return out


# =============================================================================
# 1. Crash-test base channel
# =============================================================================

qv = brentq(lambda x: hq(x) - 1.5, 1.0 / 3.0 + 1e-12, 0.5 - 1e-12)

P = np.array([0.5, 0.25, 0.25], dtype=float)
Q = np.array([qv, qv, 1.0 - 2.0 * qv], dtype=float)

assert abs(H(P) - 1.5) < 2e-12
assert abs(H(Q) - 1.5) < 2e-12
assert 3.0 / 8.0 < qv < 5.0 / 12.0

mec_atoms = np.array(
    [qv, 0.25, 1.0 - 2.0 * qv, 0.5 - qv, 2.0 * qv - 0.75],
    dtype=float,
)
assert abs(mec_atoms.sum() - 1.0) < 2e-12
assert mec_atoms.min() > 0.0

C1 = H(mec_atoms)
E1 = C1 - 1.5


# =============================================================================
# 2. Exact symbolic 23-atom synchronous witness
# =============================================================================

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
        float(sp.N(entries.get(a, 0).subs(q, qv), 50))
        if hasattr(entries.get(a, 0), "subs")
        else float(entries.get(a, 0))
        for a in atoms4
    ],
    dtype=float,
)

positive_idx = np.where(probs23 > 1e-14)[0]
positive23 = probs23[positive_idx]
assert len(positive23) == 23
assert positive23.min() > 1e-3
assert abs(positive23.sum() - 1.0) < 2e-12
H23 = H(positive23)


# =============================================================================
# 3. One-vector / commutative-algebra representation
# =============================================================================

# Restrict to positive support. In this basis the multiplication algebra is C^23.
support23 = [atoms4[i] for i in positive_idx]
psi23 = np.sqrt(positive23)

assert abs(float(np.dot(psi23, psi23)) - 1.0) < 2e-12

# Each transversal event is a diagonal projection. Check <psi,Pi psi>.
max_projection_error = 0.0
for a1, a2 in itertools.product([0, 1], repeat=2):
    for y1, y2 in itertools.product(range(3), repeat=2):
        indicator = np.array(
            [
                1.0
                if ((atom[0] if a1 == 0 else atom[1]) == y1
                    and (atom[2] if a2 == 0 else atom[3]) == y2)
                else 0.0
                for atom in support23
            ]
        )
        expectation = float(np.dot(psi23, indicator * psi23))
        target = float((P if a1 == 0 else Q)[y1] * (P if a2 == 0 else Q)[y2])
        max_projection_error = max(max_projection_error, abs(expectation - target))
assert max_projection_error < 3e-12


# =============================================================================
# 4. Surprisal operator / full spectral ledger
# =============================================================================

def audit_surprisal_operator(probs: np.ndarray):
    psi = np.sqrt(probs)
    Ldiag = -np.log2(probs)

    H_op = float(np.dot(psi, Ldiag * psi))
    V_op = float(np.dot(psi, (Ldiag - H_op) ** 2 * psi))

    assert abs(H_op - H(probs)) < 2e-12
    assert abs(V_op - varentropy(probs)) < 2e-12

    max_phi_error = 0.0
    for t in [0.0, 0.1, 0.7, math.pi, 2.3]:
        a = phi_surprisal(probs, t)
        b = power_sum_vertical(probs, t)
        max_phi_error = max(max_phi_error, abs(a - b))
    assert max_phi_error < 2e-12
    return H_op, V_op, max_phi_error

HP_op, VP_op, phiP_err = audit_surprisal_operator(P)
HQ_op, VQ_op, phiQ_err = audit_surprisal_operator(Q)

# Spectral-ledger completeness: recover multiplicities m_u / p.
def recovered_multiplicities(probs: np.ndarray):
    law = aggregate_surprisal_law(probs)
    vals = []
    for u, mass in law:
        p = 2.0 ** (-u)
        mult = mass / p
        vals.append((u, p, mult))
        assert abs(mult - round(mult)) < 2e-11
    return vals

recP = recovered_multiplicities(P)
recQ = recovered_multiplicities(Q)


# =============================================================================
# 5. W1 = d_Lip independently via primal quantiles and dual LP
# =============================================================================

muP_base = aggregate_surprisal_law(P)
muQ_base = aggregate_surprisal_law(Q)

W1_base = w1_1d_atoms(muP_base, muQ_base)
W1_dual = w1_dual_lp(muP_base, muQ_base)

assert abs(W1_base - W1_dual) < 3e-11

LB_sync = 1.5 + 0.5 * W1_base
UB_sync = H23 / 2.0

assert LB_sync > 1.5
assert LB_sync < UB_sync
assert UB_sync < C1


# =============================================================================
# 6. Exact k=0 reciprocal-scale identity
# =============================================================================

p, r = sp.symbols("p r", positive=True)

# Case p >= r: max(1/p,1/r)=1/r, so 2F0=r=min(p,r)
assert sp.simplify(2 * (sp.Rational(1, 2) / (1 / r)) - r) == 0
# Case r >= p: max(1/p,1/r)=1/p, so 2F0=p=min(p,r)
assert sp.simplify(2 * (sp.Rational(1, 2) / (1 / p)) - p) == 0


# =============================================================================
# 7. One-step spectral rigidity
# =============================================================================

lam_star = np.array([qv, 0.25, 1.0 - 2.0 * qv, 0.5 - qv, 2.0 * qv - 0.75])
lam_A = np.array([0.25, 0.25, 1.0 - 2.0 * qv, qv - 0.25, qv - 0.25])
lam_B = np.array([0.75 - qv, 0.25, 1.0 - 2.0 * qv, qv - 0.25, 2.0 * qv - 0.75])
lam_C = np.array([2.0 * qv - 0.5, 0.25, 1.0 - 2.0 * qv, qv - 0.25, 0.5 - qv])
lam_D = np.array([qv, 0.25, qv - 0.25, 0.5 - qv, 0.5 - qv])

star_sorted = np.sort(lam_star)[::-1]
for other in [lam_A, lam_B, lam_C, lam_D]:
    o = np.sort(other)[::-1]
    cs_diff = np.cumsum(star_sorted) - np.cumsum(o)
    assert cs_diff[:-1].min() > -2e-12
    assert abs(cs_diff[-1]) < 2e-12
    assert H(lam_star) < H(other) - 1e-10


# =============================================================================
# 8. Product-block causal ingredients (Version 4 retained)
# =============================================================================

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

    w1 = w1_1d_atoms(surprisal_law_of_dist(PL), surprisal_law_of_dist(QL))
    moment_w1_upper = L * abs(H(P) - H(Q)) + math.sqrt(L * VP) + math.sqrt(L * VQ)
    profile_value = 0.5 * (HPL + HQL + w1)
    analytic_upper = L * max(H(P), H(Q)) + Aconst * math.sqrt(L) + c0

    assert w1 <= moment_w1_upper + 2e-10

    # This invokes the published profile/greedy theorem as an EXTERNAL theorem.
    # We merely check the inequality numerically on these instances.
    assert HJ <= profile_value + c0 + 4e-10
    assert HJ <= analytic_upper + 4e-10

    block_rows.append((L, HJ, w1, profile_value, analytic_upper))


# =============================================================================
# 9. Exhaustive small anti-diagonal construction
# =============================================================================

M_TEST = 3
L_TEST = 1
N_TEST = M_TEST * L_TEST

source, PL1, QL1, JL1 = enumerate_antidiagonal_source(P, Q, M_TEST, L_TEST)
assert abs(sum(prob for _, _, prob in source) - 1.0) < 2e-12

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

H_source_enum = H(prob for _, _, prob in source)
H_source_formula = (
    L_TEST * H(P)
    + L_TEST * H(Q)
    + (M_TEST - 1) * entropy_of_dist(JL1)
)
assert abs(H_source_enum - H_source_formula) < 3e-11

for j in range(1, M_TEST):
    need_P = j * L_TEST + 1
    need_Q = N_TEST - j * L_TEST + 1
    assert need_P + need_Q == N_TEST + 2


# =============================================================================
# 10. Block-resolution CLT coefficient for the crash test
# =============================================================================

p_info_vals = (1.0, 2.0)
p_info_weights = (0.5, 0.5)

q_info_vals = (-math.log2(qv), -math.log2(1.0 - 2.0 * qv))
q_info_weights = (2.0 * qv, 1.0 - 2.0 * qv)

c_crash = math.sqrt(2.0 / math.pi) * abs(math.sqrt(VP) - math.sqrt(VQ))

clt_rows = []
for L in [16, 64, 256, 1024, 4096]:
    lawP_L = two_point_convolution_law(
        p_info_vals[0], p_info_weights[0],
        p_info_vals[1], p_info_weights[1],
        L,
    )
    lawQ_L = two_point_convolution_law(
        q_info_vals[0], q_info_weights[0],
        q_info_vals[1], q_info_weights[1],
        L,
    )
    w1L = w1_1d_atoms(lawP_L, lawQ_L)
    scaled = w1L / math.sqrt(L)
    clt_rows.append((L, w1L, scaled, scaled / c_crash))

# Strong numerical sanity check that the largest-L normalized value is close
# to the predicted Gaussian coefficient. This is an illustration, not proof.
assert abs(clt_rows[-1][2] / c_crash - 1.0) < 0.03


# =============================================================================
# 11. Dyadic parity ladder -- exact moment camouflage + arithmetic barrier
# =============================================================================

parity_rows = []

for R in range(2, 8):
    PE = parity_law_exact(R, True)
    PO = parity_law_exact(R, False)

    assert sp.simplify(sum(PE.values()) - 1) == 0
    assert sp.simplify(sum(PO.values()) - 1) == 0

    for m in range(R + 1):
        me = sp.simplify(sum(w * (u ** m) for u, w in PE.items()))
        mo = sp.simplify(sum(w * (u ** m) for u, w in PO.items()))
        assert sp.simplify(me - mo) == 0

    next_diff = sp.simplify(
        sum(w * (u ** (R + 1)) for u, w in PE.items())
        -
        sum(w * (u ** (R + 1)) for u, w in PO.items())
    )
    target_abs = sp.factorial(R + 1) / (2 ** R)
    assert sp.simplify(abs(next_diff) - target_abs) == 0

    PEf = {u: float(w) for u, w in PE.items()}
    POf = {u: float(w) for u, w in PO.items()}

    w1base = w1_1d_atoms(sorted(PEf.items()), sorted(POf.items()))
    assert abs(w1base - 1.0) < 2e-12

    meanE = sum(float(w) * u for u, w in PE.items())
    meanO = sum(float(w) * u for u, w in PO.items())
    varE = sum(float(w) * (u - meanE) ** 2 for u, w in PE.items())
    varO = sum(float(w) * (u - meanO) ** 2 for u, w in PO.items())

    h_formula = (3 * R + 1) / 2.0
    v_formula = (R + 1) / 4.0

    assert abs(meanE - h_formula) < 2e-12
    assert abs(meanO - h_formula) < 2e-12
    assert abs(varE - v_formula) < 2e-12
    assert abs(varO - v_formula) < 2e-12

    odd_w1s = []
    for L in [1, 3, 5]:
        cE = convolve_integer_law(PEf, L)
        cO = convolve_integer_law(POf, L)
        w = w1_1d_atoms(sorted(cE.items()), sorted(cO.items()))
        assert w >= 1.0 - 3e-12
        odd_w1s.append(w)

    parity_rows.append((R, h_formula, v_formula, float(abs(next_diff)), odd_w1s))


# =============================================================================
# 12. REAL Mellin spectrum: normalization, transform, energy, tail degree
# =============================================================================

mellin_rows = []
max_fourier_error = 0.0

xi_sym = sp.symbols("xi", real=True)

for k in range(0, 6):
    norm_half, _ = quad(
        lambda x: wk_density(x, k),
        0.0,
        np.inf,
        epsabs=2e-12,
        epsrel=2e-12,
        limit=600,
    )
    norm = 2.0 * norm_half
    assert abs(norm - 1.0) < 2e-10

    # Exact tail-degree check by symbolic leading power for each tested integer k.
    denom = sp.prod(xi_sym**2 + sp.Rational(2*j + 1, 2) ** 2 for j in range(k + 1))
    rational_shape = 1 / denom
    tail_limit = sp.limit(
        rational_shape * xi_sym ** (2 * k + 2),
        xi_sym,
        sp.oo,
    )
    assert sp.simplify(tail_limit - 1) == 0

    # Fourier transform = orbit correlation at several t.
    errs = []
    for t in [0.3, 1.0, 2.0]:
        ft = wk_fourier(k, t)
        target = Rk_analytic(k, t)
        err = abs(ft - target)
        errs.append(err)
        max_fourier_error = max(max_fourier_error, err)
        assert err < FT_TOL

    second = math.inf
    second_formula = math.inf

    if k >= 1:
        second_half, _ = quad(
            lambda x: x * x * wk_density(x, k),
            0.0,
            np.inf,
            epsabs=2e-11,
            epsrel=2e-11,
            limit=600,
        )
        second = 2.0 * second_half
        second_formula = (2 * k + 1) / (4.0 * (2 * k - 1))
        assert abs(second - second_formula) < 2e-9

    mellin_rows.append((k, norm, max(errs), second, second_formula))

# k=0 exact Cauchy correlation check
for t in [0.1, 0.7, 1.5, 3.0]:
    assert abs(Rk_analytic(0, t) - math.exp(-abs(t) / 2.0)) < 2e-12


# =============================================================================
# 13. Direct reciprocal-scale overlap and hard/smooth local classes
# =============================================================================

# A_k = sqrt(pq) R_k(|ln(p/q)|) and min <= A_k <= sqrt(pq)
max_A_error = 0.0

test_pairs = [
    (0.5, 0.25),
    (0.41, 0.18),
    (0.10, 0.07),
    (0.8, 0.2),
]

for k in range(0, 6):
    for pp, qq in test_pairs:
        a = A_k(k, pp, qq)
        b = A_k_orbit(k, pp, qq)
        max_A_error = max(max_A_error, abs(a - b))
        assert abs(a - b) < 2e-11
        assert a >= min(pp, qq) - 2e-12
        assert a <= math.sqrt(pp * qq) + 2e-12

# k=0 exact max surprisal
for u, v in [(1.0, 2.0), (1.2, 1.7), (4.0, 0.8), (3.3, 3.3)]:
    assert abs(c_k(0, u, v) - max(u, v)) < 2e-11

# Local edge coefficients:
# k=0: cost excess / |delta| -> 1/2
delta = 1e-5
u0 = 2.0
excess0 = c_k(0, u0 - delta / 2.0, u0 + delta / 2.0) - u0
assert abs(excess0 / abs(delta) - 0.5) < 2e-6

smooth_local_rows = []
for k in [1, 2, 3, 5]:
    E_k = (2 * k + 1) / (4.0 * (2 * k - 1))
    predicted = math.log(2.0) * E_k / 2.0

    # Use a not-too-small delta to avoid cancellation in log2.
    delta = 2e-4
    u = 2.0 - delta / 2.0
    v = 2.0 + delta / 2.0
    excess = c_k(k, u, v) - (u + v) / 2.0
    observed = excess / (delta * delta)

    assert abs(observed / predicted - 1.0) < 3e-4
    smooth_local_rows.append((k, observed, predicted))


# =============================================================================
# 14. Large-k limits: harmonic mean and sech
# =============================================================================

large_k_rows = []

for k in [5, 20, 100]:
    err_H = max(
        abs(Hk_ratio(k, r0) - 2.0 / (1.0 + r0))
        for r0 in [0.2, 0.5, 0.8]
    )

    err_R = max(
        abs(Rk_analytic(k, t) - 1.0 / math.cosh(t / 2.0))
        for t in [0.5, 1.0, 2.0]
    )

    err_w = max(
        abs(wk_density(x, k) - 1.0 / math.cosh(math.pi * x))
        for x in [0.0, 0.5, 1.0]
    )

    large_k_rows.append((k, err_H, err_R, err_w))

# Require visible improvement from k=5 to k=100.
assert large_k_rows[-1][1] < large_k_rows[0][1]
assert large_k_rows[-1][2] < large_k_rows[0][2]
assert large_k_rows[-1][3] < large_k_rows[0][3]


# =============================================================================
# 15. Spectral-probe inequality D_eta <= sigma_eta W1
# =============================================================================

probe_rows = []

for k in [1, 2, 3, 4]:
    D = D_eta_REAL(P, Q, k)
    E_k = (2 * k + 1) / (4.0 * (2 * k - 1))
    sigma = math.sqrt(E_k)
    rhs = sigma * W1_base

    assert D <= rhs + 2e-9

    counterfactual_increment_certificate = D / (2.0 * sigma)
    assert counterfactual_increment_certificate <= 0.5 * W1_base + 2e-9

    probe_rows.append(
        (k, D, sigma, rhs, D / rhs, counterfactual_increment_certificate)
    )


# =============================================================================
# 16. Analytic causal-rate upper-bound illustrations
# =============================================================================

def causal_rate_upper(m: int, L: int) -> float:
    hstar = max(H(P), H(Q))
    hmin = min(H(P), H(Q))
    return hstar + hmin / m + Aconst / math.sqrt(L) + c0 / L


rate_examples = [(8, 16), (16, 64), (32, 256), (64, 1024)]
rate_values = [(m, L, causal_rate_upper(m, L)) for m, L in rate_examples]

assert rate_values[-1][2] < rate_values[0][2]
assert rate_values[-1][2] > 1.5


# =============================================================================
# REPORT
# =============================================================================

print("WHO PAYS THE BILL? v5 -- REPRODUCIBILITY / PROOF-INGREDIENT AUDIT")
print("=" * 82)
print()
print("STATUS LEGEND")
print("  [EXACT]   symbolic identity or exhaustive finite check")
print("  [NUMERIC] independent numerical verification of an analytic identity")
print("  [ASYMPT]  finite numerical illustration of a separately proved asymptotic theorem")
print("  [EXTERNAL] uses a published theorem as an input; not proved by this script")
print()

print("BASE CRASH TEST")
print(f"q                                       = {qv:.17f}")
print(f"H(P), H(Q)                              = {H(P):.15f}, {H(Q):.15f}")
print(f"C1 one-shot witness                     = {C1:.15f}")
print(f"E1                                      = {E1:.15f}")
print()

print("[EXACT] SYNCHRONOUS WITNESS + VECTOR ALGEBRA")
print(f"23-atom witness entropy                 = {H23:.15f}")
print(f"23-atom witness rate                    = {UB_sync:.15f}")
print(f"max <psi,Pi psi>-target error           = {max_projection_error:.3e}")
print()

print("[EXACT/NUMERIC] SURPRISAL SPECTRAL LEDGER")
print(f"H via surprisal operator P,Q            = {HP_op:.15f}, {HQ_op:.15f}")
print(f"V via surprisal operator P,Q            = {VP_op:.15f}, {VQ_op:.15f}")
print(f"max Phi_P(t)-Z_P(1-it/ln2) error        = {max(phiP_err, phiQ_err):.3e}")
print(f"W1 primal quantile                      = {W1_base:.15f}")
print(f"d_Lip dual LP                           = {W1_dual:.15f}")
print(f"sync lower bound                        = {LB_sync:.15f}")
print(f"certified sync interval                 = [{LB_sync:.15f}, {UB_sync:.15f}]")
print()

print("[EXACT] LEDGER COMPLETENESS -- RECOVERED MULTIPLICITIES")
print("P:")
for u, pp, mult in recP:
    print(f"  surprisal={u: .12f}  p={pp:.12f}  multiplicity={mult:.12f}")
print("Q:")
for u, pp, mult in recQ:
    print(f"  surprisal={u: .12f}  p={pp:.12f}  multiplicity={mult:.12f}")
print()

print("[EXTERNAL + NUMERIC] PRODUCT-BLOCK COUPLING INGREDIENT")
print(f"V_P                                     = {VP:.15f}")
print(f"V_Q                                     = {VQ:.15f}")
print(f"A=(sqrt(VP)+sqrt(VQ))/2                 = {Aconst:.15f}")
print(f"c0=log2(e)/e                            = {c0:.15f}")
print("L   H(greedy J_L)      W1(product)        profile value       analytic upper")
for L, HJ, w1, profile_value, analytic_upper in block_rows:
    print(
        f"{L:<2d}  {HJ:>16.12f}   {w1:>16.12f}   "
        f"{profile_value:>16.12f}   {analytic_upper:>16.12f}"
    )
print()

print("[EXACT FINITE] ANTI-DIAGONAL ENUMERATION")
print(f"test m,L,n                              = {M_TEST},{L_TEST},{N_TEST}")
print(f"enumerated source atoms                 = {len(source)}")
print(f"max path-law absolute error             = {max_path_error:.3e}")
print(f"H(source), formula                      = {H_source_enum:.15f}, {H_source_formula:.15f}")
print()

print("[ASYMPT] CRASH BLOCK-RESOLUTION COEFFICIENT")
print(f"predicted Gaussian coefficient          = {c_crash:.15f}")
print("L      W1_L               W1_L/sqrt(L)       ratio to prediction")
for L, w1L, scaled, ratio in clt_rows:
    print(f"{L:<5d}  {w1L:>16.12f}   {scaled:>16.12f}   {ratio:>16.12f}")
print()

print("[EXACT] DYADIC PARITY LADDER")
print("R   common H       common V       |first unmatched moment|    W1 odd L=1,3,5")
for R, hh, vv, nxt, oddw in parity_rows:
    print(
        f"{R:<2d}  {hh:>10.6f}   {vv:>10.6f}   {nxt:>18.8f}   "
        + ", ".join(f"{x:.9f}" for x in oddw)
    )
print()

print("[NUMERIC + SYMBOLIC DEGREE] REAL MELLIN SPECTRUM")
print(f"max Fourier-vs-orbit error              = {max_fourier_error:.3e}")
print("k   integral w_k      max FT error       second moment       formula")
for k, norm, ferr, sec, secf in mellin_rows:
    if k == 0:
        print(f"{k:<2d}  {norm:>14.12f}   {ferr:>14.3e}   {'infinite':>16s}   {'infinite':>16s}")
    else:
        print(f"{k:<2d}  {norm:>14.12f}   {ferr:>14.3e}   {sec:>16.12f}   {secf:>16.12f}")
print()

print("[NUMERIC] RECIPROCAL-SCALE / LOCAL EDGE")
print(f"max A_k formula-vs-orbit error          = {max_A_error:.3e}")
print("smooth k   observed quadratic coeff     predicted coeff")
for k, obs, pred in smooth_local_rows:
    print(f"{k:<8d}   {obs:>20.12f}   {pred:>20.12f}")
print()

print("[ASYMPT] LARGE-k REAL LIMITS")
print("k   max H-ratio error   max R->sech error   max w->sech(pi x) error")
for row in large_k_rows:
    print(f"{row[0]:<3d} {row[1]:>18.10e} {row[2]:>20.10e} {row[3]:>24.10e}")
print()

print("[NUMERIC] ORBIT--LEDGER SPECTRAL PROBE")
print("k   D_eta             sigma*W1          ratio       sync increment cert.")
for k, D, sigma, rhs, ratio, cert in probe_rows:
    print(f"{k:<2d}  {D:>14.12f}   {rhs:>14.12f}   {ratio:>9.6f}   {cert:>18.12f}")
print()

print("[ASYMPT] ANALYTIC CAUSAL RATE UPPER BOUNDS")
for m, L, rate in rate_values:
    print(f"m={m:<3d} L={L:<5d} n={m*L:<7d} rate <= {rate:.12f}")
print()

print("CRASH-TEST LEDGER")
print("C_inf^causal                            = 1.500000000000000  [analytic theorem]")
print(f"C_inf^syn lower                         = {LB_sync:.15f}")
print(f"strict asymptotic gap >=                = {LB_sync - 1.5:.15f}")
print()
print("IMPORTANT:")
print("  ALL FINITE / SYMBOLIC CHECKS PASSED.")
print("  The asymptotic causal Shannon-floor theorem, CLT limits, spectral theorem,")
print("  profile coupling theorem, and related infinite-limit statements still require")
print("  the analytic proofs given/cited in the manuscript; this script audits their")
print("  hypotheses, identities, constants, finite constructions, and numerical behavior.")
print()
print("ALL CHECKS PASSED")
