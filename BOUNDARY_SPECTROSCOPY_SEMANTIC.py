#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOUNDARY_SPECTROSCOPY_SELECTIVE.py
=================================

Boundary Semantic Compiler / CEGAR prototype
--------------------------------------------

This is deliberately NOT a tree-search solver.

The old spectroscopy prototype sampled many concrete witness trees and tried
to recognize a pattern afterwards.  This rewrite follows the relation-calculus
strategy used elsewhere in DREAM6:

    exact local relation
        -> quotient to a small boundary interface
        -> compile transition relation on that interface
        -> detect when the quotient forgets information
        -> use an LP dual separator to add exactly the missing observable
        -> repeat until sampled relation closure stabilizes.

The local zero-repair step is represented directly as a transport relation

    pi(x,s) >= 0,
    sum_s pi(x,s) = P_b(x),
    sum_x pi(x,s) = (Q_b * B)(s),
    pi(x,s) = 0 for s < x.

A child law is not an independent LP object.  It is the residual row law

    B_x(j) = pi(x, x+j) / P_b(x).

Thus the internal realization space is quotiented before we do any recursive
reasoning.

Core coarse block
-----------------
    P_b = (1,12,38,12,1)/64
    Q_b = (0,16,32,16,0)/64

Exact primitive defect
----------------------
    P_b(t) - Q_b(t) = (1-t)^4 / 64.

At r=0 the interior terminal rule exposes exactly two illegal cells:
    (x,s) = (2,1) with mass 3/128,
    (x,s) = (4,3) with mass 1/128,
total boundary debt = 1/32.

What this program tries to discover
-----------------------------------
A minimal scale-stable boundary interface.  It starts with a tiny set of
linear observables on a buffer law B, compiles the exact one-step transport
relation through support functions, and automatically refines the interface
whenever two states that look identical through the current quotient have
different local relation geometry.

The refinement observable is not guessed.  It is pulled back from the
column-dual of the LP that witnessed the discrepancy.

This is a discovery/compiler prototype, not a proof of boundary closure or
O(log n).

Outputs
-------
<out>_atlas.json
    candidate semantic types, observables, transitions and refinement ledger

<out>_states.csv
    concrete states used to compile/refine the quotient

<out>_relations.csv
    support-function fingerprints of exact one-step relations

<out>_refinements.csv
    every CEGAR refinement and its dual-derived observable

Typical run
-----------
python BOUNDARY_SPECTROSCOPY_SELECTIVE.py ^
    --closure-rounds 4 ^
    --refine-rounds 8 ^
    --support 12 ^
    --anchor-k 1 ^
    --out boundary_semantic
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

try:
    import sympy as sp
except Exception:
    sp = None

# ---------------------------------------------------------------------------
# Exact problem constants
# ---------------------------------------------------------------------------

P_NUM = np.array([1, 12, 38, 12, 1], dtype=np.int64)
Q_NUM = np.array([0, 16, 32, 16, 0], dtype=np.int64)
DEN = 64

P = P_NUM.astype(float) / DEN
Q = Q_NUM.astype(float) / DEN

CRITICAL_BRANCHES = (0, 2, 4)

# Exact r=0 terminal debt.
DEBT_21 = Fraction(3, 128)
DEBT_43 = Fraction(1, 128)
DEBT_TOTAL = DEBT_21 + DEBT_43

STATUS_FEASIBLE = "FEASIBLE"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Probability-law utilities
# ---------------------------------------------------------------------------

def trim(a: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    j = len(a)
    while j > 1 and abs(a[j - 1]) <= tol:
        j -= 1
    return a[:j].copy()


def normalize(a: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    a[np.abs(a) <= tol] = 0.0
    if np.min(a) < -1e-9:
        raise ValueError("negative probability mass")
    a[a < 0] = 0.0
    s = float(np.sum(a))
    if s <= 0:
        raise ValueError("zero probability mass")
    return trim(a / s)


def conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.convolve(np.asarray(a, float), np.asarray(b, float))


def coeff(a: np.ndarray, j: int) -> float:
    return float(a[j]) if 0 <= j < len(a) else 0.0


def mean(a: np.ndarray) -> float:
    return float(np.dot(np.arange(len(a), dtype=float), np.asarray(a, float)))


def valuation(a: np.ndarray, tol: float = 1e-11) -> int:
    z = np.flatnonzero(np.asarray(a, float) > tol)
    return int(z[0]) if len(z) else 10**9


def top_support(a: np.ndarray, tol: float = 1e-11) -> int:
    z = np.flatnonzero(np.asarray(a, float) > tol)
    return int(z[-1]) if len(z) else 0


def pad(a: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=float)
    out[:min(n, len(a))] = a[:n]
    return out


def rational_hint(x: float, max_den: int = 65536) -> str:
    try:
        return str(Fraction(float(x)).limit_denominator(max_den))
    except Exception:
        return ""


def canonical_law_key(a: np.ndarray, digits: int = 11) -> tuple:
    a = trim(a)
    return tuple(np.round(a, digits))


def f_orbit(k: int) -> np.ndarray:
    f = np.array([0.0, 1.0])  # f0=t
    for _ in range(k):
        sq = conv(f, f)
        g = np.zeros(max(2, len(sq)), dtype=float)
        g[1] += 0.5
        g[:len(sq)] += 0.5 * sq
        f = normalize(g)
    return f


def G0(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    D = conv(A, C)
    g = np.zeros(max(1, len(D)), dtype=float)
    g[0] += 0.5
    g[:len(D)] += 0.5 * D
    return normalize(g)


# ---------------------------------------------------------------------------
# Sanity identities
# ---------------------------------------------------------------------------

def primitive_defect_coefficients() -> list[int]:
    # 64(P-Q) should be (1,-4,6,-4,1)
    return [int(x) for x in (P_NUM - Q_NUM)]


def exact_sanity() -> dict:
    return {
        "P_minus_Q_numerator": primitive_defect_coefficients(),
        "expected_quartic": [1, -4, 6, -4, 1],
        "quartic_identity_exact": primitive_defect_coefficients() == [1, -4, 6, -4, 1],
        "boundary_debt_21": str(DEBT_21),
        "boundary_debt_43": str(DEBT_43),
        "boundary_debt_total": str(DEBT_TOTAL),
    }


# ---------------------------------------------------------------------------
# Linear state observables
# ---------------------------------------------------------------------------

@dataclass
class Observable:
    name: str
    weights: np.ndarray
    origin: str

    def eval(self, B: np.ndarray) -> float:
        w = self.weights
        return float(np.dot(w, pad(B, len(w))))

    def normalized(self, tol: float = 1e-14) -> "Observable":
        w = np.asarray(self.weights, float).copy()
        w[np.abs(w) < tol] = 0.0
        m = float(np.max(np.abs(w))) if len(w) else 0.0
        if m > tol:
            w /= m
        # deterministic sign convention
        nz = np.flatnonzero(np.abs(w) > tol)
        if len(nz) and w[nz[0]] < 0:
            w *= -1
        return Observable(self.name, w, self.origin)


def q_prefix_parent_weights(k: int, support: int) -> np.ndarray:
    """
    Parent functional B -> F_{Q*B}(k).
    """
    w = np.zeros(support + 1, dtype=float)
    for j in range(support + 1):
        total = 0.0
        for y, q in enumerate(Q):
            if j + y <= k:
                total += q
        w[j] = total
    return w


def initial_observables(support: int) -> list[Observable]:
    """
    Start intentionally small.  If this quotient is insufficient, CEGAR must
    discover the missing coordinates from exact local-relation duals.
    """
    obs: list[Observable] = []

    e0 = np.zeros(support + 1); e0[0] = 1.0
    obs.append(Observable("B0", e0, "seed:hard_edge"))

    # The second nontrivial one-step Hall prefix.  k=1 is proportional to B0,
    # so k=3 is the first new low-edge combination.
    obs.append(Observable(
        "Qprefix3",
        q_prefix_parent_weights(3, support),
        "seed:critical_Hall_prefix"
    ).normalized())

    # Mean is affine bookkeeping and useful for rejecting spurious quotient merges.
    m = np.arange(support + 1, dtype=float)
    obs.append(Observable("mean", m, "seed:inventory_mean").normalized())

    return independent_observables(obs)


def independent_observables(obs: list[Observable], tol: float = 1e-10) -> list[Observable]:
    out: list[Observable] = []
    if not obs:
        return out
    for o in obs:
        o = o.normalized()
        w = o.weights
        if np.max(np.abs(w)) < tol:
            continue
        if not out:
            out.append(o)
            continue
        M = np.vstack([x.weights for x in out])
        rank0 = np.linalg.matrix_rank(M, tol)
        rank1 = np.linalg.matrix_rank(np.vstack([M, w]), tol)
        if rank1 > rank0:
            out.append(o)
    return out


def add_observable_if_new(obs: list[Observable], candidate: Observable,
                          tol: float = 1e-9) -> bool:
    candidate = candidate.normalized()
    if np.max(np.abs(candidate.weights)) < tol:
        return False
    if not obs:
        obs.append(candidate)
        return True
    M = np.vstack([o.weights for o in obs])
    rank0 = np.linalg.matrix_rank(M, tol)
    rank1 = np.linalg.matrix_rank(np.vstack([M, candidate.weights]), tol)
    if rank1 <= rank0:
        return False
    obs.append(candidate)
    return True


# ---------------------------------------------------------------------------
# Exact one-step transport relation
# ---------------------------------------------------------------------------

@dataclass
class TransportLP:
    parent: np.ndarray
    support: int
    cells: list[tuple[int, int]]
    cell_index: dict[tuple[int, int], int]
    Aeq: sparse.csr_matrix
    beq: np.ndarray
    qB: np.ndarray


@dataclass
class SolveResult:
    status: str
    value: Optional[float]
    pi: Optional[np.ndarray]
    children: Optional[list[np.ndarray]]
    parent_dual_weight: Optional[np.ndarray]
    raw: object


def build_transport_lp(B: np.ndarray, support: int) -> Optional[TransportLP]:
    """
    Variables are admissible coupling cells pi[x,s] with s>=x.

    Row marginals are P.
    Column marginals are Q*B.

    This is exactly the one-step zero-repair relation.
    """
    B = normalize(B)
    if top_support(B) > support:
        return None

    qB = conv(Q, pad(B, support + 1))
    max_s = len(qB) - 1

    cells: list[tuple[int, int]] = []
    for x in range(5):
        for s in range(max_s + 1):
            if s >= x:
                cells.append((x, s))
    idx = {c: i for i, c in enumerate(cells)}

    ri: list[int] = []
    ci: list[int] = []
    va: list[float] = []
    rhs: list[float] = []
    row = 0

    # Row marginals.
    for x in range(5):
        for s in range(max_s + 1):
            k = idx.get((x, s))
            if k is not None:
                ri.append(row); ci.append(k); va.append(1.0)
        rhs.append(float(P[x]))
        row += 1

    # Column marginals.
    for s in range(max_s + 1):
        for x in range(5):
            k = idx.get((x, s))
            if k is not None:
                ri.append(row); ci.append(k); va.append(1.0)
        rhs.append(float(qB[s]))
        row += 1

    Aeq = sparse.coo_matrix((va, (ri, ci)),
                            shape=(row, len(cells))).tocsr()

    return TransportLP(
        parent=B, support=support, cells=cells, cell_index=idx,
        Aeq=Aeq, beq=np.asarray(rhs, float), qB=qB
    )


def pi_matrix(lp: TransportLP, xvec: np.ndarray) -> np.ndarray:
    M = np.zeros((5, len(lp.qB)), dtype=float)
    for k, (x, s) in enumerate(lp.cells):
        M[x, s] = xvec[k]
    return M


def children_from_pi(lp: TransportLP, xvec: np.ndarray) -> list[np.ndarray]:
    M = pi_matrix(lp, xvec)
    out = []
    for x in range(5):
        # B_x(j) = pi(x,x+j)/P_x
        max_j = max(0, M.shape[1] - 1 - x)
        b = np.zeros(max_j + 1, dtype=float)
        for j in range(max_j + 1):
            b[j] = M[x, x + j] / P[x]
        out.append(normalize(b))
    return out


def parent_dual_from_column_multipliers(beta: np.ndarray,
                                        support: int) -> np.ndarray:
    """
    If the transport LP dual has column multipliers beta_s, then

       value(B) = const + <beta, Q*B>,

    so the pullback to the parent law is

       w_j = sum_y Q_y beta_{j+y}.

    This is the CEGAR refinement observable.
    """
    w = np.zeros(support + 1, dtype=float)
    for j in range(support + 1):
        z = 0.0
        for y, q in enumerate(Q):
            s = j + y
            if 0 <= s < len(beta):
                z += q * beta[s]
        w[j] = z
    return w


def solve_transport_direction(lp: TransportLP,
                              objective: np.ndarray,
                              maximize: bool = False) -> SolveResult:
    """
    Solve one support-function query of the local relation.

    Three-valued contract:
      FEASIBLE   : optimization solved
      INFEASIBLE : HiGHS explicitly reports infeasibility
      UNKNOWN    : timeout/degeneracy/other numerical status
    """
    c = np.asarray(objective, float)
    if maximize:
        c = -c

    res = linprog(
        c,
        A_eq=lp.Aeq,
        b_eq=lp.beq,
        bounds=[(0, None)] * len(lp.cells),
        method="highs"
    )

    if res.status == 2:
        return SolveResult(STATUS_INFEASIBLE, None, None, None, None, res)
    if not res.success:
        return SolveResult(STATUS_UNKNOWN, None, None, None, None, res)

    value = float(np.dot(objective, res.x))
    M = pi_matrix(lp, res.x)
    children = children_from_pi(lp, res.x)

    parent_w = None
    try:
        dual = np.asarray(res.eqlin.marginals, float)
        # First 5 equality rows are row marginals, remaining are columns.
        beta = dual[5:]
        if maximize:
            # We minimized -objective, so flip the dual back to the requested
            # support-function orientation.
            beta = -beta
        parent_w = parent_dual_from_column_multipliers(beta, lp.support)
    except Exception:
        parent_w = None

    return SolveResult(STATUS_FEASIBLE, value, M, children, parent_w, res)


# ---------------------------------------------------------------------------
# Relation-output coordinates
# ---------------------------------------------------------------------------

@dataclass
class RelationDirection:
    name: str
    objective: np.ndarray
    origin: str


def child_observable_objective(lp: TransportLP, branch: int,
                               observable: Observable) -> np.ndarray:
    """
    Objective pi -> observable(B_branch).
    """
    c = np.zeros(len(lp.cells), dtype=float)
    for k, (x, s) in enumerate(lp.cells):
        if x != branch:
            continue
        j = s - x
        if 0 <= j < len(observable.weights):
            c[k] = observable.weights[j] / P[branch]
    return c


def coupling_rectangle_objective(lp: TransportLP,
                                 x_max: int, s_min: int) -> np.ndarray:
    c = np.zeros(len(lp.cells), dtype=float)
    for k, (x, s) in enumerate(lp.cells):
        if x <= x_max and s >= s_min:
            c[k] = 1.0
    return c


def relation_directions(lp: TransportLP,
                        observables: list[Observable]) -> list[RelationDirection]:
    """
    Minimal boundary-semantic readout.

    We deliberately do NOT audit arbitrary child coefficients here.  The first
    semantic quotient is the part of the local transport phase that the known
    r=0 boundary can actually see:

        child H1  ~ F_{Q*B_x}(1)
        child H3  ~ F_{Q*B_x}(3)

    for x in {0,2,4}, plus the two donor capacities and their shared overlap.

    That gives nine raw boundary outputs.  Exact marginal-kernel reduction below
    determines how many are genuine phase coordinates.
    """
    dirs: list[RelationDirection] = []

    H1 = Observable(
        "H1",
        q_prefix_parent_weights(1, lp.support),
        "boundary_readout:Hall_prefix_1"
    )
    H3 = Observable(
        "H3",
        q_prefix_parent_weights(3, lp.support),
        "boundary_readout:Hall_prefix_3"
    )

    for x in CRITICAL_BRANCHES:
        for o in (H1, H3):
            dirs.append(RelationDirection(
                f"x{x}:{o.name}",
                child_observable_objective(lp, x, o),
                "boundary_child_Hall_readout"
            ))

    dirs.append(RelationDirection(
        "donor21",
        coupling_rectangle_objective(lp, x_max=1, s_min=2),
        "boundary_donor_rectangle"
    ))
    dirs.append(RelationDirection(
        "donor43",
        coupling_rectangle_objective(lp, x_max=3, s_min=4),
        "boundary_donor_rectangle"
    ))
    dirs.append(RelationDirection(
        "donor_shared",
        coupling_rectangle_objective(lp, x_max=1, s_min=4),
        "boundary_donor_overlap"
    ))

    return dirs



# ---------------------------------------------------------------------------
# Exact boundary-visible phase quotient
# ---------------------------------------------------------------------------

def marginal_kernel_matrix(support: int) -> tuple[list[tuple[int,int]], np.ndarray]:
    """
    Matrix M of row/column marginals on the admissible coupling cells.
    The internal crossing/phase space is ker M.
    """
    max_s = support + 4
    cells = [(x,s) for x in range(5) for s in range(max_s+1) if s >= x]
    M = np.zeros((5 + max_s + 1, len(cells)), dtype=float)
    for k,(x,s) in enumerate(cells):
        M[x,k] = 1.0
        M[5+s,k] = 1.0
    return cells, M


def boundary_readout_matrix(support: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Return names, R, M where R maps a coupling to the nine raw boundary outputs
    and M is the marginal map.
    """
    # Any parent with this support gives the same admissible cell ordering.
    dummy = np.zeros(support+1, dtype=float)
    dummy[min(1,support)] = 1.0
    lp = build_transport_lp(dummy, support)
    if lp is None:
        raise RuntimeError("failed to build boundary readout geometry")

    dirs = relation_directions(lp, [])
    names = [d.name for d in dirs]
    R = np.vstack([d.objective for d in dirs])

    cells, M = marginal_kernel_matrix(support)

    # build_transport_lp and marginal_kernel_matrix must use identical ordering.
    if cells != lp.cells:
        raise RuntimeError("coupling-cell ordering mismatch")

    return names, R, M


def phase_invariant_direction(objective: np.ndarray,
                              support: int,
                              tol: float = 1e-10) -> bool:
    """
    A coupling readout is phase-invariant iff it lies in the row span of the
    marginal map, equivalently iff it vanishes on ker M.
    """
    _, M = marginal_kernel_matrix(support)
    r0 = np.linalg.matrix_rank(M, tol)
    r1 = np.linalg.matrix_rank(np.vstack([M, objective]), tol)
    return bool(r1 == r0)


def _primitive_integer_vector(values) -> list[int]:
    if sp is None:
        return []
    vals = [sp.Rational(v) for v in values]
    L = 1
    for v in vals:
        L = sp.ilcm(L, int(v.q))
    ints = [int(v*L) for v in vals]
    g = 0
    for z in ints:
        g = math.gcd(g, abs(z))
    if g:
        ints = [z//g for z in ints]
    for z in ints:
        if z:
            if z < 0:
                ints = [-q for q in ints]
            break
    return ints


def exact_boundary_phase_quotient(support: int) -> dict:
    """
    Exact rational reduction of the nine raw boundary outputs modulo the
    row/column marginal map.

    phase_rank = rank([M;R]) - rank(M)

    A greedy basis of rows of R gives explicit phase coordinates.
    The left dependency basis records boundary-output combinations that are
    already fixed by marginals and therefore are not latent phase.
    """
    names, Rf, Mf = boundary_readout_matrix(support)

    if sp is None:
        rM = int(np.linalg.matrix_rank(Mf))
        rMR = int(np.linalg.matrix_rank(np.vstack([Mf,Rf])))
        basis_names = []
        cur = Mf.copy()
        rank = rM
        for i,name in enumerate(names):
            t = np.vstack([cur,Rf[i]])
            r = int(np.linalg.matrix_rank(t))
            if r > rank:
                basis_names.append(name)
                cur = t
                rank = r
        return {
            "status": "NUMERICAL_FALLBACK",
            "support": support,
            "raw_outputs": names,
            "marginal_rank": rM,
            "phase_rank": rMR-rM,
            "phase_basis": basis_names,
            "dependencies": [],
        }

    # Rebuild exactly with Rational entries.
    cells = [(x,s) for x in range(5) for s in range(support+5) if s>=x]
    n = len(cells)

    M = sp.zeros(5 + support + 5, n)
    for k,(x,s) in enumerate(cells):
        M[x,k] = 1
        M[5+s,k] = 1

    Pr = [sp.Rational(int(v), DEN) for v in P_NUM]
    Qr = [sp.Rational(int(v), DEN) for v in Q_NUM]

    rows = []
    exact_names = []

    for x in CRITICAL_BRANCHES:
        for kp in (1,3):
            w = []
            for j in range(support+1):
                w.append(sum(Qr[y] for y in range(5) if j+y <= kp))
            rr = [sp.Rational(0) for _ in range(n)]
            for k,(xx,scol) in enumerate(cells):
                if xx == x:
                    j = scol-x
                    if 0 <= j < len(w):
                        rr[k] = w[j] / Pr[x]
            rows.append(rr)
            exact_names.append(f"x{x}:H{kp}")

    for name,xmax,smin in (
        ("donor21",1,2),
        ("donor43",3,4),
        ("donor_shared",1,4)
    ):
        rr = [
            sp.Rational(1) if x<=xmax and scol>=smin else sp.Rational(0)
            for x,scol in cells
        ]
        rows.append(rr)
        exact_names.append(name)

    R = sp.Matrix(rows)
    rM = int(M.rank())
    rMR = int(M.col_join(R).rank())
    phase_rank = rMR-rM

    current = M.copy()
    rank = rM
    basis_names = []
    basis_indices = []
    for i,name in enumerate(exact_names):
        test = current.col_join(R[i,:])
        rr = int(test.rank())
        if rr > rank:
            basis_names.append(name)
            basis_indices.append(i)
            current = test
            rank = rr

    # lambda^T R is phase-invariant iff lambda^T R lies in rowspace(M).
    # Solve R^T lambda - M^T mu = 0.
    A = R.T.row_join(-M.T)
    raw_ns = A.nullspace()

    lambda_columns = []
    for v in raw_ns:
        lam = v[:len(exact_names),:]
        if any(z != 0 for z in lam):
            lambda_columns.append(lam)

    dependency_vectors = []
    if lambda_columns:
        L = sp.Matrix.hstack(*lambda_columns)
        for col in L.columnspace():
            ints = _primitive_integer_vector(list(col))
            if ints and any(ints):
                dependency_vectors.append(ints)

    dependencies = []
    for ints in dependency_vectors:
        terms = []
        for c,name in zip(ints, exact_names):
            if c:
                terms.append({"coefficient": int(c), "output": name})
        dependencies.append(terms)

    return {
        "status": "EXACT_RATIONAL",
        "support": support,
        "raw_outputs": exact_names,
        "marginal_rank": rM,
        "phase_rank": phase_rank,
        "phase_basis": basis_names,
        "phase_basis_indices": basis_indices,
        "dependencies": dependencies,
    }


def phase_rank_scan(max_support: int) -> list[dict]:
    out = []
    for s in range(2, max(2,max_support)+1):
        q = exact_boundary_phase_quotient(s)
        out.append({
            "support": s,
            "phase_rank": q["phase_rank"],
            "phase_basis": q["phase_basis"],
            "status": q["status"],
        })
    return out


# ---------------------------------------------------------------------------
# Parent interface / fingerprints
# ---------------------------------------------------------------------------

def parent_signature(B: np.ndarray,
                     observables: list[Observable]) -> np.ndarray:
    return np.array([o.eval(B) for o in observables], dtype=float)


def quantized_signature(sig: np.ndarray, tol: float) -> tuple[int, ...]:
    return tuple(int(np.rint(x / tol)) for x in sig)


@dataclass
class FingerprintEntry:
    direction: str
    sense: str
    status: str
    value: Optional[float]
    parent_dual_weight: Optional[np.ndarray]
    witness_children: Optional[list[np.ndarray]]
    witness_pi: Optional[np.ndarray]


def compile_relation_fingerprint(B: np.ndarray,
                                 observables: list[Observable],
                                 support: int) -> list[FingerprintEntry]:
    lp = build_transport_lp(B, support)
    if lp is None:
        return [FingerprintEntry(
            direction="support",
            sense="n/a",
            status=STATUS_UNKNOWN,
            value=None,
            parent_dual_weight=None,
            witness_children=None,
            witness_pi=None
        )]

    dirs = relation_directions(lp, observables)
    out: list[FingerprintEntry] = []

    for d in dirs:
        for sense, maximize in (("min", False), ("max", True)):
            sol = solve_transport_direction(lp, d.objective, maximize=maximize)
            out.append(FingerprintEntry(
                direction=d.name,
                sense=sense,
                status=sol.status,
                value=sol.value,
                parent_dual_weight=sol.parent_dual_weight,
                witness_children=sol.children,
                witness_pi=sol.pi
            ))
    return out


def fingerprint_value_map(fp: list[FingerprintEntry]) -> dict[tuple[str, str], FingerprintEntry]:
    return {(e.direction, e.sense): e for e in fp}


def fingerprint_distance(a: list[FingerprintEntry],
                         b: list[FingerprintEntry]) -> tuple[float, Optional[tuple[str, str]]]:
    A = fingerprint_value_map(a)
    B = fingerprint_value_map(b)
    best = 0.0
    key_best = None

    keys = sorted(set(A) & set(B))
    for k in keys:
        x, y = A[k], B[k]
        if x.status != y.status:
            return float("inf"), k
        if x.status != STATUS_FEASIBLE:
            continue
        if x.value is None or y.value is None:
            continue
        d = abs(x.value - y.value)
        if d > best:
            best = d
            key_best = k
    return best, key_best



# ---------------------------------------------------------------------------
# Interface-fiber separation
# ---------------------------------------------------------------------------

@dataclass
class FiberSolve:
    status: str
    value: Optional[float]
    parent: Optional[np.ndarray]
    pi: Optional[np.ndarray]
    raw: object


def solve_interface_fiber(reference: np.ndarray,
                          observables: list[Observable],
                          support: int,
                          relation_objective: np.ndarray,
                          maximize: bool = False) -> FiberSolve:
    """
    Optimize one local-relation support direction over *all parent laws B*
    that share the current interface signature of `reference`.

    This is the actual quotient test:

        same visible interface
            but
        different exact one-step relation geometry?

    If max-min has positive width, the current interface is insufficient.

    Variables:
        B_0,...,B_support
        admissible pi(x,s), s>=x

    Constraints:
        B is a probability law
        <phi_i,B> = <phi_i,reference> for every current observable
        row(pi)=P
        col(pi)=Q*B

    No tree is built.
    """
    reference = normalize(reference)
    if top_support(reference) > support:
        return FiberSolve(STATUS_UNKNOWN, None, None, None, None)

    # Build a fixed cell ordering using a padded reference; the ordering depends
    # only on support, not on the actual parent values.
    base_lp = build_transport_lp(reference, support)
    if base_lp is None:
        return FiberSolve(STATUS_UNKNOWN, None, None, None, None)

    cells = base_lp.cells
    nB = support + 1
    nPi = len(cells)
    nvar = nB + nPi

    ri: list[int] = []
    ci: list[int] = []
    va: list[float] = []
    rhs: list[float] = []
    row = 0

    # Parent normalization.
    for j in range(nB):
        ri.append(row); ci.append(j); va.append(1.0)
    rhs.append(1.0)
    row += 1

    # Current interface signature.
    for o in observables:
        sig = o.eval(reference)
        for j in range(nB):
            wj = float(o.weights[j]) if j < len(o.weights) else 0.0
            if abs(wj) > 0:
                ri.append(row); ci.append(j); va.append(wj)
        rhs.append(sig)
        row += 1

    # Row marginals of pi.
    for x in range(5):
        for k, (xx, ss) in enumerate(cells):
            if xx == x:
                ri.append(row); ci.append(nB + k); va.append(1.0)
        rhs.append(float(P[x]))
        row += 1

    # Column relation:
    #   sum_x pi(x,s) - (Q*B)(s) = 0
    max_s = len(base_lp.qB) - 1
    for ss in range(max_s + 1):
        for k, (x, s_col) in enumerate(cells):
            if s_col == ss:
                ri.append(row); ci.append(nB + k); va.append(1.0)

        for j in range(nB):
            qsum = 0.0
            for y, q in enumerate(Q):
                if j + y == ss:
                    qsum += float(q)
            if abs(qsum) > 0:
                ri.append(row); ci.append(j); va.append(-qsum)

        rhs.append(0.0)
        row += 1

    Aeq = sparse.coo_matrix(
        (va, (ri, ci)), shape=(row, nvar)
    ).tocsr()

    c = np.zeros(nvar, dtype=float)
    c[nB:] = np.asarray(relation_objective, float)
    if maximize:
        c = -c

    res = linprog(
        c,
        A_eq=Aeq,
        b_eq=np.asarray(rhs, float),
        bounds=[(0, None)] * nvar,
        method="highs"
    )

    if res.status == 2:
        return FiberSolve(STATUS_INFEASIBLE, None, None, None, res)
    if not res.success:
        return FiberSolve(STATUS_UNKNOWN, None, None, None, res)

    value = float(np.dot(relation_objective, res.x[nB:]))
    parent = normalize(res.x[:nB])
    M = pi_matrix(base_lp, res.x[nB:])

    return FiberSolve(
        STATUS_FEASIBLE, value, parent, M, res
    )


def semantic_fiber_width(reference: np.ndarray,
                         observables: list[Observable],
                         support: int,
                         direction: RelationDirection) -> tuple[
                             Optional[float], Optional[FiberSolve], Optional[FiberSolve]
                         ]:
    lo = solve_interface_fiber(
        reference, observables, support,
        direction.objective, maximize=False
    )
    hi = solve_interface_fiber(
        reference, observables, support,
        direction.objective, maximize=True
    )

    if lo.status != STATUS_FEASIBLE or hi.status != STATUS_FEASIBLE:
        return None, lo, hi

    return float(hi.value - lo.value), lo, hi


# ---------------------------------------------------------------------------
# Concrete relation closure
# ---------------------------------------------------------------------------

@dataclass
class ConcreteState:
    state_id: int
    law: np.ndarray
    generation: int
    origin: str
    parent_id: Optional[int]
    branch: Optional[int]


def exposed_child_states(B: np.ndarray,
                         observables: list[Observable],
                         support: int) -> list[tuple[np.ndarray, str, int]]:
    """
    Deterministic relation closure: expose the local polytope only in semantic
    directions.  No random objectives.
    """
    lp = build_transport_lp(B, support)
    if lp is None:
        return []

    dirs = relation_directions(lp, observables)
    out: list[tuple[np.ndarray, str, int]] = []
    seen = set()

    for d in dirs:
        # Donor-only directions do not select one preferred child, but their
        # witness still carries critical branch laws.  Keep them too.
        for sense, maximize in (("min", False), ("max", True)):
            sol = solve_transport_direction(lp, d.objective, maximize=maximize)
            if sol.status != STATUS_FEASIBLE or sol.children is None:
                continue
            for x in CRITICAL_BRANCHES:
                child = sol.children[x]
                key = canonical_law_key(child)
                if key in seen:
                    continue
                seen.add(key)
                out.append((child, f"{d.name}:{sense}", x))
    return out


def build_concrete_closure(anchor_k: int,
                           observables: list[Observable],
                           support: int,
                           closure_rounds: int,
                           max_states: int,
                           include_boundary_constructor: bool = True) -> list[ConcreteState]:
    """
    Compile a finite concrete basis for abstraction refinement.

    Seeds:
      f0, ..., f_{anchor_k+1}

    Closure:
      exposed critical children of the exact one-step relation
      plus G0(child, f_anchor) boundary states.

    This is relation closure, not recursive tree search.
    """
    states: list[ConcreteState] = []
    by_key: dict[tuple, int] = {}
    next_id = 0

    def add(B, generation, origin, parent_id=None, branch=None) -> Optional[int]:
        nonlocal next_id
        B = normalize(B)
        if top_support(B) > support:
            return None
        key = canonical_law_key(B)
        if key in by_key:
            return by_key[key]
        sid = next_id; next_id += 1
        by_key[key] = sid
        states.append(ConcreteState(
            sid, B, generation, origin, parent_id, branch
        ))
        return sid

    for k in range(anchor_k + 2):
        add(f_orbit(k), 0, f"seed:f{k}")

    anchor = f_orbit(anchor_k)

    frontier = list(states)
    for r in range(1, closure_rounds + 1):
        if len(states) >= max_states:
            break
        new_ids = []
        # Snapshot current frontier; do not recursively expand new states in same round.
        current = list(frontier)
        frontier = []

        for st in current:
            if len(states) >= max_states:
                break
            children = exposed_child_states(st.law, observables, support)
            for child, why, branch in children:
                sid = add(
                    child, r,
                    f"relation:{why}",
                    parent_id=st.state_id,
                    branch=branch
                )
                if sid is not None and sid >= len(states) - 1:
                    new_ids.append(sid)

                if include_boundary_constructor:
                    gb = G0(child, anchor)
                    sid2 = add(
                        gb, r,
                        f"boundary:G0({why},f{anchor_k})",
                        parent_id=st.state_id,
                        branch=branch
                    )
                    if sid2 is not None and sid2 >= len(states) - 1:
                        new_ids.append(sid2)

                if len(states) >= max_states:
                    break
            if len(states) >= max_states:
                break

        # Deduplicate frontier ids preserving order.
        seen = set()
        for sid in new_ids:
            if sid in seen or sid >= len(states):
                continue
            seen.add(sid)
            frontier.append(states[sid])

        if not frontier:
            break

    return states


# ---------------------------------------------------------------------------
# CEGAR: refine interface when local relation differs inside one quotient type
# ---------------------------------------------------------------------------

@dataclass
class Refinement:
    round: int
    reference_state: int
    relation_direction: str
    fiber_min: float
    fiber_max: float
    fiber_width: float
    observable_name: str
    observable_weights: np.ndarray
    origin: str
    witness_parent_min: np.ndarray
    witness_parent_max: np.ndarray


def cegar_refine(states: list[ConcreteState],
                 observables: list[Observable],
                 support: int,
                 refine_rounds: int,
                 relation_tol: float,
                 reference_limit: int = 24) -> tuple[
                     list[Observable], list[Refinement],
                     dict[int, list[FingerprintEntry]]
                 ]:
    """
    Counterexample-guided abstraction refinement on the *interface fiber*.

    This no longer waits for two sampled states to accidentally share the same
    rounded signature.

    For a concrete reference state B we freeze only its current interface
    coordinates and allow the parent law itself to vary over that fiber.
    We then ask whether one exact local-relation support direction has nonzero
    width on the fiber.

    If yes, the quotient forgot information.

    The missing observable is obtained from the fixed-parent transport dual at
    one of the extremal witness parents.  Thus refinement is driven by the exact
    local relation, not by an arbitrary feature list.
    """
    refinements: list[Refinement] = []
    fingerprints: dict[int, list[FingerprintEntry]] = {}

    # Deterministic reference order: orbit seeds, boundary states, then others.
    ordered = sorted(
        states,
        key=lambda st: (
            0 if st.origin.startswith("seed:") else
            1 if st.origin.startswith("boundary:") else 2,
            st.generation,
            st.state_id
        )
    )

    for rr in range(refine_rounds):
        found = None

        refs = ordered[:max(1, min(reference_limit, len(ordered)))]

        for st in refs:
            base_lp = build_transport_lp(st.law, support)
            if base_lp is None:
                continue

            dirs = [
                d for d in relation_directions(base_lp, observables)
                if phase_invariant_direction(d.objective, support)
            ]

            for d in dirs:
                width, lo, hi = semantic_fiber_width(
                    st.law, observables, support, d
                )
                if width is None or lo is None or hi is None:
                    continue
                if width <= relation_tol:
                    continue
                if lo.parent is None or hi.parent is None:
                    continue

                found = (st, d, width, lo, hi)
                break
            if found is not None:
                break

        if found is None:
            break

        st, d, width, lo, hi = found

        # Pull back a local support hyperplane from one of the two extremal
        # parent witnesses.  Try both and keep the one that separates the
        # extremal parents more strongly.
        candidates = []

        for label, parent, maximize in (
            ("min", lo.parent, False),
            ("max", hi.parent, True)
        ):
            lp = build_transport_lp(parent, support)
            if lp is None:
                continue

            # Cell ordering is support-determined, so d.objective is compatible.
            sol = solve_transport_direction(
                lp, d.objective, maximize=maximize
            )
            if sol.status != STATUS_FEASIBLE:
                continue
            if sol.parent_dual_weight is None:
                continue

            w = np.asarray(sol.parent_dual_weight, float)
            sep = abs(
                float(np.dot(w, pad(lo.parent, len(w)))) -
                float(np.dot(w, pad(hi.parent, len(w))))
            )
            candidates.append((sep, label, w))

        if candidates:
            candidates.sort(key=lambda z: z[0], reverse=True)
            _, label, candidate_w = candidates[0]
            origin = (
                f"CEGAR:fiber_transport_dual:"
                f"{d.name}:extreme_{label}"
            )
        else:
            # Explicitly marked fallback.  This keeps the compiler moving if
            # HiGHS does not expose a stable equality dual.
            candidate_w = pad(hi.parent, support + 1) - pad(lo.parent, support + 1)
            origin = f"CEGAR:fiber_extreme_difference:{d.name}"

        name = f"phi{len(observables)}_{d.name.replace(':','_')}"
        candidate = Observable(name, candidate_w, origin).normalized()

        added = add_observable_if_new(observables, candidate)
        if not added:
            # The witnessed ambiguity is not removable by a new independent
            # linear parent coordinate from this separator.  Do not fabricate
            # progress.
            break

        refinements.append(Refinement(
            round=rr,
            reference_state=st.state_id,
            relation_direction=d.name,
            fiber_min=float(lo.value),
            fiber_max=float(hi.value),
            fiber_width=float(width),
            observable_name=candidate.name,
            observable_weights=candidate.weights.copy(),
            origin=origin,
            witness_parent_min=lo.parent.copy(),
            witness_parent_max=hi.parent.copy()
        ))

        # New observables change relation directions and parent signatures.
        fingerprints = {}

    # Final fixed-parent relation fingerprints for the compiled concrete basis.
    for st in states:
        fingerprints[st.state_id] = compile_relation_fingerprint(
            st.law, observables, support
        )

    return observables, refinements, fingerprints


# ---------------------------------------------------------------------------
# Compile candidate semantic atlas
# ---------------------------------------------------------------------------

def compile_types(states: list[ConcreteState],
                  observables: list[Observable],
                  fingerprints: dict[int, list[FingerprintEntry]],
                  signature_tol: float) -> tuple[list[dict], dict[int, int]]:
    groups: dict[tuple[int, ...], list[ConcreteState]] = {}
    for st in states:
        sig = parent_signature(st.law, observables)
        groups.setdefault(quantized_signature(sig, signature_tol), []).append(st)

    types = []
    state_to_type = {}

    for tid, (key, members) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        for st in members:
            state_to_type[st.state_id] = tid

        representative = members[0]
        sig = parent_signature(representative.law, observables)
        fp = fingerprints[representative.state_id]

        fpd = {}
        for e in fp:
            fpd[f"{e.direction}:{e.sense}"] = {
                "status": e.status,
                "value": e.value,
                "value_rational_hint": rational_hint(e.value) if e.value is not None else None,
            }

        types.append({
            "type_id": tid,
            "member_state_ids": [st.state_id for st in members],
            "member_count": len(members),
            "signature": [float(x) for x in sig],
            "signature_rational_hints": [rational_hint(x) for x in sig],
            "representative_state": representative.state_id,
            "representative_origin": representative.origin,
            "relation_support": fpd,
        })

    return types, state_to_type


def compile_exposed_transitions(states: list[ConcreteState],
                                observables: list[Observable],
                                fingerprints: dict[int, list[FingerprintEntry]],
                                state_to_type: dict[int, int],
                                support: int,
                                signature_tol: float) -> list[dict]:
    """
    Record only transitions witnessed by semantic support directions.
    This is a candidate relation atlas, not an exhaustive theorem.
    """
    sig_to_type = {}
    for st in states:
        sig = quantized_signature(parent_signature(st.law, observables), signature_tol)
        sig_to_type[sig] = state_to_type[st.state_id]

    transitions = []
    seen = set()

    for st in states:
        from_type = state_to_type[st.state_id]
        for e in fingerprints[st.state_id]:
            if e.status != STATUS_FEASIBLE or e.witness_children is None:
                continue
            for x in CRITICAL_BRANCHES:
                child = e.witness_children[x]
                sig = quantized_signature(parent_signature(child, observables), signature_tol)
                to_type = sig_to_type.get(sig)
                if to_type is None:
                    continue
                key = (from_type, x, to_type, e.direction, e.sense)
                if key in seen:
                    continue
                seen.add(key)
                transitions.append({
                    "from_type": from_type,
                    "branch": x,
                    "to_type": to_type,
                    "exposed_by": e.direction,
                    "sense": e.sense,
                })

    return transitions


# ---------------------------------------------------------------------------
# Boundary donor diagnostics
# ---------------------------------------------------------------------------

def donor_status_from_fingerprint(fp: list[FingerprintEntry]) -> dict:
    m = fingerprint_value_map(fp)

    def val(name, sense):
        e = m.get((name, sense))
        return e.value if e is not None and e.status == STATUS_FEASIBLE else None

    d21_min = val("donor21", "min")
    d21_max = val("donor21", "max")
    d43_min = val("donor43", "min")
    d43_max = val("donor43", "max")
    shared_min = val("donor_shared", "min")
    shared_max = val("donor_shared", "max")

    return {
        "donor21_min": d21_min,
        "donor21_max": d21_max,
        "donor43_min": d43_min,
        "donor43_max": d43_max,
        "donor_shared_min": shared_min,
        "donor_shared_max": shared_max,
        "debt21": float(DEBT_21),
        "debt43": float(DEBT_43),
        "debt_total": float(DEBT_TOTAL),
        # These are only necessary capacity screens, not recursive closure tests.
        "can_cover_21_by_max_capacity": None if d21_max is None else bool(d21_max + 1e-10 >= float(DEBT_21)),
        "can_cover_43_by_max_capacity": None if d43_max is None else bool(d43_max + 1e-10 >= float(DEBT_43)),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_states_csv(path: str, states: list[ConcreteState],
                     observables: list[Observable],
                     fingerprints: dict[int, list[FingerprintEntry]],
                     state_to_type: dict[int, int]):
    fields = [
        "state_id", "type_id", "generation", "origin", "parent_id", "branch",
        "valuation", "top_support", "mean"
    ]
    fields += [o.name for o in observables]
    fields += [f"coef{j}" for j in range(16)]
    fields += [
        "donor21_min", "donor21_max",
        "donor43_min", "donor43_max",
        "donor_shared_min", "donor_shared_max",
        "debt21", "debt43", "debt_total",
        "can_cover_21_by_max_capacity", "can_cover_43_by_max_capacity"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for st in states:
            row = {
                "state_id": st.state_id,
                "type_id": state_to_type[st.state_id],
                "generation": st.generation,
                "origin": st.origin,
                "parent_id": st.parent_id,
                "branch": st.branch,
                "valuation": valuation(st.law),
                "top_support": top_support(st.law),
                "mean": mean(st.law),
            }
            for o in observables:
                row[o.name] = o.eval(st.law)
            for j in range(16):
                row[f"coef{j}"] = coeff(st.law, j)
            row.update(donor_status_from_fingerprint(fingerprints[st.state_id]))
            w.writerow(row)


def write_relations_csv(path: str,
                        states: list[ConcreteState],
                        fingerprints: dict[int, list[FingerprintEntry]],
                        state_to_type: dict[int, int]):
    fields = [
        "state_id", "type_id", "direction", "sense",
        "status", "value", "value_rational_hint"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for st in states:
            for e in fingerprints[st.state_id]:
                w.writerow({
                    "state_id": st.state_id,
                    "type_id": state_to_type[st.state_id],
                    "direction": e.direction,
                    "sense": e.sense,
                    "status": e.status,
                    "value": e.value,
                    "value_rational_hint": rational_hint(e.value) if e.value is not None else "",
                })


def write_refinements_csv(path: str, refs: list[Refinement]):
    fields = [
        "round", "reference_state",
        "relation_direction",
        "fiber_min", "fiber_max", "fiber_width",
        "observable_name", "origin",
        "weights", "weights_rational_hints",
        "witness_parent_min", "witness_parent_max"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in refs:
            w.writerow({
                "round": r.round,
                "reference_state": r.reference_state,
                "relation_direction": r.relation_direction,
                "fiber_min": r.fiber_min,
                "fiber_max": r.fiber_max,
                "fiber_width": r.fiber_width,
                "observable_name": r.observable_name,
                "origin": r.origin,
                "weights": json.dumps([float(x) for x in r.observable_weights]),
                "weights_rational_hints": json.dumps(
                    [rational_hint(x, 4096) for x in r.observable_weights]
                ),
                "witness_parent_min": json.dumps(
                    [float(x) for x in r.witness_parent_min]
                ),
                "witness_parent_max": json.dumps(
                    [float(x) for x in r.witness_parent_max]
                ),
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-k", type=int, default=1)
    ap.add_argument("--support", type=int, default=12)
    ap.add_argument("--closure-rounds", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=256)
    ap.add_argument("--refine-rounds", type=int, default=8)
    ap.add_argument("--fiber-reference-limit", type=int, default=24)
    ap.add_argument("--signature-tol", type=float, default=1e-8)
    ap.add_argument("--relation-tol", type=float, default=1e-7)
    ap.add_argument("--no-boundary-constructor", action="store_true")
    ap.add_argument("--out", type=str, default="boundary_semantic")
    args = ap.parse_args()

    print("=== BOUNDARY SEMANTIC COMPILER ===")
    print(f"support / anchor-k     : {args.support} / {args.anchor_k}")
    print(f"closure rounds         : {args.closure_rounds}")
    print(f"refinement rounds      : {args.refine_rounds}")
    print("tree search             : NONE")
    print("random objectives       : NONE")
    print("local object            : exact transport relation")
    print("quotient refinement     : LP column-dual pullback")
    print("boundary debt           : 3/128 + 1/128 = 1/32")
    print("=" * 88)

    sanity = exact_sanity()
    if not sanity["quartic_identity_exact"]:
        raise RuntimeError("primitive defect sanity failed")

    phase_quotient = exact_boundary_phase_quotient(args.support)
    rank_scan = phase_rank_scan(args.support)
    stable_ranks = sorted(set(int(x["phase_rank"]) for x in rank_scan))

    print("[exact phase quotient]")
    print(f"  raw boundary outputs  : {len(phase_quotient['raw_outputs'])}")
    print(f"  latent phase rank     : {phase_quotient['phase_rank']}")
    print(f"  phase basis           : {phase_quotient['phase_basis']}")
    print(f"  support-rank scan     : {[x['phase_rank'] for x in rank_scan]}")
    print(f"  stable rank           : {len(stable_ranks)==1}")

    observables = initial_observables(args.support)

    # First concrete closure under the intentionally small interface.
    states = build_concrete_closure(
        anchor_k=args.anchor_k,
        observables=observables,
        support=args.support,
        closure_rounds=args.closure_rounds,
        max_states=args.max_states,
        include_boundary_constructor=not args.no_boundary_constructor
    )

    print(f"[compile] concrete relation states : {len(states)}")
    print(f"[compile] initial observables      : {[o.name for o in observables]}")

    observables, refinements, fingerprints = cegar_refine(
        states=states,
        observables=observables,
        support=args.support,
        refine_rounds=args.refine_rounds,
        relation_tol=args.relation_tol,
        reference_limit=args.fiber_reference_limit
    )

    print(f"[parent CEGAR] refinements        : {len(refinements)}")
    for r in refinements:
        print(
            f"  round={r.round} ref={r.reference_state} "
            f"via {r.relation_direction} "
            f"fiber=[{r.fiber_min:.6g},{r.fiber_max:.6g}] "
            f"width={r.fiber_width:.3e} -> {r.observable_name}"
        )

    print(f"[parent CEGAR] final observables   : {[o.name for o in observables]}")

    types, state_to_type = compile_types(
        states, observables, fingerprints, args.signature_tol
    )
    transitions = compile_exposed_transitions(
        states, observables, fingerprints, state_to_type,
        args.support, args.signature_tol
    )

    print(f"[atlas] candidate semantic types  : {len(types)}")
    print(f"[atlas] exposed transitions       : {len(transitions)}")

    # Status ledger.
    status_counts = {STATUS_FEASIBLE: 0, STATUS_INFEASIBLE: 0, STATUS_UNKNOWN: 0}
    for fp in fingerprints.values():
        for e in fp:
            status_counts[e.status] = status_counts.get(e.status, 0) + 1

    observables_json = []
    for o in observables:
        observables_json.append({
            "name": o.name,
            "origin": o.origin,
            "weights": [float(x) for x in o.weights],
            "weights_rational_hints": [rational_hint(x, 4096) for x in o.weights],
        })

    refinements_json = []
    for r in refinements:
        refinements_json.append({
            "round": r.round,
            "reference_state": r.reference_state,
            "relation_direction": r.relation_direction,
            "fiber_min": r.fiber_min,
            "fiber_max": r.fiber_max,
            "fiber_width": r.fiber_width,
            "observable_name": r.observable_name,
            "origin": r.origin,
            "weights": [float(x) for x in r.observable_weights],
            "weights_rational_hints": [rational_hint(x, 4096) for x in r.observable_weights],
            "witness_parent_min": [float(x) for x in r.witness_parent_min],
            "witness_parent_max": [float(x) for x in r.witness_parent_max],
        })

    atlas = {
        "kind": "boundary_semantic_relation_compiler",
        "status": "DISCOVERY_ONLY",
        "problem": {
            "P_num": P_NUM.tolist(),
            "Q_num": Q_NUM.tolist(),
            "denominator": DEN,
            "primitive_defect": "(1-t)^4/64",
            "boundary_illegal_cells": {
                "(2,1)": str(DEBT_21),
                "(4,3)": str(DEBT_43),
                "total": str(DEBT_TOTAL),
            },
        },
        "contract": {
            "tree_search": False,
            "random_objectives": False,
            "local_relation": "nonnegative transport with exact row/column marginals and s>=x",
            "abstraction": "exact marginal quotient -> boundary-visible latent phase + linear parent interface",
            "refinement": "exact phase quotient first; CEGAR only on phase-invariant parent-side relation data",
            "proof_claim": False,
        },
        "args": vars(args),
        "sanity": sanity,
        "boundary_phase_quotient": phase_quotient,
        "phase_rank_scan": rank_scan,
        "status_counts": status_counts,
        "observables": observables_json,
        "refinements": refinements_json,
        "types": types,
        "transitions": transitions,
        "limitations": [
            "The exact phase rank concerns the stated nine-dimensional local boundary readout only.",
            "The concrete closure is finite and support-truncated.",
            "The semantic types are candidate quotient classes on the sampled closure, not a theorem.",
            "HiGHS duals are floating-point discovery data; rational hints require separate certification.",
            "Donor rectangle capacity is only a local boundary screen, not recursive viability.",
            "A stable atlas suggests a lemma; it does not prove boundary closure."
        ],
    }

    out = args.out
    Path(out + "_atlas.json").write_text(
        json.dumps(atlas, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    write_states_csv(out + "_states.csv", states, observables, fingerprints, state_to_type)
    write_relations_csv(out + "_relations.csv", states, fingerprints, state_to_type)
    write_refinements_csv(out + "_refinements.csv", refinements)

    print("\n=== STATUS ===")
    print(json.dumps({
        "states": len(states),
        "types": len(types),
        "observables": len(observables),
        "refinements": len(refinements),
        "transitions": len(transitions),
        "LP_status": status_counts,
    }, indent=2))

    print("\nWrote:")
    print(" ", out + "_atlas.json")
    print(" ", out + "_states.csv")
    print(" ", out + "_relations.csv")
    print(" ", out + "_refinements.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
