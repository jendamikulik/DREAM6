#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delta-stable ultra-robust causal renormalization audit for the quartic pair.

This version is designed for the current fourth-order renormalization attack.
It keeps the exact structural diagnostics from the previous audit and adds a
controlled target-chi branch.

Main features
-------------
* reduced N^2 transport LP (not the original N^4 two-step LP);
* only active support coordinates are used;
* one redundant transport marginal equation is removed;
* explicit product-coupling feasibility audit;
* no truncation of small positive probabilities;
* tiny negative solver leakage is clipped only after a full feasibility audit;
* max current causal match is solved first;
* min-W and max-W points are solved on the near-optimal match face;
* transports are reflection-symmetrized on the symmetric quartic branch;
* actual matched/residual mass is used everywhere;
* exact renormalization identities for eta, variance and chi are cross-checked;
* lower factorial defects 0..3 are checked every step;
* common affine normalization s -> (s-c)/g is applied to both residual laws;
* optional branch policies:
      minW       iterate the min-W face point;
      maxW       iterate the max-W face point;
      targetChi  choose a convex combination of minW/maxW that gets the next
                 standardized quartic defect chi as close as possible to a
                 requested target (default 1/20).

Important interpretation
------------------------
The quantity 1-match is an unmatched block mass, NOT an inventory cost.
Bounded/critical chi is a renormalization diagnostic; by itself it does not
prove B_n = O(log n).  The missing theorem-level bridge is still a uniform
prefix-compatible inventory increment bound per dyadic scale.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from math import comb, fsum
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog, minimize_scalar
from scipy.sparse import coo_matrix, csr_matrix


# -----------------------------------------------------------------------------
# Numerical tolerances
# -----------------------------------------------------------------------------
# HiGHS can return tiny negative coordinates (e.g. 1e-10) on large degenerate
# transport faces.  These are accepted only if clipping them leaves the full
# row/column marginals within TRANSPORT_TOL.
NEG_TOL = 1e-9
TRANSPORT_TOL = 2e-10
PAIR_TOL = 2e-8
LP_TOL = 1e-9
REPAIR_TOL = 5e-13
REPAIR_MAXITER = 20000
FACE_TOL_DEFAULT = 1e-8
IDENTITY_TOL = 2e-6
IDENTITY_REL_TOL = 2e-8
MOMENT_REL_TOL = 2e-10
TARGET_SEARCH_TOL = 1e-12


@dataclass
class TransportProblem:
    A: np.ndarray
    B: np.ndarray
    N: int
    xs: np.ndarray
    ys: np.ndarray
    pairs: List[Tuple[int, int]]
    c_match: np.ndarray
    W_xy: np.ndarray
    Aeq: csr_matrix
    beq: np.ndarray
    diag_success: Dict[int, List[Tuple[int, float]]]
    mean_total: float


@dataclass
class ResidualState:
    A: np.ndarray
    B: np.ndarray
    c: int
    g: int
    matched_mass: float
    residual_mass: float
    matched_measure: np.ndarray
    W_actual: float


# -----------------------------------------------------------------------------
# Probability / moment utilities
# -----------------------------------------------------------------------------
def pad_same(A: Sequence[float], B: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = max(len(A), len(B))
    if len(A) < n:
        A = np.pad(A, (0, n - len(A)))
    if len(B) < n:
        B = np.pad(B, (0, n - len(B)))
    return A, B


def sanitize_probability(x: Sequence[float], name: str = "dist") -> np.ndarray:
    """Clip only tiny negative roundoff; retain every positive coefficient."""
    x = np.asarray(x, dtype=float).copy()
    if x.size == 0:
        raise RuntimeError(f"{name}: empty distribution")
    mn = float(np.min(x))
    if mn < -NEG_TOL:
        raise RuntimeError(f"{name}: genuine negative mass {mn:.3e}")
    x[x < 0.0] = 0.0
    s = float(fsum(float(v) for v in x))
    if not math.isfinite(s) or s <= 0.0:
        raise RuntimeError(f"{name}: invalid total mass {s}")
    x /= s
    return x


def reflection_error(x: np.ndarray) -> float:
    return float(np.max(np.abs(x - x[::-1]))) if x.size else 0.0


def project_reflection_symmetric(x: np.ndarray, name: str) -> np.ndarray:
    err = reflection_error(x)
    if err > PAIR_TOL:
        raise RuntimeError(f"{name}: reflection asymmetry too large: {err:.3e}")
    return sanitize_probability(0.5 * (x + x[::-1]), name=name)


def mean_of(x: np.ndarray) -> float:
    return float(fsum(i * float(p) for i, p in enumerate(x)))


def variance(x: np.ndarray) -> float:
    mu = mean_of(x)
    return float(fsum(float(p) * (i - mu) ** 2 for i, p in enumerate(x)))


def factorial_defect(A: np.ndarray, B: np.ndarray, order: int) -> float:
    """sum_i C(i,order)(A_i-B_i), i.e. derivative defect / order!."""
    A, B = pad_same(A, B)
    return float(
        fsum(
            comb(i, order) * (float(A[i]) - float(B[i]))
            for i in range(order, len(A))
        )
    )


def eta4(A: np.ndarray, B: np.ndarray) -> float:
    return factorial_defect(A, B, 4)



def factorial_moment(X: np.ndarray, order: int) -> float:
    return float(
        fsum(
            comb(i, order) * float(X[i])
            for i in range(order, len(X))
        )
    )

def validate_pair(A: np.ndarray, B: np.ndarray, label: str) -> None:
    A, B = pad_same(A, B)
    massA = float(fsum(float(v) for v in A))
    massB = float(fsum(float(v) for v in B))
    muA = mean_of(A)
    muB = mean_of(B)

    if abs(massA - 1.0) > PAIR_TOL or abs(massB - 1.0) > PAIR_TOL:
        raise RuntimeError(f"{label}: mass invariant failed: {massA}, {massB}")
    if abs(muA - muB) > PAIR_TOL:
        raise RuntimeError(f"{label}: equal-mean invariant failed: {muA}, {muB}")

    for j in range(4):
        d = factorial_defect(A, B, j)
        # Floating-point cancellation in a zero defect scales with the size of
        # the corresponding factorial moment.  Use an absolute+relative audit.
        scale_j = 0.5 * (
            abs(factorial_moment(A, j)) + abs(factorial_moment(B, j))
        )
        tol_j = max(5e-8, MOMENT_REL_TOL * max(1.0, scale_j))
        if abs(d) > tol_j:
            raise RuntimeError(
                f"{label}: factorial defect order {j} = {d:.3e} "
                f"(tol={tol_j:.3e}, scale={scale_j:.3e})"
            )

    eA = reflection_error(A)
    eB = reflection_error(B)
    if eA > PAIR_TOL or eB > PAIR_TOL:
        raise RuntimeError(
            f"{label}: reflection symmetry failed: A={eA:.3e}, B={eB:.3e}"
        )


# -----------------------------------------------------------------------------
# LP helpers
# -----------------------------------------------------------------------------
def solve_lp(
    c: np.ndarray,
    A_eq: csr_matrix | None = None,
    b_eq: np.ndarray | None = None,
    A_ub: csr_matrix | None = None,
    b_ub: np.ndarray | None = None,
    bounds=(0, None),
    label: str = "LP",
):
    """Try HiGHS backends in a stable order."""
    methods = ["highs-ds", "highs-ipm", "highs"]
    failures = []
    for method in methods:
        try:
            res = linprog(
                c,
                A_eq=A_eq,
                b_eq=b_eq,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method=method,
                options={
                    "primal_feasibility_tolerance": LP_TOL,
                    "dual_feasibility_tolerance": LP_TOL,
                    "presolve": True,
                },
            )
            if res.success:
                return res, method
            failures.append(f"{method}: {res.message}")
        except Exception as exc:
            failures.append(f"{method}: {exc}")
    raise RuntimeError(label + " failed; " + " | ".join(failures))


def build_transport_problem(A: np.ndarray, B: np.ndarray) -> TransportProblem:
    A, B = pad_same(A, B)
    N = len(A)

    # No positive-mass truncation.
    xs = np.flatnonzero(A > 0.0)
    ys = np.flatnonzero(B > 0.0)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("empty active support")

    pairs = [(int(x), int(y)) for x in xs for y in ys]
    nvar = len(pairs)

    # Explicit product coupling proves the transport polytope is nonempty.
    product = np.outer(A[xs], B[ys])
    row_err = float(np.max(np.abs(product.sum(axis=1) - A[xs])))
    col_err = float(np.max(np.abs(product.sum(axis=0) - B[ys])))
    if row_err > 2e-12 or col_err > 2e-12:
        raise RuntimeError(
            f"product coupling audit failed: row={row_err:.3e}, col={col_err:.3e}"
        )

    # alpha_d = maximal one-step mass on displacement diagonal X-Y=d.
    diag_success: Dict[int, List[Tuple[int, float]]] = {}
    alpha: Dict[int, float] = {}
    for d in range(-(N - 1), N):
        items: List[Tuple[int, float]] = []
        total = 0.0
        for t in range(N):
            tt = t - d
            if 0 <= tt < N:
                m = min(float(A[t]), float(B[tt]))
                if m > 0.0:
                    items.append((t, m))
                    total += m
        diag_success[d] = items
        alpha[d] = total

    c_match = np.empty(nvar, dtype=float)
    for j, (x, y) in enumerate(pairs):
        c_match[j] = alpha[y - x]

    mean_total = mean_of(A) + mean_of(B)
    W_xy = np.zeros(nvar, dtype=float)
    for j, (x, y) in enumerate(pairs):
        d = y - x
        val = 0.0
        for t, m in diag_success[d]:
            s = x + t
            val += (s - mean_total) ** 2 * m
        W_xy[j] = val

    # All active X rows + all but one active Y columns (rank 2N-1).
    nrows = len(xs) + max(0, len(ys) - 1)
    rr: List[int] = []
    cc: List[int] = []
    vv: List[float] = []
    beq: List[float] = []

    x_row = {int(x): i for i, x in enumerate(xs)}
    y_cols = list(map(int, ys[:-1]))
    y_row = {y: len(xs) + i for i, y in enumerate(y_cols)}

    for j, (x, y) in enumerate(pairs):
        rr.append(x_row[x]); cc.append(j); vv.append(1.0)
        if y in y_row:
            rr.append(y_row[y]); cc.append(j); vv.append(1.0)

    for x in xs:
        beq.append(float(A[int(x)]))
    for y in y_cols:
        beq.append(float(B[y]))

    Aeq = coo_matrix((vv, (rr, cc)), shape=(nrows, nvar), dtype=float).tocsr()
    beq_arr = np.asarray(beq, dtype=float)

    return TransportProblem(
        A=A,
        B=B,
        N=N,
        xs=xs,
        ys=ys,
        pairs=pairs,
        c_match=c_match,
        W_xy=W_xy,
        Aeq=Aeq,
        beq=beq_arr,
        diag_success=diag_success,
        mean_total=mean_total,
    )


def dense_transport(tp: TransportProblem, pi: np.ndarray) -> np.ndarray:
    P = np.zeros((tp.N, tp.N), dtype=float)
    for val, (x, y) in zip(pi, tp.pairs):
        P[x, y] = float(val)
    return P


def vector_from_dense(tp: TransportProblem, P: np.ndarray) -> np.ndarray:
    return np.asarray([P[x, y] for x, y in tp.pairs], dtype=float)


def _active_targets(tp: TransportProblem) -> Tuple[np.ndarray, np.ndarray]:
    """Return exactly normalized active row/column marginals."""
    a = np.asarray(tp.A[tp.xs], dtype=float).copy()
    b = np.asarray(tp.B[tp.ys], dtype=float).copy()
    sa = float(fsum(float(x) for x in a))
    sb = float(fsum(float(x) for x in b))
    if sa <= 0.0 or sb <= 0.0:
        raise RuntimeError("non-positive active mass")
    a /= sa
    b /= sb
    return a, b


def repair_transport(tp: TransportProblem, pi: np.ndarray, label: str) -> np.ndarray:
    """
    Project a numerically approximate transport back onto the exact active
    transport polytope.  This is essential at N>=65: a 1e-10 marginal error
    can be amplified by fourth moments.

    We first clip only tiny negative solver leakage, then use RAS/Sinkhorn
    scaling.  A vanishingly small product-coupling blend is added only if
    needed to guarantee total support.  The smallest successful blend is used.
    """
    pi = np.asarray(pi, dtype=float).copy()
    if pi.size != len(tp.pairs):
        raise RuntimeError(f"{label}: wrong transport vector size")

    mn = float(np.min(pi)) if pi.size else 0.0
    if mn < -5e-8:
        raise RuntimeError(f"{label}: genuine negative transport mass {mn:.3e}")
    pi[pi < 0.0] = 0.0

    nr, nc = len(tp.xs), len(tp.ys)
    P0 = pi.reshape(nr, nc).copy()
    a, b = _active_targets(tp)
    base = np.outer(a, b)

    # Try essentially no perturbation first; increase only if the sparse
    # support selected by HiGHS cannot be matrix-scaled reliably.
    blends = [0.0, 1e-16, 1e-14, 1e-12, 1e-10]
    best = None
    for blend in blends:
        P = P0.copy()
        if blend > 0.0:
            P = (1.0 - blend) * P + blend * base

        ok = True
        for it in range(REPAIR_MAXITER):
            rs = P.sum(axis=1)
            if np.any(rs <= 0.0):
                ok = False
                break
            P *= (a / rs)[:, None]

            cs = P.sum(axis=0)
            if np.any(cs <= 0.0):
                ok = False
                break
            P *= (b / cs)[None, :]

            if it % 5 == 0:
                row_err = float(np.max(np.abs(P.sum(axis=1) - a)))
                col_err = float(np.max(np.abs(P.sum(axis=0) - b)))
                err = max(row_err, col_err)
                if best is None or err < best[0]:
                    best = (err, P.copy(), blend, it)
                if err <= REPAIR_TOL:
                    out = P.reshape(-1)
                    # Full-coordinate audit against the stored pair.
                    full = dense_transport(tp, out)
                    rerr = float(np.max(np.abs(full.sum(axis=1) - tp.A)))
                    cerr = float(np.max(np.abs(full.sum(axis=0) - tp.B)))
                    if max(rerr, cerr) <= max(TRANSPORT_TOL, 5e-12):
                        return out
                    # The active marginals differ from tp.A/B only by their
                    # ~machine-epsilon normalization.  Accept that difference.
                    if max(rerr, cerr) <= 5e-10:
                        return out
                    ok = False
                    break
        if ok:
            continue

    if best is not None:
        err, _, blend, it = best
        raise RuntimeError(
            f"{label}: transport repair did not converge; best err={err:.3e}, "
            f"blend={blend:.1e}, iter={it}"
        )
    raise RuntimeError(f"{label}: transport repair failed")


def audit_transport(tp: TransportProblem, pi: np.ndarray, label: str) -> np.ndarray:
    """Strict audit of an already repaired transport."""
    pi = np.asarray(pi, dtype=float)
    mn = float(np.min(pi)) if pi.size else 0.0
    if mn < -1e-14:
        raise RuntimeError(f"{label}: negative transport mass {mn:.3e}")
    P = dense_transport(tp, pi)
    row_err = float(np.max(np.abs(P.sum(axis=1) - tp.A)))
    col_err = float(np.max(np.abs(P.sum(axis=0) - tp.B)))
    mass_err = abs(float(P.sum()) - 1.0)
    worst = max(row_err, col_err, mass_err)
    if worst > 5e-10:
        raise RuntimeError(
            f"{label}: transport audit failed: row={row_err:.3e}, "
            f"col={col_err:.3e}, mass={mass_err:.3e}"
        )
    return pi


def _delta_base(tp: TransportProblem) -> np.ndarray:
    """
    Product-coupling base on the active support.  Delta variables satisfy zero
    row/column sums, avoiding the tiny-RHS pathology that made HiGHS declare
    the perfectly feasible N=129 transport problem infeasible.
    """
    a, b = _active_targets(tp)
    return np.outer(a, b).reshape(-1)


def solve_delta_lp(
    tp: TransportProblem,
    objective_pi: np.ndarray,
    maximize: bool,
    label: str,
    match_floor: float | None = None,
):
    """
    Solve a transport LP in product-shift coordinates

        pi = a ⊗ b + delta,
        row(delta)=col(delta)=0,
        delta >= -(a tensor b).

    All equality RHS values are zero.  This is much more stable when some
    marginals are 1e-20--1e-30.  Presolve=False is tried first because HiGHS
    presolve is precisely what misclassified the N=129 problem as infeasible.
    """
    base = _delta_base(tp)
    nvar = len(base)
    obj_raw = np.asarray(objective_pi, dtype=float)
    scale = max(1.0, float(np.max(np.abs(obj_raw))))
    obj = obj_raw / scale
    if maximize:
        obj = -obj

    bounds = [(-float(base[j]), None) for j in range(nvar)]
    zero_rhs = np.zeros(tp.Aeq.shape[0], dtype=float)

    Aub = None
    bub = None
    if match_floor is not None:
        # c*pi >= floor  <=>  -c*delta <= -(floor-c*base)
        Aub = csr_matrix((-tp.c_match).reshape(1, -1))
        rhs = -(float(match_floor) - float(tp.c_match @ base))
        bub = np.asarray([rhs], dtype=float)

    attempts = [
        ("highs-ds/delta/no-presolve", "highs-ds", False, 1e-9),
        ("highs-ipm/delta/no-presolve", "highs-ipm", False, 1e-9),
        ("highs/delta/no-presolve", "highs", False, 1e-9),
        ("highs-ds/delta/no-presolve-loose", "highs-ds", False, 1e-8),
        ("highs-ipm/delta/presolve", "highs-ipm", True, 1e-8),
    ]
    failures = []
    for method_label, method, presolve, tol in attempts:
        try:
            res = linprog(
                obj,
                A_eq=tp.Aeq,
                b_eq=zero_rhs,
                A_ub=Aub,
                b_ub=bub,
                bounds=bounds,
                method=method,
                options={
                    "primal_feasibility_tolerance": tol,
                    "dual_feasibility_tolerance": tol,
                    "presolve": presolve,
                },
            )
        except Exception as exc:
            failures.append(f"{method_label}: {exc}")
            continue
        if not res.success:
            failures.append(f"{method_label}: {res.message}")
            continue

        pi_raw = base + np.asarray(res.x, dtype=float)
        try:
            pi = repair_transport(tp, pi_raw, label)
            audit_transport(tp, pi, label)
        except RuntimeError as exc:
            failures.append(f"{method_label}: {exc}")
            continue

        if match_floor is not None:
            m = float(tp.c_match @ pi)
            # Repair changes the objective only at numerical scale.  Require
            # the requested near-optimal face to survive the projection.
            if m < match_floor - 2e-8:
                failures.append(
                    f"{method_label}: repaired match {m:.12g} below floor {match_floor:.12g}"
                )
                continue
        return res, pi, method_label

    raise RuntimeError(label + " failed; " + " | ".join(failures))


def solve_max_match(tp: TransportProblem):
    res, pi, method = solve_delta_lp(
        tp, tp.c_match, maximize=True, label="max-match"
    )
    match = float(tp.c_match @ pi)
    return match, pi, method


def solve_face_extreme(
    tp: TransportProblem,
    m_star: float,
    face_tol: float,
    minimize_W: bool,
):
    """Optimize W on c_match*pi >= m_star-face_tol using delta coordinates."""
    label = "min-W face" if minimize_W else "max-W face"
    _, pi, method = solve_delta_lp(
        tp,
        tp.W_xy,
        maximize=not minimize_W,
        label=label,
        match_floor=m_star - face_tol,
    )
    W = float(tp.W_xy @ pi)
    match = float(tp.c_match @ pi)
    return W, match, pi, method


def symmetrize_transport(tp: TransportProblem, pi: np.ndarray) -> np.ndarray:
    """Reflection symmetrization followed by exact marginal repair."""
    P = dense_transport(tp, pi)
    Ps = 0.5 * (P + P[::-1, ::-1])
    out = vector_from_dense(tp, Ps)
    out = repair_transport(tp, out, "symmetrized transport")
    return audit_transport(tp, out, "symmetrized transport")


# -----------------------------------------------------------------------------
# Residual / renormalization
# -----------------------------------------------------------------------------
def matched_measure(tp: TransportProblem, pi: np.ndarray) -> np.ndarray:
    M = np.zeros(2 * (tp.N - 1) + 1, dtype=float)
    for p, (x, y) in zip(pi, tp.pairs):
        p = float(p)
        if p <= 0.0:
            continue
        d = y - x
        for t, m in tp.diag_success[d]:
            M[x + t] += p * m
    return M


def common_affine_normalize(
    ResA: np.ndarray,
    ResB: np.ndarray,
    residual_mass: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Apply one common affine map to both residual laws.  Crucially, do NOT
    renormalize A and B independently: that would amplify machine-epsilon mass
    differences into visible third/fourth factorial defects at N~100.
    """
    ResA = np.asarray(ResA, dtype=np.longdouble)
    ResB = np.asarray(ResB, dtype=np.longdouble)
    positive_keys = [
        i for i in range(max(len(ResA), len(ResB)))
        if (i < len(ResA) and ResA[i] > 0.0)
        or (i < len(ResB) and ResB[i] > 0.0)
    ]
    if not positive_keys:
        raise RuntimeError("empty residual support")

    c = min(positive_keys)
    diffs = [k - c for k in positive_keys if k != c]
    if diffs:
        g = diffs[0]
        for d in diffs[1:]:
            g = math.gcd(g, d)
    else:
        g = 1

    max_key = max(positive_keys)
    new_len = (max_key - c) // g + 1
    Anew = np.zeros(new_len, dtype=np.longdouble)
    Bnew = np.zeros(new_len, dtype=np.longdouble)

    r = np.longdouble(residual_mass)
    for s, val in enumerate(ResA):
        if val > 0.0:
            Anew[(s - c) // g] += val / r
    for s, val in enumerate(ResB):
        if val > 0.0:
            Bnew[(s - c) // g] += val / r

    # Reflection symmetry is an exact invariant of the branch; project only
    # by averaging, with no separate probability normalization.
    Anew = 0.5 * (Anew + Anew[::-1])
    Bnew = 0.5 * (Bnew + Bnew[::-1])

    sA = float(np.sum(Anew, dtype=np.longdouble))
    sB = float(np.sum(Bnew, dtype=np.longdouble))
    if abs(sA - sB) > 2e-10:
        raise RuntimeError(f"renormalized masses differ: {sA:.16g}, {sB:.16g}")
    # A single common scale preserves every signed factorial defect.
    s_common = 0.5 * (sA + sB)
    if not math.isfinite(s_common) or s_common <= 0.0:
        raise RuntimeError("invalid common renormalized mass")
    Anew /= np.longdouble(s_common)
    Bnew /= np.longdouble(s_common)

    Aout = np.asarray(Anew, dtype=float)
    Bout = np.asarray(Bnew, dtype=float)
    Aout, Bout = pad_same(Aout, Bout)
    return Aout, Bout, c, g


def residual_from_pi(tp: TransportProblem, pi: np.ndarray) -> ResidualState:
    pi = audit_transport(tp, pi, "residual input transport")
    M64 = matched_measure(tp, pi)
    # Symmetric transport should give symmetric M. Averaging removes only
    # roundoff and preserves total matched mass exactly to working precision.
    M64 = 0.5 * (M64 + M64[::-1])

    A_ld = np.asarray(tp.A, dtype=np.longdouble)
    B_ld = np.asarray(tp.B, dtype=np.longdouble)
    M = np.asarray(M64, dtype=np.longdouble)
    ConvA = np.convolve(A_ld, A_ld)
    ConvB = np.convolve(B_ld, B_ld)

    ResA = ConvA - M
    ResB = ConvB - M
    minA = float(np.min(ResA))
    minB = float(np.min(ResB))
    if minA < -5e-10 or minB < -5e-10:
        raise RuntimeError(
            f"matched measure exceeds convolution: minA={minA:.3e}, minB={minB:.3e}"
        )
    ResA[ResA < 0.0] = 0.0
    ResB[ResB < 0.0] = 0.0

    matched_mass = float(np.sum(M, dtype=np.longdouble))
    rA = float(np.sum(ResA, dtype=np.longdouble))
    rB = float(np.sum(ResB, dtype=np.longdouble))
    # Use the measured common residual mass, not 1-M rounded in float64.
    residual_mass = 0.5 * (rA + rB)
    if residual_mass <= 0.0:
        raise RuntimeError("non-positive residual mass")
    if abs(rA - rB) > 2e-10:
        raise RuntimeError(
            f"residual masses differ: A={rA:.16g}, B={rB:.16g}, 1-M={1-matched_mass:.16g}"
        )
    if abs((1.0 - matched_mass) - residual_mass) > 2e-9:
        raise RuntimeError(
            f"residual mass inconsistency: 1-M={1-matched_mass:.16g}, r={residual_mass:.16g}"
        )

    Anew, Bnew, c, g = common_affine_normalize(ResA, ResB, residual_mass)

    W_actual = float(
        fsum((s - tp.mean_total) ** 2 * float(m) for s, m in enumerate(M64))
    )

    return ResidualState(
        A=Anew,
        B=Bnew,
        c=c,
        g=g,
        matched_mass=matched_mass,
        residual_mass=residual_mass,
        matched_measure=np.asarray(M64, dtype=float),
        W_actual=W_actual,
    )


def branch_diagnostics(
    tp: TransportProblem,
    pi: np.ndarray,
    V: float,
    eta: float,
    label: str,
    do_symmetrize: bool = True,
) -> Tuple[ResidualState, dict, np.ndarray]:
    pi_use = symmetrize_transport(tp, pi) if do_symmetrize else audit_transport(tp, pi, label)
    match_actual = float(tp.c_match @ pi_use)
    state = residual_from_pi(tp, pi_use)

    if abs(match_actual - state.matched_mass) > 3e-7:
        raise RuntimeError(
            f"{label}: match mismatch: c*pi={match_actual}, M(1)={state.matched_mass}"
        )

    newV = variance(state.A)
    new_eta = eta4(state.A, state.B)
    chi = eta / (V * V)
    new_chi = new_eta / (newV * newV)

    var_ratio = (newV * state.g * state.g) / V
    chi_ratio_direct = new_chi / chi
    chi_ratio_pred = 2.0 / (state.residual_mass * var_ratio * var_ratio)

    eta_pred = 2.0 * eta / (state.residual_mass * (state.g ** 4))
    eta_err = abs(new_eta - eta_pred)

    var_ratio_pred = (2.0 * V - state.W_actual) / (state.residual_mass * V)
    var_ratio_err = abs(var_ratio - var_ratio_pred)
    chi_ratio_err = abs(chi_ratio_direct - chi_ratio_pred)

    info = {
        "match_actual": match_actual,
        "r_actual": state.residual_mass,
        "W_actual": state.W_actual,
        "newV": newV,
        "var_ratio": var_ratio,
        "var_ratio_pred": var_ratio_pred,
        "var_ratio_err": var_ratio_err,
        "new_eta": new_eta,
        "eta_pred": eta_pred,
        "eta_err": eta_err,
        "chi": chi,
        "new_chi": new_chi,
        "chi_ratio_direct": chi_ratio_direct,
        "chi_ratio_pred": chi_ratio_pred,
        "chi_ratio_err": chi_ratio_err,
    }
    return state, info, pi_use


def print_branch(label: str, info: dict, state: ResidualState) -> None:
    print(f"  {label}:")
    print(
        f"    match_actual={info['match_actual']:.12f}, "
        f"r_actual={info['r_actual']:.12f}, W={info['W_actual']:.12f}"
    )
    print(
        f"    newV={info['newV']:.12f}, var_ratio={info['var_ratio']:.12f}, "
        f"pred={info['var_ratio_pred']:.12f}, err={info['var_ratio_err']:.3e}"
    )
    print(
        f"    new_eta={info['new_eta']:.12f}, eta_pred={info['eta_pred']:.12f}, "
        f"err={info['eta_err']:.3e}"
    )
    print(
        f"    new_chi={info['new_chi']:.12f}, "
        f"chi_ratio_direct={info['chi_ratio_direct']:.12f}, "
        f"pred={info['chi_ratio_pred']:.12f}, err={info['chi_ratio_err']:.3e}"
    )
    print(f"    affine c={state.c}, g={state.g}, next N={len(state.A)}")


# -----------------------------------------------------------------------------
# Target-chi control on the convex hull of the two face extremes
# -----------------------------------------------------------------------------
def target_chi_control(
    tp: TransportProblem,
    pi_min_sym: np.ndarray,
    pi_max_sym: np.ndarray,
    V: float,
    eta: float,
    chi_target: float,
) -> Tuple[ResidualState, dict, np.ndarray, float, bool]:
    """
    Search lambda in [0,1] for
        pi(lambda) = (1-lambda) pi_min + lambda pi_max
    that makes next_chi as close as possible to chi_target.

    The convex combination remains on the same near-optimal match face.
    Returns (state, info, pi, lambda, bracketed).
    """

    def evaluate(lam: float):
        pi = (1.0 - lam) * pi_min_sym + lam * pi_max_sym
        state, info, pi_use = branch_diagnostics(
            tp, pi, V, eta, label=f"targetChi(lambda={lam:.6g})", do_symmetrize=False
        )
        return state, info, pi_use

    state0, info0, _ = evaluate(0.0)
    state1, info1, _ = evaluate(1.0)
    f0 = info0["new_chi"] - chi_target
    f1 = info1["new_chi"] - chi_target
    bracketed = (f0 == 0.0) or (f1 == 0.0) or (f0 * f1 < 0.0)

    # If the target is bracketed, bisection is deterministic and robust.
    if bracketed:
        lo, hi = 0.0, 1.0
        best = None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            state, info, pi_use = evaluate(mid)
            fmid = info["new_chi"] - chi_target
            cand = (abs(fmid), mid, state, info, pi_use)
            if best is None or cand[0] < best[0]:
                best = cand
            if abs(fmid) <= TARGET_SEARCH_TOL:
                break
            # Use endpoint signs; no monotonicity assumption beyond local bracketing.
            if f0 == 0.0:
                best = (0.0, 0.0, state0, info0, pi_min_sym)
                break
            if f0 * fmid <= 0.0:
                hi = mid
                f1 = fmid
            else:
                lo = mid
                f0 = fmid
        assert best is not None
        _, lam, state, info, pi_use = best
        return state, info, pi_use, lam, True

    # Not bracketed: minimize |chi_next-target| over the convex segment.
    # This also handles weak non-monotonicity caused by the tiny face tolerance.
    def objective(lam: float) -> float:
        _, info, _ = evaluate(float(lam))
        return abs(info["new_chi"] - chi_target)

    res = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-9, "maxiter": 80},
    )
    candidates = [0.0, 1.0, float(res.x)]
    best = None
    for lam in candidates:
        state, info, pi_use = evaluate(lam)
        err = abs(info["new_chi"] - chi_target)
        cand = (err, lam, state, info, pi_use)
        if best is None or cand[0] < best[0]:
            best = cand
    assert best is not None
    _, lam, state, info, pi_use = best
    return state, info, pi_use, lam, False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--face-tol", type=float, default=FACE_TOL_DEFAULT)
    parser.add_argument(
        "--branch",
        choices=["minW", "maxW", "targetChi"],
        default="targetChi",
        help="Continuation policy for the next renormalization step.",
    )
    parser.add_argument(
        "--chi-target",
        type=float,
        default=0.05,
        help="Target standardized quartic defect for --branch targetChi (default 1/20).",
    )
    parser.add_argument(
        "--max-active-vars",
        type=int,
        default=60000,
        help="Stop gracefully before a transport LP exceeds this many active variables.",
    )
    args = parser.parse_args()

    E = sanitize_probability(np.asarray([1/8, 0, 6/8, 0, 1/8], dtype=float), "E")
    O = sanitize_probability(np.asarray([0, 1/2, 0, 1/2, 0], dtype=float), "O")
    A, B = E, O

    print("=" * 124)
    print("ULTRA-ROBUST CAUSAL RENORMALIZATION AUDIT -- DELTA-STABLE TARGET-CHI EDITION")
    print("=" * 124)
    print(
        f"branch={args.branch}, face_tol={args.face_tol:.3e}, "
        f"chi_target={args.chi_target:.12f}"
    )

    for k in range(args.steps):
        A, B = pad_same(A, B)
        # Only symmetry averaging here; no independent mass renormalization.
        if reflection_error(A) > PAIR_TOL or reflection_error(B) > PAIR_TOL:
            raise RuntimeError(f"step {k}: reflection symmetry drift")
        A = 0.5 * (A + A[::-1])
        B = 0.5 * (B + B[::-1])
        validate_pair(A, B, f"step {k}")

        V = variance(A)
        eta = eta4(A, B)
        chi = eta / (V * V)
        low_defects = [factorial_defect(A, B, j) for j in range(4)]

        tp = build_transport_problem(A, B)
        if len(tp.pairs) > args.max_active_vars:
            print(
                f"\nStep {k}: active transport has {len(tp.pairs)} vars > "
                f"--max-active-vars={args.max_active_vars}; stopping cleanly."
            )
            break
        try:
            m_star, _, max_method = solve_max_match(tp)
            Wmin_raw, match_min_raw, pi_min, min_method = solve_face_extreme(
                tp, m_star, args.face_tol, minimize_W=True
            )
            Wmax_raw, match_max_raw, pi_max, maxW_method = solve_face_extreme(
                tp, m_star, args.face_tol, minimize_W=False
            )
        except RuntimeError as exc:
            print(f"\nStep {k}: solver/audit stopped cleanly: {exc}")
            break

        state_min, info_min, pi_min_sym = branch_diagnostics(
            tp, pi_min, V, eta, "minW", do_symmetrize=True
        )
        state_max, info_max, pi_max_sym = branch_diagnostics(
            tp, pi_max, V, eta, "maxW", do_symmetrize=True
        )

        print(f"\nStep {k}:")
        print(
            f"  N={len(A)}, active=({len(tp.xs)}x{len(tp.ys)})={len(tp.pairs)} vars, "
            f"m*={m_star:.12f}, solver={max_method}"
        )
        print(
            f"  V={V:.12f}, eta={eta:.12f}, chi={chi:.12f}, "
            f"factorial defects 0..3={[f'{d:.2e}' for d in low_defects]}"
        )
        print(
            f"  face raw: Wmin={Wmin_raw:.12f} (match={match_min_raw:.12f}, {min_method}), "
            f"Wmax={Wmax_raw:.12f} (match={match_max_raw:.12f}, {maxW_method})"
        )
        print_branch("minW", info_min, state_min)
        print_branch("maxW", info_max, state_max)

        # Cross-check renormalization identities using both absolute and
        # scale-aware relative residuals.  At N~100 eta is O(1e4), so an
        # absolute 1e-5 discrepancy can still be a few parts in 1e9.
        for tag, info in [("minW", info_min), ("maxW", info_max)]:
            eta_rel = info["eta_err"] / max(1.0, abs(info["eta_pred"]))
            var_rel = info["var_ratio_err"] / max(1.0, abs(info["var_ratio_pred"]))
            chi_rel = info["chi_ratio_err"] / max(1.0, abs(info["chi_ratio_pred"]))
            worst_abs = max(info["eta_err"], info["var_ratio_err"], info["chi_ratio_err"])
            worst_rel = max(eta_rel, var_rel, chi_rel)
            if worst_abs > IDENTITY_TOL and worst_rel > IDENTITY_REL_TOL:
                print(
                    f"  WARNING {tag}: renormalization identity residual "
                    f"abs={worst_abs:.3e}, rel={worst_rel:.3e}"
                )

        if args.branch == "minW":
            chosen_state, chosen_info = state_min, info_min
            chosen_desc = "minW"
        elif args.branch == "maxW":
            chosen_state, chosen_info = state_max, info_max
            chosen_desc = "maxW"
        else:
            chosen_state, chosen_info, _, lam, bracketed = target_chi_control(
                tp,
                pi_min_sym,
                pi_max_sym,
                V,
                eta,
                args.chi_target,
            )
            chosen_desc = (
                f"targetChi lambda={lam:.9f}, "
                f"{'target bracketed' if bracketed else 'nearest point on minW/maxW segment'}"
            )
            print("  targetChi control:")
            print(f"    {chosen_desc}")
            print(
                f"    requested={args.chi_target:.12f}, "
                f"achieved={chosen_info['new_chi']:.12f}, "
                f"error={abs(chosen_info['new_chi']-args.chi_target):.3e}"
            )
            print(
                f"    match_actual={chosen_info['match_actual']:.12f}, "
                f"r_actual={chosen_info['r_actual']:.12f}, W={chosen_info['W_actual']:.12f}"
            )

        print(
            f"  CONTINUE -> {chosen_desc}; next chi={chosen_info['new_chi']:.12f}, "
            f"next N={len(chosen_state.A)}"
        )
        A, B = chosen_state.A, chosen_state.B


if __name__ == "__main__":
    main()
