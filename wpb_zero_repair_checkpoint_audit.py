#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WPB ZERO-REPAIR CHECKPOINT AUDIT
================================

Independent checks for the 22 Aug 2026 zero-repair checkpoint.

What this script checks
-----------------------
A. Exact 19/9 dual arithmetic and the finite-horizon D_15 certificate.
B. Exact small-a expansion behind
       liminf B_n / log n >= 1/3.
C. Exact delayed-boundary-repair algebra for the geometric reservoir.
D. Coefficientwise positivity of the shortest J=4 repair.
E. The integer tangent profiles
       (1,1,...), (3,3,1,...), (9,9,-3,-3,1,...).
F. Full-history LP feasibility for explicit binomial roots.

The LP is rebuilt directly from
    3 B_h(u-3) + B_h(u-5)
      = B_{h0}(u-2) + 3 B_{h1}(u-4),
with normalization and nonnegativity only.

Default run:
    python wpb_zero_repair_checkpoint_audit.py

Deeper finite-tree audit (about tens of seconds on a typical desktop):
    python wpb_zero_repair_checkpoint_audit.py --deep

Dependencies:
    sympy, numpy, scipy
"""

from __future__ import annotations

import argparse
import math
import time
from fractions import Fraction as F
from math import comb

import numpy as np
import sympy as sp
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix


# ---------------------------------------------------------------------------
# A. Exact rational lower certificates
# ---------------------------------------------------------------------------

def check_exact_lower_certificates() -> None:
    dual_19_9 = (
        F(8, 3) + F(1)
        - F(1, 3) * F(1, 9)
        - F(1, 3) * F(1, 3)
        - F(1, 9) * F(2, 3)
        - F(4, 9) * F(3, 4)
        - F(1) * F(1, 3)
        - F(4, 3) * F(1, 2)
    )
    assert dual_19_9 == F(19, 9)

    # Finite-horizon CDF-cap recursion.
    def T(q):
        q0, q1, q2 = q
        return (
            min(F(1), q1 / 3),
            min(F(1), q0 + q2 / 3),
            min(F(1), q1 + F(1, 3)),
        )

    qs = [(F(1), F(1), F(1))]
    for _ in range(40):
        qs.append(T(qs[-1]))

    def c1(m):
        return (F(1) + 3 * qs[m - 1][1]) / 4

    def c2(m):
        return (F(1) + 3 * qs[m - 1][2]) / 4

    def D(n):
        if n < 2:
            raise ValueError("D_n requires n>=2")
        qn = qs[n]
        qc = qs[n - 1]
        return (
            F(8, 3) + F(1)
            - F(1, 3) * qn[0]
            - F(1, 3) * qn[1]
            - F(1, 9) * qc[2]
            - F(4, 9) * c2(n - 1)
            - qc[1]
            - F(4, 3) * c1(n - 1)
        )

    assert D(10) == F(1327, 729)
    assert D(11) == F(4097, 2187)
    assert D(15) == F(39473, 19683)
    assert D(15) > 2

    print("A. exact lower certificates: PASS")
    print("   19/9 =", dual_19_9, "=", float(dual_19_9))
    print("   D_15 =", D(15), "=", float(D(15)), "> 2")


# ---------------------------------------------------------------------------
# B. Logarithmic lower-law expansion
# ---------------------------------------------------------------------------

def check_log_lower_expansion() -> None:
    a = sp.symbols("a", positive=True)
    z = sp.exp(-a)

    GP = (z**2 + 3 * z**4) / 4
    GQ = (3 * z**3 + z**5) / 4
    rho = sp.simplify(GQ / GP)

    GP_z2 = (z**4 + 3 * z**8) / 4
    Lambda = sp.simplify(GP_z2 / GP**2)

    d = -sp.log(rho)
    logL = sp.log(Lambda)

    d_series = sp.series(d, a, 0, 5).removeO().expand()
    L_series = sp.series(logL, a, 0, 4).removeO().expand()

    # Leading coefficients used in the theorem.
    assert sp.simplify(d_series.coeff(a, 3) - sp.Rational(1, 4)) == 0
    assert sp.simplify(L_series.coeff(a, 2) - sp.Rational(3, 4)) == 0

    Gamma = 2 * a + a * logL / d
    Gamma_series = sp.series(Gamma, a, 0, 2)
    Gamma_limit = sp.limit(Gamma, a, 0, dir="+")
    assert Gamma_limit == 3

    print("B. logarithmic lower expansion: PASS")
    print("   -log rho(e^-a) =", sp.series(d, a, 0, 5))
    print("   log Lambda(e^-a) =", sp.series(logL, a, 0, 4))
    print("   Gamma(a) =", Gamma_series)
    print("   lim Gamma =", Gamma_limit, "=> liminf B_n/log n >= 1/3")


# ---------------------------------------------------------------------------
# C/D. Exact delayed boundary repair
# ---------------------------------------------------------------------------

def check_delayed_boundary_repair() -> None:
    z, r = sp.symbols("z r", positive=True)

    G = (1 - r) / (1 - r * z)
    D = (3 - 2 * z**2) * G

    A = 3 * (1 - r) * (4 - 3 * r**2)
    alpha = sp.expand(A * (1 + r))

    # J=4 exact repair.
    C4 = A * z**2 + r * A * z**3 - alpha * z**4 * G
    E4 = sp.simplify((3 - 2 * z**2)**2 * G + C4)
    H4 = sp.simplify(z * D - C4 / (3 * z))

    E4_expected = (
        9 * (1 - r)
        + 9 * r * (1 - r) * z
        + (9 * r**2 - 8) * z**4 * G
    )
    H4_expected = (
        (1 - r) * (3 * r**2 - 1) * z
        + r * (1 - r) * (3 * r**2 - 1) * z**2
        + (3 * r**4 - 4 * r**2 + 2) * z**3 * G
    )

    assert sp.simplify(E4 - E4_expected) == 0
    assert sp.simplify(H4 - H4_expected) == 0
    assert sp.simplify(z * (3 + z**2) * D - (z * E4 + 3 * z**2 * H4)) == 0
    assert sp.simplify(E4.subs(z, 1) - 1) == 0
    assert sp.simplify(H4.subs(z, 1) - 1) == 0

    # J=4 positivity threshold:
    # (3r^2-2)^2 - alpha = 9r^2 - 8.
    threshold = sp.expand((3 * r**2 - 2)**2 - alpha)
    assert sp.factor(threshold) == 9 * r**2 - 8

    # General delayed identity checked symbolically for several J.
    for J in [4, 5, 8, 13]:
        CJ = A * z**2 + r * A * z**3 - alpha * z**J * G
        EJ = sp.simplify((3 - 2 * z**2)**2 * G + CJ)
        HJ = sp.simplify(z * D - CJ / (3 * z))
        ident = sp.simplify(z * (3 + z**2) * D - (z * EJ + 3 * z**2 * HJ))
        assert ident == 0
        assert sp.simplify(EJ.subs(z, 1) - 1) == 0
        assert sp.simplify(HJ.subs(z, 1) - 1) == 0

    # Small-epsilon debt size.
    eps = sp.symbols("eps", positive=True)
    alpha_eps = sp.expand(alpha.subs(r, 1 - eps))
    assert sp.series(alpha_eps, eps, 0, 2) == 6 * eps + sp.O(eps**2)

    print("C. delayed boundary-repair algebra: PASS")
    print("   alpha_r =", sp.factor(alpha))
    print("   J=4 positivity threshold = 9 r^2 - 8")
    print("   alpha_(1-eps) =", sp.series(alpha_eps, eps, 0, 4))

    # Concrete rational positivity sample and J_max.
    rr = F(80, 81)
    alpha_q = 3 * (1 - rr * rr) * (4 - 3 * rr * rr)
    base_q = (3 * rr * rr - 2) ** 2
    Jmax = 3
    for J in range(4, 10000):
        if alpha_q <= base_q * rr ** (J - 4):
            Jmax = J
        else:
            break
    print("   sample r=80/81: max delayed J satisfying positivity =", Jmax)


# ---------------------------------------------------------------------------
# E. Integer tangent model
# ---------------------------------------------------------------------------

def tangent_coefficients(poly_coeff, nmax=10):
    """Coefficients of P(z)/(1-z), given finite polynomial coefficients."""
    out = []
    running = 0
    for n in range(nmax):
        if n < len(poly_coeff):
            running += poly_coeff[n]
        out.append(running)
    return out


def check_tangent_model() -> None:
    g = tangent_coefficients([1], 10)
    d = tangent_coefficients([3, 0, -2], 10)
    d2 = tangent_coefficients([9, 0, -12, 0, 4], 10)

    assert g[:6] == [1, 1, 1, 1, 1, 1]
    assert d[:7] == [3, 3, 1, 1, 1, 1, 1]
    assert d2[:8] == [9, 9, -3, -3, 1, 1, 1, 1]

    print("D. integer tangent model: PASS")
    print("   G  :", g[:8])
    print("   D  :", d[:8])
    print("   D^2:", d2[:8])


# ---------------------------------------------------------------------------
# F. Full-history primal feasibility for fixed explicit roots
# ---------------------------------------------------------------------------

def binomial_root(k: int) -> np.ndarray:
    """R_k(z) = (1+2z)/3 * ((1+z)/2)^k."""
    out = np.zeros(k + 2, dtype=float)
    for j in range(k + 1):
        q = comb(k, j) / (2**k)
        out[j] += q / 3
        out[j + 1] += 2 * q / 3
    return out


def full_history_feasible(root: np.ndarray, depth: int, M: int,
                          time_limit: float | None = None):
    """
    Exact finite-support full-history feasibility LP.

    A feasible result is a genuine positive certificate for the unrestricted
    finite-depth problem.  An infeasible result at one cutoff is NOT used as a
    proof of unrestricted infeasibility.
    """
    N = M + 1
    if len(root) > N:
        raise ValueError("support cutoff smaller than root support")

    nnodes = 2 ** (depth + 1) - 1
    internal = 2**depth - 1
    nvars = nnodes * N

    # normalization + recurrence + fixed-root rows
    nrows = nnodes + internal * (M + 6) + N
    Aeq = lil_matrix((nrows, nvars), dtype=float)
    beq = np.zeros(nrows, dtype=float)

    def vid(node, j):
        return node * N + j

    row = 0

    for node in range(nnodes):
        Aeq[row, node * N:(node + 1) * N] = 1.0
        beq[row] = 1.0
        row += 1

    for node in range(internal):
        low = 2 * node + 1
        high = 2 * node + 2
        for u in range(M + 6):
            if 0 <= u - 3 < N:
                Aeq[row, vid(node, u - 3)] += 3.0
            if 0 <= u - 5 < N:
                Aeq[row, vid(node, u - 5)] += 1.0
            if 0 <= u - 2 < N:
                Aeq[row, vid(low, u - 2)] -= 1.0
            if 0 <= u - 4 < N:
                Aeq[row, vid(high, u - 4)] -= 3.0
            row += 1

    for j in range(N):
        Aeq[row, vid(0, j)] = 1.0
        beq[row] = root[j] if j < len(root) else 0.0
        row += 1

    Aeq = csr_matrix(Aeq[:row])
    beq = beq[:row]

    options = {}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    t0 = time.time()
    res = linprog(
        np.zeros(nvars),
        A_eq=Aeq,
        b_eq=beq,
        bounds=(0, None),
        method="highs",
        options=options,
    )
    elapsed = time.time() - t0

    eqerr = None
    if res.success:
        eqerr = float(np.max(np.abs(Aeq @ res.x - beq)))
        assert eqerr < 1e-8
        assert float(np.min(res.x)) >= -1e-9

    return {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "elapsed": elapsed,
        "eqerr": eqerr,
        "variables": nvars,
        "constraints": row,
        "M": M,
    }


def check_small_binomial_facts(run_deep: bool) -> None:
    # Analytic low-tail facts.
    R1 = [F(1, 6), F(1, 2), F(1, 3)]
    assert R1[0] <= F(1, 3)
    assert R1[0] + R1[1] == F(2, 3)

    # C1(R1) = b0 + 3/4(b1+b2) = 19/24 > 3/4,
    # which is the finite-depth obstruction used for R1 notin K3.
    C1_R1 = R1[0] + F(3, 4) * (R1[1] + R1[2])
    assert C1_R1 == F(19, 24)
    assert C1_R1 > F(3, 4)

    R2_frac = [F(1, 12), F(1, 3), F(5, 12), F(1, 6)]
    assert sum(R2_frac[:3]) == F(5, 6)
    assert F(5, 6) > F(22, 27)

    print("E. analytic binomial boundary checks: PASS")
    print("   C1(R1) = 19/24 > 3/4 => R1 notin K3")
    print("   F_R2(2) = 5/6 > 22/27 => R2 notin K7")

    # Positive full-tree certificates.
    tests = [
        (2, 6, 12, "R2 in K6"),
    ]
    if run_deep:
        tests.append((3, 11, 16, "R3 in K11"))

    for k, depth, M, label in tests:
        root = binomial_root(k)
        ans = full_history_feasible(root, depth, M, time_limit=120 if run_deep else 30)
        print(f"   {label}: success={ans['success']}  M={M}  "
              f"vars={ans['variables']}  rows={ans['constraints']}  "
              f"time={ans['elapsed']:.2f}s")
        if ans["success"]:
            print("      equality residual =", ans["eqerr"])
        else:
            print("      solver message =", ans["message"])
        assert ans["success"], f"positive certificate failed: {label}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also rerun the R3 in K11 full-history feasibility certificate",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("WPB ZERO-REPAIR CHECKPOINT AUDIT")
    print("=" * 88)

    check_exact_lower_certificates()
    check_log_lower_expansion()
    check_delayed_boundary_repair()
    check_tangent_model()
    check_small_binomial_facts(args.deep)

    print("=" * 88)
    print("ALL REQUESTED CHECKS PASSED")
    if not args.deep:
        print("Deep R3/K11 LP not run. Use --deep to rerun it.")
    print("=" * 88)


if __name__ == "__main__":
    main()
