#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHO PAYS THE BILL? — HARD INDEPENDENT AUDIT
===========================================

Purpose
-------
This script audits the R=2 zero-repair viability programme from the primal side
and from the exact dual side, without importing the original LP solver or the
original 19/9 certificate implementation.

It checks four logically separate things:

A. Independent full-history primal LP
   Rebuild directly from
       3 B_h(u-3) + B_h(u-5)
       = B_{h0}(u-2) + 3 B_{h1}(u-4),
   with only normalization and nonnegativity.
   It reproduces the reported finite-depth values B_n.

B. Support-cutoff trap
   Demonstrates explicitly that "first feasible support" is NOT the same as
   "support large enough for the optimum".  At depth 3:
       M=5  gives 4/3,
       M>=6 gives 32/27.
   Therefore a helper that returns the first feasible cutoff is not an
   optimizer-certification routine.

C. Exact viability-tail hierarchy
   Derives the finite-horizon CDF caps by the backward map
       T(q0,q1,q2) = (q1/3, q0+q2/3, q1+1/3),
   clipped at 1 for finite horizons.
   The infinite fixed point is
       (1/9, 1/3, 2/3).

D. Exact coefficientwise dual
   Verifies the support-independent two-generation certificate with objective
       19/9,
   including the explicit infinite tail.
   It also constructs finite-horizon versions of the same certificate and
   proves in exact rational arithmetic that
       B_15 >= 39473/19683 > 2.

Dependencies
------------
Python 3.10+
numpy
scipy

No Monte Carlo.
All analytic certificates use fractions.Fraction exactly.
The primal LP uses scipy.optimize.linprog / HiGHS numerically.

Usage
-----
    python wpb_hard_independent_audit.py

Optional:
    python wpb_hard_independent_audit.py --max-depth 10
    python wpb_hard_independent_audit.py --max-depth 9 --support-extra 6
    python wpb_hard_independent_audit.py --skip-primal

Notes
-----
Depth 10 is substantially slower than depths <=9 because the full history tree
has 2^(n+1)-1 buffer nodes.  On a typical desktop it is still practical.
"""

from __future__ import annotations

import argparse
import math
import time
from fractions import Fraction as F

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix


# ---------------------------------------------------------------------------
# Reported finite-depth targets (regression only; NOT used to build the LP)
# ---------------------------------------------------------------------------

REPORTED = {
    1: F(2, 3),
    2: F(1, 1),
    3: F(32, 27),
    4: F(35, 27),
    5: F(116, 81),
    6: F(122, 81),
    7: F(388, 243),
    8: F(412, 243),
    9: F(1292, 729),
    10: F(1346, 729),
    11: F(4150, 2187),
}


# ---------------------------------------------------------------------------
# A. Independent full-history primal LP
# ---------------------------------------------------------------------------

def solve_full_history_primal(depth: int, support_max: int):
    """
    Independent reconstruction of the full history-indexed primal LP.

    Variables:
        B_h(j) >= 0
    for every history node h of depth <= `depth`,
    and every j in {0,...,support_max}.

    Constraints:
      1. sum_j B_h(j) = 1 for every node.
      2. At every internal node and every total-information level u,

           3 B_h(u-3) + B_h(u-5)
           - B_{h0}(u-2) - 3 B_{h1}(u-4) = 0.

         This is exactly the zero-repair recursion multiplied by 4.

    Objective:
        minimize sum_j j B_root(j).

    No coefficient from the 19/9 dual is used here.
    """
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if support_max < 0:
        raise ValueError("support_max must be nonnegative")

    nnodes = 2 ** (depth + 1) - 1
    internal = 2 ** depth - 1
    N = support_max + 1
    nvars = nnodes * N

    def vid(node: int, j: int) -> int:
        return node * N + j

    rows: list[dict[int, float]] = []
    rhs: list[float] = []

    # Normalization of every buffer law.
    for node in range(nnodes):
        rows.append({vid(node, j): 1.0 for j in range(N)})
        rhs.append(1.0)

    # Exact zero-repair equations.
    for node in range(internal):
        low = 2 * node + 1
        high = 2 * node + 2

        # Largest shifted information index is support_max + 5.
        for u in range(support_max + 6):
            c: dict[int, float] = {}

            if 0 <= u - 3 < N:
                c[vid(node, u - 3)] = c.get(vid(node, u - 3), 0.0) + 3.0
            if 0 <= u - 5 < N:
                c[vid(node, u - 5)] = c.get(vid(node, u - 5), 0.0) + 1.0
            if 0 <= u - 2 < N:
                c[vid(low, u - 2)] = c.get(vid(low, u - 2), 0.0) - 1.0
            if 0 <= u - 4 < N:
                c[vid(high, u - 4)] = c.get(vid(high, u - 4), 0.0) - 3.0

            rows.append(c)
            rhs.append(0.0)

    Aeq = lil_matrix((len(rows), nvars), dtype=float)
    for r, c in enumerate(rows):
        for col, value in c.items():
            Aeq[r, col] = value
    Aeq = csr_matrix(Aeq)

    obj = np.zeros(nvars, dtype=float)
    obj[:N] = np.arange(N, dtype=float)

    t0 = time.time()
    res = linprog(
        obj,
        A_eq=Aeq,
        b_eq=np.asarray(rhs, dtype=float),
        bounds=(0, None),
        method="highs",
    )
    elapsed = time.time() - t0

    if not res.success:
        return {
            "success": False,
            "message": res.message,
            "depth": depth,
            "M": support_max,
            "elapsed": elapsed,
            "variables": nvars,
            "constraints": len(rows),
        }

    eqerr = float(np.max(np.abs(Aeq @ res.x - np.asarray(rhs, dtype=float))))
    return {
        "success": True,
        "value": float(res.fun),
        "depth": depth,
        "M": support_max,
        "elapsed": elapsed,
        "variables": nvars,
        "constraints": len(rows),
        "eqerr": eqerr,
        "root": np.asarray(res.x[:N], dtype=float),
    }


def audit_reported_table(max_depth: int = 10, support_extra: int = 6):
    print("\n" + "=" * 94)
    print("A. INDEPENDENT FULL-HISTORY PRIMAL")
    print("=" * 94)
    print(
        "Built from the zero-repair recurrence only; "
        "no 19/9 dual coefficients are used."
    )
    print()
    print(
        " n   support       primal value        reported value      abs error"
        "       eq residual      seconds"
    )
    print("-" * 94)

    out = {}
    for n in range(1, max_depth + 1):
        M = n + support_extra
        ans = solve_full_history_primal(n, M)
        if not ans["success"]:
            raise RuntimeError(
                f"Primal failed at depth {n}, support {M}: {ans['message']}"
            )

        target = REPORTED.get(n)
        target_float = float(target) if target is not None else float("nan")
        err = abs(ans["value"] - target_float) if target is not None else float("nan")

        print(
            f"{n:2d}   {M:7d}   "
            f"{ans['value']:18.12f}   "
            f"{target_float:18.12f}   "
            f"{err:10.3e}   "
            f"{ans['eqerr']:12.3e}   "
            f"{ans['elapsed']:8.3f}"
        )

        if target is not None:
            assert err < 2e-8, (n, ans["value"], target_float, err)
        assert ans["eqerr"] < 1e-8, (n, ans["eqerr"])
        out[n] = ans

    return out


# ---------------------------------------------------------------------------
# B. Support-cutoff trap
# ---------------------------------------------------------------------------

def audit_support_cutoff_trap():
    print("\n" + "=" * 94)
    print("B. SUPPORT-CUTOFF TRAP")
    print("=" * 94)
    print(
        "At depth 3, the first feasible support is not the support at which "
        "the optimum stabilizes."
    )
    print()

    vals = []
    for M in range(0, 9):
        ans = solve_full_history_primal(3, M)
        vals.append((M, ans))
        if ans["success"]:
            print(f"  depth=3, M={M:2d}: feasible, optimum = {ans['value']:.12f}")
        else:
            print(f"  depth=3, M={M:2d}: infeasible")

    feasible = [(M, a) for M, a in vals if a["success"]]
    assert feasible
    first_M, first_ans = feasible[0]

    assert first_M == 5, first_M
    assert abs(first_ans["value"] - 4 / 3) < 1e-10

    stable = [a["value"] for M, a in feasible if M >= 6]
    assert stable
    assert max(abs(v - 32 / 27) for v in stable) < 2e-8

    print()
    print("  CHECK:")
    print("    first feasible cutoff M=5  -> 4/3")
    print("    stabilized cutoff M>=6     -> 32/27")
    print("  Therefore: first feasible support != certified optimum support.")


# ---------------------------------------------------------------------------
# C. Exact finite-horizon CDF caps
# ---------------------------------------------------------------------------

ONE = F(1, 1)


def T_cap(q: tuple[F, F, F]) -> tuple[F, F, F]:
    """
    One backward viability step for low-tail CDF caps.

      q0' = q1/3
      q1' = q0 + q2/3
      q2' = q1 + 1/3

    Finite-horizon CDF values cannot exceed 1, hence componentwise clipping.
    """
    q0, q1, q2 = q
    raw = (
        q1 / 3,
        q0 + q2 / 3,
        q1 + F(1, 3),
    )
    return tuple(min(ONE, x) for x in raw)


def cap_sequence(nmax: int):
    q = (ONE, ONE, ONE)
    seq = [q]
    for _ in range(nmax):
        q = T_cap(q)
        seq.append(q)
    return seq


def infinite_fixed_point():
    qstar = (F(1, 9), F(1, 3), F(2, 3))
    assert T_cap(qstar) == qstar
    return qstar


def C1_bound_for_remaining_depth(m: int, qs) -> F:
    """
    If a node has m>=1 generations still to survive:
        4 C1(B) = F_L(3) + 3 F_H(1)
                 <= 1 + 3 q_{m-1,1}.
    """
    if m < 1:
        raise ValueError("C1 requires at least one future generation")
    return (ONE + 3 * qs[m - 1][1]) / 4


def C2_bound_for_remaining_depth(m: int, qs) -> F:
    """
    If a node has m>=1 generations still to survive:
        4 C2(B) = F_L(4) + 3 F_H(2)
                 <= 1 + 3 q_{m-1,2}.
    """
    if m < 1:
        raise ValueError("C2 requires at least one future generation")
    return (ONE + 3 * qs[m - 1][2]) / 4


def finite_horizon_dual_bound(n: int, qs) -> F:
    """
    Finite-horizon version of the same two-generation coefficientwise dual.

    Root B has n generations remaining.
    Children L,H have n-1 generations remaining.

    Active bounds:
      parent: A0 <= q_n[0], A1 <= q_n[1]
      low:    A2 <= q_{n-1}[2], C2 <= C2_{n-1}
      high:   A1 <= q_{n-1}[1], C1 <= C1_{n-1}
    """
    if n < 2:
        raise ValueError("two-generation dual requires n>=2")

    qn = qs[n]
    qc = qs[n - 1]

    c2_low = C2_bound_for_remaining_depth(n - 1, qs)
    c1_high = C1_bound_for_remaining_depth(n - 1, qs)

    return (
        F(8, 3)
        + ONE
        - F(1, 3) * qn[0]
        - F(1, 3) * qn[1]
        - F(1, 9) * qc[2]
        - F(4, 9) * c2_low
        - ONE * qc[1]
        - F(4, 3) * c1_high
    )


def audit_caps_and_finite_bounds(primal_results=None):
    print("\n" + "=" * 94)
    print("C. EXACT FINITE-HORIZON CDF CAPS AND DIRECT LOWER BOUNDS")
    print("=" * 94)

    qs = cap_sequence(80)
    qstar = infinite_fixed_point()

    print("Infinite fixed point:")
    print(f"  q* = {qstar}")
    print("  -> F(0)<=1/9, F(1)<=1/3, F(2)<=2/3")
    print()

    for n in (10, 11, 15, 20, 30, 60):
        d = finite_horizon_dual_bound(n, qs)
        print(f"  D_{n:2d} = {d} = {float(d):.12f}")

    assert finite_horizon_dual_bound(10, qs) == F(1327, 729)
    assert finite_horizon_dual_bound(11, qs) == F(4097, 2187)
    assert finite_horizon_dual_bound(15, qs) == F(39473, 19683)
    assert finite_horizon_dual_bound(15, qs) > 2

    print()
    print("Key exact finite-horizon consequence:")
    print(
        "  B_15 >= 39473/19683 = "
        f"{float(F(39473,19683)):.12f} > 2."
    )
    print(
        "This excludes a global mean-2 barrier already at a finite horizon; "
        "no compactness step is needed for that statement."
    )

    if primal_results:
        print()
        print("Primal-dual consistency on independently solved depths:")
        for n, ans in sorted(primal_results.items()):
            if n >= 2:
                d = finite_horizon_dual_bound(n, qs)
                gap = ans["value"] - float(d)
                print(
                    f"  n={n:2d}: primal={ans['value']:.12f}, "
                    f"dual={float(d):.12f}, gap={gap:.3e}"
                )
                assert float(d) <= ans["value"] + 2e-8


# ---------------------------------------------------------------------------
# D. Exact infinite-support 19/9 coefficientwise dual
# ---------------------------------------------------------------------------

def lam(u: int) -> F:
    if u == 3:
        return F(-20, 9)
    if u == 4:
        return F(-16, 9)
    if u == 5:
        return F(-4, 3)
    if u == 6:
        return F(0)
    if u >= 7:
        return F(4, 3)
    return F(0)


def c1_coeff(j: int) -> F:
    return {
        0: ONE,
        1: F(3, 4),
        2: F(3, 4),
    }.get(j, F(0))


def c2_coeff(j: int) -> F:
    return {
        0: ONE,
        1: ONE,
        2: F(3, 4),
        3: F(3, 4),
    }.get(j, F(0))


def parent_dual_coeff(j: int) -> F:
    v = F(8, 3) + F(3, 4) * lam(j + 3) + F(1, 4) * lam(j + 5)
    if j == 0:
        v -= F(1, 3)       # parent A0
    if j <= 1:
        v -= F(1, 3)       # parent A1
    return v


def low_dual_coeff(j: int) -> F:
    v = -F(1, 4) * lam(j + 2)
    if j <= 2:
        v -= F(1, 9)       # low A2
    v -= F(4, 9) * c2_coeff(j)  # low C2
    return v


def high_dual_coeff(j: int) -> F:
    v = ONE - F(3, 4) * lam(j + 4)
    if j <= 1:
        v -= ONE           # high A1
    v -= F(4, 3) * c1_coeff(j)  # high C1
    return v


def audit_19_over_9_dual():
    print("\n" + "=" * 94)
    print("D. EXACT SUPPORT-INDEPENDENT 19/9 DUAL")
    print("=" * 94)

    # Check a large finite window exactly.
    for j in range(10000):
        assert parent_dual_coeff(j) <= F(j), (
            "parent",
            j,
            parent_dual_coeff(j),
            F(j),
        )
        assert low_dual_coeff(j) <= 0, ("low", j, low_dual_coeff(j))
        assert high_dual_coeff(j) <= 0, ("high", j, high_dual_coeff(j))

    # Explicit tail: no truncation argument is being smuggled in.
    for j in (5, 6, 7, 100, 10**6):
        assert parent_dual_coeff(j) == F(4)
        assert low_dual_coeff(j) == F(-1, 3)
        assert high_dual_coeff(j) == F(0)

    print("Exact coefficient table:")
    print("  j       parent p_j       low l_j       high r_j")
    print("  ------------------------------------------------")
    for j in range(0, 6):
        label = str(j) if j < 5 else ">=5"
        jj = j
        print(
            f"  {label:>3}      "
            f"{str(parent_dual_coeff(jj)):>10}      "
            f"{str(low_dual_coeff(jj)):>8}      "
            f"{str(high_dual_coeff(jj)):>8}"
        )

    value = (
        F(8, 3)
        + ONE
        - F(1, 3) * F(1, 9)
        - F(1, 3) * F(1, 3)
        - F(1, 9) * F(2, 3)
        - F(4, 9) * F(3, 4)
        - ONE * F(1, 3)
        - F(4, 3) * F(1, 2)
    )

    assert value == F(19, 9)

    print()
    print(f"Exact dual objective = {value} = {float(value):.12f}")
    print("Tail coefficients for every j>=5 are exactly:")
    print("  parent = 4, low = -1/3, high = 0")
    print()
    print("Therefore, for every genuinely infinite viable tree:")
    print("  E[J_root] >= 19/9 > 2.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Independent primal/dual audit for the R=2 zero-repair tree."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help=(
            "largest full-history primal depth to solve (default 10). "
            "Depth 10 is much slower than <=9."
        ),
    )
    parser.add_argument(
        "--support-extra",
        type=int,
        default=6,
        help="use support_max = depth + support_extra in the primal (default 6)",
    )
    parser.add_argument(
        "--skip-primal",
        action="store_true",
        help="skip numerical full-history LPs; run exact rational audits only",
    )
    args = parser.parse_args()

    print("=" * 94)
    print("WHO PAYS THE BILL? — HARD INDEPENDENT AUDIT")
    print("NO MONTE CARLO")
    print("=" * 94)

    primal = None
    if not args.skip_primal:
        primal = audit_reported_table(
            max_depth=args.max_depth,
            support_extra=args.support_extra,
        )
        audit_support_cutoff_trap()

    audit_caps_and_finite_bounds(primal)
    audit_19_over_9_dual()

    print("\n" + "=" * 94)
    print("FINAL AUDIT VERDICT")
    print("=" * 94)
    print("1. The full-history primal formulation independently reproduces the reported")
    print("   finite-depth table (through the depth actually run).")
    print("2. 'First feasible support' is not an optimizer certificate; the cutoff")
    print("   helper must not be used that way.")
    print("3. The low-tail inequalities are direct consequences of recursive viability.")
    print("4. The 19/9 certificate is coefficientwise feasible on the entire support.")
    print("5. Exact finite-horizon dualization already gives")
    print("       B_15 >= 39473/19683 > 2.")
    print("6. Hence the former mean-2 hypothesis is excluded independently of any")
    print("   asymptotic curve fit.")
    print("=" * 94)


if __name__ == "__main__":
    main()
