#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WPB_COMMON_ANCHOR_SOLVER.py
===========================

Direct finite-support LP attack on Conjecture 7 from
"The Three-Twentieths Common-Anchor Conjecture".

For the quartic pair

    P = (1/8, 1/4, 1/4, 1/4, 1/8)
    Q = (7/64, 5/16, 5/32, 5/16, 7/64)

and the exact Bernoulli reserve

    R* = (17/20) delta_0 + (3/20) delta_1,

this script computes, at a finite support cutoff M:

  B_{n,M} = minimum root mean over exact depth-n zero-repair trees;
  A_{n,M} = minimum root mean over exact depth-n trees that ALSO admit
            a single common depth-n viable anchor C satisfying

      (Q * B_h) * R* >=_st P * C     for every |h| = n-1.

The finite-cutoff anchor loss is

      eta_{n,M} = A_{n,M} - B_{n,M}.

At the unrestricted level Conjecture 7 says the infimum loss is zero for
all n.  A uniform bounded loss C0 would still imply

      B_{2n} <= B_n + 3/20 + C0

and therefore B_n = O(log n).

IMPORTANT
---------
This is a numerical finite-support attack.  It preserves the FULL 5-ary
history tree and the exact ZR equalities, but constrains every node law to
support {0,...,M}.  HiGHS floating-point feasibility is evidence, not an
exact unrestricted proof.

Dependencies: numpy, scipy

Examples
--------
  python WPB_COMMON_ANCHOR_SOLVER.py --n-max 3 --support 8
  python WPB_COMMON_ANCHOR_SOLVER.py --n 3 --supports 6,8,10,12
  python WPB_COMMON_ANCHOR_SOLVER.py --n 3 --support 10 --save-primal
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

# ---------------------------------------------------------------------------
# Exact data
# ---------------------------------------------------------------------------

P_FR = (Fraction(1, 8), Fraction(1, 4), Fraction(1, 4), Fraction(1, 4), Fraction(1, 8))
Q_FR = (Fraction(7, 64), Fraction(5, 16), Fraction(5, 32), Fraction(5, 16), Fraction(7, 64))
RSTAR_FR = (Fraction(17, 20), Fraction(3, 20))

P = np.array([float(x) for x in P_FR], dtype=float)
Q = np.array([float(x) for x in Q_FR], dtype=float)
RSTAR = np.array([float(x) for x in RSTAR_FR], dtype=float)
ALPHABET = tuple(range(5))
History = Tuple[int, ...]


def frac_convolve(a: Sequence[Fraction], b: Sequence[Fraction]) -> List[Fraction]:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def frac_cdf(a: Sequence[Fraction]) -> List[Fraction]:
    s = Fraction(0)
    out = []
    for x in a:
        s += x
        out.append(s)
    return out


def exact_self_audit() -> Dict[str, str]:
    assert sum(P_FR) == 1 and sum(Q_FR) == 1 and sum(RSTAR_FR) == 1
    assert [P_FR[i] - Q_FR[i] for i in range(5)] == [
        Fraction(1, 64), Fraction(-4, 64), Fraction(6, 64),
        Fraction(-4, 64), Fraction(1, 64)
    ]
    mp = sum(Fraction(i) * P_FR[i] for i in range(5))
    mq = sum(Fraction(i) * Q_FR[i] for i in range(5))
    assert mp == mq == 2
    assert RSTAR_FR[1] == Fraction(3, 20)

    qr = frac_convolve(Q_FR, RSTAR_FR)
    pp = list(P_FR) + [Fraction(0)] * (len(qr) - len(P_FR))
    assert all(a <= b for a, b in zip(frac_cdf(qr), frac_cdf(pp)))
    eps = Fraction(3, 20)
    assert -3 + 20 * eps == 0
    return {
        "quartic_identity": "PASS",
        "equal_mean": str(mp),
        "Rstar_mean": str(RSTAR_FR[1]),
        "Q*Rstar_stoch_dominates_P": "PASS",
        "sharp_Bernoulli_threshold": str(eps),
    }


# ---------------------------------------------------------------------------
# Full history tree
# ---------------------------------------------------------------------------

def histories_at_depth(d: int) -> List[History]:
    return [()] if d == 0 else list(product(ALPHABET, repeat=d))


def histories_by_depth(n: int) -> List[List[History]]:
    return [histories_at_depth(d) for d in range(n + 1)]


def node_count(n: int) -> int:
    return sum(5 ** d for d in range(n + 1))


def estimate_problem_size(n: int, M: int, anchor: bool) -> Dict[str, int]:
    nodes = node_count(n)
    internal = sum(5 ** d for d in range(n))
    mult = 2 if anchor else 1
    return {
        "variables": mult * nodes * (M + 1),
        "equalities": mult * (nodes + internal * (M + 5)),
        "inequalities": (5 ** (n - 1)) * (M + 6) if anchor and n >= 1 else 0,
    }


class SparseRows:
    def __init__(self):
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.data: List[float] = []
        self.rhs: List[float] = []
        self.names: List[str] = []

    def add(self, coeffs: Iterable[Tuple[int, float]], rhs: float, name: str):
        r = len(self.rhs)
        for c, v in coeffs:
            if v != 0.0:
                self.rows.append(r)
                self.cols.append(c)
                self.data.append(float(v))
        self.rhs.append(float(rhs))
        self.names.append(name)

    def matrix(self, ncols: int) -> csr_matrix:
        return coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(len(self.rhs), ncols),
            dtype=float,
        ).tocsr()

    def vector(self) -> np.ndarray:
        return np.asarray(self.rhs, dtype=float)


class VarIndex:
    def __init__(self, n: int, M: int, trees: Sequence[str]):
        self.n, self.M = n, M
        self.h_by_d = histories_by_depth(n)
        self.map: Dict[Tuple[str, History, int], int] = {}
        self.rev: List[Tuple[str, History, int]] = []
        k = 0
        for tree in trees:
            for d in range(n + 1):
                for h in self.h_by_d[d]:
                    for j in range(M + 1):
                        self.map[(tree, h, j)] = k
                        self.rev.append((tree, h, j))
                        k += 1
        self.nvars = k

    def __call__(self, tree: str, h: History, j: int) -> int:
        return self.map[(tree, h, j)]


def add_tree_constraints(eq: SparseRows, vidx: VarIndex, tree: str, n: int, M: int):
    # Every node is a probability law.
    for d in range(n + 1):
        for h in vidx.h_by_d[d]:
            eq.add(((vidx(tree, h, j), 1.0) for j in range(M + 1)), 1.0,
                   f"{tree}:norm:{h}")

    # Exact coefficient form of Q*B_h = sum_x P_x delta_x * B_hx.
    for d in range(n):
        for h in vidx.h_by_d[d]:
            for u in range(M + 5):
                coeffs: List[Tuple[int, float]] = []
                for y, qy in enumerate(Q):
                    j = u - y
                    if 0 <= j <= M:
                        coeffs.append((vidx(tree, h, j), qy))
                for x, px in enumerate(P):
                    j = u - x
                    if 0 <= j <= M:
                        coeffs.append((vidx(tree, h + (x,), j), -px))
                eq.add(coeffs, 0.0, f"{tree}:ZR:{h}:u={u}")


def cdf_shift_weights(base: np.ndarray, M: int, t: int) -> np.ndarray:
    cdf = np.cumsum(base)
    w = np.zeros(M + 1)
    for j in range(M + 1):
        r = t - j
        if r < 0:
            w[j] = 0.0
        elif r >= len(cdf):
            w[j] = 1.0
        else:
            w[j] = cdf[r]
    return w


def add_anchor_constraints(ub: SparseRows, vidx: VarIndex, n: int, M: int, mode: str):
    if n < 1:
        return
    croot = ()
    last_internal = vidx.h_by_d[n - 1]

    if mode == "insured":
        qr = np.convolve(Q, RSTAR)
        # mu >=_st nu  <=>  F_mu(t) <= F_nu(t)
        # mu=(Q*B_h)*R*, nu=P*C.
        for h in last_internal:
            for t in range(M + 6):
                wl = cdf_shift_weights(qr, M, t)
                wr = cdf_shift_weights(P, M, t)
                coeffs: List[Tuple[int, float]] = []
                for j in range(M + 1):
                    if wl[j] != 0.0:
                        coeffs.append((vidx("B", h, j), wl[j]))
                    if wr[j] != 0.0:
                        coeffs.append((vidx("C", croot, j), -wr[j]))
                ub.add(coeffs, 0.0, f"anchor:insured:{h}:t={t}")

    elif mode == "floor":
        # Stronger sufficient condition B_h >=_st C.
        for h in last_internal:
            for t in range(M + 1):
                coeffs: List[Tuple[int, float]] = []
                for j in range(t + 1):
                    coeffs.append((vidx("B", h, j), 1.0))
                    coeffs.append((vidx("C", croot, j), -1.0))
                ub.add(coeffs, 0.0, f"anchor:floor:{h}:t={t}")
    else:
        raise ValueError(mode)


@dataclass
class SolveResult:
    success: bool
    status: int
    message: str
    objective: Optional[float]
    runtime: float
    n: int
    M: int
    anchor: bool
    nvars: int
    neq: int
    nineq: int
    eq_resid_max: Optional[float] = None
    ub_violation_max: Optional[float] = None
    root: Optional[np.ndarray] = None
    anchor_root: Optional[np.ndarray] = None
    res: object = None
    vidx: Optional[VarIndex] = None
    eq_names: Optional[List[str]] = None
    ub_names: Optional[List[str]] = None


def solve_lp(n: int, M: int, anchor: bool, anchor_mode: str,
             presolve: bool = True, time_limit: Optional[float] = None) -> SolveResult:
    trees = ("B", "C") if anchor else ("B",)
    vidx = VarIndex(n, M, trees)
    eq = SparseRows()
    add_tree_constraints(eq, vidx, "B", n, M)
    if anchor:
        add_tree_constraints(eq, vidx, "C", n, M)

    ub = SparseRows()
    if anchor:
        add_anchor_constraints(ub, vidx, n, M, anchor_mode)

    Aeq, beq = eq.matrix(vidx.nvars), eq.vector()
    Aub = ub.matrix(vidx.nvars) if ub.rhs else None
    bub = ub.vector() if ub.rhs else None

    c = np.zeros(vidx.nvars)
    for j in range(M + 1):
        c[vidx("B", (), j)] = j

    options = {"presolve": presolve}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    t0 = time.time()
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                  bounds=(0.0, None), method="highs", options=options)
    runtime = time.time() - t0

    out = SolveResult(bool(res.success), int(res.status), str(res.message),
                      float(res.fun) if res.success else None, runtime, n, M,
                      anchor, vidx.nvars, Aeq.shape[0], 0 if Aub is None else Aub.shape[0],
                      res=res, vidx=vidx, eq_names=eq.names, ub_names=ub.names)

    if res.x is not None:
        out.eq_resid_max = float(np.max(np.abs(Aeq @ res.x - beq)))
        out.ub_violation_max = 0.0 if Aub is None else float(max(0.0, np.max(Aub @ res.x - bub)))
    if res.success:
        out.root = np.array([res.x[vidx("B", (), j)] for j in range(M + 1)])
        if anchor:
            out.anchor_root = np.array([res.x[vidx("C", (), j)] for j in range(M + 1)])
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def trim_law(a: Optional[np.ndarray], tol: float = 1e-10) -> List[float]:
    if a is None:
        return []
    k = len(a)
    while k > 1 and abs(a[k - 1]) <= tol:
        k -= 1
    return [float(x) for x in a[:k]]


def rational_guess(a: Optional[np.ndarray], max_den: int = 1_000_000) -> List[str]:
    return [str(Fraction(x).limit_denominator(max_den)) for x in trim_law(a)] if a is not None else []


def dual_summary(sol: SolveResult, top_k: int = 25) -> Dict:
    if sol.res is None:
        return {}
    out = {}
    try:
        d = np.asarray(sol.res.eqlin.marginals)
        ids = np.argsort(-np.abs(d))[:top_k]
        out["largest_equality_duals"] = [
            {"row": int(i), "name": sol.eq_names[int(i)], "dual": float(d[i])}
            for i in ids if abs(d[i]) > 1e-12
        ]
    except Exception:
        pass
    if sol.anchor:
        try:
            d = np.asarray(sol.res.ineqlin.marginals)
            ids = np.argsort(-np.abs(d))[:top_k]
            out["largest_anchor_duals"] = [
                {"row": int(i), "name": sol.ub_names[int(i)], "dual": float(d[i])}
                for i in ids if abs(d[i]) > 1e-12
            ]
        except Exception:
            pass
    return out


def active_anchor(sol: SolveResult, tol: float = 1e-8, top_k: int = 50) -> List[Dict]:
    if not (sol.anchor and sol.success):
        return []
    try:
        slack = np.asarray(sol.res.ineqlin.residual)
        dual = np.asarray(sol.res.ineqlin.marginals)
    except Exception:
        return []
    ids = np.where(slack <= tol)[0]
    ids = sorted(ids, key=lambda i: -abs(dual[i]))[:top_k]
    return [{"row": int(i), "name": sol.ub_names[int(i)],
             "slack": float(slack[i]), "dual": float(dual[i])} for i in ids]


def canonical_shape(a: np.ndarray, tol: float = 1e-9, digits: int = 10) -> Tuple[float, ...]:
    ids = np.where(a > tol)[0]
    if not len(ids):
        return tuple()
    a = a[int(ids[0]):]
    while len(a) > 1 and a[-1] <= tol:
        a = a[:-1]
    return tuple(np.round(a, digits))


def shape_counts(sol: SolveResult, tree: str) -> Dict:
    if not sol.success:
        return {}
    out = {}
    for d in range(sol.n + 1):
        counts: Dict[Tuple[float, ...], int] = {}
        ex: Dict[Tuple[float, ...], History] = {}
        for h in sol.vidx.h_by_d[d]:
            a = np.array([sol.res.x[sol.vidx(tree, h, j)] for j in range(sol.M + 1)])
            k = canonical_shape(a)
            counts[k] = counts.get(k, 0) + 1
            ex.setdefault(k, h)
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        out[str(d)] = {
            "distinct_shift_canonical_shapes": len(ranked),
            "top_shapes": [
                {"count": int(cnt), "example_history": list(ex[k]), "law": list(k)}
                for k, cnt in ranked[:20]
            ],
        }
    return out


def result_dict(sol: SolveResult, patterns: bool) -> Dict:
    d = {
        "success": sol.success, "status": sol.status, "message": sol.message,
        "objective": sol.objective, "runtime_sec": sol.runtime,
        "n": sol.n, "M": sol.M, "anchor": sol.anchor,
        "nvars": sol.nvars, "neq": sol.neq, "nineq": sol.nineq,
        "eq_resid_max": sol.eq_resid_max, "ub_violation_max": sol.ub_violation_max,
        "root_law": trim_law(sol.root), "root_law_rational_guess": rational_guess(sol.root),
        "dual_summary": dual_summary(sol),
    }
    if sol.anchor:
        d["anchor_root_law"] = trim_law(sol.anchor_root)
        d["anchor_root_law_rational_guess"] = rational_guess(sol.anchor_root)
        d["active_anchor_constraints"] = active_anchor(sol)
    if patterns and sol.success:
        d["B_shape_counts"] = shape_counts(sol, "B")
        if sol.anchor:
            d["C_shape_counts"] = shape_counts(sol, "C")
    return d


def export_primal(sol: SolveResult, path: Path):
    if sol.success:
        np.savez_compressed(path, x=np.asarray(sol.res.x), objective=sol.objective,
                            n=sol.n, M=sol.M, anchor=int(sol.anchor))


def empirical_interpretation(rows: List[Dict], eta_tol: float) -> Dict:
    good = [r for r in rows if r["B_success"] and r["A_success"] and r["eta"] is not None]
    if not good:
        return {}
    etas = [max(0.0, float(r["eta"])) for r in good]
    C0 = max(etas)
    Cstep = 3.0 / 20.0 + C0
    return {
        "max_tested_eta": C0,
        "all_tested_eta_within_tolerance": all(abs(float(r["eta"])) <= eta_tol for r in good),
        "empirical_dyadic_step_constant": Cstep,
        "empirical_ln_coefficient": Cstep / math.log(2.0),
        "candidate_recurrence": f"B_(2n) <= B_n + {Cstep:.12g}",
        "candidate_asymptotic": f"B_n <= O(1) + {(Cstep/math.log(2.0)):.12g} ln(n)",
        "warning": "Empirical finite-support extrapolation only; uniformity in n and removal of M remain unproved.",
    }


def print_sol(tag: str, s: SolveResult):
    if s.success:
        print(f"{tag:<11} n={s.n:<2} M={s.M:<3} obj={s.objective:.12g} "
              f"eq={s.eq_resid_max:.2e} ub={s.ub_violation_max:.2e} time={s.runtime:.2f}s")
    else:
        print(f"{tag:<11} n={s.n:<2} M={s.M:<3} FAILED status={s.status} time={s.runtime:.2f}s :: {s.message}")


def parse_supports(args) -> List[int]:
    if args.supports:
        return sorted(set(int(x) for x in args.supports.split(",") if x.strip()))
    return [args.support]


def run(args):
    audit = exact_self_audit()
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ns = [args.n] if args.n is not None else list(range(args.n_min, args.n_max + 1))
    supports = parse_supports(args)

    print("=" * 88)
    print("THREE-TWENTIETHS COMMON-ANCHOR SOLVER")
    print("=" * 88)
    for k, v in audit.items():
        print(f"audit {k}: {v}")
    print(f"depths={ns} supports={supports} anchor_mode={args.anchor_mode}")

    rows, details = [], {}
    for M in supports:
        for n in ns:
            print("-" * 88)
            sb = estimate_problem_size(n, M, False)
            sa = estimate_problem_size(n, M, True)
            print(f"size n={n} M={M} | B vars={sb['variables']:,} eq={sb['equalities']:,} | "
                  f"A vars={sa['variables']:,} eq={sa['equalities']:,} ineq={sa['inequalities']:,}")

            B = solve_lp(n, M, False, args.anchor_mode, not args.no_presolve, args.time_limit)
            A = solve_lp(n, M, True, args.anchor_mode, not args.no_presolve, args.time_limit)
            print_sol("B_unconstr", B)
            print_sol("A_anchor", A)
            eta = A.objective - B.objective if A.success and B.success else None
            if eta is not None:
                print(f"eta=A-B    n={n:<2} M={M:<3} eta={eta:.12g}")

            row = {
                "n": n, "M": M, "B_success": B.success, "A_success": A.success,
                "B": B.objective, "A": A.objective, "eta": eta,
                "B_runtime_sec": B.runtime, "A_runtime_sec": A.runtime,
                "B_eq_resid_max": B.eq_resid_max, "A_eq_resid_max": A.eq_resid_max,
                "A_ub_violation_max": A.ub_violation_max,
            }
            rows.append(row)
            key = f"n{n}_M{M}"
            details[key] = {
                "unconstrained": result_dict(B, not args.no_pattern_analysis),
                "common_anchor": result_dict(A, not args.no_pattern_analysis),
                "eta": eta,
            }
            if args.save_primal:
                export_primal(B, outdir / f"{key}_B_primal.npz")
                export_primal(A, outdir / f"{key}_A_primal.npz")

    interp = empirical_interpretation(rows, args.eta_tol)

    csv_path = outdir / "common_anchor_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    json_path = outdir / "common_anchor_detailed.json"
    json_path.write_text(json.dumps({
        "problem": {
            "P": [str(x) for x in P_FR], "Q": [str(x) for x in Q_FR],
            "Rstar": [str(x) for x in RSTAR_FR], "Rstar_mean": "3/20",
            "finite_support_warning": "Every node law is restricted to support {0,...,M}.",
        },
        "self_audit": audit,
        "runs": details,
        "upper_bound_interpretation": interp,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("UPPER-BOUND INTERPRETATION")
    print("=" * 88)
    if interp:
        print(f"max tested eta       = {interp['max_tested_eta']:.12g}")
        print(f"all eta ~ 0          = {interp['all_tested_eta_within_tolerance']} (tol={args.eta_tol:g})")
        print(f"empirical dyadic step= {interp['empirical_dyadic_step_constant']:.12g}")
        print(f"candidate ln coeff   = {interp['empirical_ln_coefficient']:.12g}")
        print(interp["candidate_recurrence"])
        print(interp["candidate_asymptotic"])
        print("WARNING:", interp["warning"])
    else:
        print("No successful paired runs.")
    print("saved", csv_path)
    print("saved", json_path)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Direct LP test of the Three-Twentieths Common-Anchor Conjecture")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--n", type=int, default=None)
    g.add_argument("--n-max", type=int, default=3)
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--support", type=int, default=8)
    p.add_argument("--supports", type=str, default=None, help="e.g. 6,8,10,12")
    p.add_argument("--anchor-mode", choices=("insured", "floor"), default="insured")
    p.add_argument("--out-dir", default="common_anchor_out")
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--eta-tol", type=float, default=1e-8)
    p.add_argument("--no-presolve", action="store_true")
    p.add_argument("--save-primal", action="store_true")
    p.add_argument("--no-pattern-analysis", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.n is None and args.n_min > args.n_max:
        raise SystemExit("--n-min must be <= --n-max")
    run(args)


if __name__ == "__main__":
    main()
