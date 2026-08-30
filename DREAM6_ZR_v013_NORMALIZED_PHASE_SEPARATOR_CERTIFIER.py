#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6-ZR v0.13 — NORMALIZED PHASE SEPARATOR CERTIFIER
=======================================================

Mission
-------
Turn the depth-3, s=1 normalized-phase obstruction into the smallest possible
load-bearing certificate.

The candidate obstruction is a rational threshold q=a/b separating two
histories h+ and h-:

    G_{h+}(x|s) > q > G_{h-}(x|s)

on the ENTIRE numerical optimal face.

Because G=W/R and R>0, division is unnecessary.  The two statements are

    b W_{h+}(s,x) - a R_{h+}(s) > 0,                 (S+)
    a R_{h-}(s) - b W_{h-}(s,x) > 0.                 (S-)

For q=1/3 this becomes

    3 W_{h+} - R_{h+} > 0,
    R_{h-} - 3 W_{h-} > 0.

v0.13 attacks the claim in TWO independent LP formulations.

A. FACE-MIN CERTIFICATE
   Freeze the numerical optimal face c^T z=v* and minimize S+ and S- directly.
   Positive minima certify the signs on that numerical face.

B. CROSSING-PENALTY CERTIFICATE
   Do NOT freeze the optimal face.  Instead solve

       v_cross,+ = min c^T z  subject to F and S+ <= 0,
       v_cross,- = min c^T z  subject to F and S- <= 0.

   If v_cross,+ > v* and v_cross,- > v*, then any solution crossing either
   rational threshold pays a strictly positive objective penalty.  Therefore
   no true optimizer of the finite LP can cross either side.

The crossing formulation is especially useful because it avoids deriving the
obstruction from a floating equality c^T z=v*.  It asks directly how much
objective value must be sacrificed before the forbidden half-space can even
be reached.

AUTO mode
---------
The script can reconstruct the candidate separator from the optimal-face
R/W ranges of all histories.  It finds

    L_h = W_min(h)/R_max(h),
    U_h = W_max(h)/R_min(h),

then chooses
    h+ = argmax L_h,
    h- = argmin U_h,

and searches for the simplest rational q strictly between U_{h-} and L_{h+}.
For the current depth-3, s=1, x=0 data the expected simple separator is 1/3.

Epistemic contract
------------------
* This is a finite numerical LP certificate, not an infinite theorem.
* Both HiGHS algorithms are cross-checked.
* Solver "Unknown" or timeout is never interpreted as infeasible.
* A PROVED-style finite statement is printed only if:
    - both R denominators are positively bounded;
    - both face minima have positive margins;
    - both crossing penalties are positive;
    - independent solver methods agree within configured tolerances;
    - KKT/dual residuals are small.
* No O(log n) or scale-uniform claim is made.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

VERSION = "DREAM6_ZR_v0.13_NORMALIZED_PHASE_SEPARATOR_CERTIFIER"

P = np.asarray([1 / 8, 1 / 4, 1 / 4, 1 / 4, 1 / 8], dtype=float)
Q = np.asarray([7 / 64, 5 / 16, 5 / 32, 5 / 16, 7 / 64], dtype=float)
B8 = np.asarray([
    0.53562333995590705,
    0.42450840479209673,
    0.039868255251995686,
], dtype=float)
D = 4


def highs_options() -> dict:
    return {
        "dual_feasibility_tolerance": 1e-9,
        "primal_feasibility_tolerance": 1e-9,
        "ipm_optimality_tolerance": 1e-10,
        "simplex_dual_edge_weight_strategy": "dantzig",
    }


def face_highs_options() -> dict:
    # SciPy/HiGHS effectively bottoms out around 1e-10 for feasibility.
    return {
        "dual_feasibility_tolerance": 1e-10,
        "primal_feasibility_tolerance": 1e-10,
        "ipm_optimality_tolerance": 1e-12,
        "simplex_dual_edge_weight_strategy": "dantzig",
    }


@dataclass(frozen=True)
class LightTree:
    nodes: tuple
    node_index: dict
    internals: tuple
    leaves: tuple


@dataclass(frozen=True)
class LightChart:
    M: int
    N: int
    ztuples: tuple


def build_tree(depth: int) -> LightTree:
    nodes = [()]
    for d in range(1, depth + 1):
        nodes.extend(itertools.product(range(5), repeat=d))
    nodes = tuple(nodes)
    node_index = {h: i for i, h in enumerate(nodes)}
    internals = tuple(h for h in nodes if len(h) < depth)
    leaves = tuple(h for h in nodes if len(h) == depth)
    return LightTree(nodes, node_index, internals, leaves)


def build_chart(M: int) -> LightChart:
    N = M + 1
    ztuples = []
    for j in range(N):
        for y in range(5):
            for x in range(5):
                k = j + y - x
                if 0 <= k <= M:
                    ztuples.append((j, y, x, k))
    return LightChart(M=M, N=N, ztuples=tuple(ztuples))


def build_exact_lp(tree: LightTree, chart: LightChart) -> dict:
    """Full finite-depth carry LP with row metadata for dual explanation."""
    N = chart.N
    Z = len(chart.ztuples)
    n_nodes = len(tree.nodes)
    n_internal = len(tree.internals)
    Aoff = {h: i * N for i, h in enumerate(tree.nodes)}
    base_z = n_nodes * N
    Zoff = {h: base_z + i * Z for i, h in enumerate(tree.internals)}
    nv = base_z + n_internal * Z

    eq_r: list[int] = []
    eq_c: list[int] = []
    eq_v: list[float] = []
    beq: list[float] = []
    eq_meta: list[tuple] = []

    ub_r: list[int] = []
    ub_c: list[int] = []
    ub_v: list[float] = []
    bub: list[float] = []
    ub_meta: list[tuple] = []

    def add_eq(entries: Iterable[tuple[int, float]], rhs: float, meta: tuple):
        rr = len(beq)
        for col, val in entries:
            if val != 0.0:
                eq_r.append(rr)
                eq_c.append(int(col))
                eq_v.append(float(val))
        beq.append(float(rhs))
        eq_meta.append(meta)

    def add_ub(entries: Iterable[tuple[int, float]], rhs: float, meta: tuple):
        rr = len(bub)
        for col, val in entries:
            if val != 0.0:
                ub_r.append(rr)
                ub_c.append(int(col))
                ub_v.append(float(val))
        bub.append(float(rhs))
        ub_meta.append(meta)

    z_by_jy: dict[tuple[int, int], list[int]] = {}
    z_by_xk: dict[tuple[int, int], list[int]] = {}
    for zi, (j, y, x, k) in enumerate(chart.ztuples):
        z_by_jy.setdefault((j, y), []).append(zi)
        z_by_xk.setdefault((x, k), []).append(zi)

    # Every buffer law is normalized.
    for h in tree.nodes:
        add_eq(
            [(Aoff[h] + j, 1.0) for j in range(N)],
            1.0,
            ("norm", h),
        )

    # Exact local carry semantics.
    for h in tree.internals:
        zo = Zoff[h]
        for j in range(N):
            for y in range(5):
                row = [(zo + zi, 1.0) for zi in z_by_jy.get((j, y), [])]
                row.append((Aoff[h] + j, -float(Q[y])))
                add_eq(row, 0.0, ("jy_marginal", h, int(j), int(y)))

        for x in range(5):
            child = h + (x,)
            for k in range(N):
                row = [(zo + zi, 1.0) for zi in z_by_xk.get((x, k), [])]
                row.append((Aoff[child] + k, -float(P[x])))
                add_eq(row, 0.0, ("child_marginal", h, int(x), int(k)))

    # Fixed continuation target B8: first-order stochastic dominance at leaves.
    cdf = np.cumsum(B8)
    for h in tree.leaves:
        add_ub([(Aoff[h], 1.0)], float(cdf[0]), ("leaf_cdf0", h))
        add_ub(
            [(Aoff[h] + j, 1.0) for j in range(2)],
            float(cdf[1]),
            ("leaf_cdf1", h),
        )

    Aeq = coo_matrix(
        (eq_v, (eq_r, eq_c)), shape=(len(beq), nv)
    ).tocsr()
    Aub = coo_matrix(
        (ub_v, (ub_r, ub_c)), shape=(len(bub), nv)
    ).tocsr()

    objective = np.zeros(nv, dtype=float)
    objective[Aoff[()] : Aoff[()] + N] = np.arange(N, dtype=float)

    return {
        "Aeq": Aeq,
        "beq": np.asarray(beq, dtype=float),
        "eq_meta": eq_meta,
        "Aub": Aub,
        "bub": np.asarray(bub, dtype=float),
        "ub_meta": ub_meta,
        "objective": objective,
        "bounds": [(0.0, None)] * nv,
        "Aoff": Aoff,
        "Zoff": Zoff,
        "nv": nv,
        "N": N,
        "Z": Z,
    }


def solve_base(lp: dict, method: str = "highs"):
    t0 = time.time()
    res = linprog(
        lp["objective"],
        A_ub=lp["Aub"],
        b_ub=lp["bub"],
        A_eq=lp["Aeq"],
        b_eq=lp["beq"],
        bounds=lp["bounds"],
        method=method,
        options=highs_options(),
    )
    return res, time.time() - t0


def linear_vec(nv: int, entries: Iterable[tuple[int, float]]) -> np.ndarray:
    out = np.zeros(nv, dtype=float)
    for idx, coef in entries:
        out[int(idx)] += float(coef)
    return out


def w_entries(chart: LightChart, lp: dict, h: tuple, s: int, x: int):
    zo = lp["Zoff"][h]
    out = []
    for zi, (j, y, xx, _k) in enumerate(chart.ztuples):
        if xx == x and j + y == s:
            out.append((zo + zi, 1.0))
    return out


def r_entries(chart: LightChart, lp: dict, h: tuple, s: int):
    """Total mass on anti-diagonal s, summed over all admissible x."""
    out = []
    for x in range(5):
        out.extend(w_entries(chart, lp, h, s, x))
    return out


def a_convolution_entries(lp: dict, h: tuple, s: int):
    """R_h(s)=sum_{j+y=s} A_h(j) Q(y), directly in A coordinates."""
    out = []
    N = lp["N"]
    for y, q in enumerate(Q):
        j = s - y
        if 0 <= j < N:
            out.append((lp["Aoff"][h] + j, float(q)))
    return out


def parse_sx(label: str):
    try:
        parts = dict(piece.split("=") for piece in label.split(","))
        return int(parts["s"]), int(parts["x"])
    except Exception:
        return None


def rational_string(x: float, max_den: int, tol: float) -> str | None:
    f = Fraction(float(x)).limit_denominator(max_den)
    if abs(float(f) - float(x)) <= tol:
        return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)
    return None


def _base_exact_face_eq(lp: dict, optimum: float, ext_cols: int = 0):
    Aeq = lp["Aeq"]
    if ext_cols:
        Aeq = hstack([Aeq, csr_matrix((Aeq.shape[0], ext_cols))], format="csr")
    obj = np.concatenate([lp["objective"], np.zeros(ext_cols, dtype=float)])
    Aeq = vstack([Aeq, csr_matrix(obj.reshape(1, -1))], format="csr")
    beq = np.concatenate([lp["beq"], np.asarray([float(optimum)])])
    meta = list(lp["eq_meta"]) + [("objective_face",)]
    return Aeq, beq, meta


def _base_slab_face_ub(lp: dict, optimum: float, face_tol: float, ext_cols: int = 0):
    c = lp["objective"]
    rows = csr_matrix(np.vstack([c, -c]))
    rhs = np.asarray([optimum + face_tol, -(optimum - face_tol)], dtype=float)
    Aub = vstack([lp["Aub"], rows], format="csr")
    if ext_cols:
        Aub = hstack([Aub, csr_matrix((Aub.shape[0], ext_cols))], format="csr")
    bub = np.concatenate([lp["bub"], rhs])
    meta = list(lp["ub_meta"]) + [("face_upper",), ("face_lower",)]
    return Aub, bub, meta


def build_joint_problem(
    chart: LightChart,
    lp: dict,
    optimum: float,
    histories: Sequence[tuple],
    observables: Sequence[tuple[str, Callable[[tuple], list[tuple[int, float]]]]],
    *,
    face_mode: str,
    face_tol: float,
):
    """Build the joint min-max LP and retain exact row metadata."""
    nv = lp["nv"]
    prepared = []
    for label, fn in observables:
        rows_h = []
        for h in histories:
            ent = fn(h)
            if ent:
                rows_h.append((h, linear_vec(nv, ent)))
        if len(rows_h) >= 2:
            prepared.append((str(label), rows_h))

    if not prepared:
        raise RuntimeError("no jointly testable observables")

    m = len(prepared)
    ext_nv = nv + 2 * m + 1
    t_idx = ext_nv - 1
    c = np.zeros(ext_nv, dtype=float)
    c[t_idx] = 1.0

    if face_mode == "exact":
        Aub = hstack(
            [lp["Aub"], csr_matrix((lp["Aub"].shape[0], ext_nv - nv))],
            format="csr",
        )
        bub = np.asarray(lp["bub"], dtype=float).copy()
        ub_meta = list(lp["ub_meta"])
        Aeq, beq, eq_meta = _base_exact_face_eq(lp, optimum, ext_nv - nv)
    elif face_mode == "slab":
        Aub, bub, ub_meta = _base_slab_face_ub(
            lp, optimum, face_tol, ext_nv - nv
        )
        Aeq = hstack(
            [lp["Aeq"], csr_matrix((lp["Aeq"].shape[0], ext_nv - nv))],
            format="csr",
        )
        beq = np.asarray(lp["beq"], dtype=float).copy()
        eq_meta = list(lp["eq_meta"])
    else:
        raise ValueError(face_mode)

    extra_rows = []
    extra_rhs = []
    extra_meta = []

    for qi, (label, rows_h) in enumerate(prepared):
        u_idx = nv + 2 * qi
        l_idx = nv + 2 * qi + 1

        for h, lv in rows_h:
            rr = np.zeros(ext_nv, dtype=float)
            rr[:nv] = lv
            rr[u_idx] = -1.0
            extra_rows.append(csr_matrix(rr.reshape(1, -1)))
            extra_rhs.append(0.0)
            extra_meta.append(("hist_upper", qi, label, h))

            rr = np.zeros(ext_nv, dtype=float)
            rr[:nv] = -lv
            rr[l_idx] = 1.0
            extra_rows.append(csr_matrix(rr.reshape(1, -1)))
            extra_rhs.append(0.0)
            extra_meta.append(("hist_lower", qi, label, h))

        rr = np.zeros(ext_nv, dtype=float)
        rr[u_idx] = 1.0
        rr[l_idx] = -1.0
        rr[t_idx] = -1.0
        extra_rows.append(csr_matrix(rr.reshape(1, -1)))
        extra_rhs.append(0.0)
        extra_meta.append(("spread_cap", qi, label))

    Aub = vstack([Aub] + extra_rows, format="csr")
    bub = np.concatenate([bub, np.asarray(extra_rhs, dtype=float)])
    ub_meta = ub_meta + extra_meta

    bounds = list(lp["bounds"]) + [(None, None)] * (2 * m) + [(0.0, None)]

    return {
        "c": c,
        "Aub": Aub,
        "bub": bub,
        "ub_meta": ub_meta,
        "Aeq": Aeq,
        "beq": beq,
        "eq_meta": eq_meta,
        "bounds": bounds,
        "prepared": prepared,
        "nv": nv,
        "ext_nv": ext_nv,
        "t_idx": t_idx,
        "m": m,
        "face_mode": face_mode,
    }


def solve_joint(problem: dict):
    t0 = time.time()
    res = linprog(
        problem["c"],
        A_ub=problem["Aub"],
        b_ub=problem["bub"],
        A_eq=problem["Aeq"],
        b_eq=problem["beq"],
        bounds=problem["bounds"],
        method="highs",
        options=face_highs_options(),
    )
    return res, time.time() - t0


def face_range_exact(lp: dict, optimum: float, vec: np.ndarray) -> dict:
    """Min/max a linear functional on c^T z = optimum."""
    Aeq = vstack(
        [lp["Aeq"], csr_matrix(lp["objective"].reshape(1, -1))],
        format="csr",
    )
    beq = np.concatenate([lp["beq"], [float(optimum)]])
    t0 = time.time()
    rmin = linprog(
        vec,
        A_ub=lp["Aub"], b_ub=lp["bub"],
        A_eq=Aeq, b_eq=beq, bounds=lp["bounds"],
        method="highs", options=face_highs_options(),
    )
    rmax = linprog(
        -vec,
        A_ub=lp["Aub"], b_ub=lp["bub"],
        A_eq=Aeq, b_eq=beq, bounds=lp["bounds"],
        method="highs", options=face_highs_options(),
    )
    dt = time.time() - t0
    if not (rmin.success and rmax.success):
        return {
            "success": False,
            "min_error": None if rmin.success else rmin.message,
            "max_error": None if rmax.success else rmax.message,
            "seconds": dt,
        }
    lo = float(rmin.fun)
    hi = -float(rmax.fun)
    return {
        "success": True,
        "min": lo,
        "max": hi,
        "width": hi - lo,
        "mid": 0.5 * (lo + hi),
        "seconds": dt,
    }


def kkt_numeric_audit(problem: dict, res) -> dict:
    """Audit HiGHS marginals against its own primal/dual KKT equations."""
    mu = np.asarray(res.ineqlin.marginals, dtype=float)
    lam = np.asarray(res.eqlin.marginals, dtype=float)
    lower = np.asarray(res.lower.marginals, dtype=float)
    upper = np.asarray(res.upper.marginals, dtype=float)

    # SciPy HiGHS marginals use the sign convention in which
    # c - A_ub^T mu - A_eq^T lam - lower + upper = 0.
    stat = (
        problem["c"]
        - problem["Aub"].T @ mu
        - problem["Aeq"].T @ lam
        - lower
        + upper
    )
    dual_value = float(problem["bub"] @ mu + problem["beq"] @ lam)
    return {
        "stationarity_max_abs": float(np.max(np.abs(stat))),
        "dual_value": dual_value,
        "primal_value": float(res.fun),
        "primal_dual_gap": float(res.fun - dual_value),
        "ineq_marginal_min": float(np.min(mu)) if mu.size else 0.0,
        "ineq_marginal_max": float(np.max(mu)) if mu.size else 0.0,
    }


def extract_dual_support(
    problem: dict,
    res,
    *,
    weight_tol: float,
    rational_max_den: int,
    rational_tol: float,
) -> dict:
    """Extract dual weights attached to joint-spread constraints."""
    mu = np.asarray(res.ineqlin.marginals, dtype=float)
    resid = np.asarray(res.ineqlin.residual, dtype=float)

    spread_caps = []
    hist_upper = []
    hist_lower = []
    base_ub = []

    for i, meta in enumerate(problem["ub_meta"]):
        w = -float(mu[i])  # nonnegative weight in the conventional <= primal dual
        rec = {
            "row": int(i),
            "meta": list(meta),
            "marginal": float(mu[i]),
            "weight": w,
            "slack": float(resid[i]),
        }
        if meta[0] == "spread_cap":
            rec["rational_weight"] = rational_string(
                w, rational_max_den, rational_tol
            )
            spread_caps.append(rec)
        elif meta[0] == "hist_upper":
            if w > weight_tol:
                rec["rational_weight"] = rational_string(
                    w, rational_max_den, rational_tol
                )
                hist_upper.append(rec)
        elif meta[0] == "hist_lower":
            if w > weight_tol:
                rec["rational_weight"] = rational_string(
                    w, rational_max_den, rational_tol
                )
                hist_lower.append(rec)
        else:
            if w > weight_tol:
                base_ub.append(rec)

    active_caps = [r for r in spread_caps if r["weight"] > weight_tol]
    alpha_sum = float(sum(r["weight"] for r in active_caps))

    eq_marg = np.asarray(res.eqlin.marginals, dtype=float)
    top_eq = []
    for i, (meta, val) in enumerate(zip(problem["eq_meta"], eq_marg)):
        if abs(float(val)) > weight_tol:
            top_eq.append({
                "row": int(i),
                "meta": list(meta),
                "marginal": float(val),
                "abs": abs(float(val)),
            })
    top_eq.sort(key=lambda r: r["abs"], reverse=True)

    return {
        "spread_caps": spread_caps,
        "active_spread_caps": active_caps,
        "active_hist_upper": hist_upper,
        "active_hist_lower": hist_lower,
        "active_base_ub": base_ub,
        "alpha_sum": alpha_sum,
        "top_eq_multipliers": top_eq[:40],
    }


def primal_observable_table(problem: dict, res) -> list[dict]:
    nv = problem["nv"]
    out = []
    for qi, (label, rows_h) in enumerate(problem["prepared"]):
        u_idx = nv + 2 * qi
        l_idx = nv + 2 * qi + 1
        U = float(res.x[u_idx])
        L = float(res.x[l_idx])
        vals = [(h, float(lv @ res.x[:nv])) for h, lv in rows_h]
        vals.sort(key=lambda hv: hv[1])
        out.append({
            "qi": int(qi),
            "label": label,
            "U": U,
            "L": L,
            "spread": U - L,
            "argmin_history": list(vals[0][0]),
            "argmin_value": vals[0][1],
            "argmax_history": list(vals[-1][0]),
            "argmax_value": vals[-1][1],
        })
    return out


def induced_dual_functional(problem: dict, support: dict) -> np.ndarray:
    """Build D(z)=sum upper_weights*L - sum lower_weights*L in original z-space."""
    nv = problem["nv"]
    v = np.zeros(nv, dtype=float)
    prepared = problem["prepared"]

    # Lookup linear observable vector by (qi, history).
    lookup = {}
    for qi, (_label, rows_h) in enumerate(prepared):
        for h, lv in rows_h:
            lookup[(qi, tuple(h))] = lv

    for rec in support["active_hist_upper"]:
        meta = rec["meta"]
        qi = int(meta[1])
        h = tuple(meta[3])
        v += float(rec["weight"]) * lookup[(qi, h)]

    for rec in support["active_hist_lower"]:
        meta = rec["meta"]
        qi = int(meta[1])
        h = tuple(meta[3])
        v -= float(rec["weight"]) * lookup[(qi, h)]

    return v


def detect_partition_certificate(
    chart: LightChart,
    lp: dict,
    optimum: float,
    problem: dict,
    res,
    support: dict,
    *,
    weight_tol: float,
    equality_tol: float,
) -> dict:
    """Recognize a pigeonhole/conservation certificate from the dual support.

    Desired shape:
      * active spread caps all lie on one anti-diagonal s,
      * they cover every admissible x=0,...,min(4,s),
      * cap weights are equal 1/k,
      * every active upper row uses the same history h+,
      * every active lower row uses the same history h-,
      * each cell has exactly one upper and one lower weight equal to its cap weight.

    Then
        max_x spread_{s,x}
          >= (1/k) |R_{h+}(s)-R_{h-}(s)|,
    where R_h(s)=sum_x W_h(s,x)=sum_{j+y=s}A_h(j)Q(y).
    """
    caps = support["active_spread_caps"]
    if not caps:
        return {"success": False, "reason": "no active spread-cap dual weights"}

    sx = []
    for rec in caps:
        meta = rec["meta"]
        parsed = parse_sx(str(meta[2]))
        if parsed is None:
            return {"success": False, "reason": f"unparsed label {meta[2]}"}
        sx.append((parsed[0], parsed[1], float(rec["weight"]), int(meta[1])))

    s_values = {s for s, _x, _w, _qi in sx}
    if len(s_values) != 1:
        return {"success": False, "reason": "active caps span multiple anti-diagonals", "active": sx}
    s0 = next(iter(s_values))
    expected_x = list(range(0, min(4, s0) + 1))
    got_x = sorted(x for _s, x, _w, _qi in sx)
    if got_x != expected_x:
        return {
            "success": False,
            "reason": "active caps do not cover the full admissible x partition",
            "s": s0,
            "expected_x": expected_x,
            "got_x": got_x,
        }

    k = len(expected_x)
    weights = [w for _s, _x, w, _qi in sx]
    target_w = 1.0 / k
    if max(abs(w - target_w) for w in weights) > equality_tol:
        return {
            "success": False,
            "reason": "cap weights are not the uniform partition weights",
            "weights": weights,
            "target": target_w,
        }

    # Map active upper/lower weights by qi.
    up_by_q: dict[int, list[dict]] = {}
    lo_by_q: dict[int, list[dict]] = {}
    for rec in support["active_hist_upper"]:
        up_by_q.setdefault(int(rec["meta"][1]), []).append(rec)
    for rec in support["active_hist_lower"]:
        lo_by_q.setdefault(int(rec["meta"][1]), []).append(rec)

    hplus_set = set()
    hminus_set = set()
    for _s, _x, alpha, qi in sx:
        ups = up_by_q.get(qi, [])
        los = lo_by_q.get(qi, [])
        if len(ups) != 1 or len(los) != 1:
            return {
                "success": False,
                "reason": "a load-bearing cell does not have one upper and one lower history",
                "qi": qi,
                "upper_count": len(ups),
                "lower_count": len(los),
            }
        if abs(float(ups[0]["weight"]) - alpha) > equality_tol:
            return {"success": False, "reason": "upper weight != cap weight", "qi": qi}
        if abs(float(los[0]["weight"]) - alpha) > equality_tol:
            return {"success": False, "reason": "lower weight != cap weight", "qi": qi}
        hplus_set.add(tuple(ups[0]["meta"][3]))
        hminus_set.add(tuple(los[0]["meta"][3]))

    if len(hplus_set) != 1 or len(hminus_set) != 1:
        return {
            "success": False,
            "reason": "load-bearing cells use different history pairs",
            "hplus": [list(h) for h in sorted(hplus_set)],
            "hminus": [list(h) for h in sorted(hminus_set)],
        }

    hplus = next(iter(hplus_set))
    hminus = next(iter(hminus_set))

    # Rdiff in routing coordinates.
    vR = np.zeros(lp["nv"], dtype=float)
    for idx, c in r_entries(chart, lp, hplus, s0):
        vR[idx] += c
    for idx, c in r_entries(chart, lp, hminus, s0):
        vR[idx] -= c

    # Same Rdiff directly in A*Q coordinates, to verify conservation identity.
    vA = np.zeros(lp["nv"], dtype=float)
    for idx, c in a_convolution_entries(lp, hplus, s0):
        vA[idx] += c
    for idx, c in a_convolution_entries(lp, hminus, s0):
        vA[idx] -= c

    identity_range = face_range_exact(lp, optimum, vR - vA)
    r_range = face_range_exact(lp, optimum, vR)

    if not r_range.get("success"):
        return {"success": False, "reason": "could not range Rdiff on exact face", "r_range": r_range}

    rmid = float(r_range["mid"])
    cert_floor = abs(rmid) / k
    joint_floor = float(res.fun)
    floor_gap = joint_floor - cert_floor

    # At the joint optimizer, inspect the load-bearing pair cell-by-cell.
    cell_diffs = []
    for x in expected_x:
        vp = linear_vec(lp["nv"], w_entries(chart, lp, hplus, s0, x))
        vm = linear_vec(lp["nv"], w_entries(chart, lp, hminus, s0, x))
        diff = float((vp - vm) @ res.x[: lp["nv"]])
        cell_diffs.append({"x": int(x), "difference": diff})

    # Human-readable convolution coefficients.
    q_terms = []
    for y, q in enumerate(Q):
        j = s0 - y
        if 0 <= j < lp["N"]:
            fq = Fraction(float(q)).limit_denominator(64)
            q_terms.append({"j": int(j), "y": int(y), "Q": f"{fq.numerator}/{fq.denominator}"})

    return {
        "success": True,
        "kind": "ANTI_DIAGONAL_TOTAL_MASS_PIGEONHOLE",
        "s": int(s0),
        "x_partition": expected_x,
        "k": int(k),
        "uniform_dual_weight": target_w,
        "h_plus": list(hplus),
        "h_minus": list(hminus),
        "R_difference_exact_face": r_range,
        "routing_vs_AQ_identity_exact_face": identity_range,
        "certificate_floor": cert_floor,
        "joint_floor": joint_floor,
        "joint_minus_certificate": floor_gap,
        "cell_differences_at_joint_optimizer": cell_diffs,
        "Q_convolution_terms": q_terms,
        "equation": (
            f"max_x spread(s={s0},x) >= |R_{hplus}({s0})-R_{hminus}({s0})|/{k}"
        ),
    }




# ============================================================================
# v0.8 normalization-aware audit
# ============================================================================

def parse_history(text: str) -> tuple[int, ...]:
    text = text.strip()
    if not text:
        return ()
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def exact_face_matrices(lp: dict, optimum: float, extra_cols: int = 0):
    Aeq, beq, eq_meta = _base_exact_face_eq(lp, optimum, extra_cols)
    Aub = hstack(
        [lp["Aub"], csr_matrix((lp["Aub"].shape[0], extra_cols))],
        format="csr",
    ) if extra_cols else lp["Aub"]
    bub = np.asarray(lp["bub"], dtype=float).copy()
    return Aeq, beq, eq_meta, Aub, bub


def face_range_method(lp: dict, optimum: float, vec: np.ndarray, method: str) -> dict:
    Aeq = vstack(
        [lp["Aeq"], csr_matrix(lp["objective"].reshape(1, -1))],
        format="csr",
    )
    beq = np.concatenate([lp["beq"], [float(optimum)]])
    t0 = time.time()
    rmin = linprog(
        vec,
        A_ub=lp["Aub"], b_ub=lp["bub"],
        A_eq=Aeq, b_eq=beq, bounds=lp["bounds"],
        method=method, options=face_highs_options(),
    )
    rmax = linprog(
        -vec,
        A_ub=lp["Aub"], b_ub=lp["bub"],
        A_eq=Aeq, b_eq=beq, bounds=lp["bounds"],
        method=method, options=face_highs_options(),
    )
    dt = time.time() - t0
    if not (rmin.success and rmax.success):
        return {
            "success": False,
            "method": method,
            "min_error": None if rmin.success else rmin.message,
            "max_error": None if rmax.success else rmax.message,
            "seconds": dt,
        }
    lo = float(rmin.fun)
    hi = -float(rmax.fun)
    return {
        "success": True,
        "method": method,
        "min": lo,
        "max": hi,
        "mid": 0.5 * (lo + hi),
        "width": hi - lo,
        "seconds": dt,
    }


def solve_exact_face_feasibility(
    lp: dict,
    optimum: float,
    extra_equalities: Sequence[tuple[np.ndarray, float]],
    *,
    method: str = "highs",
) -> dict:
    rows = [lp["Aeq"], csr_matrix(lp["objective"].reshape(1, -1))]
    rhs = [*lp["beq"], float(optimum)]
    for vec, val in extra_equalities:
        rows.append(csr_matrix(np.asarray(vec, dtype=float).reshape(1, -1)))
        rhs.append(float(val))
    Aeq = vstack(rows, format="csr")
    beq = np.asarray(rhs, dtype=float)
    c = np.zeros(lp["nv"], dtype=float)

    t0 = time.time()
    res = linprog(
        c,
        A_ub=lp["Aub"], b_ub=lp["bub"],
        A_eq=Aeq, b_eq=beq,
        bounds=lp["bounds"],
        method=method, options=face_highs_options(),
    )
    dt = time.time() - t0
    out = {
        "success": bool(res.success),
        "method": method,
        "message": res.message,
        "seconds": dt,
    }
    if res.success:
        eq_err = float(np.max(np.abs(Aeq @ res.x - beq)))
        ub_violation = np.asarray(lp["Aub"] @ res.x - lp["bub"], dtype=float)
        out.update({
            "eq_error": eq_err,
            "max_ub_violation": float(max(0.0, np.max(ub_violation))) if ub_violation.size else 0.0,
            "x": res.x,
        })
    return out


def solve_pair_minimax(
    lp: dict,
    optimum: float,
    diff_vectors: Sequence[tuple[int, np.ndarray]],
    *,
    extra_equalities: Sequence[tuple[np.ndarray, float]] = (),
    method: str = "highs",
) -> dict:
    """min t subject to |d_x^T z| <= t simultaneously for all x."""
    nv = lp["nv"]
    ext_nv = nv + 1
    t_idx = nv

    Aeq0 = hstack(
        [lp["Aeq"], csr_matrix((lp["Aeq"].shape[0], 1))],
        format="csr",
    )
    rows_eq = [Aeq0]
    beq = list(lp["beq"])

    objrow = np.zeros(ext_nv, dtype=float)
    objrow[:nv] = lp["objective"]
    rows_eq.append(csr_matrix(objrow.reshape(1, -1)))
    beq.append(float(optimum))

    for vec, rhs in extra_equalities:
        rr = np.zeros(ext_nv, dtype=float)
        rr[:nv] = np.asarray(vec, dtype=float)
        rows_eq.append(csr_matrix(rr.reshape(1, -1)))
        beq.append(float(rhs))

    Aeq = vstack(rows_eq, format="csr")
    beq = np.asarray(beq, dtype=float)

    Aub = hstack(
        [lp["Aub"], csr_matrix((lp["Aub"].shape[0], 1))],
        format="csr",
    )
    bub = np.asarray(lp["bub"], dtype=float).copy()
    extra_rows = []
    for x, vec in diff_vectors:
        for sign in (+1.0, -1.0):
            rr = np.zeros(ext_nv, dtype=float)
            rr[:nv] = sign * np.asarray(vec, dtype=float)
            rr[t_idx] = -1.0
            extra_rows.append(csr_matrix(rr.reshape(1, -1)))
    if extra_rows:
        Aub = vstack([Aub] + extra_rows, format="csr")
        bub = np.concatenate([bub, np.zeros(len(extra_rows), dtype=float)])

    c = np.zeros(ext_nv, dtype=float)
    c[t_idx] = 1.0
    bounds = list(lp["bounds"]) + [(0.0, None)]

    t0 = time.time()
    res = linprog(
        c,
        A_ub=Aub, b_ub=bub,
        A_eq=Aeq, b_eq=beq,
        bounds=bounds,
        method=method, options=face_highs_options(),
    )
    dt = time.time() - t0
    if not res.success:
        return {
            "success": False,
            "method": method,
            "message": res.message,
            "seconds": dt,
        }

    z = res.x[:nv]
    vals = [{"x": int(x), "difference": float(vec @ z)}
            for x, vec in diff_vectors]
    return {
        "success": True,
        "method": method,
        "floor": float(res.fun),
        "seconds": dt,
        "differences": vals,
        "eq_error": float(np.max(np.abs(Aeq @ res.x - beq))),
        "max_ub_violation": float(max(0.0, np.max(Aub @ res.x - bub))),
        "x": z,
    }


def raw_auto_target(
    chart: LightChart,
    lp: dict,
    optimum: float,
    histories: Sequence[tuple],
    structural_s: Sequence[int],
    args,
) -> dict:
    observables = []
    for s0 in structural_s:
        for x0 in range(5):
            observables.append((
                f"s={s0},x={x0}",
                lambda h, ss=s0, xx=x0: w_entries(chart, lp, h, ss, xx),
            ))

    problem = build_joint_problem(
        chart, lp, optimum, histories, observables,
        face_mode="exact", face_tol=float(args.face_tol),
    )
    res, sec = solve_joint(problem)
    if not res.success:
        return {"success": False, "error": res.message, "seconds": sec}

    support = extract_dual_support(
        problem, res,
        weight_tol=float(args.dual_weight_tol),
        rational_max_den=int(args.rational_max_den),
        rational_tol=float(args.rational_tol),
    )
    partition = detect_partition_certificate(
        chart, lp, optimum, problem, res, support,
        weight_tol=float(args.dual_weight_tol),
        equality_tol=float(args.dual_equal_tol),
    )
    return {
        "success": True,
        "joint_floor": float(res.fun),
        "seconds": sec,
        "dual_support": support,
        "partition_certificate": partition,
        "kkt_numeric_audit": kkt_numeric_audit(problem, res),
    }


def denominator_audit(
    chart: LightChart,
    lp: dict,
    optimum: float,
    h_plus: tuple,
    h_minus: tuple,
    s0: int,
    methods: Sequence[str],
) -> dict:
    vp = linear_vec(lp["nv"], r_entries(chart, lp, h_plus, s0))
    vm = linear_vec(lp["nv"], r_entries(chart, lp, h_minus, s0))
    vd = vp - vm

    by_method = {}
    for method in methods:
        by_method[method] = {
            "R_plus": face_range_method(lp, optimum, vp, method),
            "R_minus": face_range_method(lp, optimum, vm, method),
            "R_diff": face_range_method(lp, optimum, vd, method),
        }

    # Canonical readout uses the first successful method; cross-method spread is reported.
    canonical = None
    for method in methods:
        r = by_method[method]
        if r["R_plus"].get("success") and r["R_minus"].get("success") and r["R_diff"].get("success"):
            canonical = r
            break

    return {
        "success": canonical is not None,
        "by_method": by_method,
        "canonical": canonical,
        "vectors": {"plus": vp, "minus": vm, "diff": vd},
    }


def true_shape_differences(
    chart: LightChart,
    lp: dict,
    z: np.ndarray,
    h_plus: tuple,
    h_minus: tuple,
    s0: int,
    xs: Sequence[int],
) -> dict:
    vpR = linear_vec(lp["nv"], r_entries(chart, lp, h_plus, s0))
    vmR = linear_vec(lp["nv"], r_entries(chart, lp, h_minus, s0))
    rp = float(vpR @ z)
    rm = float(vmR @ z)
    if rp <= 0.0 or rm <= 0.0:
        return {"success": False, "R_plus": rp, "R_minus": rm}

    rows = []
    for x0 in xs:
        wp = float(linear_vec(lp["nv"], w_entries(chart, lp, h_plus, s0, x0)) @ z)
        wm = float(linear_vec(lp["nv"], w_entries(chart, lp, h_minus, s0, x0)) @ z)
        gp = wp / rp
        gm = wm / rm
        rows.append({
            "x": int(x0),
            "W_plus": wp,
            "W_minus": wm,
            "G_plus": gp,
            "G_minus": gm,
            "G_difference": gp - gm,
        })
    return {
        "success": True,
        "R_plus": rp,
        "R_minus": rm,
        "ratio_Rplus_over_Rminus": rp / rm,
        "rows": rows,
        "max_abs_G_difference": max(abs(r["G_difference"]) for r in rows) if rows else 0.0,
        "sum_G_plus": sum(r["G_plus"] for r in rows),
        "sum_G_minus": sum(r["G_minus"] for r in rows),
    }


def ratio_locked_witness(
    chart: LightChart,
    lp: dict,
    optimum: float,
    h_plus: tuple,
    h_minus: tuple,
    s0: int,
    xs: Sequence[int],
    rho: float,
    methods: Sequence[str],
) -> dict:
    eqs = []
    for x0 in xs:
        vp = linear_vec(lp["nv"], w_entries(chart, lp, h_plus, s0, x0))
        vm = linear_vec(lp["nv"], w_entries(chart, lp, h_minus, s0, x0))
        eqs.append((vp - float(rho) * vm, 0.0))

    runs = []
    for method in methods:
        fr = solve_exact_face_feasibility(lp, optimum, eqs, method=method)
        rec = {k: v for k, v in fr.items() if k != "x"}
        if fr.get("success"):
            shape = true_shape_differences(
                chart, lp, fr["x"], h_plus, h_minus, s0, xs
            )
            rec["true_shape"] = shape
        runs.append(rec)
    return {"rho": float(rho), "runs": runs}


def equal_mass_slice_audit(
    chart: LightChart,
    lp: dict,
    optimum: float,
    h_plus: tuple,
    h_minus: tuple,
    s0: int,
    xs: Sequence[int],
    mass_zero_tol: float,
    methods: Sequence[str],
    rdiff_range: dict,
) -> dict:
    vRp = linear_vec(lp["nv"], r_entries(chart, lp, h_plus, s0))
    vRm = linear_vec(lp["nv"], r_entries(chart, lp, h_minus, s0))
    vdiff = vRp - vRm

    if rdiff_range.get("success"):
        lo, hi = float(rdiff_range["min"]), float(rdiff_range["max"])
        if lo > mass_zero_tol or hi < -mass_zero_tol:
            # Confirm with one explicit feasibility solve, but the range already tells the story.
            confirm = solve_exact_face_feasibility(
                lp, optimum, [(vdiff, 0.0)], method=methods[0]
            )
            return {
                "status": "INFEASIBLE_ON_OPTIMAL_FACE__FORCED_MASS_MISMATCH",
                "R_diff_range": rdiff_range,
                "explicit_feasibility_check": {
                    k: v for k, v in confirm.items() if k != "x"
                },
            }

    raw_diffs = []
    for x0 in xs:
        vp = linear_vec(lp["nv"], w_entries(chart, lp, h_plus, s0, x0))
        vm = linear_vec(lp["nv"], w_entries(chart, lp, h_minus, s0, x0))
        raw_diffs.append((x0, vp - vm))
    ans = solve_pair_minimax(
        lp, optimum, raw_diffs,
        extra_equalities=[(vdiff, 0.0)],
        method=methods[0],
    )
    rec = {k: v for k, v in ans.items() if k != "x"}
    return {
        "status": "FEASIBLE_EQUAL_MASS_SLICE",
        "equal_mass_raw_floor": rec,
    }






# ============================================================================
# v0.13 separator certificate
# ============================================================================

def parse_history(text: str) -> tuple[int, ...]:
    return tuple(int(v.strip()) for v in text.split(",") if v.strip())


def _face_eq(lp: dict, optimum: float):
    Aeq = vstack(
        [lp["Aeq"], csr_matrix(lp["objective"].reshape(1, -1))],
        format="csr",
    )
    beq = np.concatenate([lp["beq"], [float(optimum)]])
    meta = list(lp["eq_meta"]) + [("objective_face",)]
    return Aeq, beq, meta


def solve_face_linear(
    lp: dict,
    optimum: float,
    vec: np.ndarray,
    *,
    method: str,
    time_limit: float,
) -> dict:
    Aeq, beq, eq_meta = _face_eq(lp, optimum)
    opts = dict(face_highs_options())
    if time_limit > 0:
        opts["time_limit"] = float(time_limit)
    t0 = time.time()
    res = linprog(
        vec,
        A_ub=lp["Aub"], b_ub=lp["bub"],
        A_eq=Aeq, b_eq=beq,
        bounds=lp["bounds"],
        method=method, options=opts,
    )
    sec = time.time() - t0
    out = {
        "success": bool(res.success),
        "status_code": int(res.status),
        "message": res.message,
        "method": method,
        "seconds": sec,
    }
    if res.success:
        out["value"] = float(res.fun)
        out["kkt"] = dual_kkt_audit(
            res, vec, Aeq, beq, lp["Aub"], lp["bub"],
            eq_meta=eq_meta, ub_meta=lp["ub_meta"],
        )
    return out


def solve_face_range(
    lp: dict,
    optimum: float,
    vec: np.ndarray,
    *,
    method: str,
    time_limit: float,
) -> dict:
    lo = solve_face_linear(lp, optimum, vec, method=method, time_limit=time_limit)
    hi0 = solve_face_linear(lp, optimum, -vec, method=method, time_limit=time_limit)
    if not (lo.get("success") and hi0.get("success")):
        return {"success": False, "min_run": lo, "max_run": hi0}
    low = float(lo["value"])
    high = -float(hi0["value"])
    if high < low and abs(high-low) <= 1e-10:
        low, high = high, low
    return {
        "success": True,
        "min": low,
        "max": high,
        "mid": 0.5*(low+high),
        "width": high-low,
        "min_run": lo,
        "max_run": hi0,
    }


def solve_crossing_penalty(
    lp: dict,
    separator_vec: np.ndarray,
    *,
    method: str,
    time_limit: float,
    label: str,
) -> dict:
    # Add separator_vec^T z <= 0.
    row = csr_matrix(separator_vec.reshape(1, -1))
    Aub = vstack([lp["Aub"], row], format="csr")
    bub = np.concatenate([lp["bub"], [0.0]])
    ub_meta = list(lp["ub_meta"]) + [("separator_crossing", label)]

    opts = dict(face_highs_options())
    if time_limit > 0:
        opts["time_limit"] = float(time_limit)
    t0 = time.time()
    res = linprog(
        lp["objective"],
        A_ub=Aub, b_ub=bub,
        A_eq=lp["Aeq"], b_eq=lp["beq"],
        bounds=lp["bounds"],
        method=method, options=opts,
    )
    sec = time.time() - t0
    out = {
        "success": bool(res.success),
        "status_code": int(res.status),
        "message": res.message,
        "method": method,
        "seconds": sec,
        "label": label,
    }
    if res.success:
        out["objective"] = float(res.fun)
        out["separator_value_at_cross_optimum"] = float(separator_vec @ res.x)
        out["kkt"] = dual_kkt_audit(
            res, lp["objective"], lp["Aeq"], lp["beq"], Aub, bub,
            eq_meta=lp["eq_meta"], ub_meta=ub_meta,
        )
        try:
            out["separator_dual_marginal"] = float(res.ineqlin.marginals[-1])
            out["separator_slack"] = float(res.ineqlin.residual[-1])
        except Exception:
            out["separator_dual_marginal"] = None
            out["separator_slack"] = None
    return out


def dual_kkt_audit(
    res,
    objective: np.ndarray,
    Aeq,
    beq,
    Aub,
    bub,
    *,
    eq_meta: Sequence[tuple],
    ub_meta: Sequence[tuple],
    support_tol: float = 1e-8,
    top_k: int = 30,
) -> dict:
    """Numerical primal/dual audit using SciPy/HiGHS marginals."""
    try:
        yeq = np.asarray(res.eqlin.marginals, dtype=float)
        yub = np.asarray(res.ineqlin.marginals, dtype=float)
        ylo = np.asarray(res.lower.marginals, dtype=float)
        yup = np.asarray(res.upper.marginals, dtype=float)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    # SciPy marginals satisfy c = Aeq^T yeq + Aub^T yub + lower + upper.
    stationarity = (
        np.asarray(objective, dtype=float)
        - np.asarray(Aeq.T @ yeq).ravel()
        - np.asarray(Aub.T @ yub).ravel()
        - ylo
        - yup
    )

    primal_eq = np.asarray(Aeq @ res.x - beq, dtype=float)
    primal_ub = np.asarray(Aub @ res.x - bub, dtype=float)

    dual_obj = float(np.dot(beq, yeq) + np.dot(bub, yub))
    primal_obj = float(np.dot(objective, res.x))

    # Complementarity diagnostics.
    ub_slack = np.asarray(bub - Aub @ res.x, dtype=float)
    lower_slack = np.asarray(res.x, dtype=float)
    comp_ub = yub * ub_slack
    comp_lo = ylo * lower_slack

    eq_support = sorted(
        [
            (abs(float(v)), i, float(v), eq_meta[i] if i < len(eq_meta) else ("eq", i))
            for i, v in enumerate(yeq)
            if abs(float(v)) > support_tol
        ],
        reverse=True,
    )[:top_k]
    ub_support = sorted(
        [
            (abs(float(v)), i, float(v), ub_meta[i] if i < len(ub_meta) else ("ub", i))
            for i, v in enumerate(yub)
            if abs(float(v)) > support_tol
        ],
        reverse=True,
    )[:top_k]

    return {
        "available": True,
        "primal_objective": primal_obj,
        "dual_objective": dual_obj,
        "duality_gap_abs": abs(primal_obj-dual_obj),
        "stationarity_inf": float(np.max(np.abs(stationarity))),
        "primal_eq_inf": float(np.max(np.abs(primal_eq))) if primal_eq.size else 0.0,
        "primal_ub_violation": float(max(0.0, np.max(primal_ub))) if primal_ub.size else 0.0,
        "complementarity_ub_inf": float(np.max(np.abs(comp_ub))) if comp_ub.size else 0.0,
        "complementarity_lower_inf": float(np.max(np.abs(comp_lo))) if comp_lo.size else 0.0,
        "ineq_dual_max_positive": float(max(0.0, np.max(yub))) if yub.size else 0.0,
        "lower_dual_min_negative": float(min(0.0, np.min(ylo))) if ylo.size else 0.0,
        "top_eq_support": [
            {"row": i, "multiplier": v, "meta": list(meta)}
            for _, i, v, meta in eq_support
        ],
        "top_ub_support": [
            {"row": i, "multiplier": v, "meta": list(meta)}
            for _, i, v, meta in ub_support
        ],
    }


def simple_rational_between(
    lower: float,
    upper: float,
    max_den: int,
) -> Fraction | None:
    """Return simplest positive rational q with lower < q < upper."""
    if not lower < upper:
        return None
    best = None
    for d in range(1, int(max_den)+1):
        # Strictly inside.
        n0 = math.floor(lower*d) + 1
        n1 = math.ceil(upper*d) - 1
        if n0 <= n1:
            # Prefer numerator closest to midpoint, then smaller |numerator|.
            mid = 0.5*(lower+upper)
            candidates = list(range(n0, n1+1))
            n = min(candidates, key=lambda k: (abs(k/d-mid), abs(k), k))
            best = Fraction(n, d)
            break
    return best


def automatic_separator_scan(
    chart: LightChart,
    lp: dict,
    optimum: float,
    histories: Sequence[tuple],
    s0: int,
    x0: int,
    *,
    method: str,
    time_limit: float,
    max_den: int,
    progress_every: int,
) -> dict:
    rows = []
    for k, h in enumerate(histories, start=1):
        rv = linear_vec(lp["nv"], r_entries(chart, lp, tuple(h), int(s0)))
        wv = linear_vec(lp["nv"], w_entries(chart, lp, tuple(h), int(s0), int(x0)))
        rr = solve_face_range(lp, optimum, rv, method=method, time_limit=time_limit)
        wr = solve_face_range(lp, optimum, wv, method=method, time_limit=time_limit)
        if not (rr.get("success") and wr.get("success")):
            raise RuntimeError(f"range solve failed for h={h}")
        rmin, rmax = float(rr["min"]), float(rr["max"])
        wmin, wmax = float(wr["min"]), float(wr["max"])
        if rmin <= 0:
            lower, upper = 0.0, 1.0
        else:
            lower = wmin / rmax
            upper = wmax / rmin
        rows.append({
            "history": list(h),
            "R": [rmin, rmax],
            "W": [wmin, wmax],
            "ratio_outer": [lower, upper],
        })
        if progress_every and (k % progress_every == 0 or k == len(histories)):
            print(f"[auto] {k:3d}/{len(histories)}", flush=True)

    plus = max(rows, key=lambda r: r["ratio_outer"][0])
    minus = min(rows, key=lambda r: r["ratio_outer"][1])
    L = float(plus["ratio_outer"][0])
    U = float(minus["ratio_outer"][1])
    q = simple_rational_between(U, L, max_den=max_den)

    return {
        "rows": rows,
        "h_plus": plus["history"],
        "h_minus": minus["history"],
        "lower_plus": L,
        "upper_minus": U,
        "gap": L-U,
        "separator_fraction": None if q is None else {
            "numerator": q.numerator,
            "denominator": q.denominator,
            "value": float(q),
        },
    }


def method_agreement(values: Sequence[float], abs_tol: float, rel_tol: float) -> bool:
    vals = [float(v) for v in values if np.isfinite(v)]
    if len(vals) < 2:
        return True
    spread = max(vals)-min(vals)
    scale = max(max(abs(v) for v in vals), 1.0)
    return spread <= abs_tol + rel_tol*scale


def run_v013(args) -> dict:
    depth = int(args.depth)
    M = int(args.M)
    tree = build_tree(depth)
    chart = build_chart(M)
    lp = build_exact_lp(tree, chart)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    base_runs = {}
    for method in methods:
        br, sec = solve_base(lp, method=method)
        base_runs[method] = {
            "success": bool(br.success),
            "objective": None if not br.success else float(br.fun),
            "seconds": sec,
            "message": br.message,
        }
    good_base = [r["objective"] for r in base_runs.values() if r["success"]]
    if not good_base:
        raise RuntimeError("no base LP method succeeded")
    optimum = min(good_base)

    hdepth = depth - 1 if args.history_depth < 0 else int(args.history_depth)
    histories = [h for h in tree.internals if len(h) == hdepth]

    print("="*120)
    print("DREAM6-ZR v0.13  NORMALIZED PHASE SEPARATOR CERTIFIER")
    print("="*120)
    print(f"depth / M                 : {depth} / {M}")
    print(f"history depth             : {hdepth}")
    print(f"target s,x                : ({args.s},{args.x})")
    print(f"methods                   : {methods}")
    print(f"base optimum              : {optimum:.15g}")
    print(f"base method spread        : {max(good_base)-min(good_base):.3e}")
    print("crossing penalty          : ENABLED")
    print("dual/KKT audit            : ENABLED")
    print("="*120, flush=True)

    auto = None
    if args.h_plus.lower() == "auto" or args.h_minus.lower() == "auto" or args.separator.lower() == "auto":
        print("-"*120)
        print("AUTO RATIONAL SEPARATOR DISCOVERY")
        auto = automatic_separator_scan(
            chart, lp, optimum, histories, int(args.s), int(args.x),
            method=args.scan_method,
            time_limit=float(args.time_limit),
            max_den=int(args.max_den),
            progress_every=int(args.progress_every),
        )
        print(f"strongest lower history   : {tuple(auto['h_plus'])}")
        print(f"strongest upper history   : {tuple(auto['h_minus'])}")
        print(f"outer ratio gap           : {auto['gap']:.15e}")
        print(f"lower_plus / upper_minus  : {auto['lower_plus']:.15e} / {auto['upper_minus']:.15e}")
        print(f"simple separator          : {auto['separator_fraction']}")

    h_plus = tuple(auto["h_plus"]) if args.h_plus.lower() == "auto" else parse_history(args.h_plus)
    h_minus = tuple(auto["h_minus"]) if args.h_minus.lower() == "auto" else parse_history(args.h_minus)

    if args.separator.lower() == "auto":
        if not auto or not auto["separator_fraction"]:
            raise RuntimeError("AUTO could not find a rational separator")
        a = int(auto["separator_fraction"]["numerator"])
        b = int(auto["separator_fraction"]["denominator"])
    else:
        q = Fraction(args.separator)
        a, b = int(q.numerator), int(q.denominator)

    if h_plus not in lp["Zoff"] or h_minus not in lp["Zoff"]:
        raise RuntimeError("selected histories are not internal coupling histories")

    Rp = linear_vec(lp["nv"], r_entries(chart, lp, h_plus, int(args.s)))
    Wp = linear_vec(lp["nv"], w_entries(chart, lp, h_plus, int(args.s), int(args.x)))
    Rm = linear_vec(lp["nv"], r_entries(chart, lp, h_minus, int(args.s)))
    Wm = linear_vec(lp["nv"], w_entries(chart, lp, h_minus, int(args.s), int(args.x)))

    Splus = b*Wp - a*Rp
    Sminus = a*Rm - b*Wm

    print("-"*120)
    print("RATIONAL SEPARATOR")
    print(f"h+ / h-                   : {h_plus} / {h_minus}")
    print(f"q = a/b                   : {a}/{b} = {a/b:.15g}")
    print(f"S+                        : {b} W(h+) - {a} R(h+)")
    print(f"S-                        : {a} R(h-) - {b} W(h-)")

    # Denominator positivity.
    denom = {}
    for label, vec in [("R_plus", Rp), ("R_minus", Rm)]:
        denom[label] = {}
        for method in methods:
            denom[label][method] = solve_face_range(
                lp, optimum, vec, method=method, time_limit=float(args.time_limit)
            )
        mins = [r["min"] for r in denom[label].values() if r.get("success")]
        print(f"{label:27s}: min={min(mins):.15e}")

    face = {"S_plus": {}, "S_minus": {}}
    print("-"*120)
    print("A. DIRECT FACE MINIMA")
    for label, vec in [("S_plus", Splus), ("S_minus", Sminus)]:
        vals = []
        for method in methods:
            rr = solve_face_linear(
                lp, optimum, vec, method=method, time_limit=float(args.time_limit)
            )
            face[label][method] = rr
            if rr.get("success"):
                vals.append(float(rr["value"]))
                print(
                    f"{label:8s} {method:10s}: min={rr['value']:.15e} "
                    f"KKTstat={rr['kkt'].get('stationarity_inf', math.nan):.2e} "
                    f"dualgap={rr['kkt'].get('duality_gap_abs', math.nan):.2e}"
                )
            else:
                print(f"{label:8s} {method:10s}: FAILED {rr['message']}")
        if vals:
            print(f"{label:8s} conservative min: {min(vals):.15e}")

    print("-"*120)
    print("B. OBJECTIVE PENALTY TO CROSS THE THRESHOLD")
    crossing = {"S_plus": {}, "S_minus": {}}
    for label, vec in [("S_plus", Splus), ("S_minus", Sminus)]:
        vals = []
        for method in methods:
            rr = solve_crossing_penalty(
                lp, vec, method=method, time_limit=float(args.time_limit), label=label
            )
            crossing[label][method] = rr
            if rr.get("success"):
                penalty = float(rr["objective"] - optimum)
                rr["penalty_over_optimum"] = penalty
                vals.append(penalty)
                print(
                    f"{label:8s} {method:10s}: v_cross={rr['objective']:.15e} "
                    f"penalty={penalty:.15e} "
                    f"sep@cross={rr['separator_value_at_cross_optimum']:.3e} "
                    f"dual_sep={rr.get('separator_dual_marginal')}"
                )
            elif rr.get("status_code") == 2:
                print(f"{label:8s} {method:10s}: CROSSING HALFSPACE INFEASIBLE")
            else:
                print(f"{label:8s} {method:10s}: UNRESOLVED {rr['message']}")
        if vals:
            print(f"{label:8s} conservative crossing penalty: {min(vals):.15e}")

    # Decision ledger.
    Rpos = all(
        any(r.get("success") and float(r["min"]) > float(args.margin_tol)
            for r in denom[label].values())
        for label in ("R_plus", "R_minus")
    )

    face_mins = {}
    face_ok = True
    for label in ("S_plus", "S_minus"):
        vals = [float(r["value"]) for r in face[label].values() if r.get("success")]
        face_mins[label] = min(vals) if vals else -math.inf
        if not vals or face_mins[label] <= float(args.margin_tol):
            face_ok = False
        if vals and not method_agreement(vals, float(args.method_abs_tol), float(args.method_rel_tol)):
            face_ok = False

    cross_pen = {}
    cross_ok = True
    for label in ("S_plus", "S_minus"):
        vals = []
        statuses = []
        for r in crossing[label].values():
            statuses.append(int(r.get("status_code", 99)))
            if r.get("success"):
                vals.append(float(r["objective"] - optimum))
        if vals:
            cross_pen[label] = min(vals)
            if cross_pen[label] <= float(args.objective_gap_tol):
                cross_ok = False
            if not method_agreement(vals, float(args.method_abs_tol), float(args.method_rel_tol)):
                cross_ok = False
        elif statuses and all(s == 2 for s in statuses):
            cross_pen[label] = math.inf
        else:
            cross_pen[label] = -math.inf
            cross_ok = False

    kkt_ok = True
    for block in (face, crossing):
        for label in block:
            for r in block[label].values():
                if r.get("success") and r.get("kkt", {}).get("available"):
                    k = r["kkt"]
                    if (
                        float(k["stationarity_inf"]) > float(args.kkt_tol)
                        or float(k["primal_eq_inf"]) > float(args.kkt_tol)
                        or float(k["primal_ub_violation"]) > float(args.kkt_tol)
                        or float(k["duality_gap_abs"]) > float(args.kkt_tol)
                    ):
                        kkt_ok = False

    base_agree = method_agreement(
        good_base, float(args.method_abs_tol), float(args.method_rel_tol)
    )

    if Rpos and face_ok and cross_ok and kkt_ok and base_agree:
        verdict = "NORMALIZED_FRESH_PHASE_SEPARATED_BY_RATIONAL_THRESHOLD"
    else:
        verdict = "UNRESOLVED__CERTIFICATE_REQUIREMENTS_NOT_ALL_MET"

    print("="*120)
    print("FINAL CERTIFICATE")
    print("="*120)
    print(f"denominators positive      : {Rpos}")
    print(f"face sign margins          : S+={face_mins['S_plus']:.15e}  S-={face_mins['S_minus']:.15e}")
    print(f"crossing penalties         : S+={cross_pen['S_plus']:.15e}  S-={cross_pen['S_minus']:.15e}")
    print(f"KKT/dual audit             : {kkt_ok}")
    print(f"independent method agreement: {base_agree and face_ok and cross_ok}")
    print(f"VERDICT                    : {verdict}")
    if verdict.startswith("NORMALIZED_FRESH_PHASE"):
        print(f">>> FOR EVERY FINITE OPTIMIZER: G_{h_plus}(x={args.x}|s={args.s}) > {a}/{b}. <<<")
        print(f">>> FOR EVERY FINITE OPTIMIZER: G_{h_minus}(x={args.x}|s={args.s}) < {a}/{b}. <<<")
        print(">>> THEREFORE NO HISTORY-INDEPENDENT NORMALIZED ROUTING PROFILE EXISTS ON THIS ANTIDIAGONAL. <<<")
    print(">>> FINITE NUMERICAL LP CERTIFICATE ONLY; NO SCALE OR O(log n) CLAIM. <<<")
    print("="*120)

    return {
        "version": VERSION,
        "epistemic_contract": {
            "finite_numerical_certificate": True,
            "infinite_horizon_claim": False,
            "objective_face_only": False,
            "crossing_penalty_used": True,
            "dual_kkt_audited": True,
        },
        "parameters": vars(args),
        "base_runs": base_runs,
        "optimum_used": optimum,
        "auto_separator_scan": auto,
        "separator": {
            "s": int(args.s), "x": int(args.x),
            "h_plus": list(h_plus), "h_minus": list(h_minus),
            "numerator": a, "denominator": b, "value": a/b,
            "S_plus_formula": f"{b}*W(h_plus)-{a}*R(h_plus)",
            "S_minus_formula": f"{a}*R(h_minus)-{b}*W(h_minus)",
        },
        "denominator_audit": denom,
        "face_minima": face,
        "crossing_penalties": crossing,
        "decision": {
            "R_positive": Rpos,
            "face_ok": face_ok,
            "crossing_ok": cross_ok,
            "kkt_ok": kkt_ok,
            "base_methods_agree": base_agree,
            "face_min_conservative": face_mins,
            "crossing_penalty_conservative": cross_pen,
            "verdict": verdict,
        },
        "final_readout": verdict,
    }


def main():
    ap = argparse.ArgumentParser(
        description="DREAM6-ZR v0.13 rational normalized-phase separator certifier"
    )
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--history-depth", type=int, default=-1)
    ap.add_argument("--s", type=int, default=1)
    ap.add_argument("--x", type=int, default=0)
    ap.add_argument("--h-plus", default="auto")
    ap.add_argument("--h-minus", default="auto")
    ap.add_argument("--separator", default="auto", help='e.g. "1/3" or auto')
    ap.add_argument("--max-den", type=int, default=64)
    ap.add_argument("--scan-method", default="highs-ipm")
    ap.add_argument("--methods", default="highs-ds,highs-ipm")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--progress-every", type=int, default=5)

    ap.add_argument("--margin-tol", type=float, default=1e-8)
    ap.add_argument("--objective-gap-tol", type=float, default=1e-9)
    ap.add_argument("--kkt-tol", type=float, default=5e-8)
    ap.add_argument("--method-abs-tol", type=float, default=1e-8)
    ap.add_argument("--method-rel-tol", type=float, default=1e-8)
    ap.add_argument("--out", default="v013_normalized_phase_separator.json")
    args = ap.parse_args()

    allowed = {"highs", "highs-ds", "highs-ipm"}
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.scan_method not in allowed or any(m not in allowed for m in methods):
        raise SystemExit("unsupported HiGHS method")

    payload = run_v013(args)
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", out.resolve())


if __name__ == "__main__":
    main()
