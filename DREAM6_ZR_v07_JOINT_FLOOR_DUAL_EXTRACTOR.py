#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6-ZR v0.7 — JOINT FLOOR DUAL EXTRACTOR
=============================================

Purpose
-------
Explain the nonzero joint anti-diagonal history-spread floor on the ENTIRE
optimal face of the finite quartic zero-repair LP.

This is not a deeper-horizon optimizer and not a curve-fit program.  It asks:

    WHY can the joint bulk floor not be zero?

Pipeline
--------
    exact carry semantics (finite M)
      -> solve the base finite LP
      -> freeze the optimal face
      -> solve ONE joint min-max history-spread LP
      -> extract HiGHS dual multipliers of the joint LP
      -> identify which spread cells/histories are load-bearing
      -> try to collapse the dual support to an anti-diagonal mass-conservation
         certificate
      -> verify the induced conservation functional over the whole frozen face

The key distinction is:

    PRIMAL ACTIVE != DUAL LOAD-BEARING.

Many cells can tie at the joint optimum while only a strict subset is needed by
one valid dual lower-bound certificate.

Epistemic contract
------------------
* A single LP vertex is never interpreted as structure.
* A positive joint floor is a finite numerical optimal-face fact, not an
  infinite-horizon theorem.
* Rational reconstruction is used only on dimensionless dual weights (e.g.
  1/3); it is not used to pretend the B8-dependent floor is exact.
* The word PROVED is reserved for algebraic implications conditional on the
  displayed finite-LP quantities; floating HiGHS values remain NUMERICAL.

Default reproduction target
---------------------------
Depth 3, M=8, history depth 2, bulk s={2,3}.  The expected joint floor is about
1.475854e-2.  A clean dual certificate is expected to place weight 1/3 on the
three s=2 cells x=0,1,2, comparing histories (3,0) and (2,1).
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

VERSION = "DREAM6_ZR_v0.7_JOINT_FLOOR_DUAL_EXTRACTOR"

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


def run_case(args) -> dict:
    depth = int(args.depth)
    M = int(args.M)
    history_depth = depth - 1 if args.history_depth < 0 else int(args.history_depth)
    structural_s = (
        [depth - 1, depth]
        if args.s.strip().lower() == "auto"
        else [int(v.strip()) for v in args.s.split(",") if v.strip()]
    )

    print("=" * 104)
    print("DREAM6-ZR v0.7  JOINT FLOOR DUAL EXTRACTOR")
    print("=" * 104)
    print(f"depth / M           : {depth} / {M}")
    print(f"history depth       : {history_depth}")
    print(f"bulk anti-diagonals : {structural_s}")
    print(f"face slab tolerance : {args.face_tol:.3e}")
    print("vertex interpretation: FORBIDDEN")
    print("mission             : explain the joint floor, not merely measure it")
    print("=" * 104, flush=True)

    tree = build_tree(depth)
    chart = build_chart(M)
    lp = build_exact_lp(tree, chart)
    base, base_sec = solve_base(lp)
    if not base.success:
        raise RuntimeError(base.message)
    optimum = float(base.fun)
    histories = [h for h in tree.internals if len(h) == history_depth]
    print(f"base optimum        : {optimum:.15g}")
    print(f"histories tested    : {len(histories)}")

    observables = []
    for s0 in structural_s:
        for x0 in range(5):
            # Keep only cells that exist for at least one history. build_joint_problem
            # will discard cells with fewer than two nonempty histories.
            observables.append((
                f"s={s0},x={x0}",
                lambda h, ss=s0, xx=x0: w_entries(chart, lp, h, ss, xx),
            ))

    results = {}
    for mode in args.face_mode:
        problem = build_joint_problem(
            chart, lp, optimum, histories, observables,
            face_mode=mode, face_tol=float(args.face_tol),
        )
        res, sec = solve_joint(problem)
        if not res.success:
            results[mode] = {"success": False, "error": res.message, "seconds": sec}
            print(f"{mode.upper():5s} joint floor  : FAILED  {res.message}")
            continue

        primal_table = primal_observable_table(problem, res)
        support = extract_dual_support(
            problem, res,
            weight_tol=float(args.dual_weight_tol),
            rational_max_den=int(args.rational_max_den),
            rational_tol=float(args.rational_tol),
        )
        kkt = kkt_numeric_audit(problem, res)
        induced = induced_dual_functional(problem, support)
        induced_range = face_range_exact(lp, optimum, induced)
        partition = detect_partition_certificate(
            chart, lp, optimum, problem, res, support,
            weight_tol=float(args.dual_weight_tol),
            equality_tol=float(args.dual_equal_tol),
        )

        active_primal = [
            r for r in primal_table
            if abs(float(r["spread"]) - float(res.fun)) <= float(args.active_tol)
        ]
        load_bearing = [
            {
                "label": rec["meta"][2],
                "weight": rec["weight"],
                "rational_weight": rec.get("rational_weight"),
                "slack": rec["slack"],
            }
            for rec in support["active_spread_caps"]
        ]

        results[mode] = {
            "success": True,
            "joint_floor": float(res.fun),
            "seconds": sec,
            "observables_tested": int(problem["m"]),
            "primal_observables": primal_table,
            "primal_active_at_floor": active_primal,
            "dual_support": support,
            "kkt_numeric_audit": kkt,
            "induced_dual_functional_exact_face_range": induced_range,
            "partition_certificate": partition,
        }

        print("-" * 104)
        print(f"{mode.upper():5s} joint floor  : {res.fun:.15e}")
        print(f"primal cells @ floor: {len(active_primal)} / {problem['m']}")
        print("dual load-bearing caps:")
        for rec in load_bearing:
            rw = rec["rational_weight"] or "-"
            print(
                f"  {rec['label']:10s} weight={rec['weight']:.15g}  rational={rw}  slack={rec['slack']:.3e}"
            )
        print(f"sum dual cap weights: {support['alpha_sum']:.15g}")
        print(
            "KKT numeric audit   : "
            f"stationarity={kkt['stationarity_max_abs']:.3e}  "
            f"gap={kkt['primal_dual_gap']:.3e}"
        )

        if induced_range.get("success"):
            print(
                "dual-induced L(z) : "
                f"[{induced_range['min']:.15e}, {induced_range['max']:.15e}]  "
                f"width={induced_range['width']:.3e}"
            )

        if partition.get("success"):
            print("COMMON BOTTLENECK    : FOUND")
            print(f"  type               : {partition['kind']}")
            print(f"  h+ / h-            : {tuple(partition['h_plus'])} / {tuple(partition['h_minus'])}")
            print(f"  anti-diagonal s    : {partition['s']}")
            print(f"  x partition        : {partition['x_partition']}")
            rr = partition["R_difference_exact_face"]
            print(
                "  R-difference range : "
                f"[{rr['min']:.15e}, {rr['max']:.15e}]  width={rr['width']:.3e}"
            )
            print(f"  pigeonhole floor   : {partition['certificate_floor']:.15e}")
            print(f"  joint floor        : {partition['joint_floor']:.15e}")
            print(f"  difference         : {partition['joint_minus_certificate']:.3e}")
            print(f"  equation           : {partition['equation']}")
        else:
            print("COMMON BOTTLENECK    : NOT REDUCED TO ONE PARTITION CERTIFICATE")
            print(f"  reason             : {partition.get('reason')}")

    # Cross-mode consistency: slab should approach exact as tolerance shrinks.
    exact = results.get("exact", {})
    slab = results.get("slab", {})
    if exact.get("success") and slab.get("success"):
        consistency = {
            "exact_minus_slab": float(exact["joint_floor"] - slab["joint_floor"]),
            "scaled_by_face_tol": float(
                (exact["joint_floor"] - slab["joint_floor"]) / max(args.face_tol, 1e-300)
            ),
        }
    else:
        consistency = None

    final = "UNRESOLVED"
    exact_part = exact.get("partition_certificate", {}) if exact else {}
    if exact.get("success") and exact_part.get("success"):
        rr = exact_part["R_difference_exact_face"]
        idr = exact_part.get("routing_vs_AQ_identity_exact_face", {})
        floor_gap = abs(float(exact_part["joint_minus_certificate"]))
        if (
            rr.get("success")
            and float(rr["width"]) <= args.invariant_tol
            and idr.get("success")
            and max(abs(float(idr["min"])), abs(float(idr["max"]))) <= args.invariant_tol
            and floor_gap <= args.floor_match_tol
        ):
            final = "FINITE_JOINT_FLOOR_EXPLAINED_BY_ANTIDIAGONAL_MASS_CONSERVATION"
        else:
            final = "PARTITION_CANDIDATE_FOUND_BUT_NUMERIC_CERTIFICATION_INCOMPLETE"

    print("=" * 104)
    print("FINAL READOUT")
    print("=" * 104)
    print("joint-floor mechanism:", final)
    if final.startswith("FINITE_JOINT_FLOOR_EXPLAINED"):
        print(">>> THE DEPTH-FINITE JOINT FLOOR IS NOT A MYSTERIOUS SIX-CELL EFFECT. <<<")
        print(">>> ONE LOAD-BEARING DUAL CERTIFICATE REDUCES IT TO AN ANTI-DIAGONAL TOTAL-MASS MISMATCH. <<<")
        print(">>> THIS DOES NOT YET DECIDE THE INFINITE-HORIZON O(log n) UPPER BOUND. <<<")
    else:
        print(">>> NO COMPLETE FINITE CONSERVATION EXPLANATION CERTIFIED BY THIS RUN. <<<")
    print("=" * 104)

    return {
        "version": VERSION,
        "epistemic_contract": {
            "single_vertex_is_structure": False,
            "finite_face_is_infinite_theorem": False,
            "rationalize_floor": False,
            "rationalize_dimensionless_dual_weights_only": True,
        },
        "parameters": {
            "depth": depth,
            "M": M,
            "history_depth": history_depth,
            "s": structural_s,
            "face_tol": float(args.face_tol),
            "face_mode": list(args.face_mode),
        },
        "base": {
            "success": True,
            "optimum": optimum,
            "seconds": base_sec,
            "variables": int(lp["nv"]),
            "eq_rows": int(lp["Aeq"].shape[0]),
            "ub_rows": int(lp["Aub"].shape[0]),
        },
        "joint": results,
        "exact_vs_slab": consistency,
        "final_readout": final,
    }


def parse_face_modes(text: str) -> list[str]:
    t = text.strip().lower()
    if t == "both":
        return ["slab", "exact"]
    vals = [v.strip() for v in t.split(",") if v.strip()]
    bad = [v for v in vals if v not in {"slab", "exact"}]
    if bad or not vals:
        raise argparse.ArgumentTypeError("--face-mode must be slab, exact, or both")
    return vals


def main():
    ap = argparse.ArgumentParser(
        description="DREAM6-ZR v0.7: explain the joint bulk floor by extracting its dual certificate"
    )
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--history-depth", type=int, default=-1)
    ap.add_argument(
        "--s", default="auto",
        help="comma-separated anti-diagonals; auto means depth-1,depth (depth3 -> 2,3; depth4 -> 3,4)",
    )
    ap.add_argument("--face-tol", type=float, default=1e-9)
    ap.add_argument("--face-mode", type=parse_face_modes, default=["slab", "exact"])
    ap.add_argument("--dual-weight-tol", type=float, default=1e-8)
    ap.add_argument("--dual-equal-tol", type=float, default=1e-8)
    ap.add_argument("--active-tol", type=float, default=1e-8)
    ap.add_argument("--rational-max-den", type=int, default=4096)
    ap.add_argument("--rational-tol", type=float, default=1e-10)
    ap.add_argument("--invariant-tol", type=float, default=1e-9)
    ap.add_argument("--floor-match-tol", type=float, default=1e-9)
    ap.add_argument("--out", default="DREAM6_ZR_v07_joint_floor_dual.json")
    args = ap.parse_args()

    if args.depth < 1:
        raise SystemExit("--depth must be >=1")
    if args.M < 0:
        raise SystemExit("--M must be >=0")
    if args.history_depth >= args.depth:
        raise SystemExit("--history-depth must be < depth")

    payload = run_case(args)
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", out.resolve())


if __name__ == "__main__":
    main()
