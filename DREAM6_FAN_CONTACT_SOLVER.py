#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6 FAN-CONTACT SOLVER / AUDITOR v0.1
========================================

Purpose
-------
This program is designed to test the structural statement

    "the bill belongs to the fan, not the path"

inside the exact finite-horizon quartic zero-repair problem.

It does two logically separate jobs:

(A) EXACT FULL-TREE SOLVE (finite horizon)
    It can solve the original occupancy LP, with all output histories, for
    modest horizons n.  The formulation is exact at finite n (up to the
    numerical LP solve): fresh Y~Q, exact IID X~P, nonnegative buffer, and
    exact causal flow.

(B) FRONTIER / CONTACT AUDIT
    Given solved horizons 1,...,N, it reconstructs every conditional buffer
    law B_h from the active state occupancies and computes

        Delta_h = E[B_h] - B_{n-|h|},

    where B_r is the optimal root bill at remaining horizon r.

    It then checks the theorem

        sum_x P(x) Delta_{hx}
          = Delta_h + B_r - B_{r-1},

    at every internal node for which the needed reference values are known.

    It also searches for frontier contacts Delta_h ~= 0 and, crucially,
    recursive contacts whose conditional law matches a lower-horizon optimal
    root law.  If a contact h returns to the frontier at descendant hg*, the
    solver verifies the multiscale counterfactual export identity

        B_r - B_{r-s}
          = sum_{|g|=s, g != g*} P(g) Delta_{hg}

    (for contact h and contact hg*).

The output is therefore not merely "a contact exists".  It reports exactly
which counterfactual branches carry the missing bill and checks the numerical
residual of the conservation identity.

Asymptotics
-----------
The program also writes descriptive growth diagnostics for the available
sequence B_n: increments, dyadic increments, log/sqrt/n^(1/4) least-squares
fits, and an optional free power fit a+b*n^alpha.  These diagnostics are NOT
proofs of asymptotic growth.  Full-tree LP size grows exponentially, so large-n
asymptotics require either a separate compressed exact theorem or external
series supplied with --series-csv.

Default quartic pair
--------------------
    P = (1/8, 1/4, 1/4, 1/4, 1/8)
    Q = (7/64, 5/16, 5/32, 5/16, 7/64)
    G_P(z)-G_Q(z) = (1-z)^4/64
    E P = E Q = 2

Typical use
-----------
1) Audit an existing DREAM6-IMBA directory containing
   horizon_XXX_signature.json and horizon_XXX_mu_active.csv:

    python DREAM6_FAN_CONTACT_SOLVER.py --audit-dir .

2) Solve all horizons through 4 and audit them:

    python DREAM6_FAN_CONTACT_SOLVER.py --solve-through 4 --out-dir fan_run

3) Re-use existing n<=6 outputs and ask for a looser contact tolerance at a
   large numerical solve:

    python DREAM6_FAN_CONTACT_SOLVER.py --audit-dir . --contact-tol 1e-7

4) Add an external long-n series (CSV columns: n,bill[,label]):

    python DREAM6_FAN_CONTACT_SOLVER.py --audit-dir . --series-csv mem_series.csv

Dependencies
------------
    Python >= 3.10
    numpy
    scipy

No local finisher, no heuristic repair, no hidden branch simplification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit, linprog
from scipy.sparse import coo_matrix, csr_matrix


VERSION = "DREAM6 FAN-CONTACT v0.1"

VALUES = (0, 1, 2, 3, 4)
P_FRAC = (
    Fraction(1, 8),
    Fraction(1, 4),
    Fraction(1, 4),
    Fraction(1, 4),
    Fraction(1, 8),
)
Q_FRAC = (
    Fraction(7, 64),
    Fraction(5, 16),
    Fraction(5, 32),
    Fraction(5, 16),
    Fraction(7, 64),
)
P = np.asarray([float(x) for x in P_FRAC], dtype=float)
Q = np.asarray([float(x) for x in Q_FRAC], dtype=float)
P_BY_X = {x: float(p) for x, p in zip(VALUES, P_FRAC)}
Q_BY_Y = {y: float(q) for y, q in zip(VALUES, Q_FRAC)}


def history_key(h: Tuple[int, ...]) -> str:
    return "-" if not h else ",".join(map(str, h))


def parse_history(s: str) -> Tuple[int, ...]:
    s = s.strip()
    if s in ("", "-"):
        return ()
    return tuple(int(x) for x in s.split(","))


def p_word(g: Sequence[int]) -> float:
    out = 1.0
    for x in g:
        out *= P_BY_X[int(x)]
    return float(out)


def rational_hint(x: float, max_den: int = 1_000_000, tol: float = 1e-11) -> str | None:
    if not np.isfinite(x):
        return None
    q = Fraction(float(x)).limit_denominator(max_den)
    if abs(float(q) - float(x)) <= tol * max(1.0, abs(float(x))):
        return str(q)
    return None


def l1_pmf(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in set(a) | set(b)))


# ===========================================================================
# Exact finite-horizon occupancy LP
# ===========================================================================

@dataclass
class LPModel:
    n: int
    mu: Dict[Tuple[int, int, Tuple[int, ...]], int]
    nu: Dict[Tuple[int, int, Tuple[int, ...], int, int], int]
    c: np.ndarray
    A_eq: csr_matrix
    b_eq: np.ndarray
    j0_max: int
    jmax_by_t: Tuple[int, ...]


def estimate_model_size(n: int) -> dict:
    if n < 1:
        raise ValueError("n must be >= 1")
    xs = VALUES
    ys = VALUES
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    xmin = min(xs)
    j0_max = n * max(0, xmax - ymin)
    up_step = max(0, ymax - xmin)
    jmax_by_t = [j0_max + t * up_step for t in range(n + 1)]

    mu_est = 0
    nu_est = 0
    for t in range(n + 1):
        nh = len(xs) ** t
        mu_est += nh * (jmax_by_t[t] + 1)
        if t < n:
            nu_est += nh * (jmax_by_t[t] + 1) * len(ys) * len(xs)

    return {
        "mu_upper": int(mu_est),
        "nu_upper": int(nu_est),
        "variables_upper": int(mu_est + nu_est),
        "j0_max": int(j0_max),
        "jmax_by_t": jmax_by_t,
    }


def build_model(n: int, max_variables: int) -> LPModel:
    size = estimate_model_size(n)
    if size["variables_upper"] > max_variables:
        raise RuntimeError(
            f"horizon n={n}: estimated upper bound {size['variables_upper']:,} variables "
            f"exceeds --max-variables={max_variables:,}. Increase deliberately if desired."
        )

    xs = VALUES
    ys = VALUES
    j0_max = int(size["j0_max"])
    jmax_by_t = tuple(int(x) for x in size["jmax_by_t"])

    mu: Dict[Tuple[int, int, Tuple[int, ...]], int] = {}
    nu: Dict[Tuple[int, int, Tuple[int, ...], int, int], int] = {}
    costs: List[float] = []

    def add_var(cost: float = 0.0) -> int:
        i = len(costs)
        costs.append(float(cost))
        return i

    for t in range(n + 1):
        for h in product(xs, repeat=t):
            for j in range(jmax_by_t[t] + 1):
                mu[(t, j, h)] = add_var(float(j) if t == 0 else 0.0)

    for t in range(n):
        for h in product(xs, repeat=t):
            for j in range(jmax_by_t[t] + 1):
                for y in ys:
                    for x in xs:
                        jp = j + y - x
                        if 0 <= jp <= jmax_by_t[t + 1]:
                            nu[(t, j, h, y, x)] = add_var(0.0)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    rhs: List[float] = []
    row = 0

    def coef(i: int, a: float) -> None:
        rows.append(row)
        cols.append(i)
        data.append(float(a))

    # Root normalization.
    for j in range(j0_max + 1):
        coef(mu[(0, j, ())], 1.0)
    rhs.append(1.0)
    row += 1

    # Fresh Y and causal action mass:
    # sum_x nu(j,h,y,x) = Q(y) mu(j,h).
    for t in range(n):
        for h in product(xs, repeat=t):
            for j in range(jmax_by_t[t] + 1):
                for y in ys:
                    for x in xs:
                        ni = nu.get((t, j, h, y, x))
                        if ni is not None:
                            coef(ni, 1.0)
                    coef(mu[(t, j, h)], -Q_BY_Y[y])
                    rhs.append(0.0)
                    row += 1

    # Flow.
    for t in range(n):
        predecessors: Dict[Tuple[int, Tuple[int, ...]], List[int]] = defaultdict(list)
        for (tt, j, h, y, x), ni in nu.items():
            if tt != t:
                continue
            jp = j + y - x
            predecessors[(jp, h + (x,))].append(ni)

        for h2 in product(xs, repeat=t + 1):
            for jp in range(jmax_by_t[t + 1] + 1):
                coef(mu[(t + 1, jp, h2)], 1.0)
                for ni in predecessors.get((jp, h2), ()):
                    coef(ni, -1.0)
                rhs.append(0.0)
                row += 1

    # Exact IID P output:
    # mass(X^{t+1}=hx) = P(x) mass(X^t=h).
    for t in range(n):
        for h in product(xs, repeat=t):
            for x in xs:
                for jp in range(jmax_by_t[t + 1] + 1):
                    coef(mu[(t + 1, jp, h + (x,))], 1.0)
                for j in range(jmax_by_t[t] + 1):
                    coef(mu[(t, j, h)], -P_BY_X[x])
                rhs.append(0.0)
                row += 1

    A_eq = coo_matrix(
        (data, (rows, cols)),
        shape=(row, len(costs)),
        dtype=float,
    ).tocsr()

    return LPModel(
        n=n,
        mu=mu,
        nu=nu,
        c=np.asarray(costs, dtype=float),
        A_eq=A_eq,
        b_eq=np.asarray(rhs, dtype=float),
        j0_max=j0_max,
        jmax_by_t=jmax_by_t,
    )


def solve_model(model: LPModel, time_limit: float | None):
    options = {"presolve": True}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    t0 = time.time()
    result = linprog(
        model.c,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=(0.0, None),
        method="highs",
        options=options,
    )
    return result, time.time() - t0


def export_solution(out_dir: Path, model: LPModel, result, solve_seconds: float, tol: float) -> None:
    if not result.success:
        raise RuntimeError(f"LP n={model.n} failed: {result.message}")
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.asarray(result.x, dtype=float)

    mu_path = out_dir / f"horizon_{model.n:03d}_mu_active.csv"
    with mu_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "buffer", "x_history", "mass"])
        for (t, j, h), idx in sorted(model.mu.items()):
            mass = float(z[idx])
            if mass > tol:
                w.writerow([t, j, history_key(h), f"{mass:.17g}"])

    root_pmf = []
    for j in range(model.j0_max + 1):
        m = float(z[model.mu[(0, j, ())]])
        if m > tol:
            root_pmf.append({"j": int(j), "mass": m, "rational_hint": rational_hint(m)})

    eqlin_resid = np.asarray(result.eqlin.residual, dtype=float)
    max_resid = float(np.max(np.abs(eqlin_resid))) if eqlin_resid.size else 0.0
    sig = {
        "version": VERSION,
        "horizon": int(model.n),
        "optimal_bill": float(result.fun),
        "optimal_bill_rational_hint": rational_hint(float(result.fun)),
        "initial_buffer_distribution": root_pmf,
        "solver": {
            "success": bool(result.success),
            "message": str(result.message),
            "solve_seconds": float(solve_seconds),
            "variables": int(len(model.c)),
            "equalities": int(model.A_eq.shape[0]),
            "nnz": int(model.A_eq.nnz),
            "max_equality_residual": max_resid,
        },
        "pair": {
            "P": [str(x) for x in P_FRAC],
            "Q": [str(x) for x in Q_FRAC],
            "identity": "P(z)-Q(z)=(1-z)^4/64",
        },
    }
    with (out_dir / f"horizon_{model.n:03d}_signature.json").open("w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2, ensure_ascii=False)


# ===========================================================================
# Loading and reconstructing conditional laws from occupancy exports
# ===========================================================================

@dataclass
class HorizonData:
    n: int
    bill: float
    node_mass: Dict[Tuple[int, ...], float]
    node_mean: Dict[Tuple[int, ...], float]
    node_pmf: Dict[Tuple[int, ...], Dict[int, float]]
    signature_path: Path
    mu_path: Path
    solver_residual: float | None = None


def load_signature(path: Path) -> tuple[int, float, float | None]:
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    n = int(d["horizon"])
    bill = float(d["optimal_bill"])
    resid = None
    if isinstance(d.get("solver"), dict) and d["solver"].get("max_equality_residual") is not None:
        resid = float(d["solver"]["max_equality_residual"])
    return n, bill, resid


def load_mu(path: Path) -> tuple[
    Dict[Tuple[int, ...], float],
    Dict[Tuple[int, ...], float],
    Dict[Tuple[int, ...], Dict[int, float]],
]:
    raw: Dict[Tuple[int, ...], Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        needed = {"buffer", "x_history", "mass"}
        if not needed.issubset(set(r.fieldnames or [])):
            raise ValueError(f"{path}: expected columns {sorted(needed)}")
        for row in r:
            h = parse_history(row["x_history"])
            j = int(row["buffer"])
            m = float(row["mass"])
            raw[h][j] += m

    node_mass: Dict[Tuple[int, ...], float] = {}
    node_mean: Dict[Tuple[int, ...], float] = {}
    node_pmf: Dict[Tuple[int, ...], Dict[int, float]] = {}
    for h, d in raw.items():
        total = float(sum(d.values()))
        if total <= 0:
            continue
        node_mass[h] = total
        node_mean[h] = float(sum(j * m for j, m in d.items()) / total)
        node_pmf[h] = {int(j): float(m / total) for j, m in d.items() if m > 0}
    return node_mass, node_mean, node_pmf


def discover_horizons(path: Path) -> Dict[int, HorizonData]:
    out: Dict[int, HorizonData] = {}
    for sp in sorted(path.glob("horizon_*_signature.json")):
        try:
            n, bill, resid = load_signature(sp)
        except Exception:
            continue
        mp = path / f"horizon_{n:03d}_mu_active.csv"
        if not mp.exists():
            continue
        mass, mean, pmf = load_mu(mp)
        if () not in mean:
            continue
        out[n] = HorizonData(
            n=n,
            bill=bill,
            node_mass=mass,
            node_mean=mean,
            node_pmf=pmf,
            signature_path=sp,
            mu_path=mp,
            solver_residual=resid,
        )
    return out


# ===========================================================================
# Frontier/contact theorem audit
# ===========================================================================

@dataclass
class NodeInfo:
    h: Tuple[int, ...]
    remaining: int
    mean: float
    delta: float
    contact: bool
    root_l1: float | None
    law_match: bool


def node_infos(
    tree: HorizonData,
    refs: Mapping[int, HorizonData],
    contact_tol: float,
    law_tol: float,
) -> Dict[Tuple[int, ...], NodeInfo]:
    info: Dict[Tuple[int, ...], NodeInfo] = {}
    for h, mean in tree.node_mean.items():
        r = tree.n - len(h)
        if r not in refs and r != 0:
            continue
        b = 0.0 if r == 0 else refs[r].bill
        delta = float(mean - b)
        contact = abs(delta) <= contact_tol
        root_l1 = None
        law_match = False
        if r in refs and h in tree.node_pmf and () in refs[r].node_pmf:
            root_l1 = l1_pmf(tree.node_pmf[h], refs[r].node_pmf[()])
            law_match = root_l1 <= law_tol
        info[h] = NodeInfo(
            h=h,
            remaining=r,
            mean=float(mean),
            delta=delta,
            contact=contact,
            root_l1=root_l1,
            law_match=law_match,
        )
    return info


def descendants_at_distance(parent: Tuple[int, ...], s: int) -> Iterable[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    for g in product(VALUES, repeat=s):
        yield parent + tuple(g), tuple(g)


def audit_one_step(
    tree: HorizonData,
    info: Mapping[Tuple[int, ...], NodeInfo],
    refs: Mapping[int, HorizonData],
) -> tuple[list[dict], float]:
    rows = []
    max_abs = 0.0
    for h, ni in info.items():
        r = ni.remaining
        if r < 1 or (r - 1 not in refs and r - 1 != 0):
            continue
        child_infos = []
        ok = True
        for x in VALUES:
            hx = h + (x,)
            if hx not in info:
                ok = False
                break
            child_infos.append(info[hx])
        if not ok:
            continue
        lhs = float(sum(P_BY_X[x] * child_infos[x].delta for x in VALUES))
        bprev = 0.0 if r - 1 == 0 else refs[r - 1].bill
        target = float(ni.delta + refs[r].bill - bprev)
        resid = lhs - target
        max_abs = max(max_abs, abs(resid))
        rows.append({
            "history": history_key(h),
            "remaining": r,
            "delta_parent": ni.delta,
            "lhs_weighted_child_delta": lhs,
            "target": target,
            "residual": resid,
            "parent_contact": ni.contact,
            "contact_children": [x for x in VALUES if child_infos[x].contact],
            "law_match_children": [x for x in VALUES if child_infos[x].law_match],
        })
    return rows, max_abs


def contact_candidates(
    parent: Tuple[int, ...],
    info: Mapping[Tuple[int, ...], NodeInfo],
    max_jump: int,
    require_law_match: bool = True,
) -> list[tuple[int, Tuple[int, ...], NodeInfo]]:
    if parent not in info:
        return []
    r = info[parent].remaining
    out = []
    for s in range(1, min(max_jump, r) + 1):
        local = []
        for hg, g in descendants_at_distance(parent, s):
            ni = info.get(hg)
            if ni is None or ni.remaining < 1:
                continue
            if ni.contact and ((not require_law_match) or ni.law_match):
                local.append((s, g, ni))
        if local:
            local.sort(key=lambda x: (
                float("inf") if x[2].root_l1 is None else x[2].root_l1,
                x[1],
            ))
            return local
    return out


def build_contact_chain(
    tree: HorizonData,
    info: Mapping[Tuple[int, ...], NodeInfo],
    max_jump: int,
) -> list[Tuple[int, ...]]:
    if () not in info or not info[()].contact:
        return []
    chain = [()]
    parent = ()
    while info[parent].remaining > 0:
        cands = contact_candidates(parent, info, max_jump=max_jump, require_law_match=True)
        if not cands:
            break
        _, g, _ = cands[0]
        nxt = parent + g
        if nxt == parent:
            break
        chain.append(nxt)
        parent = nxt
    return chain


def fan_export_for_jump(
    parent: Tuple[int, ...],
    child: Tuple[int, ...],
    info: Mapping[Tuple[int, ...], NodeInfo],
    refs: Mapping[int, HorizonData],
) -> tuple[dict, list[dict]]:
    pinfo = info[parent]
    cinfo = info[child]
    s = len(child) - len(parent)
    if s <= 0:
        raise ValueError("child must be a strict descendant")
    gstar = child[len(parent):]
    r = pinfo.remaining
    if r - s < 0:
        raise ValueError("jump beyond leaf")
    bsmall = 0.0 if r - s == 0 else refs[r - s].bill
    target_increment = float(refs[r].bill - bsmall)

    branches = []
    total = 0.0
    total_excluding = 0.0
    for hg, g in descendants_at_distance(parent, s):
        ni = info.get(hg)
        if ni is None:
            continue
        w = p_word(g)
        c = float(w * ni.delta)
        total += c
        is_star = tuple(g) == tuple(gstar)
        if not is_star:
            total_excluding += c
        branches.append({
            "parent": history_key(parent),
            "g": history_key(tuple(g)),
            "descendant": history_key(hg),
            "jump": s,
            "probability": w,
            "mean": ni.mean,
            "delta": ni.delta,
            "weighted_delta": c,
            "is_contact": ni.contact,
            "law_match": ni.law_match,
            "root_l1": ni.root_l1,
            "selected_contact": is_star,
        })

    # General theorem target includes Delta_parent.
    theorem_target = float(pinfo.delta + target_increment)
    return {
        "parent": history_key(parent),
        "child_contact": history_key(child),
        "remaining_parent": r,
        "remaining_child": r - s,
        "jump": s,
        "delta_parent": pinfo.delta,
        "delta_child_contact": cinfo.delta,
        "child_law_match": cinfo.law_match,
        "child_root_l1": cinfo.root_l1,
        "B_r_minus_B_r_minus_s": target_increment,
        "weighted_all_descendants": total,
        "theorem_target": theorem_target,
        "theorem_residual": total - theorem_target,
        "weighted_counterfactual_excluding_contact": total_excluding,
        "counterfactual_residual": total_excluding - target_increment,
    }, branches


def audit_horizon(
    tree: HorizonData,
    refs: Mapping[int, HorizonData],
    contact_tol: float,
    law_tol: float,
    max_jump: int,
) -> dict:
    info = node_infos(tree, refs, contact_tol=contact_tol, law_tol=law_tol)
    one_rows, max_one_resid = audit_one_step(tree, info, refs)
    contact_parent_resids = [abs(r['residual']) for r in one_rows if r['parent_contact']]
    max_contact_parent_resid = max(contact_parent_resids, default=0.0)

    by_depth = []
    for t in range(tree.n + 1):
        nodes = [ni for h, ni in info.items() if len(h) == t]
        contacts = [ni for ni in nodes if ni.contact]
        law_matches = [ni for ni in nodes if ni.law_match]
        by_depth.append({
            "depth": t,
            "remaining": tree.n - t,
            "node_count": len(nodes),
            "contact_count": len(contacts),
            "law_match_count": len(law_matches),
            "contacts": [history_key(x.h) for x in contacts[:50]],
            "law_matches": [history_key(x.h) for x in law_matches[:50]],
        })

    chain = build_contact_chain(tree, info, max_jump=max_jump)
    jumps = []
    branch_rows = []
    for a, b in zip(chain, chain[1:]):
        jr, br = fan_export_for_jump(a, b, info, refs)
        jumps.append(jr)
        branch_rows.extend(br)

    # List the most interesting exact/near-exact lower-horizon law copies.
    matches = []
    for h, ni in sorted(info.items(), key=lambda kv: (len(kv[0]), kv[0])):
        if h == () or ni.remaining == 0:
            continue
        if ni.law_match:
            matches.append({
                "history": history_key(h),
                "depth": len(h),
                "remaining": ni.remaining,
                "mean": ni.mean,
                "delta": ni.delta,
                "root_l1": ni.root_l1,
            })

    return {
        "horizon": tree.n,
        "bill": tree.bill,
        "solver_residual": tree.solver_residual,
        "root_mean_from_mu": tree.node_mean.get(()),
        "root_bill_residual": tree.node_mean.get((), float("nan")) - tree.bill,
        "max_one_step_frontier_identity_residual_from_active_export": max_one_resid,
        "max_one_step_residual_on_contact_parents": max_contact_parent_resid,
        "depth_summary": by_depth,
        "contact_chain": [history_key(h) for h in chain],
        "contact_jumps": jumps,
        "recursive_lower_horizon_law_matches": matches,
        "one_step_rows": one_rows,
        "fan_branch_rows": branch_rows,
    }


# ===========================================================================
# Growth diagnostics (descriptive, not proof)
# ===========================================================================


def linear_fit(n: np.ndarray, b: np.ndarray, phi) -> dict:
    x = np.asarray([phi(float(k)) for k in n], dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, b, rcond=None)
    pred = X @ beta
    resid = b - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "rmse": rmse}


def growth_diagnostics(series: Mapping[int, float], label: str) -> dict:
    pts = sorted((int(n), float(v)) for n, v in series.items() if n >= 1 and np.isfinite(v))
    if not pts:
        return {"label": label, "points": []}
    ns = np.asarray([x[0] for x in pts], dtype=float)
    bs = np.asarray([x[1] for x in pts], dtype=float)

    increments = []
    for (n0, b0), (n1, b1) in zip(pts, pts[1:]):
        if n1 == n0 + 1:
            d = b1 - b0
            increments.append({
                "n": n1,
                "delta": d,
                "n_times_delta": n1 * d,
                "sqrt_n_times_delta": math.sqrt(n1) * d,
            })

    dyadic = []
    dct = dict(pts)
    for n, b in pts:
        if 2 * n in dct:
            dyadic.append({
                "n": n,
                "B_n": b,
                "B_2n": dct[2 * n],
                "doubling_increment": dct[2 * n] - b,
            })

    raw_slopes = []
    for (n0, b0), (n1, b1) in zip(pts, pts[1:]):
        if b0 > 0 and b1 > 0 and n1 > n0:
            raw_slopes.append({
                "n0": n0,
                "n1": n1,
                "alpha_raw": math.log(b1 / b0) / math.log(n1 / n0),
            })

    fits = {}
    if len(pts) >= 3:
        fits["a+b_log_n"] = linear_fit(ns, bs, lambda x: math.log(x))
        fits["a+b_sqrt_n"] = linear_fit(ns, bs, lambda x: math.sqrt(x))
        fits["a+b_n_quarter"] = linear_fit(ns, bs, lambda x: x ** 0.25)

    if len(pts) >= 5:
        def power_model(x, a, c, alpha):
            return a + c * np.power(x, alpha)
        try:
            p0 = [float(bs[0]) / 2.0, max(1e-8, float(bs[-1] - bs[0])) / max(1.0, ns[-1] ** 0.5), 0.5]
            pars, _ = curve_fit(
                power_model,
                ns,
                bs,
                p0=p0,
                bounds=([-10.0, -10.0, 0.01], [10.0, 10.0, 1.5]),
                maxfev=100000,
            )
            pred = power_model(ns, *pars)
            rmse = float(np.sqrt(np.mean((bs - pred) ** 2)))
            fits["a+c_n_alpha"] = {
                "a": float(pars[0]),
                "c": float(pars[1]),
                "alpha": float(pars[2]),
                "rmse": rmse,
            }
        except Exception as exc:
            fits["a+c_n_alpha"] = {"error": str(exc)}

    return {
        "label": label,
        "epistemic_status": "descriptive_only_not_asymptotic_proof",
        "points": [{"n": n, "bill": b} for n, b in pts],
        "increments": increments,
        "dyadic": dyadic,
        "raw_loglog_slopes": raw_slopes,
        "fits": fits,
    }


def load_external_series(path: Path) -> Dict[str, Dict[int, float]]:
    series: Dict[str, Dict[int, float]] = defaultdict(dict)
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = set(r.fieldnames or [])
        if not {"n", "bill"}.issubset(fields):
            raise ValueError(f"{path}: CSV must contain n,bill and optionally label")
        for row in r:
            label = row.get("label") or path.stem
            series[str(label)][int(row["n"])] = float(row["bill"])
    return dict(series)


# ===========================================================================
# Reporting
# ===========================================================================


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            rr = {}
            for k in keys:
                v = r.get(k)
                if isinstance(v, (list, dict, tuple)):
                    rr[k] = json.dumps(v, ensure_ascii=False)
                else:
                    rr[k] = v
            w.writerow(rr)


def write_markdown(out_dir: Path, audits: Sequence[dict], growth: Sequence[dict]) -> None:
    lines = []
    lines.append("# DREAM6 fan/contact audit\n")
    lines.append("The structural identity being checked is\n")
    lines.append("$$\\sum_x P(x)\\Delta_{hx}=\\Delta_h+\\mathcal B_r-\\mathcal B_{r-1}.$$\n")
    lines.append("A recursive contact jump of length $s$ additionally checks\n")
    lines.append("$$\\mathcal B_r-\\mathcal B_{r-s}=\\sum_{g\\neq g_*}P(g)\\Delta_{hg}.$$\n")
    lines.append("Numerical contact means $|\\Delta|$ is below the configured tolerance.  A `law match` additionally requires the conditional PMF to match the independently solved lower-horizon root PMF in $L^1$.\n")

    for a in audits:
        lines.append(f"## Horizon n={a['horizon']}\n")
        lines.append(f"- bill: `{a['bill']:.17g}`")
        lines.append(f"- max one-step residual on contact parents: `{a['max_one_step_residual_on_contact_parents']:.3e}`")
        lines.append(f"- max one-step residual over all nodes reconstructed from the active CSV: `{a['max_one_step_frontier_identity_residual_from_active_export']:.3e}` (may be contaminated by active-export truncation in tiny cells)")
        if a.get("solver_residual") is not None:
            lines.append(f"- source LP equality residual: `{a['solver_residual']:.3e}`")
        chain = a.get("contact_chain", [])
        lines.append(f"- selected recursive contact chain: `{chain}`")
        matches = a.get("recursive_lower_horizon_law_matches", [])
        lines.append(f"- lower-horizon law matches: `{len(matches)}`")
        lines.append("")
        if a.get("contact_jumps"):
            lines.append("| parent | contact descendant | jump | target bill | counterfactual sum | residual | law L1 |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for j in a["contact_jumps"]:
                l1 = j.get("child_root_l1")
                lines.append(
                    f"| {j['parent']} | {j['child_contact']} | {j['jump']} | "
                    f"{j['B_r_minus_B_r_minus_s']:.12g} | "
                    f"{j['weighted_counterfactual_excluding_contact']:.12g} | "
                    f"{j['counterfactual_residual']:.3e} | "
                    f"{'' if l1 is None else f'{l1:.3e}'} |"
                )
            lines.append("")

    lines.append("## Growth diagnostics\n")
    lines.append("These are descriptive diagnostics only; they do not prove an asymptotic law.\n")
    for g in growth:
        lines.append(f"### {g['label']}\n")
        if g.get("dyadic"):
            lines.append("Dyadic increments:")
            for d in g["dyadic"]:
                lines.append(f"- n={d['n']}: B_2n-B_n = `{d['doubling_increment']:.12g}`")
        fits = g.get("fits", {})
        if fits:
            lines.append("Fits (smaller RMSE only means better fit on the supplied finite sample):")
            for name, spec in fits.items():
                lines.append(f"- `{name}`: `{spec}`")
        lines.append("")

    (out_dir / "fan_contact_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    audit_dir: Path,
    out_dir: Path,
    contact_tol: float,
    law_tol: float,
    max_jump: int,
    series_csv: Path | None,
) -> tuple[list[dict], list[dict]]:
    refs = discover_horizons(audit_dir)
    if not refs:
        raise RuntimeError(
            f"No compatible horizon_XXX_signature.json + horizon_XXX_mu_active.csv pairs in {audit_dir}"
        )

    # Need all intermediate reference bills to evaluate every remaining horizon.
    available = sorted(refs)
    missing = [r for r in range(1, max(available) + 1) if r not in refs]
    if missing:
        print(f"[warning] missing reference horizons: {missing}; some Delta values cannot be evaluated")

    audits = []
    all_one = []
    all_branches = []
    all_jumps = []
    all_matches = []

    for n in available:
        # Audit only if all remaining references 1..n are available.
        needed = set(range(1, n + 1))
        if not needed.issubset(refs):
            print(f"[skip] n={n}: missing lower-horizon references")
            continue
        a = audit_horizon(
            refs[n], refs,
            contact_tol=contact_tol,
            law_tol=law_tol,
            max_jump=max_jump,
        )
        audits.append(a)
        for r in a["one_step_rows"]:
            all_one.append({"horizon": n, **r})
        for r in a["fan_branch_rows"]:
            all_branches.append({"horizon": n, **r})
        for r in a["contact_jumps"]:
            all_jumps.append({"horizon": n, **r})
        for r in a["recursive_lower_horizon_law_matches"]:
            all_matches.append({"horizon": n, **r})

        print(
            f"[audit n={n}] B={a['bill']:.12g} "
            f"max_contact_identity_res={a['max_one_step_residual_on_contact_parents']:.3e} "
            f"chain={a['contact_chain']}"
        )
        for j in a["contact_jumps"]:
            print(
                f"  [fan] {j['parent']} -> {j['child_contact']} (s={j['jump']}): "
                f"target={j['B_r_minus_B_r_minus_s']:.12g} "
                f"counterfactual={j['weighted_counterfactual_excluding_contact']:.12g} "
                f"res={j['counterfactual_residual']:.3e} "
                f"lawL1={j['child_root_l1'] if j['child_root_l1'] is not None else float('nan'):.3e}"
            )

    exact_series = {n: refs[n].bill for n in available}
    growth = [growth_diagnostics(exact_series, "full_tree_finite_horizon")]
    if series_csv is not None:
        for label, s in load_external_series(series_csv).items():
            growth.append(growth_diagnostics(s, label))

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "contact_tolerance": contact_tol,
        "law_l1_tolerance": law_tol,
        "max_contact_jump": max_jump,
        "epistemic_note": (
            "Finite-horizon LP and conservation residuals are numerical unless separately rationalized. "
            "Growth fits are descriptive only."
        ),
        "audits": audits,
        "growth": growth,
    }
    with (out_dir / "fan_contact_report.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_csv(out_dir / "one_step_frontier_identity.csv", all_one)
    write_csv(out_dir / "contact_jumps.csv", all_jumps)
    write_csv(out_dir / "fan_branch_contributions.csv", all_branches)
    write_csv(out_dir / "recursive_law_matches.csv", all_matches)

    growth_rows = []
    for g in growth:
        for p in g.get("points", []):
            growth_rows.append({"label": g["label"], **p})
    write_csv(out_dir / "growth_series.csv", growth_rows)
    write_markdown(out_dir, audits, growth)
    return audits, growth


# ===========================================================================
# CLI
# ===========================================================================


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Exact quartic zero-repair full-tree solver + frontier-contact/fan audit."
    )
    ap.add_argument(
        "--solve-through",
        type=int,
        default=None,
        help="solve every exact full-tree horizon 1..N before auditing",
    )
    ap.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="directory containing existing horizon_XXX_signature.json and *_mu_active.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dream6_fan_contact_run"),
        help="output directory for solves/reports",
    )
    ap.add_argument(
        "--max-variables",
        type=int,
        default=700_000,
        help="safety cap for exact full-tree LP variable estimate",
    )
    ap.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="HiGHS time limit per solved horizon in seconds",
    )
    ap.add_argument(
        "--lp-export-tol",
        type=float,
        default=1e-12,
        help="active occupancy export threshold",
    )
    ap.add_argument(
        "--contact-tol",
        type=float,
        default=1e-8,
        help="absolute tolerance for Delta_h=0 contact detection",
    )
    ap.add_argument(
        "--law-tol",
        type=float,
        default=1e-8,
        help="L1 tolerance for matching B_h to an independently solved lower-horizon root law",
    )
    ap.add_argument(
        "--max-contact-jump",
        type=int,
        default=4,
        help="maximum number of levels skipped when searching for the next recursive contact",
    )
    ap.add_argument(
        "--series-csv",
        type=Path,
        default=None,
        help="optional external growth series CSV with columns n,bill[,label]",
    )
    args = ap.parse_args()

    print("=" * 92)
    print(VERSION)
    print("P =", [str(x) for x in P_FRAC])
    print("Q =", [str(x) for x in Q_FRAC])
    print("identity: G_P-G_Q=(1-z)^4/64; E P = E Q = 2")
    print("goal: locate recursive frontier contacts and audit where the bill is exported")
    print("=" * 92)

    if args.solve_through is None and args.audit_dir is None:
        # Convenient default: audit current directory if compatible data exist;
        # otherwise solve a modest exact calibration through n=4.
        here = Path(".")
        if discover_horizons(here):
            args.audit_dir = here
        else:
            args.solve_through = 4

    if args.solve_through is not None:
        if args.solve_through < 1:
            raise SystemExit("--solve-through must be >= 1")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for n in range(1, args.solve_through + 1):
            sig = args.out_dir / f"horizon_{n:03d}_signature.json"
            mu = args.out_dir / f"horizon_{n:03d}_mu_active.csv"
            if sig.exists() and mu.exists():
                print(f"[reuse] n={n}: existing solution in {args.out_dir}")
                continue
            est = estimate_model_size(n)
            print(
                f"[build] n={n} estimated vars<={est['variables_upper']:,} "
                f"j0_max={est['j0_max']}"
            )
            model = build_model(n, max_variables=args.max_variables)
            print(
                f"[lp] n={n} vars={len(model.c):,} eq={model.A_eq.shape[0]:,} "
                f"nnz={model.A_eq.nnz:,}"
            )
            result, sec = solve_model(model, args.time_limit)
            if not result.success:
                raise SystemExit(f"LP n={n} failed: {result.message}")
            print(f"[solve] n={n} B_n={result.fun:.15g} time={sec:.3f}s")
            export_solution(args.out_dir, model, result, sec, tol=args.lp_export_tol)
        audit_dir = args.out_dir
    else:
        audit_dir = args.audit_dir

    if audit_dir is None:
        raise SystemExit("No audit directory resolved")

    report_dir = args.out_dir
    run_audit(
        audit_dir=audit_dir,
        out_dir=report_dir,
        contact_tol=args.contact_tol,
        law_tol=args.law_tol,
        max_jump=args.max_contact_jump,
        series_csv=args.series_csv,
    )

    print("[done]", report_dir / "fan_contact_report.md")
    print("[done]", report_dir / "fan_contact_report.json")
    print("[done]", report_dir / "contact_jumps.csv")
    print("[done]", report_dir / "fan_branch_contributions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
