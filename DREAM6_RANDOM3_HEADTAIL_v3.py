#!/usr/bin/env python3
"""
DREAM6 RANDOM3 HEAD-TAIL v3
==========================

Specialized deterministic prototype for standard sparse random 3-SAT.

Active path
-----------

    pure 3-CNF
      -> fixed reinforced BP/cavity transport
      -> continuous field-geometry test
      -> [diffuse basin] direct final continuous field
         OR
         [structured basin]
            -> head-tail survey transport
            -> continuous BP/SP energy comparison
            -> adaptive continuous analog-SAT escape
      -> ONE simultaneous IEEE-754 binary32 sign readout
      -> independent exact CNF verifier

There are no random seeds, random restarts, Boolean flips, WalkSAT/GSAT,
branching, Boolean decimation, residual-guided repair, verifier feedback,
or intermediate Boolean assignments.

The head-tail idea is methodologically inspired by:

  T. Misiakiewicz and G. G. Wen,
  "The sharp SAT/UNSAT phase transition in random ellipsoid fitting",
  arXiv:2608.10184v1 (2026).

Their theorem is NOT used as a theorem about SAT.  The borrowed principle is
only the decomposition strategy: isolate a structured/high-influence component
and use a universal transport on the diffuse low-influence bulk.

Soundness
---------
A SAT verdict is emitted only after direct verification of every original CNF
clause.  Failure to find a satisfying assignment is reported as UNCLASSIFIED,
never UNSAT.  In particular, UUF instances require a separate UNSAT certificate
before an UNSAT claim can be made.

Complexity
----------
For pure width-3 CNF, every BP, survey, and analog sweep is O(n + L), where L
is the number of literal occurrences.  All default sweep counts are fixed
constants, so the implemented arithmetic-work count is O(n + L).  This is an
implementation complexity statement, not a completeness theorem for random
3-SAT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False
    njit = None

VERSION = "DREAM6_RANDOM3_HEADTAIL_v3"

# Fast diffuse BP channel: inherited from RANDOM3 FAST v2.
DEFAULT_BP_SWEEPS = 250
DEFAULT_BP_DAMPING = 0.78
DEFAULT_RHO_LOW = 0.065
DEFAULT_RHO_HIGH = 0.50
DEFAULT_RHO_HOLD = 100
DEFAULT_RHO_POWER = 1.50
DEFAULT_LOG_CLIP = 50.0
DEFAULT_EPSILON = 1.0e-12

# Continuous basin test.  This is NOT a Boolean residual test.
DEFAULT_WEAK_Q = 0.01
DEFAULT_STRUCTURED_TRIGGER = 1.0e-5

# Head-tail survey channel.
DEFAULT_SP_SWEEPS = 40
DEFAULT_SP_ETA0 = 0.60
DEFAULT_SP_BULK_DAMPING = 0.25
DEFAULT_SP_HEAD_DAMPING = 0.70
DEFAULT_SP_HEAD_THETA = 1.00
DEFAULT_SP_HEAD_GAMMA = 4.00
DEFAULT_SP_ENERGY_RATIO = 1.20
DEFAULT_SP_INIT_SCALE = 0.75

# Continuous analog escape for structured basins.
DEFAULT_ANALOG_STEPS = 20000
DEFAULT_ANALOG_MAX_DELTA = 0.02


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_dimacs_3sat(path: str | Path) -> tuple[int, np.ndarray]:
    nvars = None
    declared_clauses = None
    clauses: list[tuple[int, int, int]] = []
    current: list[int] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("c") or line.startswith("%"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) < 4 or parts[1].lower() != "cnf":
                    raise ValueError("expected DIMACS 'p cnf n m' header")
                nvars = int(parts[2])
                declared_clauses = int(parts[3])
                continue

            for token in line.split():
                lit = int(token)
                if lit == 0:
                    if not current:
                        continue
                    if len(current) != 3:
                        raise ValueError(
                            f"HEADTAIL v3 accepts pure 3-CNF only; got clause width {len(current)}"
                        )
                    clauses.append((current[0], current[1], current[2]))
                    current.clear()
                else:
                    current.append(lit)

    if current:
        raise ValueError("unterminated DIMACS clause")
    if nvars is None:
        raise ValueError("missing DIMACS header")
    if declared_clauses is not None and declared_clauses != len(clauses):
        raise ValueError(
            f"DIMACS header says {declared_clauses} clauses, parsed {len(clauses)}"
        )
    if not clauses:
        raise ValueError("empty formula")

    arr = np.asarray(clauses, dtype=np.int32)
    if int(np.max(np.abs(arr))) > int(nvars):
        raise ValueError("literal variable index outside DIMACS range")
    return int(nvars), arr


def compile_incidence(nvars: int, clauses: np.ndarray) -> dict[str, np.ndarray]:
    m = int(clauses.shape[0])
    flat = clauses.reshape(-1)
    edge_var = np.abs(flat).astype(np.int64, copy=False) - 1
    edge_sign = np.where(flat > 0, 1.0, -1.0).astype(np.float64, copy=False)
    edge_factor = np.repeat(np.arange(m, dtype=np.int64), 3)
    factor_offsets = np.arange(0, 3 * m + 1, 3, dtype=np.int64)
    occurrence = np.bincount(edge_var, minlength=nvars).astype(np.int64, copy=False)
    return {
        "edge_var": edge_var,
        "edge_sign": edge_sign,
        "edge_factor": edge_factor,
        "factor_offsets": factor_offsets,
        "occurrence": occurrence,
    }


def reinforced_bp_transport(
    nvars: int,
    incidence: dict[str, np.ndarray],
    *,
    sweeps: int,
    damping: float,
    rho_low: float,
    rho_high: float,
    rho_hold: int,
    rho_power: float,
    log_clip: float,
    epsilon: float,
) -> tuple[np.ndarray, dict]:
    edge_var = incidence["edge_var"]
    edge_sign = incidence["edge_sign"]
    edge_factor = incidence["edge_factor"]
    factor_offsets = incidence["factor_offsets"]
    E = int(edge_var.size)

    variable_to_factor = np.zeros(E, dtype=np.float64)
    factor_to_variable = np.zeros(E, dtype=np.float64)

    alpha = float(damping)
    rho0 = float(rho_low)
    rho1 = float(rho_high)
    hold = int(rho_hold)
    rpow = float(rho_power)
    clip = max(float(log_clip), 1.0)
    eps = max(float(epsilon), np.finfo(np.float64).tiny)

    started = time.perf_counter()
    final_update = math.inf
    final_rho = rho0

    for sweep in range(int(sweeps)):
        if sweep < hold:
            rho = rho0
        else:
            frac = float(sweep - hold + 1) / float(max(1, int(sweeps) - hold))
            rho = rho0 + (rho1 - rho0) * (frac ** rpow)
        final_rho = rho

        cavity = np.clip(variable_to_factor, -clip, clip)
        p_true = 1.0 / (1.0 + np.exp(-cavity))
        p_violate = np.where(edge_sign > 0.0, 1.0 - p_true, p_true)

        clause_product = np.multiply.reduceat(p_violate, factor_offsets[:-1])
        product_other = clause_product[edge_factor] / np.maximum(p_violate, eps)
        new_factor = edge_sign * (-np.log(np.maximum(eps, 1.0 - product_other)))

        total_field = np.bincount(
            edge_var, weights=new_factor, minlength=nvars
        ).astype(np.float64, copy=False)

        new_variable = total_field[edge_var] - new_factor + rho * total_field[edge_var]

        final_update = max(
            float(np.max(np.abs(new_factor - factor_to_variable))),
            float(np.max(np.abs(new_variable - variable_to_factor))),
        )

        factor_to_variable = (1.0 - alpha) * factor_to_variable + alpha * new_factor
        variable_to_factor = (1.0 - alpha) * variable_to_factor + alpha * new_variable

    belief = np.bincount(
        edge_var, weights=factor_to_variable, minlength=nvars
    ).astype(np.float64, copy=False)

    return belief, {
        "kind": "reinforced_OR_cavity",
        "sweeps": int(sweeps),
        "damping": alpha,
        "rho_low": rho0,
        "rho_high": rho1,
        "rho_hold": hold,
        "rho_power": rpow,
        "rho_final": final_rho,
        "final_update_norm": float(final_update),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def normalized_bp_spin(field: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.abs(np.asarray(field, dtype=np.float64))
    positive = a[a > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, 1.0e-12)
    return np.tanh(np.asarray(field, dtype=np.float64) / scale), scale


def field_tail_ratio(field: np.ndarray, q: float) -> tuple[float, float, float]:
    a = np.abs(np.asarray(field, dtype=np.float64))
    med = max(float(np.median(a)), 1.0e-30)
    low = float(np.quantile(a, float(q)))
    return low / med, low, med


def soft_clause_energy(clauses: np.ndarray, spin: np.ndarray) -> tuple[float, float]:
    idx = np.abs(clauses).astype(np.int64) - 1
    signs = np.where(clauses > 0, 1.0, -1.0)
    values = np.asarray(spin, dtype=np.float64)[idx]
    k = np.prod(np.maximum((1.0 - signs * values) * 0.5, 0.0), axis=1)
    return float(np.mean(k * k)), float(np.max(k))


def head_tail_survey_transport(
    nvars: int,
    clauses: np.ndarray,
    incidence: dict[str, np.ndarray],
    *,
    sweeps: int,
    eta0: float,
    bulk_damping: float,
    head_damping: float,
    head_theta: float,
    head_gamma: float,
    epsilon: float,
) -> tuple[np.ndarray, dict]:
    """Survey propagation with a smooth high-influence/diffuse damping split.

    For each directed clause->variable survey edge, the ordinary SP update is
    computed first.  Its continuous influence is

        q_e = |eta_new - eta_old| (eta_new + eps).

    Normalized influence z_e = q_e / RMS(q) feeds a smooth head weight

        w_e = sigmoid(gamma (z_e - theta)).

    The actual damping is one blended law

        alpha_e = (1-w_e) alpha_bulk + w_e alpha_head.

    No edge is selected from a Boolean residual set.
    """
    edge_var = incidence["edge_var"]
    edge_sign = incidence["edge_sign"]
    m = int(clauses.shape[0])
    E = int(edge_var.size)
    eps = max(float(epsilon), 1.0e-15)

    eta = np.full(E, float(eta0), dtype=np.float64)
    last_head_mass = 0.0
    last_update = math.inf
    started = time.perf_counter()

    pos_mask = edge_sign > 0.0
    neg_mask = ~pos_mask

    for _ in range(int(sweeps)):
        log_not = np.log(np.maximum(1.0 - eta, eps))
        sum_pos = np.bincount(
            edge_var[pos_mask], weights=log_not[pos_mask], minlength=nvars
        )
        sum_neg = np.bincount(
            edge_var[neg_mask], weights=log_not[neg_mask], minlength=nvars
        )

        same_log = np.where(pos_mask, sum_pos[edge_var], sum_neg[edge_var]) - log_not
        opp_log = np.where(pos_mask, sum_neg[edge_var], sum_pos[edge_var])

        p_same = np.exp(np.clip(same_log, -50.0, 0.0))
        p_opp = np.exp(np.clip(opp_log, -50.0, 0.0))

        pi0 = p_same * p_opp
        pi_u = (1.0 - p_opp) * p_same
        pi_s = (1.0 - p_same) * p_opp
        p_u = pi_u / np.maximum(pi0 + pi_u + pi_s, eps)

        pu3 = p_u.reshape(m, 3)
        new3 = np.empty_like(pu3)
        new3[:, 0] = pu3[:, 1] * pu3[:, 2]
        new3[:, 1] = pu3[:, 0] * pu3[:, 2]
        new3[:, 2] = pu3[:, 0] * pu3[:, 1]
        new_eta = new3.reshape(-1)

        q = np.abs(new_eta - eta) * (new_eta + eps)
        rms = math.sqrt(float(np.mean(q * q)) + eps * eps)
        z = q / rms
        x = np.clip(float(head_gamma) * (z - float(head_theta)), -40.0, 40.0)
        head_weight = 1.0 / (1.0 + np.exp(-x))
        alpha = float(bulk_damping) + head_weight * (
            float(head_damping) - float(bulk_damping)
        )

        last_update = float(np.max(np.abs(new_eta - eta)))
        last_head_mass = float(np.mean(head_weight))
        eta = (1.0 - alpha) * eta + alpha * new_eta

    log_not = np.log(np.maximum(1.0 - eta, eps))
    sum_pos = np.bincount(
        edge_var[pos_mask], weights=log_not[pos_mask], minlength=nvars
    )
    sum_neg = np.bincount(
        edge_var[neg_mask], weights=log_not[neg_mask], minlength=nvars
    )

    p_pos = np.exp(np.clip(sum_pos, -50.0, 0.0))
    p_neg = np.exp(np.clip(sum_neg, -50.0, 0.0))
    pi0 = p_pos * p_neg
    pi_plus = (1.0 - p_pos) * p_neg
    pi_minus = (1.0 - p_neg) * p_pos
    denom = np.maximum(pi0 + pi_plus + pi_minus, eps)
    bias = (pi_plus - pi_minus) / denom

    return bias.astype(np.float64, copy=False), {
        "kind": "smooth_head_tail_survey",
        "sweeps": int(sweeps),
        "eta0": float(eta0),
        "bulk_damping": float(bulk_damping),
        "head_damping": float(head_damping),
        "head_theta": float(head_theta),
        "head_gamma": float(head_gamma),
        "final_update_norm": float(last_update),
        "final_mean_head_weight": float(last_head_mass),
        "mean_eta": float(np.mean(eta)),
        "max_eta": float(np.max(eta)),
        "runtime_seconds": float(time.perf_counter() - started),
    }


if HAVE_NUMBA:
    @njit(cache=True)
    def _analog_derivative_numba(s, loga, vars3, signs3, ds, dl):
        n = s.size
        m = vars3.shape[0]
        for i in range(n):
            ds[i] = 0.0

        mx = -1.0e300
        for a in range(m):
            if loga[a] > mx:
                mx = loga[a]

        for a in range(m):
            i0 = vars3[a, 0]
            i1 = vars3[a, 1]
            i2 = vars3[a, 2]
            f0 = max((1.0 - signs3[a, 0] * s[i0]) * 0.5, 1.0e-12)
            f1 = max((1.0 - signs3[a, 1] * s[i1]) * 0.5, 1.0e-12)
            f2 = max((1.0 - signs3[a, 2] * s[i2]) * 0.5, 1.0e-12)
            k = f0 * f1 * f2
            dl[a] = k
            aa = math.exp(max(loga[a] - mx, -50.0))
            kk = aa * k * k
            ds[i0] += kk * signs3[a, 0] / f0
            ds[i1] += kk * signs3[a, 1] / f1
            ds[i2] += kk * signs3[a, 2] / f2

    @njit(cache=True)
    def _analog_heun_numba(s0, vars3, signs3, steps, max_delta):
        n = s0.size
        m = vars3.shape[0]
        s = s0.copy()
        loga = np.zeros(m, dtype=np.float64)
        ds1 = np.empty(n, dtype=np.float64)
        dl1 = np.empty(m, dtype=np.float64)
        ds2 = np.empty(n, dtype=np.float64)
        dl2 = np.empty(m, dtype=np.float64)
        sp = np.empty(n, dtype=np.float64)
        lp = np.empty(m, dtype=np.float64)

        for t in range(steps):
            _analog_derivative_numba(s, loga, vars3, signs3, ds1, dl1)
            md = 0.0
            for i in range(n):
                z = abs(ds1[i])
                if z > md:
                    md = z
            dt = 1.0
            if md > max_delta:
                dt = max_delta / md

            for i in range(n):
                x = s[i] + dt * ds1[i]
                sp[i] = min(0.999999, max(-0.999999, x))
            for a in range(m):
                lp[a] = loga[a] + dt * dl1[a]

            _analog_derivative_numba(sp, lp, vars3, signs3, ds2, dl2)

            for i in range(n):
                x = s[i] + 0.5 * dt * (ds1[i] + ds2[i])
                s[i] = min(0.999999, max(-0.999999, x))
            for a in range(m):
                loga[a] += 0.5 * dt * (dl1[a] + dl2[a])

            if (t + 1) % 256 == 0:
                mx = -1.0e300
                for a in range(m):
                    if loga[a] > mx:
                        mx = loga[a]
                for a in range(m):
                    loga[a] -= mx

        return s


def _analog_derivative_numpy(
    s: np.ndarray,
    loga: np.ndarray,
    vars3: np.ndarray,
    signs3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = s[vars3]
    f = np.maximum((1.0 - signs3 * values) * 0.5, 1.0e-12)
    k = np.prod(f, axis=1)
    a = np.exp(np.clip(loga - np.max(loga), -50.0, 0.0))
    edge = (a[:, None] * (k[:, None] ** 2) * signs3 / f).reshape(-1)
    ds = np.bincount(vars3.reshape(-1), weights=edge, minlength=s.size).astype(np.float64)
    return ds, k


def _analog_heun_numpy(
    s0: np.ndarray,
    vars3: np.ndarray,
    signs3: np.ndarray,
    steps: int,
    max_delta: float,
) -> np.ndarray:
    s = np.clip(np.asarray(s0, dtype=np.float64).copy(), -0.999999, 0.999999)
    loga = np.zeros(vars3.shape[0], dtype=np.float64)
    for t in range(int(steps)):
        ds1, dl1 = _analog_derivative_numpy(s, loga, vars3, signs3)
        md = float(np.max(np.abs(ds1)))
        dt = 1.0 if md <= max_delta else float(max_delta) / md
        sp = np.clip(s + dt * ds1, -0.999999, 0.999999)
        lp = loga + dt * dl1
        ds2, dl2 = _analog_derivative_numpy(sp, lp, vars3, signs3)
        s = np.clip(s + 0.5 * dt * (ds1 + ds2), -0.999999, 0.999999)
        loga = loga + 0.5 * dt * (dl1 + dl2)
        if (t + 1) % 256 == 0:
            loga -= np.max(loga)
    return s


def analog_escape(
    clauses: np.ndarray,
    spin0: np.ndarray,
    *,
    steps: int,
    max_delta: float,
) -> tuple[np.ndarray, dict]:
    vars3 = np.abs(clauses).astype(np.int64) - 1
    signs3 = np.where(clauses > 0, 1.0, -1.0).astype(np.float64)
    s0 = np.clip(np.asarray(spin0, dtype=np.float64), -0.999999, 0.999999)
    started = time.perf_counter()

    if HAVE_NUMBA:
        s = _analog_heun_numba(s0, vars3, signs3, int(steps), float(max_delta))
        engine = "numba_heun_rk2"
    else:
        s = _analog_heun_numpy(s0, vars3, signs3, int(steps), float(max_delta))
        engine = "numpy_heun_rk2"

    mean_e, max_k = soft_clause_energy(clauses, s)
    return s, {
        "kind": "adaptive_analog_SAT_escape",
        "integrator": engine,
        "steps": int(steps),
        "max_delta_s": float(max_delta),
        "final_soft_energy": float(mean_e),
        "final_max_clause_K": float(max_k),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def one_binary32_readout(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    field32 = np.asarray(field, dtype=np.float32)
    assignment = field32 >= np.float32(0.0)
    return assignment, field32


def verify(clauses: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    values = assignment[np.abs(clauses).astype(np.int64) - 1]
    literal_true = np.where(clauses > 0, values, ~values)
    satisfied = np.any(literal_true, axis=1)
    return np.flatnonzero(~satisfied).astype(np.int64, copy=False)


def write_model(path: str | Path, assignment: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    literals = [str(i + 1 if bool(v) else -(i + 1)) for i, v in enumerate(assignment)]
    with p.open("w", encoding="utf-8") as fh:
        fh.write(f"c model generated by {VERSION}\n")
        fh.write("s SATISFIABLE\n")
        for start in range(0, len(literals), 20):
            chunk = literals[start:start + 20]
            suffix = " 0" if start + 20 >= len(literals) else ""
            fh.write("v " + " ".join(chunk) + suffix + "\n")


def write_unsat_diagnostic(path: str | Path, clauses: np.ndarray, ids: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(f"# post-readout diagnostic only; never fed back\n")
        fh.write(f"# unsatisfied clause count: {len(ids)} / {len(clauses)}\n")
        for cid in ids:
            clause = clauses[int(cid)]
            fh.write(f"{int(cid)+1}: {' '.join(map(str, clause.tolist()))} 0\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DREAM6 RANDOM3 HEADTAIL v3 -- deterministic continuous random 3-SAT prototype"
    )
    ap.add_argument("--cnf-path", required=True)
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--unsat-out", default=None)
    ap.add_argument("--json-out", default=None)

    ap.add_argument("--bp-sweeps", type=int, default=DEFAULT_BP_SWEEPS)
    ap.add_argument("--bp-damping", type=float, default=DEFAULT_BP_DAMPING)
    ap.add_argument("--rho-low", type=float, default=DEFAULT_RHO_LOW)
    ap.add_argument("--rho-high", type=float, default=DEFAULT_RHO_HIGH)
    ap.add_argument("--rho-hold", type=int, default=DEFAULT_RHO_HOLD)
    ap.add_argument("--rho-power", type=float, default=DEFAULT_RHO_POWER)

    ap.add_argument("--weak-q", type=float, default=DEFAULT_WEAK_Q)
    ap.add_argument("--structured-trigger", type=float, default=DEFAULT_STRUCTURED_TRIGGER)

    ap.add_argument("--sp-sweeps", type=int, default=DEFAULT_SP_SWEEPS)
    ap.add_argument("--sp-eta0", type=float, default=DEFAULT_SP_ETA0)
    ap.add_argument("--sp-bulk-damping", type=float, default=DEFAULT_SP_BULK_DAMPING)
    ap.add_argument("--sp-head-damping", type=float, default=DEFAULT_SP_HEAD_DAMPING)
    ap.add_argument("--sp-head-theta", type=float, default=DEFAULT_SP_HEAD_THETA)
    ap.add_argument("--sp-head-gamma", type=float, default=DEFAULT_SP_HEAD_GAMMA)
    ap.add_argument("--sp-energy-ratio", type=float, default=DEFAULT_SP_ENERGY_RATIO)
    ap.add_argument("--sp-init-scale", type=float, default=DEFAULT_SP_INIT_SCALE)

    ap.add_argument("--analog-steps", type=int, default=DEFAULT_ANALOG_STEPS)
    ap.add_argument("--analog-max-delta", type=float, default=DEFAULT_ANALOG_MAX_DELTA)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    total_started = time.perf_counter()
    input_hash = sha256_file(args.cnf_path)
    nvars, clauses = read_dimacs_3sat(args.cnf_path)

    compile_started = time.perf_counter()
    incidence = compile_incidence(nvars, clauses)
    compile_seconds = time.perf_counter() - compile_started

    print(f"=== {VERSION} ===")
    print(f"CNF                 : {args.cnf_path}")
    print(f"variables/clauses   : {nvars}/{len(clauses)}")
    print(f"literal incidences  : {incidence['edge_var'].size}")
    print("contract            : no RNG / restart / flips / branching / residual feedback")
    print("Boolean states      : exactly one, after all continuous dynamics")
    print("UNSAT claim         : NEVER; failure => UNCLASSIFIED")
    print("method reference    : Misiakiewicz--Wen 2026, head/tail principle only")
    print("=" * 92)

    bp_field, bp_meta = reinforced_bp_transport(
        nvars,
        incidence,
        sweeps=int(args.bp_sweeps),
        damping=float(args.bp_damping),
        rho_low=float(args.rho_low),
        rho_high=float(args.rho_high),
        rho_hold=int(args.rho_hold),
        rho_power=float(args.rho_power),
        log_clip=DEFAULT_LOG_CLIP,
        epsilon=DEFAULT_EPSILON,
    )

    tail_ratio, tail_q, tail_median = field_tail_ratio(bp_field, float(args.weak_q))
    structured = bool(tail_ratio > float(args.structured_trigger))

    print(
        "[bp]"
        f" T={bp_meta['sweeps']}"
        f" update={bp_meta['final_update_norm']:.6g}"
        f" time={bp_meta['runtime_seconds']:.6f}s"
    )
    print(
        "[continuous geometry]"
        f" Q{args.weak_q:.3g}(|H|)/median={tail_ratio:.6g}"
        f" trigger={args.structured_trigger:.6g}"
        f" basin={'structured' if structured else 'diffuse'}"
    )

    sp_meta = None
    analog_meta = None
    channel = "bp_diffuse"
    bp_energy = None
    sp_energy = None

    if not structured:
        final_continuous_field = bp_field
    else:
        bp_spin, bp_scale = normalized_bp_spin(bp_field)
        bp_energy, _ = soft_clause_energy(clauses, bp_spin)

        sp_bias, sp_meta = head_tail_survey_transport(
            nvars,
            clauses,
            incidence,
            sweeps=int(args.sp_sweeps),
            eta0=float(args.sp_eta0),
            bulk_damping=float(args.sp_bulk_damping),
            head_damping=float(args.sp_head_damping),
            head_theta=float(args.sp_head_theta),
            head_gamma=float(args.sp_head_gamma),
            epsilon=DEFAULT_EPSILON,
        )
        sp_energy, _ = soft_clause_energy(clauses, sp_bias)
        ratio = sp_energy / max(bp_energy, 1.0e-30)

        if ratio <= float(args.sp_energy_ratio):
            sp_field = np.arctanh(np.clip(sp_bias, -0.999, 0.999))
            spin0 = np.tanh(float(args.sp_init_scale) * sp_field)
            channel = "survey_head_tail"
        else:
            spin0 = bp_spin
            channel = "bp_structured"

        print(
            "[head-tail survey]"
            f" T={sp_meta['sweeps']}"
            f" head_mass={sp_meta['final_mean_head_weight']:.6g}"
            f" E_sp/E_bp={ratio:.6g}"
            f" source={channel}"
            f" time={sp_meta['runtime_seconds']:.6f}s"
        )

        final_continuous_field, analog_meta = analog_escape(
            clauses,
            spin0,
            steps=int(args.analog_steps),
            max_delta=float(args.analog_max_delta),
        )
        print(
            "[analog escape]"
            f" steps={analog_meta['steps']}"
            f" maxds={analog_meta['max_delta_s']:.6g}"
            f" E={analog_meta['final_soft_energy']:.6g}"
            f" time={analog_meta['runtime_seconds']:.6f}s"
        )

    # FIRST and ONLY Boolean state.
    assignment, field32 = one_binary32_readout(final_continuous_field)
    unsat_ids = verify(clauses, assignment)
    unsat = int(unsat_ids.size)
    sat = unsat == 0

    stem = Path(args.cnf_path).stem
    model_path = None
    if sat:
        model_path = Path(args.model_out or f"{stem}_headtail_v3.model")
        write_model(model_path, assignment)

    unsat_path = None
    if args.unsat_out or not sat:
        unsat_path = Path(args.unsat_out or f"{stem}_headtail_v3.unsat.txt")
        write_unsat_diagnostic(unsat_path, clauses, unsat_ids)

    runtime = time.perf_counter() - total_started
    abs_field = np.abs(field32.astype(np.float64))
    report = {
        "version": VERSION,
        "cnf_path": str(Path(args.cnf_path).resolve()),
        "cnf_sha256": input_hash,
        "nvars": int(nvars),
        "nclauses": int(len(clauses)),
        "literal_occurrences": int(incidence["edge_var"].size),
        "decision": "SAT" if sat else "UNCLASSIFIED",
        "verified_sat": bool(sat),
        "satisfied_clauses": int(len(clauses) - unsat),
        "unsatisfied_clauses": int(unsat),
        "compile_seconds": float(compile_seconds),
        "bp": bp_meta,
        "continuous_geometry": {
            "weak_quantile": float(args.weak_q),
            "weak_abs_field": float(tail_q),
            "median_abs_field": float(tail_median),
            "weak_to_median_ratio": float(tail_ratio),
            "structured_trigger": float(args.structured_trigger),
            "structured": bool(structured),
        },
        "head_tail_survey": sp_meta,
        "bp_soft_energy": None if bp_energy is None else float(bp_energy),
        "sp_soft_energy": None if sp_energy is None else float(sp_energy),
        "selected_continuous_channel": channel,
        "analog_escape": analog_meta,
        "readout": {
            "kind": "one_simultaneous_binary32_sign_projection",
            "field_min_abs": float(np.min(abs_field)),
            "field_q01_abs": float(np.quantile(abs_field, 0.01)),
            "field_median_abs": float(np.median(abs_field)),
            "field_max_abs": float(np.max(abs_field)),
            "intermediate_boolean_states": 0,
        },
        "contract": {
            "random_seed": False,
            "random_restart": False,
            "walksat_or_boolean_flips": False,
            "branching": False,
            "decimation": False,
            "residual_feedback": False,
            "verifier_feedback": False,
            "external_sat_solver": False,
            "one_boolean_readout": True,
            "unsat_certificate": False,
            "failure_semantics": "UNCLASSIFIED, never UNSAT",
            "fixed_sweep_arithmetic_work": "O(n+L) for pure width-3 CNF",
            "completeness_claim": "none; SATLIB/random completeness is an experimental target",
        },
        "reference": {
            "authors": "Theodor Misiakiewicz and Garrett G. Wen",
            "title": "The sharp SAT/UNSAT phase transition in random ellipsoid fitting",
            "arxiv": "2608.10184v1",
            "use": "methodological head-tail decomposition principle only; no theorem transfer",
        },
        "numba_available": bool(HAVE_NUMBA),
        "model_path": str(model_path) if model_path is not None else None,
        "unsat_path": str(unsat_path) if unsat_path is not None else None,
        "runtime_seconds": float(runtime),
    }

    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 92)
    print("FINAL RESULT")
    print(f"continuous channel  : {channel}")
    print(f"satisfied clauses   : {len(clauses)-unsat}/{len(clauses)}")
    print(f"unsatisfied clauses : {unsat}/{len(clauses)}")
    print("SAT soundness       : " + ("PASS" if sat else "PRESERVED — no SAT verdict"))
    print("decision            : " + ("SAT" if sat else "UNCLASSIFIED"))
    print(f"compile time        : {compile_seconds:.6f} s")
    print(f"runtime total       : {runtime:.6f} s")
    if model_path is not None:
        print(f"valid model         : {model_path}")
    if unsat_path is not None:
        print(f"diagnostic only     : {unsat_path}  [never fed back]")

    return 0 if sat else 2


if __name__ == "__main__":
    raise SystemExit(main())
