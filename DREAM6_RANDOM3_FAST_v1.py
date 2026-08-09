#!/usr/bin/env python3
"""
DREAM6 RANDOM3 FAST v1
=====================

Purpose
-------
A deliberately specialized speed prototype for satisfiable sparse random 3-SAT.
It is NOT the universal DREAM6 compiler and it does not attempt to recognize
semantic benchmark families.

Active path:

    3-CNF incidence
        -> one fixed global continuous OR-cavity transport
        -> one simultaneous binary32 sign readout
        -> independent exact verification

There are no random seeds, restarts, Boolean flips, WalkSAT/GSAT moves,
branching, decimation, residual feedback, candidate ranking, or external SAT
solver.  The verifier is called only after the single Boolean readout.

The default hot-loop constants are fixed globally:

    sweeps        T     = 200
    damping       alpha = 0.78
    reinforcement rho   = 0.065
    log clip            = 50
    epsilon             = 1e-12

For fixed clause width 3 and fixed T, the implemented transport is O(n + L)
in time and memory, where L is the number of literal occurrences.  This is an
implementation-complexity statement, not a SAT-completeness theorem.

Numerical note
--------------
The hot OR-cavity map is the frozen float64 v158/v159-style response used by
recent DREAM6 versions.  The one final readout is explicitly cast to IEEE-754
binary32 before the sign decision.  This speed prototype therefore does NOT
claim the full cross-platform bit contract of the larger DREAM6 kernel; that
contract can be restored later if this reduced random-3SAT architecture proves
worth keeping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

VERSION = "DREAM6_RANDOM3_FAST_v1"

DEFAULT_SWEEPS = 200
DEFAULT_DAMPING = 0.78
DEFAULT_REINFORCEMENT = 0.065
DEFAULT_LOG_CLIP = 50.0
DEFAULT_EPSILON = 1.0e-12


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_dimacs_3sat(path: str | Path) -> tuple[int, np.ndarray]:
    """Read DIMACS and require a pure width-3 CNF."""
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
                    # Several SATLIB files end with a standalone 0 after a '%' line.
                    # It is a stream terminator, not an empty clause.
                    if not current:
                        continue
                    if len(current) != 3:
                        raise ValueError(
                            f"RANDOM3 FAST accepts pure 3-CNF only; got clause width {len(current)}"
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
    if int(np.max(np.abs(arr))) > int(nvars) or int(np.min(np.abs(arr))) < 1:
        raise ValueError("literal variable index outside declared DIMACS range")
    return int(nvars), arr


def compile_incidence(nvars: int, clauses: np.ndarray) -> dict[str, np.ndarray]:
    """Compile the raw 3-CNF incidence once; no semantic motif recognition."""
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


def random3_cavity_transport(
    nvars: int,
    clauses: np.ndarray,
    incidence: dict[str, np.ndarray],
    *,
    sweeps: int,
    damping: float,
    reinforcement: float,
    log_clip: float,
    epsilon: float,
) -> tuple[np.ndarray, dict]:
    """One fixed synchronous OR-cavity map over the complete 3-CNF incidence.

    For edge e=(a,i), q_e is the cavity probability that literal e violates
    clause a.  The factor response is

        u_e = s_e * [-log(1 - prod_{f in a\\{e}} q_f)].

    Variable concurrence is the global occurrence sum H_i = sum_{e->i} u_e,
    and the reverse message is

        v_e = H_i - u_e + rho H_i.

    Both directed message families are updated synchronously with damping
    alpha.  No Boolean assignment exists inside the loop.
    """
    if sweeps <= 0:
        raise ValueError("sweeps must be positive")
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must lie in (0,1]")
    if reinforcement < 0.0:
        raise ValueError("reinforcement must be nonnegative")

    edge_var = incidence["edge_var"]
    edge_sign = incidence["edge_sign"]
    edge_factor = incidence["edge_factor"]
    factor_offsets = incidence["factor_offsets"]
    E = int(edge_var.size)

    variable_to_factor = np.zeros(E, dtype=np.float64)
    factor_to_variable = np.zeros(E, dtype=np.float64)

    alpha = float(damping)
    rho = float(reinforcement)
    clip = max(float(log_clip), 1.0)
    eps = max(float(epsilon), np.finfo(np.float64).tiny)

    started = time.perf_counter()
    final_update = math.inf

    for _ in range(int(sweeps)):
        cavity = np.clip(variable_to_factor, -clip, clip)

        # Exact OR factor response in the frozen arithmetic order.
        p_true = 1.0 / (1.0 + np.exp(-cavity))
        p_violate = np.where(edge_sign > 0.0, 1.0 - p_true, p_true)
        clause_product = np.multiply.reduceat(p_violate, factor_offsets[:-1])
        product_other = clause_product[edge_factor] / np.maximum(p_violate, eps)
        new_factor = edge_sign * (
            -np.log(np.maximum(eps, 1.0 - product_other))
        )

        # Stable incidence-order accumulation used by the frozen random-SAT map.
        total_field = np.bincount(
            edge_var,
            weights=new_factor,
            minlength=nvars,
        ).astype(np.float64, copy=False)

        new_variable = (
            total_field[edge_var]
            - new_factor
            + rho * total_field[edge_var]
        )

        final_update = max(
            float(np.max(np.abs(new_factor - factor_to_variable))),
            float(np.max(np.abs(new_variable - variable_to_factor))),
        )

        factor_to_variable = (
            (1.0 - alpha) * factor_to_variable + alpha * new_factor
        )
        variable_to_factor = (
            (1.0 - alpha) * variable_to_factor + alpha * new_variable
        )

    belief64 = np.bincount(
        edge_var,
        weights=factor_to_variable,
        minlength=nvars,
    ).astype(np.float64, copy=False)

    meta = {
        "kind": "fixed_random3_global_OR_cavity_transport",
        "sweeps": int(sweeps),
        "damping": float(alpha),
        "reinforcement": float(rho),
        "log_clip": float(clip),
        "epsilon": float(eps),
        "edges": int(E),
        "final_update_norm": float(final_update),
        "runtime_seconds": float(time.perf_counter() - started),
        "state_precision": "float64 inherited OR-cavity hot map",
        "boolean_state_inside_loop": False,
        "reads_residuals": False,
        "restarts": False,
        "boolean_flips": False,
    }
    return belief64, meta


def one_binary32_readout(belief64: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The first and only Boolean state constructed by the algorithm."""
    field32 = np.asarray(belief64, dtype=np.float32)
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
        fh.write("c model generated by DREAM6 RANDOM3 FAST v1\n")
        fh.write("s SATISFIABLE\n")
        for start in range(0, len(literals), 20):
            chunk = literals[start:start + 20]
            suffix = " 0" if start + 20 >= len(literals) else ""
            fh.write("v " + " ".join(chunk) + suffix + "\n")


def write_unsat(path: str | Path, clauses: np.ndarray, ids: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(f"# UNSAT clause count: {len(ids)} / {len(clauses)}\n")
        for cid in ids:
            clause = clauses[int(cid)]
            fh.write(f"{int(cid)+1}: {' '.join(map(str, clause.tolist()))} 0\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DREAM6 RANDOM3 FAST v1 -- fixed one-path random 3-SAT speed prototype"
    )
    ap.add_argument("--cnf-path", required=True)
    ap.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS)
    ap.add_argument("--damping", type=float, default=DEFAULT_DAMPING)
    ap.add_argument("--reinforcement", type=float, default=DEFAULT_REINFORCEMENT)
    ap.add_argument("--log-clip", type=float, default=DEFAULT_LOG_CLIP)
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--unsat-out", default=None)
    ap.add_argument("--json-out", default=None)
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
    print("INPUT")
    print(f"  CNF              : {args.cnf_path}")
    print(f"  variables/clauses: {nvars}/{len(clauses)}")
    print(f"  literal edges    : {incidence['edge_var'].size}")
    print("SPECIALIZATION")
    print("  accepted geometry: pure width-3 CNF only")
    print("  semantic atlas   : NONE")
    print("  compression      : NONE; raw incidence")
    print("ONE GLOBAL RANDOM3 TRANSPORT")
    print(f"  sweeps           : {args.sweeps}")
    print(f"  damping alpha    : {args.damping:.9g}")
    print(f"  reinforcement rho: {args.reinforcement:.9g}")
    print("  seeds/restarts   : NONE")
    print("  Boolean flips    : NONE")
    print("  residual feedback: NONE")
    print("ONE READOUT")
    print("  operation        : one simultaneous binary32 sign projection")
    print("  verifier         : original CNF; post-readout only")
    print("=" * 88)

    belief64, op_meta = random3_cavity_transport(
        nvars,
        clauses,
        incidence,
        sweeps=int(args.sweeps),
        damping=float(args.damping),
        reinforcement=float(args.reinforcement),
        log_clip=float(args.log_clip),
        epsilon=float(args.epsilon),
    )
    assignment, field32 = one_binary32_readout(belief64)
    unsat_ids = verify(clauses, assignment)
    unsat = int(unsat_ids.size)
    sat = unsat == 0

    stem = Path(args.cnf_path).stem
    if sat:
        model_path = Path(args.model_out or f"{stem}_random3_fast.model")
        write_model(model_path, assignment)
    else:
        model_path = None

    unsat_path = None
    if args.unsat_out or not sat:
        unsat_path = Path(args.unsat_out or f"{stem}_random3_fast.unsat.txt")
        write_unsat(unsat_path, clauses, unsat_ids)

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
        "operator": op_meta,
        "readout": {
            "kind": "single_simultaneous_binary32_sign_projection",
            "field_min_abs": float(np.min(abs_field)),
            "field_q01_abs": float(np.quantile(abs_field, 0.01)),
            "field_median_abs": float(np.median(abs_field)),
            "field_max_abs": float(np.max(abs_field)),
            "intermediate_boolean_states": 0,
        },
        "contract": {
            "random_seed": False,
            "restart_portfolio": False,
            "walksat_or_local_flips": False,
            "branching": False,
            "decimation": False,
            "residual_feedback": False,
            "verifier_feedback": False,
            "candidate_ranking": False,
            "external_sat_solver": False,
            "one_boolean_readout": True,
            "specialized_scope": "pure 3-CNF random-SAT speed laboratory",
            "fixed_sweep_complexity": "O(n+L) for fixed T=200 and width=3",
            "sat_completeness_claim": "none",
        },
        "model_path": str(model_path) if model_path is not None else None,
        "unsat_path": str(unsat_path) if unsat_path is not None else None,
        "runtime_seconds": float(runtime),
    }

    if args.json_out:
        jp = Path(args.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(
        "[random3 flow]"
        f" T={op_meta['sweeps']}"
        f" alpha={op_meta['damping']:.6g}"
        f" rho={op_meta['reinforcement']:.6g}"
        f" update={op_meta['final_update_norm']:.6g}"
        f" time={op_meta['runtime_seconds']:.6f}s"
    )
    print("=" * 88)
    print("FINAL RESULT")
    print(f"satisfied clauses   : {len(clauses)-unsat}/{len(clauses)}")
    print(f"unsatisfied clauses : {unsat}/{len(clauses)}")
    print("SAT soundness       : " + ("PASS" if sat else "PRESERVED — no SAT verdict"))
    print("decision            : " + ("SAT" if sat else "UNCLASSIFIED"))
    print(f"compile time        : {compile_seconds:.6f} s")
    print(f"runtime total       : {runtime:.6f} s")
    if model_path is not None:
        print(f"valid model         : {model_path}")
    if unsat_path is not None:
        print(f"residual diagnostic : {unsat_path}  [never fed back]")
    return 0 if sat else 2


if __name__ == "__main__":
    raise SystemExit(main())
