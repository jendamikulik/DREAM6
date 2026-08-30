#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6-ZR v0.25 LAST-LAYER NORMAL-FORM DECIDER

Purpose
-------
Numerically decide whether the positive block-optimal prefix filling problem
admits a last-layer normal form:

    r_h = 0 for 1 <= |h| <= n-2,

while keeping the same block-optimal root cap and minimizing only the
independent last-layer KR residual at |h| = n-1.

The script reuses the v0.24 backward-viability/Farkas oracle and checkpoint
format.  It is deliberately a numerical positivity test; the signed residual
pushing lemma is analytic and solver-free, but positivity is not assumed.
"""
import argparse, itertools, json, time, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

try:
    import DREAM6_ZR_v024_PREFIX_MASTER_VIABILITY_CUTTING_PLANE as v24
except ImportError:
    # Allows execution when both files live in /mnt/data during verification.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import DREAM6_ZR_v024_PREFIX_MASTER_VIABILITY_CUTTING_PLANE as v24

VERSION = "DREAM6-ZR v0.25 LAST-LAYER NORMAL-FORM DECIDER"


def build_last_layer_master(PM, target_cuts, base_value, root_slack=2e-9):
    """Build the positive last-layer normal-form LP.

    Core block identity and all law normalizations are inherited from v0.24.
    Independent ZR rows at depths 1,...,n-2 are imposed exactly.
    The objective is weighted KR only at depth n-1.
    """
    n, M, nA = PM.n, PM.M, PM.nA
    A0, b0 = PM.core_eq()

    # Exact old-prefix rows: r_h(u)=0 for 1 <= |h| <= n-2.
    er, ec, ev, eb = [], [], [], []
    rr = 0
    for d in range(1, max(1, n-1)):
        if d > n-2:
            break
        for h in itertools.product(range(5), repeat=d):
            for u in range(M + v24.D + 1):
                for col, z in PM.zr_coeff(h, u):
                    if z:
                        er.append(rr); ec.append(col); ev.append(z)
                eb.append(0.0); rr += 1

    if rr:
        E = coo_matrix((ev, (er, ec)), shape=(rr, nA)).tocsr()
        AeqA = vstack([A0, E], format='csr')
        beq = np.concatenate([b0, np.asarray(eb, float)])
    else:
        AeqA, beq = A0, b0

    # KR epigraph variables only for the last independent prefix layer.
    dlast = n - 1
    titems = [
        (h, k, v24.hprob(h))
        for h in itertools.product(range(5), repeat=dlast)
        for k in range(M + v24.D)
    ] if dlast >= 1 else []
    nt = len(titems)
    off = nA
    Aeq = hstack([AeqA, csr_matrix((AeqA.shape[0], nt))], format='csr')

    ur, uc, uv, ub = [], [], [], []
    r = 0

    # Boundary viability outer cuts.
    for a, b, label in target_cuts:
        for h in PM.leaves:
            for j, z in enumerate(a):
                if z:
                    ur.append(r); uc.append(PM.av(h, j)); uv.append(z)
            ub.append(float(b)); r += 1

    # Block-optimal root cap.
    for j in range(PM.N):
        if j:
            ur.append(r); uc.append(j); uv.append(float(j))
    ub.append(float(base_value) + float(root_slack)); r += 1

    # |CDF residual| epigraph at depth n-1.
    for ti, (h, k, w) in enumerate(titems):
        cf = PM.cum_coeff(h, k)
        for col, z in cf.items():
            ur.append(r); uc.append(col); uv.append(z)
        ur.append(r); uc.append(off + ti); uv.append(-1.0)
        ub.append(0.0); r += 1

        for col, z in cf.items():
            ur.append(r); uc.append(col); uv.append(-z)
        ur.append(r); uc.append(off + ti); uv.append(-1.0)
        ub.append(0.0); r += 1

    Aub = coo_matrix((uv, (ur, uc)), shape=(r, nA + nt)).tocsr()
    obj = np.zeros(nA + nt)
    for ti, (_, _, w) in enumerate(titems):
        obj[off + ti] = float(w)

    return Aeq, beq, Aub, np.asarray(ub, float), obj, nA, nt


def solve_master(PM, cuts, base_value, root_slack, time_limit, method):
    Aeq, beq, Aub, bub, obj, nA, nt = build_last_layer_master(
        PM, cuts, base_value, root_slack
    )
    opts = v24.highs_options(time_limit)
    t0 = time.time()
    res = linprog(
        obj,
        A_ub=Aub,
        b_ub=bub,
        A_eq=Aeq,
        b_eq=beq,
        bounds=[(0, None)] * len(obj),
        method=method,
        options=opts,
    )
    return res, time.time() - t0, nA


def layer_diagnostics(PM, x):
    out = {}
    for d in range(1, PM.n):
        weighted = 0.0
        mx = 0.0
        for h in itertools.product(range(5), repeat=d):
            rv = np.asarray([
                sum(z * x[col] for col, z in PM.zr_coeff(h, u))
                for u in range(PM.M + v24.D + 1)
            ])
            kr = float(np.abs(np.cumsum(rv)[:-1]).sum())
            weighted += v24.hprob(h) * kr
            mx = max(mx, kr)
        out[str(d)] = {
            'weighted_kr': float(weighted),
            'max_history_kr': float(mx),
        }
    return out


def solve_case(args):
    PM = v24.PrefixMaster(args.n, args.M)
    O = v24.ViabilityOracle(
        args.M,
        oracle_tol=args.oracle_tol,
        boundary_tol=args.boundary_tol,
        max_loops=args.max_local_loops,
    )
    v24.load_oracle_cache(O, args.checkpoint)

    ledger = []
    final = None
    for it in range(1, args.max_master + 1):
        res, sec, nA = solve_master(
            PM,
            O.cache[args.n],
            args.base_value,
            args.root_slack,
            args.time_limit,
            args.method,
        )
        if not res.success:
            return {
                'status': 'MASTER_FAILED',
                'message': res.message,
                'ledger': ledger,
            }

        ok, fail = v24.validate_boundary(
            PM, res.x, O, args.n, batch_cuts=args.batch_cuts
        )
        v24.save_oracle_cache(O, args.checkpoint)
        rec = {
            'it': it,
            'last_fill': float(res.fun),
            'seconds': sec,
            'target_cuts': len(O.cache[args.n]),
            'boundary_ok': ok,
            'new_failures': len(fail),
            'boundary_accepts': O.boundary_accepts,
            'max_boundary_phi': O.max_boundary_phi,
        }
        ledger.append(rec)
        print(
            f"[last {it:03d}] F_last={res.fun:.15g} ok={ok} "
            f"E{args.n}cuts={len(O.cache[args.n])} sec={sec:.3f} "
            f"BA={O.boundary_accepts}",
            flush=True,
        )
        if ok:
            final = res
            break

    if final is None:
        return {'status': 'MAX_ITER', 'ledger': ledger}

    x = np.asarray(final.x[:PM.nA], float)
    by_depth = layer_diagnostics(PM, x)
    gap = None
    if args.reference_fill is not None:
        gap = float(final.fun - args.reference_fill)

    verdict = 'POSITIVE_LAST_LAYER_NORMAL_FORM_NUMERICALLY_VERIFIED'
    if gap is not None and gap > args.equality_tol:
        verdict = 'POSITIVE_LAST_LAYER_NORMAL_FORM_HAS_POSITIVE_PREMIUM'

    return {
        'status': 'PASS',
        'verdict': verdict,
        'n': args.n,
        'M': args.M,
        'base_value': float(args.base_value),
        'reference_fill': None if args.reference_fill is None else float(args.reference_fill),
        'last_layer_fill': float(final.fun),
        'last_minus_reference': gap,
        'equality_tol': float(args.equality_tol),
        'by_depth': by_depth,
        'E_target_cuts': len(O.cache[args.n]),
        'cut_counts': {str(k): len(v) for k, v in O.cache.items()},
        'boundary_accepts': O.boundary_accepts,
        'max_boundary_phi': O.max_boundary_phi,
        'oracle_stats': dict(O.stats),
        'ledger': ledger,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--M', type=int, required=True)
    ap.add_argument('--base-value', type=float, required=True)
    ap.add_argument('--reference-fill', type=float, default=None)
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--max-master', type=int, default=100)
    ap.add_argument('--batch-cuts', type=int, default=4)
    ap.add_argument('--time-limit', type=float, default=300)
    ap.add_argument('--oracle-tol', type=float, default=2e-8)
    ap.add_argument('--boundary-tol', type=float, default=1e-8)
    ap.add_argument('--max-local-loops', type=int, default=1000)
    ap.add_argument('--root-slack', type=float, default=2e-9)
    ap.add_argument('--equality-tol', type=float, default=2e-7)
    ap.add_argument('--method', choices=['highs','highs-ds','highs-ipm'], default='highs-ipm')
    args = ap.parse_args()

    if args.n < 2:
        raise SystemExit('n must be >= 2')

    print(VERSION)
    print('=' * 100)
    print(
        f"n={args.n} M={args.M} base={args.base_value:.15g} "
        f"reference_fill={args.reference_fill}"
    )
    print(
        "contract: exact ZR on depths 1..n-2; KR objective only on depth n-1; "
        "positive laws; boundary E_n recursively audited"
    )

    out = solve_case(args)
    print(json.dumps({k:v for k,v in out.items() if k != 'ledger'}, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print('WROTE', Path(args.out).resolve())


if __name__ == '__main__':
    main()
