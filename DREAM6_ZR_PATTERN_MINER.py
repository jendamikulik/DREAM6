#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6_ZR_PATTERN_MINER.py

Pattern miner for the exact quartic zero-repair LP.
Requires WPB_COMMON_ANCHOR_SOLVER.py in the same directory.

It does NOT use the common-anchor constraints.  It solves the unconstrained
full-history LP for n=1..N and mines the optimal trees for:
  * root support activation and reduced-cost wavefronts,
  * exact numerical copies of older optimal roots B_k,
  * continuation signatures B_k -> B_{k-1},
  * hidden continuation phase: same B_k law, different signatures.

Example:
    python DREAM6_ZR_PATTERN_MINER.py --n-max 5 --M 6
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from collections import Counter
from pathlib import Path
from fractions import Fraction
import numpy as np


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("wpb_ca", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wpb_ca"] = mod
    spec.loader.exec_module(mod)
    return mod


def law(sol, h):
    v=sol.vidx
    return np.array([sol.res.x[v("B",h,j)] for j in range(sol.M+1)], dtype=float)


def support(a,tol=1e-9):
    return tuple(np.where(a>tol)[0].tolist())


def reduced_costs(sol):
    d=np.asarray(sol.res.lower.marginals,dtype=float)
    v=sol.vidx
    return np.array([d[v("B",(),j)] for j in range(sol.M+1)])


def matches(a,b,tol):
    return float(np.max(np.abs(a-b))) <= tol


def continuation_signature(sol,h,target,tol):
    if len(h)>=sol.n: return ()
    return tuple(x for x in range(5) if matches(law(sol,h+(x,)),target,tol))


def rational(x,den=1000000):
    return str(Fraction(float(x)).limit_denominator(den))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',type=Path,default=Path('WPB_COMMON_ANCHOR_SOLVER.py'))
    ap.add_argument('--n-max',type=int,default=5)
    ap.add_argument('--M',type=int,default=6)
    ap.add_argument('--time-limit',type=float,default=120.0)
    ap.add_argument('--tol',type=float,default=2e-8)
    ap.add_argument('--json-out',type=Path,default=Path('dream6_zr_pattern.json'))
    args=ap.parse_args()
    base=load_base(args.base)

    sols={}; roots={}; payload={'roots':{}}
    print('='*100)
    print('DREAM6-ZR GLOBAL PATTERN MINER')
    print('full exact 5-ary history LP; NO common anchor; NO Phi/carrier ansatz')
    print('='*100)

    for n in range(1,args.n_max+1):
        s=base.solve_lp(n=n,M=args.M,anchor=False,anchor_mode='insured',time_limit=args.time_limit)
        if not s.success:
            print(f'n={n}: FAILED: {s.message}')
            break
        sols[n]=s; roots[n]=s.root.copy()
        rc=reduced_costs(s)
        print(f'n={n}: B_n={s.objective:.12f}  support={support(s.root)}  runtime={s.runtime:.2f}s')
        print('  root =',np.array2string(s.root,precision=9,suppress_small=True))
        print('  rc   =',np.array2string(rc,precision=7,suppress_small=True))
        payload['roots'][str(n)]={
            'mean':s.objective,'mean_rational_guess':rational(s.objective),
            'law':s.root.tolist(),'support':list(support(s.root)),
            'reduced_costs':rc.tolist(),'eq_residual':s.eq_resid_max,
        }

    if not sols: return 2
    N=max(sols); deep=sols[N]
    print('\n'+'='*100)
    print(f'SELF-SIMILARITY INSIDE OPTIMAL DEPTH-{N} TREE')
    print('='*100)
    recurrence={}; hidden=[]

    for k in range(1,N):
        hits=[]
        for d,hs in enumerate(deep.vidx.h_by_d):
            for h in hs:
                if matches(law(deep,h),roots[k],args.tol): hits.append(h)
        bydepth=Counter(map(len,hits))
        sig=Counter()
        examples={}
        if k>=2:
            for h in hits:
                if len(h)<N:
                    z=continuation_signature(deep,h,roots[k-1],args.tol)
                    sig[z]+=1; examples.setdefault(z,h)
        print(f'B_{k}: occurrences={len(hits)} by_depth={dict(bydepth)}')
        if k>=2:
            print(f'     continuation signatures to B_{k-1}: {dict(sig)}')
            if len(sig)>1: hidden.append(k)
        recurrence[str(k)]={
            'occurrences':len(hits),'by_depth':{str(a):b for a,b in bydepth.items()},
            'continuation_signatures':{str(a):b for a,b in sig.items()},
            'examples':{str(a):list(h) for a,h in examples.items()},
        }

    print('\nStandalone root continuation signatures:')
    standalone={}
    for k in range(2,N+1):
        z=continuation_signature(sols[k],(),roots[k-1],args.tol)
        standalone[str(k)]=list(z)
        print(f'  B_{k} -> B_{k-1}: {z}')

    # Follow exact B_N -> B_{N-1} -> ... spines.
    spines=[]
    def rec(h,k,path):
        if k<=1:
            spines.append(path.copy()); return
        found=False
        for x in range(5):
            ch=h+(x,)
            if len(ch)<=N and matches(law(deep,ch),roots[k-1],args.tol):
                found=True; path.append(x); rec(ch,k-1,path); path.pop()
        if not found: spines.append(path.copy())
    rec((),N,[])
    print('regenerative spines:',spines[:30])

    print('\n'+'='*100)
    if hidden:
        print('HIDDEN CONTINUATION PHASE DETECTED AT:', ', '.join('B_'+str(k) for k in hidden))
        print('Same buffer law occurs with different continuation signatures.')
        print('=> dynamically relevant state is strictly richer than B_h alone.')
    else:
        print('No multi-signature repeated law at tested depth.')
    print('='*100)

    payload.update({
        'deepest_success':N,'M':args.M,'recurrence':recurrence,
        'standalone_signatures':standalone,'spines':spines,
        'hidden_phase_detected_at':hidden,
        'warning':'finite-support binary64 LP discovery; exact certification still required',
    })
    args.json_out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('saved:',args.json_out)
    return 0

if __name__=='__main__': raise SystemExit(main())
