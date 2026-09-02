#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6_ZR_ENTRY_CERT_v01.py

Independent numerical crash-test for the exact graded Entry Lemma.

This script does NOT discover the theorem and is not part of its proof.
It verifies, by independent transportation LPs, the constrained couplings used
in the proof for:
  * the peeling step,
  * the landing step,
  * extreme one-atom tails,
  * random finite-support tails.

Exact target classes
--------------------
T(cbar) = { B : B_1 = 0, 0 <= B_0 <= cbar }.

Peeling map:
    tau(c) = max(0, (48 c - 9)/38).

Landing threshold:
    c <= 13/48.

CORE:
    B_0 = 0, B_1 <= 13/16.

Critical-branch contract checked by LP
--------------------------------------
Peeling:
  x=0 child CORE via pi(0,1)=0,
  x=2 child has B'_1=0 and B'_0 <= tau(B_0),
  x=4 child is CORE or terminal t.

Landing:
  x=0 child CORE,
  x=2 child CORE,
  x=4 child CORE or terminal t.

The x=4 disjunction is resolved deterministically:
  if q_5 >= P_4, set row 4 entirely at column 5 (terminal t);
  otherwise set row 4 entirely in columns >= 6 (CORE).

Usage
-----
python DREAM6_ZR_ENTRY_CERT_v01.py --random-tests 10000 --max-tail-index 32
"""

import argparse
import json
from pathlib import Path
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

P = np.array([1,12,38,12,1], dtype=float)/64.0
Q = np.array([0,16,32,16,0], dtype=float)/64.0

C0 = Fraction(1,2)
C1 = Fraction(15,38)
C2 = Fraction(189,722)
LAND = Fraction(13,48)

def tau_frac(c):
    z = (48*c - 9)/38
    return max(Fraction(0), z)

def delta_frac(c):
    return Fraction(23,128) - c/4

def normalize(B):
    B=np.asarray(B,float)
    if np.min(B)<-1e-13:
        raise ValueError("negative law")
    B=np.maximum(B,0)
    return B/B.sum()

def geom(support):
    maxu=support+4
    cells=[(x,u) for x in range(5) for u in range(maxu+1) if u>=x]
    idx={c:i for i,c in enumerate(cells)}
    return maxu,cells,idx

def solve_contract(B, mode):
    B=normalize(B)
    if abs(B[1] if len(B)>1 else 0.0)>1e-10:
        return False, "B1_nonzero", None

    c=float(B[0])
    q=np.convolve(Q,B)
    maxu,cells,idx=geom(len(B)-1)
    n=len(cells)

    Aeq=[];beq=[]
    Aub=[];bub=[]

    # Exact row marginals.
    for x in range(5):
        r=np.zeros(n)
        for u in range(maxu+1):
            k=idx.get((x,u))
            if k is not None:
                r[k]=1
        Aeq.append(r);beq.append(P[x])

    # Exact column marginals.
    for u in range(maxu+1):
        r=np.zeros(n)
        for x in range(5):
            k=idx.get((x,u))
            if k is not None:
                r[k]=1
        Aeq.append(r);beq.append(q[u] if u<len(q) else 0.0)

    # x=0 -> CORE: no residual 1.
    k=idx.get((0,1))
    if k is not None:
        r=np.zeros(n);r[k]=1
        Aeq.append(r);beq.append(0.0)

    if mode=="peel":
        # x=2: no residual 1, and residual-0 mass bounded by tau(c).
        k=idx.get((2,3))
        if k is not None:
            r=np.zeros(n);r[k]=1
            Aeq.append(r);beq.append(0.0)

        k=idx[(2,2)]
        r=np.zeros(n);r[k]=1
        Aub.append(r);bub.append(P[2]*max(0.0,(48*c-9)/38))

    elif mode=="land":
        # x=2 -> CORE: zero constant.
        k=idx[(2,2)]
        r=np.zeros(n);r[k]=1
        Aeq.append(r);beq.append(0.0)

        # child B1 <= 13/16.
        k=idx.get((2,3))
        if k is not None:
            r=np.zeros(n);r[k]=1
            Aub.append(r);bub.append(P[2]*13/16)
    else:
        raise ValueError(mode)

    # x=4 -> terminal or CORE.
    q5=q[5] if len(q)>5 else 0.0
    if q5 >= P[4]-1e-13:
        # terminal t: row 4 entirely at column 5.
        for u in range(maxu+1):
            k=idx.get((4,u))
            if k is not None and u!=5:
                r=np.zeros(n);r[k]=1
                Aeq.append(r);beq.append(0.0)
    else:
        # CORE: no residual 0 or 1, so no columns 4 or 5.
        for u in (4,5):
            k=idx.get((4,u))
            if k is not None:
                r=np.zeros(n);r[k]=1
                Aeq.append(r);beq.append(0.0)

    res=linprog(
        np.zeros(n),
        A_ub=np.asarray(Aub) if Aub else None,
        b_ub=np.asarray(bub) if bub else None,
        A_eq=np.asarray(Aeq),
        b_eq=np.asarray(beq),
        bounds=[(0,None)]*n,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9
        }
    )
    if not res.success:
        return False, res.message, res

    eq=np.max(np.abs(np.asarray(Aeq)@res.x-np.asarray(beq)))
    ub=0.0
    if Aub:
        ub=max(0.0,float(np.max(np.asarray(Aub)@res.x-np.asarray(bub))))
    neg=max(0.0,float(-np.min(res.x)))
    return bool(eq<1e-8 and ub<1e-8 and neg<1e-10), {
        "eq_resid":float(eq),"ub_resid":float(ub),"negative":float(neg)
    }, res

def extreme_law(c,j):
    B=np.zeros(max(2,j+1),float)
    B[0]=float(c)
    B[j]=1-float(c)
    return B

def random_law(rng,c,max_tail_index):
    B=np.zeros(max_tail_index+1,float)
    B[0]=c
    B[1]=0
    tail=rng.dirichlet(np.ones(max_tail_index-1))*(1-c)
    B[2:]=tail
    return B

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--random-tests",type=int,default=10000)
    ap.add_argument("--max-tail-index",type=int,default=32)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",type=str,default="entry_cert_v01_summary.json")
    args=ap.parse_args()

    exact={
        "c0":str(C0),
        "tau_c0":str(tau_frac(C0)),
        "c1":str(C1),
        "tau_c1":str(tau_frac(C1)),
        "c2":str(C2),
        "landing_threshold":str(LAND),
        "c2_below_landing":bool(C2<LAND),
        "delta_c0":str(delta_frac(C0)),
        "delta_c1":str(delta_frac(C1)),
        "delta_c2":str(delta_frac(C2)),
        "core_delta":str(delta_frac(Fraction(0))),
    }

    failures=[]
    max_eq=0.0
    max_ub=0.0

    # Extreme tails.
    critical_cs=[Fraction(0),Fraction(3,16),LAND,C2,C1,C0]
    extreme_count=0
    for c in critical_cs:
        for j in range(2,args.max_tail_index+1):
            B=extreme_law(c,j)

            ok,diag,_=solve_contract(B,"peel")
            extreme_count+=1
            if not ok:
                failures.append({"kind":"extreme_peel","c":str(c),"j":j,"diag":str(diag)})
                break
            if isinstance(diag,dict):
                max_eq=max(max_eq,diag["eq_resid"]);max_ub=max(max_ub,diag["ub_resid"])

            if c<=LAND:
                ok,diag,_=solve_contract(B,"land")
                extreme_count+=1
                if not ok:
                    failures.append({"kind":"extreme_land","c":str(c),"j":j,"diag":str(diag)})
                    break
                if isinstance(diag,dict):
                    max_eq=max(max_eq,diag["eq_resid"]);max_ub=max(max_ub,diag["ub_resid"])

    # Random finite-support laws.
    rng=np.random.default_rng(args.seed)
    random_count=0
    for i in range(args.random_tests):
        c=float(rng.random()*0.5)
        B=random_law(rng,c,args.max_tail_index)

        mode="land" if c<=float(LAND) else "peel"
        ok,diag,_=solve_contract(B,mode)
        random_count+=1
        if not ok:
            failures.append({"kind":"random","i":i,"c":c,"mode":mode,"diag":str(diag)})
            break
        if isinstance(diag,dict):
            max_eq=max(max_eq,diag["eq_resid"]);max_ub=max(max_ub,diag["ub_resid"])

    summary={
        "kind":"DREAM6-ZR Entry Lemma independent LP crash-test",
        "proof_claim":False,
        "exact_arithmetic":exact,
        "tested":{
            "extreme_contract_checks":extreme_count,
            "random_contract_checks":random_count,
            "max_tail_index":args.max_tail_index
        },
        "failures":failures,
        "numerical_audit":{
            "max_eq_residual":max_eq,
            "max_ub_residual":max_ub
        },
        "result":"PASS" if not failures else "FAIL",
        "note":"This is an independent numerical regression test of a separately proved transportation construction."
    }

    Path(args.out).write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
