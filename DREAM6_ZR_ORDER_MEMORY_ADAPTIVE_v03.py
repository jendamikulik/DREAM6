#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DREAM6_ZR_ORDER_MEMORY_ONE_READOUT_v01.py
=========================================

DREAM6-style order-memory attack on the zero-repair problem.

This is deliberately NOT a grid scanner.

One run means exactly:

    exact semantic chart sigma_K(h)=(count(h), suffix_K(h))
        -> ONE deterministic global IEEE-754 binary32 field
        -> ONE simultaneous selector-support readout
        -> at most ONE exactification LP on that frozen face
        -> independent exact verifier
        -> post-readout projection ledger for k=0,...,K

The projection ledger NEVER feeds back into the field and NEVER triggers a
second readout.  It asks, after the fact, whether the single K-memory solution
actually collapses to smaller memory signatures.

Thus a depth-7, K=3 run simultaneously tells us how the ONE selected global
section behaves under count-only, last-1, last-2, and last-3 projections,
at every absolute tree depth 0,...,6.

This is a discovery solver.  A successful exactification gives a rigorous
finite-depth causal construction in the chosen K-memory class.  The projection
ledger is diagnostic: failure to collapse to k<K does not prove that no other
k-memory optimum exists.

No RNG, no restarts, no Boolean/discrete flips, no branching, no verifier
feedback, no residual-guided repair, no intermediate readout.

Example
-------
python DREAM6_ZR_ORDER_MEMORY_ONE_READOUT_v01.py ^
    --depth 7 ^
    --memory 3 ^
    --iterations 1200 ^
    --json-out ZR_ONE_r7_K3.json ^
    --npz-out ZR_ONE_r7_K3.npz

Go deeper:
python DREAM6_ZR_ORDER_MEMORY_ONE_READOUT_v01.py ^
    --depth 9 ^
    --memory 3 ^
    --iterations 1200 ^
    --json-out ZR_ONE_r9_K3.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

try:
    import torch
except Exception as exc:
    raise RuntimeError("PyTorch is required") from exc

VERSION = "DREAM6_ZR_ORDER_MEMORY_ADAPTIVE_v0.3_MARGIN_READOUT"

P = np.asarray([1/8, 1/4, 1/4, 1/4, 1/8], dtype=np.float64)
Q = np.asarray([7/64, 5/16, 5/32, 5/16, 7/64], dtype=np.float64)
B8 = np.asarray([
    0.53562333995590705,
    0.42450840479209673,
    0.039868255251995686,
], dtype=np.float64)


@dataclass(frozen=True)
class SigDAG:
    K: int
    nodes: tuple
    node_index: dict
    internals: tuple
    internal_index: np.ndarray
    child_index: np.ndarray
    leaf_index: np.ndarray
    node_depth: np.ndarray
    internal_depth: np.ndarray


@dataclass(frozen=True)
class Chart:
    M: int
    N: int
    valid: np.ndarray
    transition: np.ndarray
    ztuples: tuple
    by_jy: tuple
    by_xk: tuple


def update_signature(sig, x: int, K: int):
    counts, suffix = sig
    c = list(counts)
    c[x] += 1
    if K <= 0:
        s2 = ()
    else:
        s2 = (suffix + (x,))[-K:]
    return (tuple(c), tuple(s2))


def project_signature(sig, k: int):
    counts, suffix = sig
    if k <= 0:
        return (counts, ())
    return (counts, tuple(suffix[-k:]))


def build_signature_dag(depth: int, K: int) -> SigDAG:
    root = ((0,0,0,0,0), ())
    layers = [[root]]
    for _d in range(depth):
        nxt = set()
        for sig in layers[-1]:
            for x in range(5):
                nxt.add(update_signature(sig, x, K))
        layers.append(sorted(nxt))

    nodes = tuple(sig for layer in layers for sig in layer)
    idx = {sig:i for i,sig in enumerate(nodes)}
    internals = tuple(sig for layer in layers[:-1] for sig in layer)

    child = np.zeros((len(internals),5), dtype=np.int64)
    for ii,sig in enumerate(internals):
        for x in range(5):
            child[ii,x] = idx[update_signature(sig,x,K)]

    return SigDAG(
        K=K,
        nodes=nodes,
        node_index=idx,
        internals=internals,
        internal_index=np.asarray([idx[s] for s in internals],dtype=np.int64),
        child_index=child,
        leaf_index=np.asarray([idx[s] for s in layers[-1]],dtype=np.int64),
        node_depth=np.asarray([sum(s[0]) for s in nodes],dtype=np.int64),
        internal_depth=np.asarray([sum(s[0]) for s in internals],dtype=np.int64),
    )


def build_chart(M: int) -> Chart:
    N=M+1
    valid=np.zeros((N,5,5),dtype=bool)
    transition=np.zeros((5,N,N,5),dtype=np.float32)
    ztuples=[]
    by_jy=[[[] for _ in range(5)] for _ in range(N)]
    by_xk=[[[] for _ in range(N)] for _ in range(5)]
    for j in range(N):
        for y in range(5):
            for x in range(5):
                k=j+y-x
                if 0 <= k <= M:
                    zi=len(ztuples)
                    valid[j,y,x]=True
                    transition[x,k,j,y]=1.0
                    ztuples.append((j,y,x,k))
                    by_jy[j][y].append(zi)
                    by_xk[x][k].append(zi)
    return Chart(
        M=M,N=N,valid=valid,transition=transition,
        ztuples=tuple(ztuples),
        by_jy=tuple(tuple(tuple(v) for v in row) for row in by_jy),
        by_xk=tuple(tuple(tuple(v) for v in row) for row in by_xk),
    )


def configure_torch(threads: int):
    torch.set_num_threads(max(1,int(threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


def masked_softmax_x(logits, valid):
    dt=torch.float32
    masked=torch.where(valid[None],logits,torch.tensor(-80.0,dtype=dt))
    out=torch.softmax(masked,dim=3)*valid[None]
    return out/(out.sum(dim=3,keepdim=True)+torch.tensor(1e-30,dtype=dt))


def run_global_field(
    dag: SigDAG, chart: Chart, *,
    iterations: int,
    learning_rate: float,
    root_weight: float,
    temperature_stop: float,
    branch_weight: float,
    concurrence_weight: float,
    leaf_weight: float,
    entropy_weight: float,
    threads: int,
    verbose_every: int,
):
    configure_torch(threads)
    dt=torch.float32

    P_t=torch.tensor(P,dtype=dt)
    Q_t=torch.tensor(Q,dtype=dt)
    valid_t=torch.tensor(chart.valid,dtype=torch.bool)
    T_t=torch.tensor(chart.transition,dtype=dt)

    ii=torch.tensor(dag.internal_index,dtype=torch.long)
    ci=torch.tensor(dag.child_index,dtype=torch.long)
    li=torch.tensor(dag.leaf_index,dtype=torch.long)

    A_logits=torch.zeros((len(dag.nodes),chart.N),dtype=dt,requires_grad=True)
    K_logits=torch.zeros((len(dag.internals),chart.N,5,5),dtype=dt,requires_grad=True)

    opt=torch.optim.Adam(
        [A_logits,K_logits],
        lr=np.float32(learning_rate).item(),
        betas=(0.9,0.999),
        eps=1e-8,
    )

    jvals=torch.arange(chart.N,dtype=dt)
    c0=np.float32(B8[0]).item()
    c1=np.float32(B8[:2].sum()).item()

    history=[]
    started=time.perf_counter()

    for it in range(iterations):
        frac=np.float32(it/max(1,iterations-1)).item()
        temp=np.float32(float(temperature_stop)**float(frac)).item()
        ramp=max(0.0,(float(frac)-0.25)/0.75)
        ow=np.float32(root_weight*ramp).item()

        opt.zero_grad(set_to_none=True)

        A=torch.softmax(A_logits/np.float32(temp).item(),dim=1)
        K=masked_softmax_x(K_logits/np.float32(temp).item(),valid_t)

        z=A[ii,:,None,None]*Q_t[None,None,:,None]*K
        branch=z.sum(dim=(1,2))
        derived=torch.einsum("njyx,xkjy->nxk",z,T_t)/P_t[None,:,None]
        declared=A[ci]

        branch_loss=(((branch-P_t[None])**2)/(P_t[None]**2)).mean()
        concurrence_loss=(
            (derived-declared)**2/
            (declared.detach()+torch.tensor(0.03,dtype=dt))
        ).mean()
        leaf=A[li]
        leaf_loss=(
            torch.relu(leaf[:,0]-c0)**2+
            torch.relu(leaf[:,:2].sum(dim=1)-c1)**2
        ).mean()
        root_mean=(A[0]*jvals).sum()

        entA=-(A*torch.log(A+1e-20)).sum(dim=1).mean()
        entKrows=-(K*torch.log(K+1e-20)).sum(dim=3)
        row_exists=valid_t.any(dim=2)[None].expand(len(dag.internals),-1,-1)
        entK=entKrows[row_exists].mean()

        loss=(
            np.float32(branch_weight).item()*branch_loss+
            np.float32(concurrence_weight).item()*concurrence_loss+
            np.float32(leaf_weight).item()*leaf_loss+
            np.float32(ow).item()*root_mean+
            np.float32(entropy_weight).item()*(entA+entK)
        )
        loss.backward()
        opt.step()

        if it==0 or it==iterations-1 or (verbose_every>0 and (it+1)%verbose_every==0):
            row={
                "iteration":int(it),
                "loss":float(loss.detach()),
                "branch_loss":float(branch_loss.detach()),
                "concurrence_loss":float(concurrence_loss.detach()),
                "leaf_loss":float(leaf_loss.detach()),
                "root_mean_soft":float(root_mean.detach()),
                "temperature":float(temp),
            }
            history.append(row)
            print(
                f"[field] it={it:5d} loss={row['loss']:.6g} "
                f"branch={row['branch_loss']:.3e} "
                f"child={row['concurrence_loss']:.3e} "
                f"leaf={row['leaf_loss']:.3e} "
                f"root={row['root_mean_soft']:.9g}",
                flush=True,
            )

    with torch.no_grad():
        A=torch.softmax(
            A_logits/np.float32(temperature_stop).item(),dim=1
        )
        K=masked_softmax_x(
            K_logits/np.float32(temperature_stop).item(),valid_t
        )

    return (
        np.asarray(A.cpu(),dtype=np.float32),
        np.asarray(K.cpu(),dtype=np.float32),
        history,
        {
            "kind":"one_global_deterministic_binary32_partition_field",
            "iterations":int(iterations),
            "learning_rate":float(learning_rate),
            "root_weight":float(root_weight),
            "temperature_stop":float(temperature_stop),
            "threads":int(threads),
            "runtime_seconds":float(time.perf_counter()-started),
            "zero_initialization":True,
            "random_noise":False,
            "branching":False,
            "restarts":False,
            "discrete_flips":False,
            "intermediate_readout":False,
            "verifier_feedback":False,
            "residual_guided_repair":False,
        },
    )


def one_shot_readout(Kprob: np.ndarray, chart: Chart, topk: int):
    """
    FIRST and ONLY discrete state.

    One simultaneous support projection over every (node,j,y) row.
    """
    n=Kprob.shape[0]
    support=np.zeros_like(Kprob,dtype=bool)
    for ii in range(n):
        for j in range(chart.N):
            for y in range(5):
                allowed=np.flatnonzero(chart.valid[j,y])
                if allowed.size==0:
                    continue
                take=min(int(topk),int(allowed.size))
                order=np.argsort(-Kprob[ii,j,y,allowed],kind="stable")[:take]
                support[ii,j,y,allowed[order]]=True
    return support


def adaptive_margin_readout(
    Kprob: np.ndarray, chart: Chart, topk: int, margin_threshold: float
):
    """
    SECOND readout policy: per (node,j,y) row, keep the top-`topk`
    admissible x's ONLY IF the margin between the topk-th and (topk+1)-th
    largest K-probabilities is at or above `margin_threshold` (a confident,
    safe row to compress). If the margin is below threshold -- an uncertain
    row where the discarded atom might be load-bearing -- ALL admissible x's
    for that row are kept instead, i.e. no compression happens there.

    This is still exactly ONE deterministic pass over the frozen field
    output: no field feedback, no iteration, no second field run. It differs
    from one_shot_readout only in WHICH atoms a single readout keeps.

    Returns (support, stats) where stats records how many rows were
    compressed vs kept in full, and the realised margin_threshold used.
    """
    n = Kprob.shape[0]
    support = np.zeros_like(Kprob, dtype=bool)
    rows_compressed = 0
    rows_kept_full = 0
    rows_trivial = 0  # allowed.size <= topk already, nothing to decide
    for ii in range(n):
        for j in range(chart.N):
            for y in range(5):
                allowed = np.flatnonzero(chart.valid[j, y])
                if allowed.size == 0:
                    continue
                if allowed.size <= topk:
                    support[ii, j, y, allowed] = True
                    rows_trivial += 1
                    continue
                order = np.argsort(-Kprob[ii, j, y, allowed], kind="stable")
                sorted_vals = Kprob[ii, j, y, allowed][order]
                margin = float(sorted_vals[topk - 1] - sorted_vals[topk])
                if margin >= margin_threshold:
                    take = topk
                    rows_compressed += 1
                else:
                    take = int(allowed.size)
                    rows_kept_full += 1
                support[ii, j, y, allowed[order[:take]]] = True
    stats = {
        "policy": "adaptive_margin",
        "topk": int(topk),
        "margin_threshold": float(margin_threshold),
        "rows_trivial": int(rows_trivial),
        "rows_compressed_to_topk": int(rows_compressed),
        "rows_kept_full_due_to_low_margin": int(rows_kept_full),
    }
    return support, stats


def topk_margin(Kprob: np.ndarray, chart: Chart, topk: int):
    vals=[]
    for ii in range(Kprob.shape[0]):
        for j in range(chart.N):
            for y in range(5):
                allowed=np.flatnonzero(chart.valid[j,y])
                if len(allowed)<=topk:
                    continue
                v=np.sort(Kprob[ii,j,y,allowed])[::-1]
                vals.append(float(v[topk-1]-v[topk]))
    a=np.asarray(vals,dtype=np.float64)
    if not len(a):
        return {"count":0}
    return {
        "count":int(len(a)),
        "min":float(np.min(a)),
        "p10":float(np.quantile(a,0.10)),
        "median":float(np.median(a)),
        "p90":float(np.quantile(a,0.90)),
    }


def build_exact_lp(dag: SigDAG, chart: Chart, support: np.ndarray):
    """
    One exactification LP on the one frozen readout face.

    Sparse arrays are preallocated; unsupported z-atoms are not removed as
    columns, but are fixed to zero by bounds.  There is NO add-back/retry.
    """
    N=chart.N
    Z=len(chart.ztuples)
    nnode=len(dag.nodes)
    nint=len(dag.internals)
    nleaf=len(dag.leaf_index)

    base_z=nnode*N
    nv=base_z+nint*Z
    neq=nnode+nint*(10*N)
    nub=2*nleaf

    eq_nnz=nnode*N+nint*(2*Z+10*N)
    er=np.empty(eq_nnz,dtype=np.int32)
    ec=np.empty(eq_nnz,dtype=np.int64)
    ev=np.empty(eq_nnz,dtype=np.float64)
    be=np.zeros(neq,dtype=np.float64)

    pos=0
    rr=0

    for ni in range(nnode):
        sl=slice(pos,pos+N)
        er[sl]=rr
        ec[sl]=np.arange(ni*N,(ni+1)*N,dtype=np.int64)
        ev[sl]=1.0
        be[rr]=1.0
        pos+=N
        rr+=1

    for ii in range(nint):
        parent=int(dag.internal_index[ii])
        zo=base_z+ii*Z

        for j in range(N):
            for y in range(5):
                ids=chart.by_jy[j][y]
                L=len(ids)
                if L:
                    sl=slice(pos,pos+L)
                    er[sl]=rr
                    ec[sl]=zo+np.asarray(ids,dtype=np.int64)
                    ev[sl]=1.0
                    pos+=L
                er[pos]=rr
                ec[pos]=parent*N+j
                ev[pos]=-Q[y]
                pos+=1
                rr+=1

        for x in range(5):
            child=int(dag.child_index[ii,x])
            for k in range(N):
                ids=chart.by_xk[x][k]
                L=len(ids)
                if L:
                    sl=slice(pos,pos+L)
                    er[sl]=rr
                    ec[sl]=zo+np.asarray(ids,dtype=np.int64)
                    ev[sl]=1.0
                    pos+=L
                er[pos]=rr
                ec[pos]=child*N+k
                ev[pos]=-P[x]
                pos+=1
                rr+=1

    assert pos==eq_nnz and rr==neq
    Aeq=coo_matrix((ev,(er,ec)),shape=(neq,nv)).tocsr()

    ub_nnz=3*nleaf
    ur=np.empty(ub_nnz,dtype=np.int32)
    uc=np.empty(ub_nnz,dtype=np.int64)
    uv=np.ones(ub_nnz,dtype=np.float64)
    bu=np.empty(nub,dtype=np.float64)

    pos=0
    uu=0
    cdf=np.cumsum(B8)
    for ni0 in dag.leaf_index:
        ni=int(ni0)
        ur[pos]=uu; uc[pos]=ni*N; pos+=1
        bu[uu]=cdf[0]; uu+=1

        ur[pos:pos+2]=uu
        uc[pos]=ni*N; uc[pos+1]=ni*N+1; pos+=2
        bu[uu]=cdf[1]; uu+=1

    Aub=coo_matrix((uv,(ur,uc)),shape=(nub,nv)).tocsr()

    obj=np.zeros(nv,dtype=np.float64)
    obj[:N]=np.arange(N,dtype=np.float64)

    bounds=[(0.0,None)]*(nnode*N)
    for ii in range(nint):
        for j,y,x,k in chart.ztuples:
            bounds.append((0.0,None) if support[ii,j,y,x] else (0.0,0.0))

    return obj,Aeq,be,Aub,bu,bounds,{
        "base_z":base_z,"Z":Z,"nv":nv,
        "neq":neq,"nub":nub,"eq_nnz":eq_nnz,"ub_nnz":ub_nnz,
    }


def exactify_once(dag,chart,support,tol,time_limit):
    obj,Aeq,be,Aub,bu,bounds,meta=build_exact_lp(dag,chart,support)
    opts={
        "primal_feasibility_tolerance":max(float(tol),1e-10),
        "dual_feasibility_tolerance":max(float(tol),1e-10),
    }
    if time_limit is not None:
        opts["time_limit"]=float(time_limit)
    t=time.perf_counter()
    res=linprog(
        obj,A_ub=Aub,b_ub=bu,A_eq=Aeq,b_eq=be,bounds=bounds,
        method="highs-ds",options=opts,
    )
    meta.update({"Aeq":Aeq,"be":be,"Aub":Aub,"bu":bu})
    return res,meta,float(time.perf_counter()-t)


def extract_exact(dag,chart,res,meta):
    N=chart.N
    Z=meta["Z"]
    base=meta["base_z"]
    A=np.asarray(res.x[:base],dtype=np.float64).reshape(len(dag.nodes),N)
    z=np.zeros((len(dag.internals),N,5,5),dtype=np.float64)
    for ii in range(len(dag.internals)):
        local=res.x[base+ii*Z:base+(ii+1)*Z]
        for zi,(j,y,x,k) in enumerate(chart.ztuples):
            z[ii,j,y,x]=local[zi]
    return A,z


def verify_exact(dag,chart,A,z):
    max_norm=float(np.max(np.abs(A.sum(axis=1)-1)))
    min_A=float(np.min(A))
    min_z=float(np.min(z))
    fresh=branch=child=leafv=0.0

    for ii in range(len(dag.internals)):
        p=int(dag.internal_index[ii])
        for j in range(chart.N):
            for y in range(5):
                fresh=max(fresh,abs(float(z[ii,j,y].sum()-A[p,j]*Q[y])))

        bx=z[ii].sum(axis=(0,1))
        branch=max(branch,float(np.max(np.abs(bx-P))))

        for x in range(5):
            dl=np.zeros(chart.N,dtype=np.float64)
            for j,y,xx,k in chart.ztuples:
                if xx==x:
                    dl[k]+=z[ii,j,y,x]
            dl/=P[x]
            c=int(dag.child_index[ii,x])
            child=max(child,float(np.max(np.abs(dl-A[c]))))

    c0=float(B8[0]); c1=float(B8[:2].sum())
    for ni0 in dag.leaf_index:
        a=A[int(ni0)]
        leafv=max(leafv,float(a[0]-c0),float(a[:2].sum()-c1),0.0)

    ok=(
        max_norm<=5e-8 and min_A>=-5e-9 and min_z>=-5e-9 and
        fresh<=5e-8 and branch<=5e-8 and child<=5e-8 and leafv<=5e-8
    )
    return {
        "pass":bool(ok),
        "root_mean":float(A[0]@np.arange(chart.N,dtype=np.float64)),
        "max_normalization_error":max_norm,
        "minimum_A":min_A,
        "minimum_z":min_z,
        "freshness_max_error":fresh,
        "branch_marginal_max_error":branch,
        "child_generation_max_error":child,
        "leaf_dominance_max_violation":leafv,
    }


def _whole_node_support_key(support_node):
    return np.packbits(support_node.astype(np.uint8)).tobytes()


def _row_key(row):
    return np.packbits(row.astype(np.uint8)).tobytes()


def projection_ledger(
    dag: SigDAG,
    chart: Chart,
    support: np.ndarray,
    *,
    A_exact=None,
    z_exact=None,
    quant_eps: float=1e-6,
):
    """
    Inspect ALL lower-memory quotients from the SAME one-shot readout.

    For each absolute depth d and each projection k<=K:
      * group K-memory nodes by sigma_k;
      * measure how often the selected support disagrees inside a group;
      * if exactification succeeded, also measure exact law spread.

    No quantity here is fed back into the solver.
    """
    out=[]

    for d in range(int(np.max(dag.internal_depth))+1):
        ids=np.flatnonzero(dag.internal_depth==d)

        for k in range(0,min(d,dag.K)+1):
            groups=defaultdict(list)
            for ii0 in ids:
                ii=int(ii0)
                groups[project_signature(dag.internals[ii],k)].append(ii)

            mixed_groups=0
            node_disagree=0
            row_disagree=0
            row_total=0
            max_group_node_motifs=1

            law_spread_max=0.0
            law_quant_mixed=0
            z_support_mixed=0

            for members in groups.values():
                # Whole selector-support motif.
                motifs=[_whole_node_support_key(support[ii]) for ii in members]
                cnt=Counter(motifs)
                max_group_node_motifs=max(max_group_node_motifs,len(cnt))
                if len(cnt)>1:
                    mixed_groups+=1
                node_disagree += len(members)-max(cnt.values())

                # Row-wise support disagreement.
                for j in range(chart.N):
                    for y in range(5):
                        if not np.any(chart.valid[j,y]):
                            continue
                        keys=[_row_key(support[ii,j,y]) for ii in members]
                        cc=Counter(keys)
                        row_disagree += len(members)-max(cc.values())
                        row_total += len(members)

                if A_exact is not None:
                    node_ids=np.asarray(
                        [int(dag.internal_index[ii]) for ii in members],
                        dtype=np.int64,
                    )
                    laws=A_exact[node_ids]
                    spread=float(np.max(np.max(laws,axis=0)-np.min(laws,axis=0)))
                    law_spread_max=max(law_spread_max,spread)

                    q=np.rint(laws/float(quant_eps)).astype(np.int64)
                    if len({row.tobytes() for row in q})>1:
                        law_quant_mixed+=1

                if z_exact is not None:
                    zkeys={
                        np.packbits((z_exact[ii]>1e-12).astype(np.uint8)).tobytes()
                        for ii in members
                    }
                    if len(zkeys)>1:
                        z_support_mixed+=1

            nstates=len(ids)
            ngroups=len(groups)
            row={
                "absolute_depth":int(d),
                "projected_memory_k":int(k),
                "source_memory_K":int(dag.K),
                "states":int(nstates),
                "projected_groups":int(ngroups),
                "compression_factor_states_per_group":float(nstates/max(1,ngroups)),
                "mixed_selector_groups":int(mixed_groups),
                "mixed_selector_group_fraction":float(mixed_groups/max(1,ngroups)),
                "selector_node_disagreement_fraction":float(node_disagree/max(1,nstates)),
                "selector_row_disagreement_fraction":float(row_disagree/max(1,row_total)),
                "max_selector_motifs_in_one_group":int(max_group_node_motifs),
                "exact_selector_quotient":bool(node_disagree==0),
            }
            if A_exact is not None:
                row.update({
                    "exact_law_group_linf_spread_max":float(law_spread_max),
                    "exact_law_quantized_mixed_groups":int(law_quant_mixed),
                    "exact_law_quantization_eps":float(quant_eps),
                })
            if z_exact is not None:
                row["exact_z_support_mixed_groups"]=int(z_support_mixed)

            out.append(row)

    # Collapse each k across absolute depths to one headline.
    summary=[]
    for k in range(dag.K+1):
        rows=[r for r in out if r["projected_memory_k"]==k]
        if not rows:
            continue
        weighted_states=sum(r["states"] for r in rows)
        item={
            "projected_memory_k":int(k),
            "depths_seen":int(len(rows)),
            "all_depths_exact_selector_quotient":bool(
                all(r["exact_selector_quotient"] for r in rows)
            ),
            "max_selector_node_disagreement_fraction":float(
                max(r["selector_node_disagreement_fraction"] for r in rows)
            ),
            "max_selector_row_disagreement_fraction":float(
                max(r["selector_row_disagreement_fraction"] for r in rows)
            ),
            "mean_selector_node_disagreement_fraction_weighted":float(
                sum(r["selector_node_disagreement_fraction"]*r["states"] for r in rows)
                / max(1,weighted_states)
            ),
        }
        if A_exact is not None:
            item["max_exact_law_group_linf_spread"]=float(
                max(r.get("exact_law_group_linf_spread_max",0.0) for r in rows)
            )
            item["all_depths_exact_law_quotient_at_eps"]=bool(
                all(r.get("exact_law_quantized_mixed_groups",0)==0 for r in rows)
            )
        if z_exact is not None:
            item["all_depths_exact_z_support_quotient"]=bool(
                all(r.get("exact_z_support_mixed_groups",0)==0 for r in rows)
            )
        summary.append(item)

    return {"by_depth_and_projection":out,"projection_summary":summary}


def state_counts_by_depth(dag):
    return {
        str(d):int(np.sum(dag.node_depth==d))
        for d in range(int(np.max(dag.node_depth))+1)
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--depth",type=int,default=7)
    ap.add_argument("--memory",type=int,default=3,
                    help="largest exact suffix memory K carried by the ONE global field")
    ap.add_argument("--M",type=int,default=8)
    ap.add_argument("--iterations",type=int,default=500)
    ap.add_argument("--learning-rate",type=float,default=0.02)
    ap.add_argument("--root-weight",type=float,default=0.02)
    ap.add_argument("--temperature-stop",type=float,default=0.6)
    ap.add_argument("--branch-weight",type=float,default=20.0)
    ap.add_argument("--concurrence-weight",type=float,default=60.0)
    ap.add_argument("--leaf-weight",type=float,default=150.0)
    ap.add_argument("--entropy-weight",type=float,default=0.0)
    ap.add_argument("--readout-topk",type=int,default=4)
    ap.add_argument(
        "--readout-policy",choices=["fixed","adaptive-margin"],default="adaptive-margin",
        help="fixed: original one_shot_readout (always keep exactly readout-topk). "
             "adaptive-margin: keep readout-topk only on rows whose margin to the "
             "next-best atom is >= --margin-threshold; low-margin rows keep all 5.",
    )
    ap.add_argument(
        "--margin-threshold",type=float,default=0.05,
        help="only used by --readout-policy adaptive-margin",
    )
    ap.add_argument("--threads",type=int,default=0,
                    help="0=auto (up to 8 CPU threads); use 1 for strict single-thread reproducibility")
    ap.add_argument("--verbose-every",type=int,default=400)
    ap.add_argument("--exactify",action="store_true",
                    help="after the one readout, run ONE exact LP on that frozen face")
    ap.add_argument("--resume-npz",type=Path,default=None,
                    help="reuse A_soft/K_soft/K_support from a previous run; no new field and no new readout")
    ap.add_argument("--lp-tol",type=float,default=1e-9)
    ap.add_argument("--lp-time-limit",type=float,default=None)
    ap.add_argument("--quant-eps",type=float,default=1e-6)
    ap.add_argument("--json-out",type=Path,default=None)
    ap.add_argument("--npz-out",type=Path,default=None)
    args=ap.parse_args()

    if args.depth<1:
        raise ValueError("depth must be >=1")
    K=max(0,min(int(args.memory),args.depth-1))

    effective_threads = (
        max(1, min(8, os.cpu_count() or 1))
        if int(args.threads) == 0 else max(1, int(args.threads))
    )

    print(f"=== {VERSION} ===",flush=True)
    print(
        f"ONE RUN: depth={args.depth} source-memory K={K} M={args.M}",
        flush=True,
    )
    print(
        "contract: ONE binary32 field -> ONE support readout -> "
        + ("ONE frozen-face exactification -> verifier -> ledger"
           if args.exactify else "post-readout ledger"),
        flush=True,
    )
    print(f"torch CPU threads={effective_threads}", flush=True)

    dag=build_signature_dag(args.depth,K)
    chart=build_chart(args.M)
    raw=(5**(args.depth+1)-1)//4
    print(
        f"states={len(dag.nodes):,} internals={len(dag.internals):,} "
        f"leaves={len(dag.leaf_index):,} raw-tree={raw:,} "
        f"compression={raw/len(dag.nodes):.3f}x",
        flush=True,
    )
    print("state layers:",state_counts_by_depth(dag),flush=True)

    if args.resume_npz is not None:
        saved=np.load(args.resume_npz)
        Asoft=np.asarray(saved["A_soft"],dtype=np.float32)
        Ksoft=np.asarray(saved["K_soft"],dtype=np.float32)
        support=np.asarray(saved["K_support"],dtype=bool)
        if Asoft.shape != (len(dag.nodes),chart.N):
            raise ValueError("resume A_soft shape does not match requested depth/memory/M")
        if Ksoft.shape != (len(dag.internals),chart.N,5,5):
            raise ValueError("resume K_soft shape does not match requested depth/memory/M")
        if support.shape != Ksoft.shape:
            raise ValueError("resume K_support shape mismatch")
        field_history=[]
        field_meta={
            "resumed_from":str(args.resume_npz),
            "field_rerun":False,
            "readout_rerun":False,
        }
        print(
            f"RESUME: reusing the SAME frozen readout from {args.resume_npz}",
            flush=True,
        )
        readout_policy_meta = {"policy": "resumed_from_npz", "source": str(args.resume_npz)}
    else:
        Asoft,Ksoft,field_history,field_meta=run_global_field(
            dag,chart,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            root_weight=args.root_weight,
            temperature_stop=args.temperature_stop,
            branch_weight=args.branch_weight,
            concurrence_weight=args.concurrence_weight,
            leaf_weight=args.leaf_weight,
            entropy_weight=args.entropy_weight,
            threads=effective_threads,
            verbose_every=args.verbose_every,
        )

        # FIRST AND ONLY READOUT.
        if args.readout_policy == "adaptive-margin":
            support, readout_policy_meta = adaptive_margin_readout(
                Ksoft, chart, args.readout_topk, args.margin_threshold
            )
        else:
            support = one_shot_readout(Ksoft, chart, args.readout_topk)
            readout_policy_meta = {"policy": "fixed", "topk": int(args.readout_topk)}

    valid=np.broadcast_to(chart.valid[None],support.shape)
    selected=int(np.sum(support & valid))
    total=int(np.sum(valid))
    margins=topk_margin(Ksoft,chart,args.readout_topk)
    print(
        f"ONE READOUT ({readout_policy_meta.get('policy')}) selected={selected:,}/{total:,} "
        f"({selected/max(1,total):.6f}) margin_median={margins.get('median')}",
        flush=True,
    )
    if readout_policy_meta.get("policy") == "adaptive_margin":
        print(
            f"  rows: trivial={readout_policy_meta['rows_trivial']:,} "
            f"compressed_to_topk={readout_policy_meta['rows_compressed_to_topk']:,} "
            f"kept_full_low_margin={readout_policy_meta['rows_kept_full_due_to_low_margin']:,} "
            f"(threshold={readout_policy_meta['margin_threshold']})",
            flush=True,
        )

    exact_meta=None
    verification=None
    Aexact=None
    zexact=None

    if args.exactify:
        print("[exactification] ONE frozen-face LP; no add-back, no second readout",flush=True)
        res,lpmeta,secs=exactify_once(
            dag,chart,support,args.lp_tol,args.lp_time_limit
        )
        exact_meta={
            "success":bool(res.success),
            "status":int(res.status),
            "message":str(res.message),
            "objective":float(res.fun) if res.success else None,
            "runtime_seconds":float(secs),
            "variables":int(lpmeta["nv"]),
            "equalities":int(lpmeta["neq"]),
            "inequalities":int(lpmeta["nub"]),
            "eq_nnz":int(lpmeta["eq_nnz"]),
        }
        if res.success:
            Aexact,zexact=extract_exact(dag,chart,res,lpmeta)
            verification=verify_exact(dag,chart,Aexact,zexact)
            print(
                f"EXACTIFICATION SUCCESS root={res.fun:.15f} "
                f"verify={verification['pass']} time={secs:.2f}s",
                flush=True,
            )
        else:
            print(
                f"EXACTIFICATION FAIL status={res.status}: {res.message}",
                flush=True,
            )

    ledger=projection_ledger(
        dag,chart,support,
        A_exact=Aexact,z_exact=zexact,
        quant_eps=args.quant_eps,
    )

    print("\nPROJECTION SUMMARY -- SAME ONE READOUT",flush=True)
    print(" k   exact-quotient   max-node-disagree   max-row-disagree",flush=True)
    print("-"*64,flush=True)
    for r in ledger["projection_summary"]:
        extra=""
        if "max_exact_law_group_linf_spread" in r:
            extra=(
                f"   law_spread={r['max_exact_law_group_linf_spread']:.3e}"
                f" lawQ={r['all_depths_exact_law_quotient_at_eps']}"
            )
        print(
            f"{r['projected_memory_k']:2d}   "
            f"{str(r['all_depths_exact_selector_quotient']):>13s}   "
            f"{r['max_selector_node_disagreement_fraction']:.9f}   "
            f"{r['max_selector_row_disagreement_fraction']:.9f}"
            f"{extra}",
            flush=True,
        )

    report={
        "version":VERSION,
        "problem":{
            "depth":int(args.depth),
            "source_memory_K":int(K),
            "M":int(args.M),
            "P":P.tolist(),
            "Q":Q.tolist(),
            "B8":B8.tolist(),
        },
        "state_space":{
            "signature":"sigma_K(h)=(count(h), suffix_K(h))",
            "states":len(dag.nodes),
            "internals":len(dag.internals),
            "leaves":len(dag.leaf_index),
            "raw_tree_nodes":raw,
            "compression_factor":raw/len(dag.nodes),
            "by_depth":state_counts_by_depth(dag),
        },
        "contract":{
            "one_global_field":True,
            "one_support_readout":True,
            "intermediate_readouts":0,
            "branching":False,
            "discrete_flips":False,
            "restarts":False,
            "verifier_feedback":False,
            "residual_guided_repair":False,
            "exactification_may_add_support":False,
            "exactification_requested":bool(args.exactify),
            "post_readout_ledger_feeds_back":False,
        },
        "field":field_meta,
        "field_history":field_history,
        "readout":{
            "topk":int(args.readout_topk),
            "selected_atoms":selected,
            "valid_atoms":total,
            "selected_fraction":selected/max(1,total),
            "topk_margin":margins,
            "policy":readout_policy_meta,
        },
        "exactification":exact_meta,
        "verification":verification,
        "projection_ledger":ledger,
        "epistemic_status":{
            "successful_exactification":
                "PROVED finite-depth construction in the chosen source-memory K class",
            "projection_collapse":
                "diagnostic property of this one readout only; not an optimality theorem for smaller k",
        },
        "environment":{
            "python":platform.python_version(),
            "numpy":np.__version__,
            "torch":torch.__version__,
        },
    }

    if args.json_out:
        args.json_out.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"JSON: {args.json_out}",flush=True)

    if args.npz_out:
        arr={
            "A_soft":Asoft,
            "K_soft":Ksoft,
            "K_support":support.astype(np.uint8),
            "internal_index":dag.internal_index,
            "child_index":dag.child_index,
            "leaf_index":dag.leaf_index,
        }
        if Aexact is not None:
            arr["A_exact"]=Aexact
            arr["z_exact"]=zexact
        np.savez_compressed(args.npz_out,**arr)
        print(f"NPZ: {args.npz_out}",flush=True)

    return 0


if __name__=="__main__":
    raise SystemExit(main())
