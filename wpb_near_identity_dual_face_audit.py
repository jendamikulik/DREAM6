#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

P = np.array([1/8,1/4,1/4,1/4,1/8], dtype=float)
Q = np.array([7/64,5/16,5/32,5/16,7/64], dtype=float)
D=4

def highs_options():
    return {
        'dual_feasibility_tolerance':1e-9,
        'primal_feasibility_tolerance':1e-9,
        'ipm_optimality_tolerance':1e-9,
        'simplex_dual_edge_weight_strategy':'dantzig',
    }

def pow_conv(a, n):
    out=np.array([1.0])
    for _ in range(n): out=np.convolve(out,a)
    return out

def tree_arrays(depth):
    nnodes=sum(5**d for d in range(depth+1))
    dep=np.empty(nnodes,dtype=np.int16); dep[0]=0
    prob=np.empty(nnodes); prob[0]=1.0
    ssum=np.empty(nnodes,dtype=np.int32); ssum[0]=0
    for node in range(sum(5**d for d in range(depth))):
        d=int(dep[node])
        first=1+5*node
        for x in range(5):
            ch=first+x
            dep[ch]=d+1; prob[ch]=prob[node]*P[x]; ssum[ch]=ssum[node]+x
    return dep,prob,ssum

def build_common(n:int,M:int, full:bool):
    depth=2*n; N=M+1
    dep,prob,ssum=tree_arrays(depth)
    nnodes=len(dep)
    nvars=nnodes*N
    rows=[]; cols=[]; vals=[]; rhs=[]; meta=[]
    r=0
    def add_row(entries, b, m):
        nonlocal r
        # entries iterable (col,val)
        anyv=False
        for c,v in entries:
            if v!=0.0:
                rows.append(r); cols.append(c); vals.append(v); anyv=True
        if not anyv: raise RuntimeError(('zero row',m))
        rhs.append(float(b)); meta.append(m); r+=1
    def vid(node,j): return node*N+j

    # normalization all lifted laws
    for node in range(nnodes):
        add_row(((vid(node,j),1.0) for j in range(N)),1.0,('norm',int(node),None))

    # continuation ZR: depths n,...,2n-1
    internal=sum(5**d for d in range(depth))
    for node in range(internal):
        d=int(dep[node])
        if not (n <= d <= 2*n-1): continue
        first=1+5*node
        for u in range(M+D+1):
            ent=[]
            for k,q in enumerate(Q):
                j=u-k
                if 0<=j<N: ent.append((vid(node,j),q))
            for x,p in enumerate(P):
                j=u-x
                if 0<=j<N: ent.append((vid(first+x,j),-p))
            add_row(ent,0.0,('cont_zr',int(node),int(u)))

    # n-block endpoint identity at root
    qn=pow_conv(Q,n)
    depthn=np.where(dep==n)[0]
    for u in range(M+D*n+1):
        ent=[]
        for k,q in enumerate(qn):
            j=u-k
            if 0<=j<N: ent.append((vid(0,j),q))
        for node in depthn:
            j=u-int(ssum[node])
            if 0<=j<N: ent.append((vid(int(node),j),-float(prob[node])))
        add_row(ent,0.0,('block',0,int(u)))

    prefix_start=r
    # omitted prefix ZR: depths 1,...,n-1. Root follows algebraically.
    if full:
        for node in range(internal):
            d=int(dep[node])
            if not (1 <= d <= n-1): continue
            first=1+5*node
            for u in range(M+D+1):
                ent=[]
                for k,q in enumerate(Q):
                    j=u-k
                    if 0<=j<N: ent.append((vid(node,j),q))
                for x,p in enumerate(P):
                    j=u-x
                    if 0<=j<N: ent.append((vid(first+x,j),-p))
                add_row(ent,0.0,('prefix_zr',int(node),int(u)))
    prefix_end=r
    A=coo_matrix((vals,(rows,cols)),shape=(r,nvars)).tocsr()
    b=np.asarray(rhs)
    c=np.zeros(nvars); c[:N]=np.arange(N,dtype=float)
    return dict(A=A,b=b,c=c,meta=meta,dep=dep,prob=prob,ssum=ssum,N=N,nnodes=nnodes,
                prefix_range=(prefix_start,prefix_end),n=n,M=M)

def solve_primal(lp):
    t=time.time()
    res=linprog(lp['c'],A_eq=lp['A'],b_eq=lp['b'],bounds=(0,None),method='highs',options=highs_options())
    dt=time.time()-t
    if not res.success: raise RuntimeError(res.message)
    eq=float(np.max(np.abs(lp['A']@res.x-lp['b'])))
    return res,dt,eq

def centered_span(a_list,uvals,rtol=1e-10):
    if not a_list: return np.zeros((0,len(uvals))),0,[]
    F=[]
    for a in a_list:
        f=np.exp(-float(a)*uvals)
        f=f-f.mean()
        norm=np.linalg.norm(f)
        if norm>0: F.append(f/norm)
    F=np.asarray(F)
    U,s,Vh=np.linalg.svd(F,full_matrices=False)
    rank=int(np.sum(s > (s[0]*rtol if len(s) else 0)))
    return Vh[:rank],rank,s.tolist()

def build_projection_eq(lp,a_list):
    A=lp['A']; meta=lp['meta']; nrows=A.shape[0]
    ps,pe=lp['prefix_range']
    # map prefix node -> row by u
    bynode={}
    for rr in range(ps,pe):
        typ,node,u=meta[rr]
        assert typ=='prefix_zr'
        bynode.setdefault(node,{})[u]=rr
    if not bynode or not a_list:
        return None,None,{'nodes':len(bynode),'rank_per_node':0,'singular_values':[]}
    uvals=np.arange(lp['M']+D+1,dtype=float)
    basis,rank,svals=centered_span(a_list,uvals)
    rr=[];cc=[];dd=[]; rowid=0
    for node,umap in sorted(bynode.items()):
        for vec in basis:
            for u,coef in enumerate(vec):
                if abs(coef)>0:
                    rr.append(rowid);cc.append(umap[u]);dd.append(float(coef))
            rowid+=1
    C=coo_matrix((dd,(rr,cc)),shape=(rowid,nrows)).tocsr()
    return C,np.zeros(rowid),{'nodes':len(bynode),'rank_per_node':rank,'singular_values':svals}

def solve_dual_restricted(lp,a_list):
    # max b^T y s.t. A^T y <= c, y free, plus projection equalities.
    C,cb,info=build_projection_eq(lp,a_list)
    t=time.time()
    res=linprog(-lp['b'],A_ub=lp['A'].T,b_ub=lp['c'],A_eq=C,b_eq=cb,
                bounds=[(None,None)]*lp['A'].shape[0],method='highs',options=highs_options())
    dt=time.time()-t
    if not res.success: return None,dt,info,res.message
    return -float(res.fun),dt,info,None


from scipy.sparse import hstack

def solve_restricted_via_primal(lp,a_list):
    C,cb,info=build_projection_eq(lp,a_list)
    if C is None or C.shape[0]==0:
        res,dt,eq=solve_primal(lp)
        return float(res.fun),dt,info,None
    Aaug=hstack([lp['A'], C.T], format='csr')
    caug=np.concatenate([lp['c'], np.zeros(C.shape[0])])
    bounds=[(0,None)]*lp['A'].shape[1] + [(None,None)]*C.shape[0]
    t=time.time()
    res=linprog(caug,A_eq=Aaug,b_eq=lp['b'],bounds=bounds,method='highs',options=highs_options())
    dt=time.time()-t
    if not res.success: return None,dt,info,res.message
    return float(res.fun),dt,info,None

def run_case(n,M,grids):
    print(f'CASE n={n} M={M}',flush=True)
    blk=build_common(n,M,False); full=build_common(n,M,True)
    rb,tb,eb=solve_primal(blk); print(' block',rb.fun,'eq',eb,'sec',tb,'vars',blk['A'].shape[1],'rows',blk['A'].shape[0],flush=True)
    rf,tf,ef=solve_primal(full); print(' full ',rf.fun,'eq',ef,'sec',tf,'rows',full['A'].shape[0],flush=True)
    pi=float(rf.fun-rb.fun)
    print(' Pi',pi,flush=True)
    val0=float(rf.fun); td0=0.0; info0={}; err=None
    print(' dual baseline from primal strong duality',val0,flush=True)
    audits=[]
    for name,alist in grids:
        val,dt,info,err=solve_restricted_via_primal(full,alist)
        loss=None if val is None else float(rf.fun-val)
        print(' ',name,'rank',info['rank_per_node'],'value',val,'loss',loss,'sec',dt,flush=True)
        audits.append(dict(name=name,a=alist,value=val,loss=loss,seconds=dt,info=info,error=err))
    return dict(n=n,M=M,block=float(rb.fun),full=float(rf.fun),Pi=pi,block_eq=eb,full_eq=ef,
                vars=full['A'].shape[1],block_rows=blk['A'].shape[0],full_rows=full['A'].shape[0],
                full_dual=val0,full_dual_loss=float(rf.fun-val0) if val0 is not None else None,
                audits=audits)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,default=2)
    ap.add_argument('--M',type=int,nargs='+',default=[8,10,12])
    ap.add_argument('--out',default='/mnt/data/wpb_master_exp/near_identity_results.json')
    args=ap.parse_args()
    base=[2.0**(-k) for k in range(1,9)]  # .5 .. .00390625
    grids=[]
    for k in range(1,len(base)+1): grids.append((f'nested_{k}',base[:k]))
    # also individual near-identity rays
    for a in base: grids.append((f'individual_{a:.8g}',[a]))
    out=[]
    for M in args.M:
        out.append(run_case(args.n,M,grids))
    Path(args.out).write_text(json.dumps({'grid':base,'cases':out},indent=2))
    print('WROTE',args.out)
if __name__=='__main__': main()
