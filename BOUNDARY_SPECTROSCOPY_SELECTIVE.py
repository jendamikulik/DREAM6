#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BOUNDARY_SPECTROSCOPY_SELECTIVE.py

Selective numerical spectroscopy for the coarse zero-repair boundary problem.

SCOUT architecture:
  * exact depth-1 LP only,
  * targeted edge objectives + a few random objectives,
  * recurse only on dangerous x in {0,2,4},
  * dynamic support,
  * local elastic duals on failures,
  * full 5-ary LP only for a few final candidates.

Exact coarse block:
  P_b = (1,12,38,12,1)/64
  Q_b = (0,16,32,16,0)/64

At every node:
  Q_b * B = sum_x P_b[x] shift_x(B_x)

This is a discovery instrument, not a proof engine.
"""

import argparse, csv, json, math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

P = np.array([1,12,38,12,1], float)/64.0
Q = np.array([0,16,32,16,0], float)/64.0

def trim(a,tol=1e-13):
    a=np.asarray(a,float)
    j=len(a)
    while j>1 and abs(a[j-1])<=tol: j-=1
    return a[:j].copy()

def normalize(a,tol=1e-12):
    a=np.asarray(a,float).copy()
    a[np.abs(a)<tol]=0
    if np.min(a)<-1e-9: raise ValueError("negative mass")
    a[a<0]=0
    s=a.sum()
    if s<=0: raise ValueError("zero mass")
    return trim(a/s)

def conv(a,b): return np.convolve(np.asarray(a,float),np.asarray(b,float))
def coeff(a,j): return float(a[j]) if 0<=j<len(a) else 0.0
def mean(a): return float(np.dot(np.arange(len(a)),a))
def variance(a):
    m=mean(a); return float(np.dot((np.arange(len(a))-m)**2,a))
def valuation(a,tol=1e-11):
    z=np.flatnonzero(np.asarray(a)>tol)
    return int(z[0]) if len(z) else 10**9
def top_support(a,tol=1e-11):
    z=np.flatnonzero(np.asarray(a)>tol)
    return int(z[-1]) if len(z) else 0
def rational_hint(x,den=4096):
    try: return str(Fraction(float(x)).limit_denominator(den))
    except Exception: return ""

def f_orbit(k):
    f=np.array([0.,1.])
    for _ in range(k):
        sq=conv(f,f)
        g=np.zeros(max(2,len(sq)))
        g[1]+=0.5
        g[:len(sq)]+=0.5*sq
        f=normalize(g)
    return f

def G0(A,C):
    d=conv(A,C)
    g=np.zeros(max(1,len(d)))
    g[0]+=0.5
    g[:len(d)]+=0.5*d
    return normalize(g)

def choose_support(B,cap=12,reserve=3):
    v=valuation(B); m=mean(B); sd=math.sqrt(max(0.,variance(B))); obs=top_support(B)
    need=max(obs,int(math.ceil(m+3*sd))+reserve,v+reserve+4)
    return max(obs,min(cap,need))

def pad_root(B,S):
    r=np.zeros(S+1); n=min(len(B),S+1); r[:n]=B[:n]
    if len(B)>S+1 and np.sum(B[S+1:])>1e-10: return None
    return r

@dataclass
class OneStep:
    S:int
    offs:tuple
    nvar:int
    Aeq:object
    beq:np.ndarray

def build_one_step(B,S):
    root=pad_root(B,S)
    if root is None: return None
    n=S+1; offs=tuple(x*n for x in range(5)); nvar=5*n
    ri=[];ci=[];va=[];rhs=[];row=0
    for x in range(5):
        off=offs[x]
        for j in range(n):
            ri.append(row);ci.append(off+j);va.append(1.)
        rhs.append(1.);row+=1
    qB=conv(Q,root)
    for s in range(max(len(qB)-1,S+4)+1):
        touched=False
        for x,px in enumerate(P):
            j=s-x
            if 0<=j<=S and px:
                ri.append(row);ci.append(offs[x]+j);va.append(-float(px));touched=True
        target=float(qB[s]) if s<len(qB) else 0.
        if touched or abs(target)>1e-15:
            rhs.append(-target);row+=1
    A=sparse.coo_matrix((va,(ri,ci)),shape=(row,nvar)).tocsr()
    return OneStep(S,offs,nvar,A,np.asarray(rhs))

def extract_children(vec,lay):
    out=[]
    for x in range(5):
        off=lay.offs[x]
        out.append(normalize(vec[off:off+lay.S+1]))
    return out

def one_step_test(B,S):
    lay=build_one_step(B,S)
    if lay is None: return False,None,None
    res=linprog(np.zeros(lay.nvar),A_eq=lay.Aeq,b_eq=lay.beq,
                bounds=[(0,None)]*lay.nvar,method="highs")
    return bool(res.success),res,lay

def hall_profile(B):
    qb=conv(Q,B); L=max(len(qb),len(P))
    fp=np.cumsum(np.pad(P,(0,L-len(P))))
    fq=np.cumsum(np.pad(qb,(0,L-len(qb))))
    sl=fp-fq; k=int(np.argmin(sl))
    d={"hall_min":float(sl[k]),"hall_k":k}
    for j in range(5): d[f"hall{j}"]=float(sl[j]) if j<len(sl) else 0.
    return d

def hall_objective(lay,x,k):
    # minimize Hall slack = maximize F_{Q*B_x}(k)
    # constant F_P(k) irrelevant, so c = - cumulative Q-convolution functional
    c=np.zeros(lay.nvar); off=lay.offs[x]
    for j in range(lay.S+1):
        w=0.
        for y,q in enumerate(Q):
            if j+y<=k: w+=q
        c[off+j]=-w
    return c

def objective_bank(lay,rng,branches,n_random):
    bank=[]
    idx=np.arange(lay.S+1,dtype=float)
    for x in branches:
        off=lay.offs[x]
        c=np.zeros(lay.nvar); c[off]=-1.; bank.append((f"max_B{x}_0",c))
        c=np.zeros(lay.nvar); c[off]=1.; bank.append((f"min_B{x}_0",c))
        c=np.zeros(lay.nvar); c[off]=-2.
        if lay.S>=1:c[off+1]=-1.
        bank.append((f"max_B{x}_edge01",c))
        for k in (1,3):
            bank.append((f"min_B{x}_hall{k}",hall_objective(lay,x,k)))
        c=np.zeros(lay.nvar);c[off:off+lay.S+1]=idx
        bank.append((f"min_B{x}_mean",c))
    w=np.exp(-np.arange(lay.S+1)/max(1.,lay.S/3))
    for r in range(n_random):
        c=np.zeros(lay.nvar)
        for x in range(5):
            scale=1.4 if x in branches else .25
            off=lay.offs[x]
            c[off:off+lay.S+1]=scale*rng.normal(size=lay.S+1)*w
        bank.append((f"rand_{r}",c))
    return bank

def generate_witnesses(B,S,rng,branches,n_random):
    lay=build_one_step(B,S)
    if lay is None:return []
    out=[];seen=set()
    for name,c in objective_bank(lay,rng,branches,n_random):
        res=linprog(c,A_eq=lay.Aeq,b_eq=lay.beq,
                    bounds=[(0,None)]*lay.nvar,method="highs")
        if not res.success: continue
        ch=extract_children(res.x,lay)
        sig=[]
        for x in branches:sig.extend(np.round(ch[x][:min(6,len(ch[x]))],9))
        sig=tuple(sig)
        if sig in seen:continue
        seen.add(sig);out.append((name,ch))
    return out

def danger(B):
    hp=hall_profile(B); h=hp["hall_min"]
    if h<0: ht=1e6
    else: ht=1./max(1e-5,h+1e-6)
    return float(5*coeff(B,0)+1.5*coeff(B,1)+.2*ht+1./(1+valuation(B)))

def local_elastic_dual(B,S,rng,probes=4):
    lay=build_one_step(B,S)
    if lay is None:return []
    test=linprog(np.zeros(lay.nvar),A_eq=lay.Aeq,b_eq=lay.beq,
                 bounds=[(0,None)]*lay.nvar,method="highs")
    if test.success:return []
    m=lay.Aeq.shape[0]
    A=sparse.hstack([lay.Aeq,sparse.eye(m),-sparse.eye(m)],format="csr")
    ans=[]
    for p in range(probes):
        c=np.zeros(lay.nvar+2*m);c[lay.nvar:]=1.
        c[:lay.nvar]=1e-9*rng.normal(size=lay.nvar)
        res=linprog(c,A_eq=A,b_eq=lay.beq,bounds=[(0,None)]*(lay.nvar+2*m),method="highs")
        if not res.success:continue
        y=np.asarray(res.eqlin.marginals,float)
        mx=np.max(np.abs(y))
        if mx>1e-12:y/=mx
        ans.append((float(res.fun),y))
    return ans

def full_tree_feasible(B,depth,S):
    root=pad_root(B,S)
    if root is None:return False
    levels=[5**k for k in range(depth+1)]
    starts=np.cumsum([0]+levels[:-1]).tolist()
    N=sum(levels);n=S+1;nvar=N*n
    def ni(l,p):return starts[l]+p
    ri=[];ci=[];va=[];rhs=[];row=0
    for node in range(N):
        off=node*n
        for j in range(n):ri.append(row);ci.append(off+j);va.append(1.)
        rhs.append(1.);row+=1
    for j in range(n):
        ri.append(row);ci.append(j);va.append(1.);rhs.append(root[j]);row+=1
    for l in range(depth):
        for pos in range(levels[l]):
            po=ni(l,pos)*n; co=[ni(l+1,pos*5+x)*n for x in range(5)]
            for s in range(S+5):
                touched=False
                for y,q in enumerate(Q):
                    j=s-y
                    if 0<=j<=S and q:
                        ri.append(row);ci.append(po+j);va.append(float(q));touched=True
                for x,p in enumerate(P):
                    j=s-x
                    if 0<=j<=S and p:
                        ri.append(row);ci.append(co[x]+j);va.append(float(-p));touched=True
                if touched:rhs.append(0.);row+=1
    A=sparse.coo_matrix((va,(ri,ci)),shape=(row,nvar)).tocsr()
    res=linprog(np.zeros(nvar),A_eq=A,b_eq=np.asarray(rhs),
                bounds=[(0,None)]*nvar,method="highs")
    return bool(res.success)

class Stream:
    def __init__(self,path,fields):
        self.f=open(path,"w",newline="",encoding="utf-8")
        self.w=csv.DictWriter(self.f,fieldnames=fields,extrasaction="ignore")
        self.w.writeheader()
    def write(self,r):self.w.writerow(r)
    def flush(self):self.f.flush()
    def close(self):self.f.close()

NODE_FIELDS=["node_id","sample","level","parent_id","branch_x","objective","valuation","top_support",
"mean","variance","B0","B1","B2","B3","B4","danger","hall_min","hall_k","hall0","hall1","hall2","hall3","hall4",
"support_used","one_step_feasible"]+[f"coef{i}" for i in range(12)]
EDGE_FIELDS=["sample","level","parent_id","child_id","branch_x","objective","danger","valuation","B0","hall_min","selected"]
DUAL_FIELDS=["sample","level","node_id","support_used","probe","soft_obj"]+[f"dual{i}" for i in range(40)]+[f"dual{i}_rat" for i in range(40)]
CAND_FIELDS=["node_id","sample","level","reason","danger","valuation","B0","B1","hall_min","hall_k","mean",
"support_used","verify_depth","verify_support","verify_pass"]+[f"coef{i}" for i in range(16)]

def record(B,node_id,sample,level,parent_id,bx,obj,S,ok):
    hp=hall_profile(B)
    r={"node_id":node_id,"sample":sample,"level":level,"parent_id":parent_id,"branch_x":bx,
       "objective":obj,"valuation":valuation(B),"top_support":top_support(B),"mean":mean(B),
       "variance":variance(B),"B0":coeff(B,0),"B1":coeff(B,1),"B2":coeff(B,2),
       "B3":coeff(B,3),"B4":coeff(B,4),"danger":danger(B),"support_used":S,
       "one_step_feasible":int(ok)}
    r.update(hp)
    for i in range(12):r[f"coef{i}"]=coeff(B,i)
    return r

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--samples",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--source-k",type=int,default=1)
    ap.add_argument("--scout-depth",type=int,default=5)
    ap.add_argument("--beam",type=int,default=4)
    ap.add_argument("--random-objectives",type=int,default=4)
    ap.add_argument("--seed-pool",type=int,default=64)
    ap.add_argument("--support-cap",type=int,default=12)
    ap.add_argument("--support-reserve",type=int,default=3)
    ap.add_argument("--branches",type=str,default="0,2,4")
    ap.add_argument("--dual-probes",type=int,default=4)
    ap.add_argument("--dual-every",type=int,default=25)
    ap.add_argument("--verify-top",type=int,default=20)
    ap.add_argument("--verify-depth",type=int,default=4)
    ap.add_argument("--verify-support",type=int,default=14)
    ap.add_argument("--status-every",type=int,default=100)
    ap.add_argument("--flush-every",type=int,default=100)
    ap.add_argument("--out",type=str,default="selective")
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    branches=tuple(int(x) for x in args.branches.split(","))
    source=f_orbit(args.source_k)

    nodes=Stream(args.out+"_nodes.csv",NODE_FIELDS)
    edges=Stream(args.out+"_edges.csv",EDGE_FIELDS)
    duals=Stream(args.out+"_duals.csv",DUAL_FIELDS)
    candidates=[];next_id=0;visited=0;fails=0

    # Precompute diverse one-step witnesses of the fixed source once.
    Ssrc=choose_support(source,args.support_cap,args.support_reserve)
    pool=[]
    for _ in range(max(1,args.seed_pool//8)):
        pool.extend(generate_witnesses(source,Ssrc,rng,branches,args.random_objectives))
        if len(pool)>=args.seed_pool:break
    if not pool:raise RuntimeError("source has no one-step witnesses")
    pool=pool[:args.seed_pool]

    sanity={}
    for k in range(3):
        f=f_orbit(k);S=choose_support(f,args.support_cap,args.support_reserve)
        ok,_,_=one_step_test(f,S)
        sanity[f"f{k}"]={"valuation":valuation(f),"mean":mean(f),"one_step":bool(ok)}

    for sample in range(args.samples):
        obj0,ch0=pool[int(rng.integers(len(pool)))]
        frontier=[]
        for bx in branches:
            B=G0(ch0[bx],source)
            S=choose_support(B,args.support_cap,args.support_reserve)
            ok,_,_=one_step_test(B,S)
            nid=next_id;next_id+=1;visited+=1
            nodes.write(record(B,nid,sample,0,-1,bx,"seed:"+obj0,S,ok))
            if ok:frontier.append((danger(B),nid,B))
            else:
                fails+=1
                candidates.append({"node_id":nid,"sample":sample,"level":0,"reason":"seed_fail","law":B,"support_used":S})

        for level in range(1,args.scout_depth+1):
            props=[]
            for _,pid,B in frontier:
                S=choose_support(B,args.support_cap,args.support_reserve)
                ws=generate_witnesses(B,S,rng,branches,args.random_objectives)
                if not ws:
                    fails+=1;candidates.append({"node_id":pid,"sample":sample,"level":level-1,"reason":"no_witness","law":B,"support_used":S})
                    if args.dual_probes and sample%max(1,args.dual_every)==0:
                        for p,(soft,y) in enumerate(local_elastic_dual(B,S,rng,args.dual_probes)):
                            dr={"sample":sample,"level":level-1,"node_id":pid,"support_used":S,"probe":p,"soft_obj":soft}
                            for i in range(min(40,len(y))):
                                dr[f"dual{i}"]=float(y[i]);dr[f"dual{i}_rat"]=rational_hint(y[i])
                            duals.write(dr)
                    continue
                seen=set()
                for obj,ch in ws:
                    for bx in branches:
                        C=ch[bx];sig=tuple(np.round(C[:min(8,len(C))],9))
                        if sig in seen:continue
                        seen.add(sig)
                        Sc=choose_support(C,args.support_cap,args.support_reserve)
                        ok,_,_=one_step_test(C,Sc)
                        nid=next_id;next_id+=1;visited+=1
                        rr=record(C,nid,sample,level,pid,bx,obj,Sc,ok);nodes.write(rr)
                        props.append((rr["danger"],nid,C,pid,bx,obj,ok,rr))
                        if not ok:
                            fails+=1;candidates.append({"node_id":nid,"sample":sample,"level":level,"reason":"one_step_fail","law":C,"support_used":Sc})
            if not props:
                frontier=[];break
            props.sort(key=lambda z:z[0],reverse=True)
            picked=[];div=set()
            for item in props:
                d,nid,B,pid,bx,obj,ok,rr=item
                if not ok:continue
                hp=hall_profile(B)
                key=(valuation(B),round(coeff(B,0),6),round(coeff(B,1),6),hp["hall_k"])
                if key in div and len(picked)>=max(1,args.beam//2):continue
                div.add(key);picked.append(item)
                if len(picked)>=args.beam:break
            keep={x[1] for x in picked}
            for item in props:
                d,nid,B,pid,bx,obj,ok,rr=item
                edges.write({"sample":sample,"level":level,"parent_id":pid,"child_id":nid,"branch_x":bx,
                             "objective":obj,"danger":d,"valuation":rr["valuation"],"B0":rr["B0"],
                             "hall_min":rr["hall_min"],"selected":int(nid in keep)})
            frontier=[(x[0],x[1],x[2]) for x in picked]
            if level==args.scout_depth:
                for d,nid,B in frontier:
                    candidates.append({"node_id":nid,"sample":sample,"level":level,"reason":"deep_survivor","law":B,
                                       "support_used":choose_support(B,args.support_cap,args.support_reserve)})

        if args.flush_every and (sample+1)%args.flush_every==0:
            nodes.flush();edges.flush();duals.flush()
        if args.status_every and (sample+1)%args.status_every==0:
            print(f"[{sample+1:8d}/{args.samples}] visited={visited} local_fail={fails} candidates={len(candidates)}",flush=True)

    nodes.close();edges.close();duals.close()

    # Enrich, dedupe and rank candidates.
    rich=[]
    for c in candidates:
        B=c["law"];hp=hall_profile(B)
        r={k:v for k,v in c.items() if k!="law"}
        r.update({"danger":danger(B),"valuation":valuation(B),"B0":coeff(B,0),"B1":coeff(B,1),
                  "hall_min":hp["hall_min"],"hall_k":hp["hall_k"],"mean":mean(B),"law":B})
        rich.append(r)
    rank={"one_step_fail":3,"seed_fail":3,"no_witness":3,"deep_survivor":1}
    rich.sort(key=lambda r:(rank.get(r["reason"],0),r["danger"]),reverse=True)
    uniq=[];seen=set()
    for r in rich:
        sig=tuple(np.round(r["law"][:min(16,len(r["law"]))],10))
        if sig in seen:continue
        seen.add(sig);uniq.append(r)

    cs=Stream(args.out+"_candidates.csv",CAND_FIELDS)
    verified=passed=0
    for i,r in enumerate(uniq):
        B=r["law"];out={k:v for k,v in r.items() if k!="law"}
        for j in range(16):out[f"coef{j}"]=coeff(B,j)
        if i<args.verify_top:
            Sv=max(top_support(B),min(args.verify_support,max(args.support_cap,top_support(B)+2)))
            ok=full_tree_feasible(B,args.verify_depth,Sv)
            out.update({"verify_depth":args.verify_depth,"verify_support":Sv,"verify_pass":int(ok)})
            verified+=1;passed+=int(ok)
        cs.write(out)
    cs.close()

    summary={"args":vars(args),"sanity":sanity,"source_pool":len(pool),"visited_nodes":visited,
             "local_failures":fails,"raw_candidates":len(candidates),"unique_candidates":len(uniq),
             "full_verified":verified,"full_verify_pass":passed,
             "warning":"Scout survival is selective, not a proof of full-tree viability; only verify_pass comes from a full tree LP."}
    with open(args.out+"_summary.json","w",encoding="utf-8") as f:json.dump(summary,f,indent=2,ensure_ascii=False)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
