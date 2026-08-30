#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, itertools, json, math, time
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack, csr_matrix

P=np.asarray([1/8,1/4,1/4,1/4,1/8],float)
Q=np.asarray([7/64,5/16,5/32,5/16,7/64],float)
D=4

def highs_options(time_limit=None):
    d={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10,'presolve':True}
    if time_limit: d['time_limit']=float(time_limit)
    return d

def powconv(a,n):
    z=np.array([1.0])
    for _ in range(n): z=np.convolve(z,a)
    return z

def hprob(h):
    v=1.0
    for x in h: v*=P[x]
    return v

def hsum(h): return sum(h)

class ViabilityOracle:
    def __init__(self,M,oracle_tol=2e-8,boundary_tol=1e-8,max_loops=1000,padding=1e-12):
        self.M=M; self.N=M+1; self.SMAX=M+4
        self.edges=tuple((s,x) for s in range(self.SMAX+1) for x in range(5) if 0<=s-x<=M)
        self.eidx={e:i for i,e in enumerate(self.edges)}; self.NE=len(self.edges)
        self.conv=np.zeros((self.SMAX+1,self.N))
        for s in range(self.SMAX+1):
            for j in range(self.N):
                y=s-j
                if 0<=y<5: self.conv[s,j]=Q[y]
        self.leq=np.zeros((self.SMAX+1+4,self.NE))
        for s,x in self.edges: self.leq[s,self.eidx[(s,x)]]=1
        for x in range(4):
            for s in range(self.SMAX+1):
                if (s,x) in self.eidx: self.leq[self.SMAX+1+x,self.eidx[(s,x)]]=1
        self.cache=defaultdict(list) # E0 simplex has no non-simplex cuts
        self.oracle_tol=oracle_tol; self.boundary_tol=boundary_tol; self.max_loops=max_loops; self.padding=padding
        self.stats=defaultdict(int); self.boundary_accepts=0; self.max_boundary_phi=0
        self._seed_exact_E1()
    def _seed_exact_E1(self):
        # Exact one-step viability for terminal simplex:
        # exists coupling X~P, S~A*Q with X <= S <= X+M iff
        # F_P(t-M) <= F_{A*Q}(t) <= F_P(t) for all t.
        FP=np.cumsum(P)
        for t in range(4):
            a=np.zeros(self.N)
            for j in range(self.N):
                a[j]=sum(Q[y] for y in range(5) if j+y<=t)
            self.add_cut(1,a,float(FP[t]),f'E1_CDF_UP_{t}')
        for q in range(4):
            t=self.M+q; a=np.zeros(self.N)
            for j in range(self.N):
                a[j]=-sum(Q[y] for y in range(5) if j+y<=t)
            self.add_cut(1,a,-float(FP[q]),f'E1_CDF_LOW_{q}')

    def canon(self,a,b):
        z=max(np.max(np.abs(a)),1e-15); return np.asarray(a)/z,float(b)/z
    def add_cut(self,lvl,a,b,label):
        ca,cb=self.canon(np.asarray(a,float),b)
        for i,(aa,bb,ll) in enumerate(self.cache[lvl]):
            xa,xb=self.canon(aa,bb)
            if np.max(np.abs(xa-ca))<1e-8:
                if cb<xb-1e-10:
                    self.cache[lvl][i]=(ca,cb,label+'*'); return 'strengthened'
                return 'duplicate'
        self.cache[lvl].append((ca,cb,label)); return 'added'
    def direct(self,lvl,A):
        if not self.cache[lvl]: return None
        vals=[(float(a@A-b),a,b,l) for a,b,l in self.cache[lvl]]
        w=max(vals,key=lambda z:z[0])
        return w if w[0]>self.oracle_tol else None
    def child_ub(self,cuts):
        if not cuts: return None,None
        G=[];h=[]
        for x in range(5):
            for a,b,l in cuts:
                row=np.zeros(self.NE)
                for k in range(self.N):
                    s=k+x
                    if (s,x) in self.eidx: row[self.eidx[(s,x)]]=a[k]
                G.append(row); h.append(P[x]*b)
        return np.asarray(G),np.asarray(h)
    def exact_local(self,A,cuts):
        G,h=self.child_ub(cuts); d=np.concatenate([self.conv@A,P[:4]])
        return linprog(np.zeros(self.NE),A_ub=G,b_ub=h,A_eq=self.leq,b_eq=d,bounds=(0,None),method='highs',options=highs_options())
    def phase1(self,A,cuts):
        G,h=self.child_ub(cuts); ni=0 if G is None else G.shape[0]; neq=self.leq.shape[0]
        nv=self.NE+2*neq+ni; c=np.zeros(nv); c[self.NE:self.NE+2*neq]=1
        if ni: c[self.NE+2*neq:]=1
        Aeq=np.zeros((neq,nv)); Aeq[:,:self.NE]=self.leq; Aeq[:,self.NE:self.NE+neq]=np.eye(neq); Aeq[:,self.NE+neq:self.NE+2*neq]=-np.eye(neq)
        d=np.concatenate([self.conv@A,P[:4]])
        if ni:
            Aub=np.zeros((ni,nv)); Aub[:,:self.NE]=G; Aub[:,self.NE+2*neq:]=-np.eye(ni); bub=h
        else: Aub=None;bub=None
        return linprog(c,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=d,bounds=(0,None),method='highs',options=highs_options())
    def separate(self,lvl,A,path=()):
        self.stats[f'calls_E{lvl}']+=1; A=np.asarray(A,float)
        dr=self.direct(lvl,A)
        if dr is not None:
            _,a,b,l=dr; return False,(a,b,l)
        if lvl<=1: return True,None
        for loop in range(self.max_loops):
            self.stats[f'loops_E{lvl}']+=1
            r=self.exact_local(A,self.cache[lvl-1]); self.stats[f'exact_E{lvl}']+=1
            if not r.success:
                ph=self.phase1(A,self.cache[lvl-1]); self.stats[f'phase1_E{lvl}']+=1
                if not ph.success: return None,('PHASE1_FAILED',lvl,ph.message)
                phi=float(ph.fun)
                if phi<=self.boundary_tol:
                    self.boundary_accepts+=1; self.max_boundary_phi=max(self.max_boundary_phi,phi); return True,None
                lam=np.asarray(ph.eqlin.marginals,float)
                g=self.conv.T@lam[:self.SMAX+1]
                beta=float(g@A-phi+self.padding)
                self.add_cut(lvl,g,beta,f'FARKAS_E{lvl}_phi={phi:.3e}')
                return False,(g,beta,f'FARKAS_E{lvl}')
            F=np.asarray(r.x,float); restart=False
            for x in range(5):
                child=np.zeros(self.N)
                for k in range(self.N):
                    s=k+x
                    if (s,x) in self.eidx: child[k]=F[self.eidx[(s,x)]]/P[x]
                before=len(self.cache[lvl-1]); ok,info=self.separate(lvl-1,child,path+(x,)); after=len(self.cache[lvl-1])
                if ok is None: return None,info
                if not ok:
                    a,b,l=info; self.add_cut(lvl-1,a,b,l); restart=True; break
            if restart: continue
            return True,None
        return None,('MAX_LOOPS',lvl)


def save_oracle_cache(O,path):
    if not path: return
    data={str(k):[{'alpha':a.tolist(),'beta':float(b),'label':l} for a,b,l in rows] for k,rows in O.cache.items()}
    with open(path,'w') as f: json.dump({'M':O.M,'cuts':data,'boundary_accepts':O.boundary_accepts,'max_boundary_phi':O.max_boundary_phi},f,indent=2)

def load_oracle_cache(O,path):
    import os
    if not path or not os.path.exists(path): return
    z=json.load(open(path))
    if int(z.get('M',-1))!=O.M: return
    for ks,rows in z.get('cuts',{}).items():
        lvl=int(ks)
        for rec in rows: O.add_cut(lvl,np.asarray(rec['alpha'],float),float(rec['beta']),rec.get('label','resume'))
    print('[oracle checkpoint] resumed',path,{k:len(v) for k,v in O.cache.items()},flush=True)

class PrefixMaster:
    def __init__(self,n,M):
        self.n=n; self.M=M; self.N=M+1
        self.nodes=tuple(h for d in range(n+1) for h in itertools.product(range(5),repeat=d))
        self.idx={h:i for i,h in enumerate(self.nodes)}
        self.leaves=tuple(itertools.product(range(5),repeat=n))
        self.pref=tuple(h for d in range(1,n) for h in itertools.product(range(5),repeat=d))
        self.nA=len(self.nodes)*self.N
    def av(self,h,j): return self.idx[h]*self.N+j
    def zr_coeff(self,h,u):
        z=[]
        for y,q in enumerate(Q):
            j=u-y
            if 0<=j<self.N: z.append((self.av(h,j),q))
        for x,p in enumerate(P):
            ch=h+(x,); j=u-x
            if 0<=j<self.N: z.append((self.av(ch,j),-p))
        return z
    def cum_coeff(self,h,k):
        d={}
        for u in range(k+1):
            for c,v in self.zr_coeff(h,u): d[c]=d.get(c,0)+v
        return d
    def core_eq(self):
        rows=[];cols=[];vals=[];rhs=[];r=0
        def add(ent,b):
            nonlocal r
            for c,v in ent:
                if v: rows.append(r);cols.append(c);vals.append(v)
            rhs.append(b);r+=1
        for h in self.nodes: add([(self.av(h,j),1) for j in range(self.N)],1)
        qn=powconv(Q,self.n)
        for u in range(self.M+D*self.n+1):
            ent=[]
            for k,q in enumerate(qn):
                j=u-k
                if 0<=j<self.N: ent.append((self.av((),j),q))
            for h in self.leaves:
                j=u-hsum(h)
                if 0<=j<self.N: ent.append((self.av(h,j),-hprob(h)))
            add(ent,0)
        return coo_matrix((vals,(rows,cols)),shape=(r,self.nA)).tocsr(),np.array(rhs)
    def root_obj(self):
        c=np.zeros(self.nA); c[:self.N]=np.arange(self.N); return c


def master_solve(PM,target_cuts,mode='root',root_cap=None,time_limit=120):
    Aeq,beq=PM.core_eq(); nA=PM.nA
    # viability cuts applied to every boundary leaf
    ur=[];uc=[];uv=[];ub=[];rr=0
    for a,b,l in target_cuts:
        for h in PM.leaves:
            for j,v in enumerate(a):
                if v: ur.append(rr);uc.append(PM.av(h,j));uv.append(v)
            ub.append(b);rr+=1
    if mode=='root':
        Aub=coo_matrix((uv,(ur,uc)),shape=(rr,nA)).tocsr() if rr else None
        obj=PM.root_obj(); bounds=(0,None)
    else:
        titems=[(h,k,hprob(h)) for h in PM.pref for k in range(PM.M+D)]
        nt=len(titems); off=nA
        # root cap
        if root_cap is not None:
            for j in range(PM.N):
                if j: ur.append(rr);uc.append(j);uv.append(j)
            ub.append(root_cap);rr+=1
        for ti,(h,k,w) in enumerate(titems):
            cf=PM.cum_coeff(h,k)
            for col,v in cf.items(): ur.append(rr);uc.append(col);uv.append(v)
            ur.append(rr);uc.append(off+ti);uv.append(-1);ub.append(0);rr+=1
            for col,v in cf.items(): ur.append(rr);uc.append(col);uv.append(-v)
            ur.append(rr);uc.append(off+ti);uv.append(-1);ub.append(0);rr+=1
        # pad Aeq with t columns
        Aeq=csr_matrix(np.hstack([Aeq.toarray(),np.zeros((Aeq.shape[0],nt))])) if nt<2000 else None
        if Aeq is None:
            from scipy.sparse import hstack
            A0,beq=PM.core_eq(); Aeq=hstack([A0,csr_matrix((A0.shape[0],nt))],format='csr')
        Aub=coo_matrix((uv,(ur,uc)),shape=(rr,nA+nt)).tocsr()
        obj=np.zeros(nA+nt)
        for ti,(_,_,w) in enumerate(titems): obj[off+ti]=w
        bounds=[(0,None)]*(nA+nt)
    opts=highs_options(time_limit); t=time.time()
    res=linprog(obj,A_ub=Aub,b_ub=None if Aub is None else np.array(ub),A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs',options=opts)
    return res,time.time()-t

def validate_boundary(PM,x,oracle,level,batch_cuts=4):
    start=len(oracle.cache[level]); failures=[]
    for h in PM.leaves:
        A=np.asarray([x[PM.av(h,j)] for j in range(PM.N)])
        ok,info=oracle.separate(level,A,path=h)
        if ok is None: return None,[info]
        if not ok: failures.append((h,info))
        if len(oracle.cache[level])-start >= int(batch_cuts):
            return False,failures
    return len(failures)==0,failures

def solve_case(n,M,max_master=200,time_limit=120,oracle_tol=2e-8,boundary_tol=1e-8,checkpoint=None,batch_cuts=4):
    PM=PrefixMaster(n,M); O=ViabilityOracle(M,oracle_tol,boundary_tol); load_oracle_cache(O,checkpoint)
    ledger=[]; base=None
    for it in range(1,max_master+1):
        r,sec=master_solve(PM,O.cache[n],mode='root',time_limit=time_limit)
        if not r.success: return {'status':'BASE_MASTER_FAILED','message':r.message,'ledger':ledger}
        ok,fail=validate_boundary(PM,r.x,O,n,batch_cuts)
        save_oracle_cache(O,checkpoint)
        ledger.append({'phase':'base','it':it,'value':float(r.fun),'sec':sec,'target_cuts':len(O.cache[n]),'ok':ok,'new_failures':len(fail)})
        print(f'[base {it:03d}] v={r.fun:.15g} ok={ok} E{n}cuts={len(O.cache[n])} sec={sec:.3f}',flush=True)
        if ok:
            base=r; break
    if base is None: return {'status':'BASE_MAX_ITER','ledger':ledger}
    v=float(base.fun)
    fill=None
    for it in range(1,max_master+1):
        r,sec=master_solve(PM,O.cache[n],mode='fill',root_cap=v+2e-9,time_limit=time_limit)
        if not r.success: return {'status':'FILL_MASTER_FAILED','message':r.message,'base_value':v,'ledger':ledger}
        ok,fail=validate_boundary(PM,r.x,O,n,batch_cuts)
        save_oracle_cache(O,checkpoint)
        ledger.append({'phase':'fill','it':it,'value':float(r.fun),'sec':sec,'target_cuts':len(O.cache[n]),'ok':ok,'new_failures':len(fail)})
        print(f'[fill {it:03d}] F={r.fun:.15g} ok={ok} E{n}cuts={len(O.cache[n])} sec={sec:.3f}',flush=True)
        if ok:
            fill=r; break
    if fill is None: return {'status':'FILL_MAX_ITER','base_value':v,'ledger':ledger}
    x=fill.x[:PM.nA]
    bydepth={}
    for d in range(1,n):
        val=0; mx=0
        for h in itertools.product(range(5),repeat=d):
            rv=[]
            for u in range(M+D+1):
                s=sum(v*x[c] for c,v in PM.zr_coeff(h,u)); rv.append(s)
            kr=sum(abs(z) for z in np.cumsum(rv)[:-1]); val+=hprob(h)*kr;mx=max(mx,kr)
        bydepth[str(d)]={'weighted_kr':float(val),'max_history_kr':float(mx)}
    # root residual diagnostic
    rv=[sum(v*x[c] for c,v in PM.zr_coeff((),u)) for u in range(M+D+1)]
    return {'status':'PASS','n':n,'M':M,'base_value':v,'fill_value':float(fill.fun),'by_depth':bydepth,
            'root_kr':float(sum(abs(z) for z in np.cumsum(rv)[:-1])),'root_residual_linf':float(max(abs(z) for z in rv)),
            'prefix_laws':len(PM.nodes),'A_vars':PM.nA,'E_target_cuts':len(O.cache[n]),'cut_counts':{str(k):len(v) for k,v in O.cache.items()},
            'boundary_accepts':O.boundary_accepts,'max_boundary_phi':O.max_boundary_phi,'oracle_stats':dict(O.stats),'ledger':ledger}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=2); ap.add_argument('--M',type=int,nargs='+',default=[8]); ap.add_argument('--max-master',type=int,default=200);ap.add_argument('--time-limit',type=float,default=120);ap.add_argument('--out'); ap.add_argument('--checkpoint'); ap.add_argument('--batch-cuts',type=int,default=4); a=ap.parse_args()
    print('DREAM6-ZR v0.24 PREFIX MASTER + BACKWARD VIABILITY CUTTING PLANE',flush=True)
    outs=[]
    for M in a.M:
        print('='*100); print(f'n={a.n} M={M}',flush=True); ck=(a.checkpoint.format(M=M,n=a.n) if a.checkpoint else None); o=solve_case(a.n,M,a.max_master,a.time_limit,checkpoint=ck,batch_cuts=a.batch_cuts); outs.append(o); print(json.dumps({k:v for k,v in o.items() if k!='ledger'},indent=2),flush=True)
    if a.out:
        with open(a.out,'w') as f: json.dump(outs,f,indent=2)
if __name__=='__main__': main()
