#!/usr/bin/env python3
"""
QZR parity-core lab: exact finite-horizon optimal trees for (E,O).

E = (1,0,6,0,1)/8 on {0,2,4},  O = (0,1,0,1,0)/2 on {1,3}.
Occupation LP: rho_h(j) = P(history h, reserve j), flows f_h(j,y,x) >= 0 with
sum_x f_h(j,y,x) = rho_h(j) O_y and x <= j+y; children rho_{hx}(j+y-x) += f.
Exactness (word law = iid E) is imposed via the word-density cap M = 1.

Modes
  bn N        exact B_n(E,O) for n = 1..N (root support <= 3n; LP, floating point)
  roots n     support of an optimal root and j*P(J>=j)
  trace n     how the optimal root pairs reserve levels with outputs; far mass by depth
  gamma L a   word-density value Gamma_L(delta_a)
"""
import numpy as np, itertools, sys, scipy.sparse as sp
from scipy.optimize import linprog
EX={0:1/8,2:6/8,4:1/8}; OY={1:0.5,3:0.5}; XS=(0,2,4)
def build(Lh, J0, B=None, M_fixed=None, extra_tail=None):
    """If B is None: B(j) j<=J0 are variables (sum 1). Minimise M (or mean if M_fixed)."""
    Jmax=J0+3*Lh
    levels=[[()]]
    for t in range(Lh): levels.append([h+(x,) for h in levels[-1] for x in XS])
    idx={}; n=0
    def var(k):
        nonlocal n
        if k not in idx: idx[k]=n; n+=1
        return idx[k]
    if B is None:
        for j in range(J0+1): var(("B",j))
    for t in range(Lh):
        for h in levels[t]:
            for j in range(Jmax+1):
                for y in (1,3):
                    for x in XS:
                        if x<=j+y and j+y-x<=Jmax: var(("f",h,j,y,x))
    for t in range(1,Lh+1):
        for h in levels[t]:
            for j in range(Jmax+1): var(("r",h,j))
    Mv=var(("M",))
    rows=[];cols=[];vals=[];beq=[];r=0
    def rho(h,j):
        if len(h)==0:
            if B is None: return ({idx[("B",j)]:1.0},0.0) if j<=J0 else ({},0.0)
            return {}, (B[j] if j<len(B) else 0.0)
        return {idx[("r",h,j)]:1.0},0.0
    for t in range(Lh):
        for h in levels[t]:
            for j in range(Jmax+1):
                for y in (1,3):
                    d,c=rho(h,j)
                    row={k:-OY[y]*v for k,v in d.items()}
                    for x in XS:
                        key=("f",h,j,y,x)
                        if key in idx: row[idx[key]]=row.get(idx[key],0)+1.0
                    if not row and abs(c)<1e-15: continue
                    for k,v in row.items(): rows.append(r);cols.append(k);vals.append(v)
                    beq.append(OY[y]*c); r+=1
    for t in range(1,Lh+1):
        for h in levels[t]:
            par,x=h[:-1],h[-1]
            for jn in range(Jmax+1):
                row={idx[("r",h,jn)]:-1.0}
                for y in (1,3):
                    j=jn-y+x; key=("f",par,j,y,x)
                    if 0<=j<=Jmax and key in idx: row[idx[key]]=row.get(idx[key],0)+1.0
                for k,v in row.items(): rows.append(r);cols.append(k);vals.append(v)
                beq.append(0.0); r+=1
    if B is None:
        for j in range(J0+1): rows.append(r);cols.append(idx[("B",j)]);vals.append(1.0)
        beq.append(1.0); r+=1
    Aeq=sp.csr_matrix((vals,(rows,cols)),shape=(r,n))
    ur=[];uc=[];uv=[];bub=[];u=0
    for h in levels[Lh]:
        p=np.prod([EX[x] for x in h])
        for j in range(Jmax+1): ur.append(u);uc.append(idx[("r",h,j)]);uv.append(1.0)
        ur.append(u);uc.append(Mv);uv.append(-p); bub.append(0.0); u+=1
    Aub=sp.csr_matrix((uv,(ur,uc)),shape=(u,n))
    c=np.zeros(n)
    bounds=[(0,None)]*n
    if M_fixed is not None:
        bounds[Mv]=(M_fixed,M_fixed)
        for j in range(J0+1): c[idx[("B",j)]]=j
    else: c[Mv]=1
    res=linprog(c,A_ub=Aub,b_ub=np.array(bub),A_eq=Aeq,b_eq=np.array(beq),bounds=bounds,method="highs")
    if res.status!=0: return None,None
    Bout=None
    if B is None: Bout=np.array([res.x[idx[("B",j)]] for j in range(J0+1)])
    build.last=(res,idx,levels,Jmax)
    return res.fun,Bout
if __name__=="__main__" and len(sys.argv)>1 and sys.argv[1].isdigit():
    for Lh in range(1,int(sys.argv[1])+1):
        J0=3*Lh
        mean,Bopt=build(Lh,J0,B=None,M_fixed=1.0)
        print(f"K_{Lh}: min mean {mean:.6f}  B_opt={np.round(Bopt[:8],4)}",flush=True)
        g,_=build(2*Lh,len(Bopt)-1,B=Bopt) if 2*Lh<=6 else (None,None)
        if g: print(f"    Gamma_{2*Lh}(B_opt_{Lh}) = {g:.6f}   (Gamma-1)*{2*Lh} = {(g-1)*2*Lh:.4f}",flush=True)

def _trace(n):
    mean,B=build(n,3*n,B=None,M_fixed=1.0)
    res,idx,levels,Jmax=build.last; x=res.x
    print(f"n={n} B_n={mean:.6f}")
    for j in range(Jmax+1):
        row=[f"(y={y},x={xx}):{x[idx[('f',(),j,y,xx)]]:.4f}" for y in (1,3) for xx in (0,2,4)
             if ("f",(),j,y,xx) in idx and x[idx[("f",(),j,y,xx)]]>1e-7]
        if row: print(f"  j={j}: "+", ".join(row))
    for t in range(1,n+1):
        far=sum(x[idx[("r",h,j)]] for h in levels[t] for j in range(5,Jmax+1))
        print(f"  depth {t}: mass at reserve>=5: {far:.5f}")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("bn","roots","trace","gamma"):
    from fractions import Fraction
    mode = sys.argv[1]
    if mode == "bn":
        for n in range(1, int(sys.argv[2]) + 1):
            mean, B = build(n, 3*n, B=None, M_fixed=1.0)
            print(f"B_{n}(E,O) = {mean:.8f} ~ {Fraction(mean).limit_denominator(8192)}", flush=True)
    elif mode == "roots":
        n = int(sys.argv[2]); mean, B = build(n, 3*n, B=None, M_fixed=1.0)
        tail = np.cumsum(B[::-1])[::-1]
        print(f"n={n} B_n={mean:.6f} support:", [(j, round(v, 5)) for j, v in enumerate(B) if v > 1e-6])
        print("  j*P(J>=j):", [(j, round(j*tail[j], 3)) for j in range(1, len(B)) if tail[j] > 1e-7])
    elif mode == "trace":
        _trace(int(sys.argv[2]))
    else:
        Lh, a = int(sys.argv[2]), int(sys.argv[3]); B = np.zeros(a+1); B[a] = 1.0
        g, _ = build(Lh, a, B=B); print(f"Gamma_{Lh}(delta_{a}) = {g:.6f}")
