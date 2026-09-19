"""Exact positive head/tail exchange and control-invariant accounting checks.
No infinite-horizon existence claim. Standard library only.
"""
from fractions import Fraction as F
from pathlib import Path
import runpy

z=runpy.run_path(str(Path(__file__).with_name('orbital_forward_repair_2026-09-19.py')))
P,Q=z['P'],z['Q']
qs=z['convolutions'](3)
ts=[z['tail_mass'](d,q) for d,q in enumerate(qs)]


def harmonic(n):
    return sum((F(1,k) for k in range(1,n+1)),F(0))


def fan(d,s,A,kappa):
    mu=z['convolve'](A,Q)
    end=8*(d+1)-s
    mu.extend([F(0)]*(end+1-len(mu)))
    for u,v in z['boundary'](d,s,qs[d]).items():mu[u]+=kappa*v
    rows=[p*(1-kappa*ts[d+1]) for p in P]
    pi=z['quantile'](mu,rows)
    z['verify_fan'](mu,pi,rows)
    kids=z['children'](pi)
    return mu,pi,kids


def main():
    mu,pi,kids=fan(0,0,[F(0)],F(1))
    kap=[F(1)]*5;eps=F(1,128);t=ts[1]
    assert eps<=pi[1][2] and eps<=P[0]*kap[0]*t
    old_h=[z['H'](a) for a in kids]
    pi[1][2]-=eps;pi[0][2]+=eps
    kap[1]+=eps/(P[1]*t);kap[0]-=eps/(P[0]*t)
    kids=z['children'](pi)
    assert kap==[F(7807,10327),F(11587,10327),F(1),F(1),F(1)]
    assert sum(p*k for p,k in zip(P,kap))==1
    assert z['H'](kids[0])==old_h[0]
    assert z['H'](kids[1])==old_h[1]-F(7,32)
    # Full coefficient identity: finite part explicitly, infinite part by
    # the common tail profile and its exactly conserved amplitude.
    for u,v in enumerate(mu):assert sum(row[u] for row in pi)==v
    for x,A in enumerate(kids):
        assert min(A)>=0 and kap[x]>=0
        assert sum(A)+kap[x]*t==1
    states=[(str(x),x,P[x],kids[x],kap[x]) for x in range(5)]
    for d in (1,2,3):
        capital=sum(p*k for h,s,p,A,k in states)
        S=sum(p*k*s for h,s,p,A,k in states)
        M=sum(p*sum(j*v for j,v in enumerate(A)) for h,s,p,A,k in states)
        V=2*sum(q*harmonic(8*d+2-y) for y,q in enumerate(qs[d]))
        assert capital==1 and 0<=S<=4*d
        assert M-ts[d]*S==V-(8*d+1)*ts[d]-2
        assert 2*harmonic(4*d+2)-6<=M<=2*harmonic(8*d+2)-2
        print('depth',d,'nodes',len(states),'exact account verified')
        if d==3:break
        nxt=[]
        for h,s,p,A,k in states:
            mu,pi,children=fan(d,s,A,k)
            for x,B in enumerate(children):nxt.append((h+str(x),s+x,p*P[x],B,k))
        states=nxt
    print('Exact head/tail exchange, positivity, ZR and martingale identities passed.')
    print('The finite checks do not prove an infinite policy.')


if __name__=='__main__':main()
