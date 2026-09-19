"""Exact constructive fan repairs. Standard library only.

Theorems and scope: orbital_forward_repair_2026-09-19.tex.
This does NOT certify an infinite policy.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import runpy

support = runpy.run_path(str(Path(__file__).with_name('verify_shifted_tail_orbit_2026-09-19.py')))
P, Q = support['P'], support['Q']
convolutions, boundary, tail_mass = (support[k] for k in ('convolutions','boundary','tail_mass'))


def convolve(a, b):
    out = [F(0)] * (len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def source(d, s, A, qs):
    out = convolve(A, Q)
    end = 8*(d+1)-s
    out.extend([F(0)]*(end+1-len(out)))
    for j, v in boundary(d, s, qs[d]).items():
        out[j] += v
    return out


def quantile(mu, rows):
    out = [[F(0)]*len(mu) for _ in rows]
    remaining = list(rows)
    x = 0
    for u, supply in enumerate(mu):
        while supply:
            while x<len(rows) and not remaining[x]: x+=1
            if x == len(rows) or x>u:
                return None
            moved = min(supply, remaining[x])
            out[x][u] += moved
            remaining[x] -= moved
            supply -= moved
    return out if all(v==0 for v in remaining) else None


def forward_repair(mu, rows, budgets):
    """Complete decision/construction for 27*pi[x,x]+7*pi[x,x+1]<=budget[x].

    Proof: each processed row has the pointwise largest possible CDF;
    the remaining source stochastically improves, preserving feasibility.
    """
    size=max(len(mu),len(rows)+1)
    rest=list(mu)+[F(0)]*(size-len(mu))
    out=[[F(0)]*size for _ in rows]
    for x, (mass, budget) in enumerate(zip(rows,budgets)):
        if any(rest[:x]): return None
        diag=rest[x]
        if diag>mass or 27*diag>budget: return None
        out[x][x]=diag;rest[x]=F(0)
        rem=mass-diag
        adj=min(rest[x+1],rem,(budget-27*diag)/7)
        out[x][x+1]=adj;rest[x+1]-=adj;rem-=adj
        for u in range(x+2,size):
            moved=min(rest[u],rem)
            out[x][u]+=moved;rest[u]-=moved;rem-=moved
            if not rem:break
        if rem:return None
    if any(rest):return None
    return out


def verify_fan(mu, pi, rows, budgets=None):
    assert pi is not None
    for x,row in enumerate(pi):
        assert min(row)>=0 and not any(row[:x])
        assert sum(row)==rows[x]
        if budgets is not None:
            assert 27*row[x]+7*row[x+1]<=budgets[x]
    for u,v in enumerate(mu):
        assert sum(row[u] for row in pi)==v
    assert all(v==0 for row in pi for v in row[len(mu):])


def children(pi):
    return [[v/P[x] for v in row[x:]] for x,row in enumerate(pi)]


def H(A):
    return 27*A[0]+7*A[1]


def two_step(d,s,A,qs,ts):
    assert d>=1
    beta,gamma=1-ts[d+1],1-ts[d+2]
    threshold=8*beta+F(128,9)*gamma
    if H(A)>threshold:return None
    mu=source(d,s,A,qs)
    rows=[p*beta for p in P]
    pi=quantile(mu,rows)
    assert pi is not None
    q1=(27*pi[1][1]+7*pi[1][2])/P[1]
    delta=max(F(0),(q1-24*gamma)/7)
    eps=delta/4
    pi[1][2]-=eps;pi[1][3]+=eps
    pi[2][2]+=eps;pi[2][3]-=eps
    verify_fan(mu,pi,rows,[24*gamma*p for p in P])
    return pi,delta


def main():
    qs=convolutions(26)
    ts=[tail_mass(d,q) for d,q in enumerate(qs)]
    demo=convolve([F(723,1000),F(0),F(277,1000)],Q)
    demo_budget=F(216,11)
    demo_pi=forward_repair(demo,P,[p*demo_budget for p in P])
    verify_fan(demo,demo_pi,P,[p*demo_budget for p in P])
    demo_base=quantile(demo,P)
    nonzero={}
    for x in range(4):
        for u in range(len(demo)-1):
            theta=sum(demo_base[i][v]-demo_pi[i][v]
                      for i in range(x+1) for v in range(u+1))
            if theta:nonzero[x,u]=theta
    assert nonzero=={(0,1):F(71,246400),(1,2):F(110137,1724800)}
    count=0
    for d in (1,2,4,8):
        alpha=1-ts[d]
        beta,gamma=1-ts[d+1],1-ts[d+2]
        assert beta<=gamma<=F(9,8)*beta
        for s in (0,2*d,4*d):
            for b0,b1 in ((F(0),F(0)),(F(1,2),F(1,2)),(F(3,4),F(1,4)),
                          (F(4,5),F(0)),(F(0),F(1))):
                A=[F(0)]*(8*d-s+1)
                A[0]=alpha*b0;A[1]=alpha*b1;A[2]=alpha*(1-b0-b1)
                result=two_step(d,s,A,qs,ts)
                if result is not None:
                    mu=source(d,s,A,qs)
                    rows=[p*beta for p in P]
                    budgets=[24*gamma*p for p in P]
                    greedy=forward_repair(mu,rows,budgets)
                    verify_fan(mu,greedy,rows,budgets)
                    count+=1
    # Reconstruct the user's specific prefix exactly, using quantile fans.
    word='0122221111111120224141'
    A=[F(0)];s=0
    for d,ch in enumerate(word):
        mu=source(d,s,A,qs)
        rows=[p*(1-ts[d+1]) for p in P]
        pi=quantile(mu,rows)
        verify_fan(mu,pi,rows)
        A=children(pi)[int(ch)];s+=int(ch)
    d=len(word)
    # An exact three-step continuation: each first child is in its exact K_2.
    mu=source(d,s,A,qs)
    beta=1-ts[d+1]
    child_bound=8*(1-ts[d+2])+F(128,9)*(1-ts[d+3])
    rows=[p*beta for p in P]
    pi=forward_repair(mu,rows,[p*child_bound for p in P])
    verify_fan(mu,pi,rows,[p*child_bound for p in P])
    first_children=children(pi)
    for x,B in enumerate(first_children):
        result=two_step(d+1,s+x,B,qs,ts)
        assert result is not None
        for z,C in enumerate(children(result[0])):
            level=d+2
            mu2=source(level,s+x+z,C,qs)
            rows2=[p*(1-ts[level+1]) for p in P]
            pi2=quantile(mu2,rows2)
            verify_fan(mu2,pi2,rows2)
    artifact={
        'scope':'Exact three-step continuation at the stated prefix; no all-horizon claim.',
        'prefix':word,'depth':d,'sum':s,
        'head':[str(v) for v in A],
        'first_transport':[[str(v) for v in row] for row in pi],
        'child_two_step_bound':str(child_bound),
        'child_H':[str(H(B)) for B in first_children],
    }
    path=Path(__file__).with_name('orbital_forward_repair_certificate_2026-09-19.json')
    path.write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    print('Exact universal-formula examples passed:',count)
    print('Exact three-step fan at prefix',word,'verified through every node.')
    print('Certificate:',path)
    print('No infinite-policy claim.')


if __name__=='__main__':
    main()
