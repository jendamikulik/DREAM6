#!/usr/bin/env python3
"""Exact arithmetic audit accompanying global_smoothing.pdf.
Finite tests supplement the universal written proof; no formal QED claim.
Python standard library only. Run with Python 3.10+.
"""
from collections import Counter
from fractions import Fraction as F
from math import comb, isqrt
from pathlib import Path
import hashlib, json, random

checks=Counter()
def check(p, group):
    checks[group]+=1
    if not p: raise AssertionError(group)

# Sparse polynomial arithmetic over Z[r,s,m]: exact universal identity check.
def const(c): return {} if c==0 else {(0,0,0):c}
def variable(i):
    e=[0,0,0]; e[i]=1
    return {tuple(e):1}
def add(*ps):
    out=Counter()
    for p in ps:
        for e,c in p.items(): out[e]+=c
    return {e:c for e,c in out.items() if c}
def neg(p): return {e:-c for e,c in p.items()}
def sub(p,q): return add(p,neg(q))
def mul(*ps):
    out=const(1)
    for p in ps:
        res=Counter()
        for e,a in out.items():
            for f,b in p.items(): res[tuple(x+y for x,y in zip(e,f))]+=a*b
        out={e:c for e,c in res.items() if c}
    return out
r,s,m=[variable(i) for i in range(3)]; one=const(1)
r1,s1=add(r,one),add(s,one)
mr,ms=sub(m,r),sub(m,s)
mr1,ms1=add(mr,one),add(ms,one)
lhs=sub(mul(const(2),mr1,s1,ms1,r1),
        add(mul(r,ms,ms1,r1),mul(s,mr,mr1,s1)))
rhs=mul(add(m,one),sub(add(mul(r1,mr1),mul(s1,ms1)),
                         mul(add(m,one),sub(r,s),sub(r,s))))
check(lhs==rhs,'symbolic_pairwise_identity')

# All allowed pairs in several boundary-tight interiors, plus larger samples.
for n in range(1,11):
    L=n*n
    for m in [2*L,2*L+n,4*L+3]:
        for r in range(L,m-L+1):
            for s in range(max(L,r-n),min(m-L,r+n)+1):
                num=(r+1)*(m-r+1)+(s+1)*(m-s+1)-(m+1)*(r-s)**2
                check(num>0,'interior_pairwise_sign')

# Verify baseline k! b_k curvature with arbitrary nonnegative linear weights.
for m in range(2,41):
    B=[comb(m,j) for j in range(m+1)]
    for a in [1,2,11]:
        for b in [0,1,17,10**8]:
            u=[a*B[j]+b*(B[j-1] if j else 0) for j in range(m+1)]
            for k in range(1,m):
                check(k*u[k]**2 >= (k+1)*u[k-1]*u[k+1], 'baseline_curvature')

# Constants at the exact integer ceiling; no square-root rounding.
def ceil_sqrt(v):
    r=isqrt(v)
    return r+(r*r<v)
def sufficient_m(n,M): return ceil_sqrt(16*(n+1)**8*M)
for n in range(1,81):
    K=n*(n+1)
    for M in [1,2,3,17,10**6,10**30+7]:
        m=sufficient_m(n,M)
        z=F((n+1)*(K+1),m-K)
        check(z<=F(1,2),'endpoint_constants')
        check(2*M*z*z<=F(1,4*K),'endpoint_constants')
        check((1+F(1,4*K))**2<1+F(1,K),'endpoint_constants')
        check(m>2*K+2,'endpoint_constants')

# Exact convolution and full-sequence verification.
def mulpoly(a,b):
    c=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b): c[i+j]+=x*y
    return c
def binomial_row(m):
    row=[1]
    for j in range(m): row.append(row[-1]*(m-j)//(j+1))
    return row
def smooth(a,m): return mulpoly(a,binomial_row(m))
def is_lc(a):
    return all(x>0 for x in a) and all(a[k]**2>=a[k-1]*a[k+1] for k in range(1,len(a)-1))
def threshold(a):
    if is_lc(a): return 0
    lo,hi=0,1
    while not is_lc(smooth(a,hi)): lo,hi=hi,2*hi
    while hi-lo>1:
        mid=(lo+hi)//2
        if is_lc(smooth(a,mid)): hi=mid
        else: lo=mid
    return hi

def family_data(family,n):
    maxima=[s for s in family if not any(s!=t and s&t==s for t in family)]
    d=max(s.bit_count() for s in family)
    a=[sum(s.bit_count()==j for s in family) for j in range(d+1)]
    return a,len(maxima)
family_count=0
for n in range(1,4):
    for mask in range(1<<(2**n-1)):
        family=[0]+[s for s in range(1,2**n) if mask>>(s-1)&1]
        a,M=family_data(family,n);d=len(a)-1
        for i in range(d+1):
            check(a[i]<=comb(n,i),'family_counts')
            check(a[d-i]<=M*sum(comb(n,j) for j in range(i+1)), 'family_counts')
        m=sufficient_m(n,M)
        c=smooth(a,m)
        for k in range(1,len(c)-1):
            check(c[k]**2>c[k-1]*c[k+1], 'full_family_smoothing')
        family_count+=1

# Non-finite-size theorem is in the paper; these stress both endpoint arguments.
rng=random.Random(20261001)
for n in range(2,13):
    K=n*(n+1)
    for M in [1,3,10**8+7]:
        a=[1]+[rng.randrange(M*(n+1)**min(i,n-i)+1) for i in range(1,n)]+[1]
        m=sufficient_m(n,M)
        row=[comb(m,j) for j in range(K+2)]
        for p in [a,a[::-1]]:
            c=[sum(p[i]*row[j-i] for i in range(min(j,n)+1)) for j in range(K+2)]
            b=[row[j]+(p[1]*row[j-1] if j else 0) for j in range(K+2)]
            for j in range(K+2):
                check(0<=4*K*(c[j]-b[j])<=b[j],'endpoint_relative_error')
            for k in range(1,K+1):
                check(c[k]**2>c[k-1]*c[k+1],'endpoint_log_concavity')

# Independent graph enumeration, not the closed polynomial formula.
def witness_graph(q,t):
    n=3*q*t+4;adj=[0]*n
    def edge(i,j): adj[i]|=1<<j;adj[j]|=1<<i
    for h in range(3):
        edge(0,h+1)
        for j in range(t):
            vs=[4+(h*t+j)*q+a for a in range(q)]
            for i in range(q):
                for k in range(i): edge(vs[i],vs[k])
                if i<q-1: edge(h+1,vs[i])
    return adj

def enumerate_independence(adj):
    full=(1<<len(adj))-1; counts=Counter();mis=0
    def rec(avail,chosen,dominated):
        nonlocal mis
        if not avail:
            counts[chosen.bit_count()]+=1
            if chosen|dominated==full: mis+=1
            return
        bit=avail&-avail;v=bit.bit_length()-1
        rec(avail^bit,chosen,dominated)
        rec((avail^bit)&~adj[v],chosen|bit,dominated|adj[v])
    rec(full,0,0)
    return [counts[i] for i in range(max(counts)+1)],mis

def witness_poly(q,t):
    base=[comb(t,j)*q**j for j in range(t+1)]+[0]
    for j in range(t+1):base[j+1]+=comb(t,j)
    p=mulpoly(mulpoly(base,base),base)
    for j in range(3*t+1):p[j+1]+=comb(3*t,j)*q**j
    return p
for q in [2,3]:
    for t in [1,2]:
        p,M=enumerate_independence(witness_graph(q,t))
        x=q**t
        check(p==witness_poly(q,t),'graph_enumeration')
        check(M==x**3+3*x*x-3*x+1,'graph_enumeration')

examples=[]
for q,t in [(2,4),(2,5),(2,6),(2,7),(2,8),(3,3),(3,4)]:
    p=witness_poly(q,t)
    g=threshold(p)
    A,B=p[-2],p[-3]
    lo,hi=-1,max(g,1)
    while hi-lo>1:
        m=(lo+hi)//2
        if A*A-B+A*m+m*(m+1)//2>=0:hi=m
        else:lo=m
    term=hi
    check(g>=term,'global_threshold_examples')
    check(is_lc(smooth(p,g)),'global_threshold_examples')
    if g: check(not is_lc(smooth(p,g-1)),'global_threshold_examples')
    examples.append({'q':q,'t':t,'vertices':3*q*t+4,'terminal':term,'global':g})

complexes=[]
for n in [8,10,12,14,16,18,20]:
    r=n//2;d=r+2;B=comb(n,r)
    a=[comb(n,j) for j in range(r+1)]+[d,1]
    g=threshold(a)
    if n>=16: check(4*g*g>B,'complex_lower_bound')
    complexes.append({'N':n,'central_binomial':B,'global_threshold':g})

report={'status':'PASS_EXACT_ARITHMETIC_AUDIT','formal_proof_assistant_certificate':False,
        'proof_of_universal_theorem':'global_smoothing.pdf / global_smoothing.tex',
        'checks':dict(checks),'total_checks':sum(checks.values()),
        'exhaustive_set_families':family_count,'graph_examples':examples,
        'complex_examples':complexes,
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(__file__).with_name('global_smoothing_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
