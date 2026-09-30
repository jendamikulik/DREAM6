#!/usr/bin/env python3
"""Exact algebra certificate and independent finite audit.

Run with Python 3.10+: python verify_degree_free.py
Standard library only. No floating-point inequality is used.

The polynomial checks are identities in Q[d,k,m,x,y,z], not samples.
The finite graph audit checks the probability identities independently by
enumeration; it is not an enumeration of all forests. The companion proof
supplies the probabilistic arguments and their universal quantifiers.
"""
from fractions import Fraction as F
from collections import deque
from pathlib import Path
import hashlib
import json

NAMES = ('d', 'k', 'm', 'x', 'y', 'z')
ZERO = (0,) * len(NAMES)
checks = []


class P:
    """Small exact multivariate polynomial ring, stored as sparse dicts."""
    def __init__(self, value=0):
        if isinstance(value, P):
            self.a = value.a.copy()
        elif isinstance(value, dict):
            self.a = {e: F(c) for e, c in value.items() if c}
        else:
            self.a = {ZERO: F(value)} if value else {}

    def __add__(self, other):
        out = self.a.copy()
        for e, c in P(other).a.items():
            out[e] = out.get(e, F(0)) + c
        return P(out)

    __radd__ = __add__

    def __neg__(self):
        return P({e: -c for e, c in self.a.items()})

    def __sub__(self, other):
        return self + (-P(other))

    def __rsub__(self, other):
        return P(other) - self

    def __mul__(self, other):
        out = {}
        for a, x in self.a.items():
            for b, y in P(other).a.items():
                e = tuple(u + v for u, v in zip(a, b))
                out[e] = out.get(e, F(0)) + x*y
        return P(out)

    __rmul__ = __mul__

    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        out, base = P(1), self
        while n:
            if n & 1:
                out = out*base
            base = base*base
            n //= 2
        return out

    def sub(self, name, value):
        index = NAMES.index(name)
        out = P()
        for e, c in self.a.items():
            remainder = list(e)
            remainder[index] = 0
            out += P({tuple(remainder): c}) * P(value)**e[index]
        return out

    def __eq__(self, other):
        return self.a == P(other).a


def variable(name):
    e = list(ZERO)
    e[NAMES.index(name)] = 1
    return P({tuple(e): 1})


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


d, k, m, x, y, z = map(variable, NAMES)
C = k*(k+1)*(d-k)*(d-k+1)
w12 = 12*k*k - 12*d*k + 2*d*(d-1)
check('quartic_dual_interior_identity', C.sub('k', k-1)-2*C+C.sub('k', k+1) == w12)
check('quartic_dual_left_boundary', C.sub('k', 1) == w12.sub('k', 0))
check('quartic_dual_right_boundary', C.sub('k', d-1) == w12.sub('k', d))
check('quartic_dual_zero_boundaries', C.sub('k', 0) == 0 and C.sub('k', d) == 0)
check('logconvex_implies_convex_identity', (x+z)**2-4*y*y == (x-z)**2+4*(x*z-y*y))
check('outside_variance_margin', (d+2)**2*F(1,4)-d*(d+2)*F(1,12) == (d+2)*(d+3)*F(1,6))

# P=1+S+D; adjoining x gives new excess D(1+x)+xS.
check('product_minus_sum_induction',(1+x)*(1+y+z)-1-(y+x)==z*(1+x)+x*y)
positive_polynomials={
 'K_m_induction':(16*m*m-(m+1)**2).sub('m',x+1),
 'm_le_two_power_induction':(2*m-(m+1)).sub('m',x+1),
 'm_plus_one_le_two_power_induction':(2*(m+1)-(m+2)).sub('m',x+1),
 'prefactor_induction':(16*(5+3*m)-(5+3*(m+1))).sub('m',x+1)}
for name,p in positive_polynomials.items():
 check(name,bool(p.a) and all(c>=0 for c in p.a.values()))
check('dyadic_base',12<=16 and 1<=2 and 2<=2 and 5+3<=2**7)
check('threshold_prefactor',24<2**5)
check('variance_contradiction',F(1,48)+F(1,48)==F(1,24)<F(1,12))
check('loglog_floor_margin',((x-8)*F(1,4)-x*F(1,8)).sub('x',y+16)==y*F(1,8))
check('starting_threshold',2**16==65536)
symbolic_count=len(checks)
thresholds=[]
for mm in (1,2,4,8,16,32,64):
 lam=4**mm;bb=2**(mm+1);kk=(12*mm*mm-1).bit_length();rr=(lam+1)*kk
 exponent=(mm+1)*rr
 check(f'K_bound_{mm}',2**kk>=12*mm*mm and kk<=4*mm)
 check(f'binomial_witness_{mm}',1+F(lam+1,lam)>2)
 check(f'R_bound_{mm}',rr<=2**(3*mm+3))
 check(f'exponent_envelope_{mm}',exponent+5+3*mm<=2**(4*mm+4))
 thresholds.append({'m':mm,'Lambda':str(lam),'B':str(bb),'K':kk,'R':str(rr),
 'N_threshold':f'{24*mm**3} * 2^{exponent}','conclusion':'M_all_forests(N) < N/m'})

def paths(n,edges,root):
 adj=[[] for _ in range(n)]
 for u,v in edges:adj[u].append(v);adj[v].append(u)
 parent,dist=[None]*n,[None]*n;dist[root]=0;queue=deque([root])
 while queue:
  u=queue.popleft()
  for v in adj[u]:
   if dist[v] is None:dist[v]=dist[u]+1;parent[v]=u;queue.append(v)
 return parent,dist

graphs=[
 ('path8',8,[(j,j+1) for j in range(7)]),
 ('star10',10,[(0,j) for j in range(1,10)]),
 ('binary7',7,[((j-1)//2,j) for j in range(1,7)]),
 ('star_of_stars13',13,[(0,j) for j in range(1,4)]+[(j,4+3*(j-1)+a) for j in range(1,4) for a in range(3)]),
 ('forest8',8,[(0,1),(1,2),(3,4),(3,5)])]
audit_pairs=0;cases=0
for name,n,edges in graphs:
 masks=[mask for mask in range(1<<n) if all(not(mask>>u&1 and mask>>v&1) for u,v in edges)]
 adj=[[] for _ in range(n)]
 for u,v in edges:adj[u].append(v);adj[v].append(u)
 vectors=[[la]*n for la in (F(1,7),F(1),F(7),F(64))]
 vectors.append([F((j%5)+1,(j%3)+1) for j in range(n)])
 for case,activity in enumerate(vectors):
  cases+=1;label=f'{name}_{case}'
  ws=[]
  for mask in masks:
   w=F(1)
   for v in range(n):
    if mask>>v&1:w*=activity[v]
   ws.append(w)
  Z=sum(ws);ps=[w/Z for w in ws]
  mus=[sum(p for mask,p in zip(masks,ps) if mask>>v&1) for v in range(n)]
  variances=[a*(1-a) for a in mus];odds=[a/(1-a) for a in mus]
  covariance=[]
  for u in range(n):
   covariance.append([sum(p for mask,p in zip(masks,ps) if mask>>u&1 and mask>>v&1)-mus[u]*mus[v] for v in range(n)])
  Lambda=max(F(1),max(activity));q=Lambda/(1+Lambda)
  # Integer B>=2 sqrt(Lambda), B>=2, using exact integer comparisons.
  B=2
  while B*B<4*Lambda:B+=1
  for v in range(n):
   budget=sum(odds[v]*odds[u] for u in adj[v])
   check(f'local_squared_correlation_budget_{label}_{v}',budget<=activity[v])
   product=F(1);sum_messages=F(0)
   for u in adj[v]:
    # q_(u->v)=mu_u/(1-mu_v), R=q/(1-q).
    cavity_q=mus[u]/(1-mus[v]);message=cavity_q/(1-cavity_q)
    product*=1+message;sum_messages+=message
    check(f'edge_correlation_identity_{label}_{v}_{u}',covariance[u][v]**2==variances[u]*variances[v]*odds[u]*odds[v])
    check(f'edge_correlation_bound_{label}_{v}_{u}',odds[u]*odds[v]<=q*q)
   check(f'root_message_identity_{label}_{v}',odds[v]*product==activity[v])
   check(f'local_budget_chain_{label}_{v}',budget<=activity[v]*sum_messages/product<=activity[v])
  for u in range(n):
   par,dist=paths(n,edges,u)
   for v in range(n):
    actual=covariance[u][v]
    if dist[v] is None:check(f'disconnected_{label}_{u}_{v}',actual==0)
    else:
     factor=F(1);w=v
     while w!=u:
      a=par[w];factor*=odds[w]*odds[a];w=a
     check(f'path_product_squared_{label}_{u}_{v}',actual**2==variances[u]*variances[v]*factor)
     check(f'path_sign_{label}_{u}_{v}',actual*(-1)**dist[v]>0)
     check(f'path_bound_{label}_{u}_{v}',abs(actual)<=F(1,4)*q**dist[v])
    audit_pairs+=1
  # Rational similarity of weighted adjacency avoids square roots.
  vec=[F(1)]*n
  for d in range(5):
   walk_weight=sum(variances[u]*vec[u] for u in range(n))
   shell=sum(abs(covariance[u][v]) for u in range(n) for v in range(n) if paths(n,edges,u)[1][v]==d)
   check(f'walk_majorizes_shell_{label}_{d}',shell<=walk_weight)
   check(f'walk_operator_bound_{label}_{d}',walk_weight<=F(n,4)*B**d)
   vec=[sum(mus[v]/(1-mus[u])*vec[v] for v in adj[u]) for u in range(n)]
  var=sum(sum(row) for row in covariance)
  for R in range(5):
   check(f'degree_free_variance_{label}_{R}',var<=F(n,2)*B**R+F(n*n,4)*q**R)
report={'status':'PASS','arithmetic':'Exact rational arithmetic and polynomial identities; no external dependency',
 'universal_symbolic_checks':symbolic_count,'finite_graph_activity_cases':cases,
 'finite_ordered_covariance_pairs':audit_pairs,'checks_passed':len(checks),
 'symbolic_checks':checks[:symbolic_count],'thresholds':thresholds,
 'local_budget':'sum_{u~v} Corr(S_u,S_v)^2 <= lambda_v',
 'variance_bound':'Var X <= (2 n B^R + n^2 q^R)/4; B=2 sqrt(Lambda), q=Lambda/(1+Lambda), Lambda>=max(1,lambda_v)',
 'run_bound':'M_all_forests(N) < 8 N / log_2(log_2 N) for N >= 2^65536',
 'degree_restriction':None,
 'scope':'Universal operator and probability arguments are written in degree_free_ceiling.tex. Finite enumeration is an independent sanity audit, not a proof for all forests. No proof-assistant certification or established publication priority is claimed.',
 'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(__file__).with_name('degree_free_certificate.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS: {symbolic_count} universal algebra checks; {cases} graph/activity cases; {audit_pairs} covariance pairs; {len(checks)} total checks.')
