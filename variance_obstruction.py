#!/usr/bin/env python3
"""A variance obstruction to deducing sparse log-concavity failures.

This is NOT a construction of forests or simplicial complexes, and NOT a
claim of publication priority. It is an explicit set-family counterexample
to an inference that uses only uniform linear tilted variance and short
runs of failures. The universal argument is below; the executable part
checks finite instances using integer arithmetic only.

THEOREM. For every even m >= 16, put N=m^3. There is a family F subset
2^[N], containing both the empty and full sets, whose rank counts a_k have:
  (1) at least N-9m^2 strict log-concavity failures;
  (2) no consecutive failure run longer than m-1;
  (3) Var_lambda(|S|) <= 3N p(1-p) <= 3N/4 for EVERY lambda>0,
      where p=lambda/(1+lambda) and Pr_lambda(S) is proportional to
      lambda^|S| over S in F.
In particular the fraction of failing indices tends to one, even though
all failure runs have length O(N^(1/3)) and the variance is uniformly O(N).

CONSTRUCTION. Write B_k=binom(N,k), d_k=min(k mod m,m-(k mod m)), and
f_k=m^2+d_k^2. Put a_0=a_N=1 and, for 1<=k<N,
    a_k=floor(B_k*f_k/(2m^2)).
Take any a_k distinct k-subsets of [N] at every rank (for example the
first a_k in lexicographic order). Since 1<=a_k<=B_k, this defines F.
It contains [N], so its only inclusion-maximal member is [N]. It is not
a downset. In particular it is not the independence family of a graph.

PROOF OF (3). For every k, 1/3<=a_k/B_k<=1: interior B_k>=N>=6, so
floor(B_k*f_k/(2m^2))>=B_k/2-1>=B_k/3; endpoints have ratio one.
The tilted rank law is the Bin(N,p) law reweighted by a function w in
[1/3,1]. Its normalizing expectation is >=1/3. Thus
    Var(X)<=E[(X-Np)^2]<=3 Np(1-p).
This is uniform over all positive lambda, including lambda depending on N.

PROOF OF (1). Set b_k=B_k*f_k/(2m^2). At every non-peak residue
k mod m != m/2, the values f_(k-1),f_k,f_(k+1) are m^2+(r-1)^2,
m^2+r^2,m^2+(r+1)^2 for some integer |r|<=m/2-1. Consequently
    f_(k-1)*f_(k+1)-f_k^2 = 2m^2-2r^2+1 >= 3m^2/2,
and f_k<=5m^2/4, giving
    f_(k-1)*f_(k+1)/f_k^2 >= 1+24/(25m^2).
For 4m^2<=k<=N-4m^2, m>=16,
    B_k^2/(B_(k-1)*B_(k+1))
      =1+(N+1)/(k(N-k)) <= 1+1/(2m^2).
Indeed k(N-k)>=4m^2(N-4m^2)>=3m^2 N, and N+1<=3N/2.
It follows that
    b_(k-1)*b_(k+1) >= (1+1/(4m^2))*b_k^2.
Here all three ranks lie between 2 and N-2, so b_j>=binom(N,2)/2
=N(N-1)/4>=16m^2. Put x=1/(16m^2). Rounding gives
    a_(k-1)*a_(k+1)
      >= (1-x)^2*b_(k-1)*b_(k+1)
      >= (1-x)^2*(1+4x)*b_k^2 > b_k^2 >= a_k^2,
since (1-x)^2(1+4x)-1=x(2-7x+4x^2)>0.
The interval contains N-8m^2+1 indices, of which exactly m^2-8m are
peak residues. Thus it certifies N-9m^2+8m+1 failures, proving (1).

PROOF OF (2). At a peak k mod m=m/2, f_k=5m^2/4=:f and
f_(k-1)=f_(k+1)=f-(m-1). Binomial log-concavity implies
    sqrt(b_(k-1)*b_(k+1)) <= B_k*(f-(m-1))/(2m^2).
Since B_k>=N=m^3, we have B_k*(m-1)/(2m^2)>=1. Therefore
    a_k >= b_k-1 >= sqrt(b_(k-1)*b_(k+1))
                     >= sqrt(a_(k-1)*a_(k+1)).
Every peak is a non-failing index, and successive peaks are m apart.
The initial and final index intervals are shorter than m as well.
This proves (2). QED.

WHAT THIS DOES NOT PROVE. It does not settle the total number of failures
for forests. It does not contradict central limit theorems for forests.
In particular no Gaussian characteristic-function envelope is claimed.
The variance bound alone cannot provide that envelope or a local limit
approximation. The forest-specific input is a remaining research issue.
"""
from math import comb
import json
import argparse


def coefficients(m):
    assert m >= 16 and m % 2 == 0
    n = m ** 3
    a, B = [], 1
    for k in range(n + 1):
        d = min(k % m, m - k % m)
        value = 1 if k in (0, n) else B * (m*m + d*d) // (2*m*m)
        assert 3*value >= B and value <= B
        a.append(value)
        if k < n:
            B = B * (n-k) // (k+1)
    return a


def audit(m):
    n = m**3
    a = coefficients(m)
    bad = [False] + [a[k]*a[k] < a[k-1]*a[k+1]
                     for k in range(1, n)] + [False]
    eligible = [k for k in range(4*m*m, n-4*m*m+1) if k % m != m//2]
    assert len(eligible) == n-9*m*m+8*m+1
    assert all(bad[k] for k in eligible)
    assert not any(bad[k] for k in range(m//2, n, m))
    longest = current = 0
    for fails in bad:
        current = current+1 if fails else 0
        longest = max(longest, current)
    assert longest <= m-1
    assert sum(bad) >= n-9*m*m
    # The local polynomial identity is checked independently at every residue.
    for r in range(-m//2+1, m//2):
        f = m*m+r*r
        assert (m*m+(r-1)**2)*(m*m+(r+1)**2)-f*f == 2*m*m-2*r*r+1
    return dict(m=m, N=n, failing_indices=sum(bad),
                total_interior_indices=n-1, longest_failure_run=longest,
                guaranteed_failures=len(eligible), status='PASS_EXACT')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--m', nargs='+', type=int, default=[16, 20, 24])
    args = parser.parse_args()
    result = dict(theorem='variance obstruction; arbitrary set families',
                  forest_claim=False, proof_assistant_certificate=False,
                  cases=[audit(m) for m in args.m])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
