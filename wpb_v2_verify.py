#!/usr/bin/env python3
"""Reproducibility checks for Who Pays the Bill? v2.

Dependencies: mpmath, sympy.
Checks:
  * equal-entropy root q;
  * the 9x9 hidden-synergy witness G is normalized;
  * all four two-step transversal projections are exact symbolically;
  * positivity at the chosen q;
  * entropy and mutual-information values;
  * self-information Wasserstein lower bound.
"""
from __future__ import annotations

import math
import mpmath as mp
import sympy as sp

mp.mp.dps = 80
q = mp.findroot(
    lambda x: -2*x*mp.log(x, 2) - (1-2*x)*mp.log(1-2*x, 2) - mp.mpf("1.5"),
    (mp.mpf("0.40"), mp.mpf("0.42")),
)

qs = sp.symbols("q", real=True)
G = sp.Matrix([
    [0, qs**2, 0, qs/4, 0, 0, 0, 0, 0],
    [0, (3*qs-1)*(4*qs-1)/4, -qs*(16*qs-7)/4, 0, 0, -(2*qs-1)/8,
     (4*qs-1)**2/4, -(16*qs**2-9*qs+1)/4, 0],
    [0, 0, 0, 0, 0, 0, -(16*qs**2-9*qs+1)/4, 0, (4*qs-1)*(8*qs-3)/8],
    [-(8*qs**2+4*qs-3)/8, 0, 0, 0, 0, 0, (32*qs**2+4*qs-7)/16, 0,
     -(16*qs**2-3)/8],
    [-(2*qs-1)*(4*qs-1)/8, qs/4, (16*qs**2-4*qs-1)/8,
     -(2*qs-1)*(8*qs-1)/8, 0, 0, 0, 0, 0],
    [0, 0, 0, (4*qs-1)*(8*qs-3)/16, 0, 0, -(2*qs-1)/8, 0, 0],
    [0, 0, (8*qs-3)/8, 0, 0, 0, sp.Rational(1, 16), 0, 0],
    [0, 0, 0, 0, qs*(8*qs-3)/4, 0, 0, 0, 0],
    [(2*qs-1)**2/2, -qs*(2*qs-1), 0, -(4*qs-3)*(8*qs-3)/16, 0,
     -(12*qs-5)/8, 0, 0, 0],
])

states = [(i, j) for i in range(3) for j in range(3)]
P = [sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 4)]
Q = [qs, qs, 1-2*qs]

assert sp.simplify(sum(G) - 1) == 0

def projection(a: int, b: int) -> sp.Matrix:
    M = sp.zeros(3, 3)
    for ri, r in enumerate(states):
        for si, s in enumerate(states):
            M[r[a], s[b]] += G[ri, si]
    return M.applyfunc(sp.simplify)

for a in (0, 1):
    for b in (0, 1):
        pa = sp.Matrix(P if a == 0 else Q)
        pb = sp.Matrix(P if b == 0 else Q)
        target = pa * pb.T
        D = projection(a, b) - target
        assert all(sp.simplify(D[i, j]) == 0 for i in range(3) for j in range(3))

# Numeric matrix at q.
subs = {qs: sp.Float(str(q), 80)}
Gn = [[mp.mpf(str(sp.N(G[i, j].subs(subs), 80))) for j in range(9)] for i in range(9)]
positive = [x for row in Gn for x in row if x > mp.mpf("1e-60")]
negative = [x for row in Gn for x in row if x < -mp.mpf("1e-60")]
assert not negative
assert len(positive) == 25


def H(vals):
    flat = []
    for x in vals:
        if isinstance(x, (list, tuple)):
            flat.extend(x)
        else:
            flat.append(x)
    return -sum(x*mp.log(x, 2) for x in flat if x > 0)

row = [sum(Gn[i][j] for j in range(9)) for i in range(9)]
col = [sum(Gn[i][j] for i in range(9)) for j in range(9)]
HG, H1, H2 = H(Gn), H(row), H(col)
MI12 = H1 + H2 - HG

# One-shot MEC value from the five nonzero cells in the paper's 3x3 witness.
mec_atoms = [mp.mpf("0.5")-q, q, 2*q-mp.mpf("0.75"), 1-2*q, mp.mpf("0.25")]
C1 = H(mec_atoms)

# Mutual informations between first selected coordinate and full/selected second table.
def mi(matrix):
    nr, nc = len(matrix), len(matrix[0])
    pr = [sum(matrix[i][j] for j in range(nc)) for i in range(nr)]
    pc = [sum(matrix[i][j] for i in range(nr)) for j in range(nc)]
    out = mp.mpf("0")
    for i in range(nr):
        for j in range(nc):
            p = matrix[i][j]
            if p > 0:
                out += p*mp.log(p/(pr[i]*pc[j]), 2)
    return out

MI_coord_full = []
MI_coord_coord = {}
for a in (0, 1):
    J = [[mp.mpf("0") for _ in range(9)] for __ in range(3)]
    for ri, r in enumerate(states):
        for si, s in enumerate(states):
            J[r[a]][si] += Gn[ri][si]
    MI_coord_full.append(mi(J))
    for b in (0, 1):
        K = [[mp.mpf("0") for _ in range(3)] for __ in range(3)]
        for ri, r in enumerate(states):
            for si, s in enumerate(states):
                K[r[a]][s[b]] += Gn[ri][si]
        MI_coord_coord[(a, b)] = mi(K)

# Wasserstein distance between one-step surprisal laws (1D quantile formula).
a = -mp.log(q, 2)
b = -mp.log(1-2*q, 2)
W1 = (mp.mpf("0.5")*abs(1-a)
      + (2*q-mp.mpf("0.5"))*abs(2-a)
      + (1-2*q)*abs(2-b))
L = mp.mpf("1.5") + W1/2

print("q                         =", mp.nstr(q, 25))
print("positive atoms in G       =", len(positive))
print("smallest positive atom    =", mp.nstr(min(positive), 18))
print("H(F1)                     =", mp.nstr(H1, 18))
print("H(F2)                     =", mp.nstr(H2, 18))
print("H(G)                      =", mp.nstr(HG, 18))
print("I(F1;F2)                  =", mp.nstr(MI12, 18))
print("I(F1(P);F2)               =", mp.nstr(MI_coord_full[0], 18))
print("I(F1(Q);F2)               =", mp.nstr(MI_coord_full[1], 18))
for key, val in MI_coord_coord.items():
    print(f"I(F1({key[0]});F2({key[1]}))          =", mp.nstr(val, 6))
print("C1 = MEC(P,Q)              =", mp.nstr(C1, 18))
print("2*C1-H(G)                 =", mp.nstr(2*C1-HG, 18))
print("H(G)/2                    =", mp.nstr(HG/2, 18))
print("W1(surprisal laws)         =", mp.nstr(W1, 18))
print("L = 1.5 + W1/2            =", mp.nstr(L, 18))

assert 2*C1 > HG
assert L > mp.mpf("1.5")
assert HG/2 > L
assert all(abs(v) < mp.mpf("1e-12") for v in MI_coord_coord.values())
print("ALL CHECKS PASSED")
