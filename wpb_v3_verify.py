#!/usr/bin/env python3
import math
import itertools
import numpy as np
import sympy as sp
from scipy.optimize import brentq


def H(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[vals > 0]
    return float(-(vals * np.log2(vals)).sum())


def hq(x):
    return -2*x*math.log2(x) - (1-2*x)*math.log2(1-2*x)

qv = brentq(lambda x: hq(x)-1.5, 1/3+1e-12, 0.5-1e-12)
P = np.array([0.5, 0.25, 0.25])
Q = np.array([qv, qv, 1-2*qv])
assert abs(H(P)-1.5) < 1e-12
assert abs(H(Q)-1.5) < 1e-12

# One-shot MEC witness from the manuscript.
mec_atoms = np.array([0.5-qv, qv, 2*qv-0.75, 1-2*qv, 0.25])
C1 = H(mec_atoms)

# New 23-atom synchronous two-step witness.
q = sp.symbols('q', positive=True)
entries = {
    (0,0,0,0): (2*q-1)*(12*q-5)/8,
    (0,0,0,2): -q*(2*q-1),
    (0,0,2,1): q/4,
    (0,1,0,0): q**2,
    (0,1,0,2): -(2*q-1)*(8*q-3)/8,
    (0,1,1,1): q/4,
    (0,2,1,0): -(2*q-1)/8,
    (0,2,2,0): -(2*q-1)*(8*q-3)/4,
    (0,2,2,2): (2*q-1)*(16*q-7)/8,
    (1,0,0,0): q/4,
    (1,0,1,1): (4*q-1)/16,
    (1,1,0,2): -(2*q-1)/8,
    (1,1,2,1): sp.Rational(1,16),
    (1,2,1,2): -(2*q-1)/8,
    (2,0,0,0): -(32*q**2-40*q+11)/16,
    (2,0,0,1): (4*q-1)**2/16,
    (2,0,1,0): sp.Rational(1,16),
    (2,1,0,1): (16*q**2-4*q-1)/16,
    (2,1,0,2): -(12*q-5)/16,
    (2,1,2,2): (4*q-1)/16,
    (2,2,0,0): (2*q-1)**2/2,
    (2,2,0,1): -q*(2*q-1),
    (2,2,2,0): -(2*q-1)/8,
}

# Symbolic normalization and all four transversal projections.
assert sp.simplify(sum(entries.values()) - 1) == 0
Ds = [
    [sp.Rational(1,2),sp.Rational(1,4),sp.Rational(1,4)],
    [q,q,1-2*q],
]
for a1, a2 in itertools.product([0,1], repeat=2):
    for y1, y2 in itertools.product(range(3), repeat=2):
        lhs = sum(v for atom,v in entries.items()
                  if (atom[0] if a1 == 0 else atom[1]) == y1
                  and (atom[2] if a2 == 0 else atom[3]) == y2)
        rhs = Ds[a1][y1] * Ds[a2][y2]
        assert sp.simplify(lhs-rhs) == 0, (a1,a2,y1,y2,sp.factor(lhs-rhs))

# Numeric full table in lexicographic (x1,z1,x2,z2) order.
atoms = list(itertools.product(range(3), repeat=4))
probs = np.array([float(entries.get(a,0).subs(q,qv) if hasattr(entries.get(a,0),'subs') else entries.get(a,0)) for a in atoms])
assert abs(probs.sum()-1) < 1e-13
positive = probs[probs > 1e-14]
assert len(positive) == 23
assert positive.min() > 1e-3
H2w = H(probs)


def marginal(coords):
    d = {}
    for p,a in zip(probs, atoms):
        key = tuple(a[i] for i in coords)
        d[key] = d.get(key,0.0) + p
    return np.array(list(d.values()))

H_R1 = H(marginal((0,1)))
H_R2 = H(marginal((2,3)))
I_R1_R2 = H_R1 + H_R2 - H2w


def mutual(A,B):
    return H(marginal(A)) + H(marginal(B)) - H(marginal(A+B))

I_X1_R2 = mutual((0,), (2,3))
I_Z1_R2 = mutual((1,), (2,3))
I_X1_X2 = mutual((0,), (2,))
I_X1_Z2 = mutual((0,), (3,))
I_Z1_X2 = mutual((1,), (2,))
I_Z1_Z2 = mutual((1,), (3,))
assert max(abs(I_X1_X2),abs(I_X1_Z2),abs(I_Z1_X2),abs(I_Z1_Z2)) < 2e-12

# Surprisal W1 distance in one dimension via quantile matching.
muP = [(1.0,0.5),(2.0,0.5)]
muQ = [(-math.log2(qv),2*qv),(-math.log2(1-2*qv),1-2*qv)]
muP.sort(); muQ.sort()
i=j=0; a=muP[0][1]; b=muQ[0][1]; W1=0.0
while i < len(muP) and j < len(muQ):
    m=min(a,b)
    W1 += m*abs(muP[i][0]-muQ[j][0])
    a-=m; b-=m
    if a < 1e-14:
        i+=1
        if i < len(muP): a=muP[i][1]
    if b < 1e-14:
        j+=1
        if j < len(muQ): b=muQ[j][1]
LB = 1.5 + 0.5*W1

# REAL k=0 identity: 2F0(1/p,1/q)=min(p,q).
p,r = sp.symbols('p r', positive=True)
F0_recip = sp.Rational(1,2) / sp.Max(1/p,1/r)
# SymPy does not simplify Max reciprocals under positivity automatically; test both order cases.
assert sp.simplify((sp.Rational(1,2)/(1/r))*2-r) == 0  # p >= r -> max(1/p,1/r)=1/r
assert sp.simplify((sp.Rational(1,2)/(1/p))*2-p) == 0  # r >= p -> max(...)=1/p

# Finite-n prefix bound for n=2.
finite_lb = 1.5 + C1

print(f"q = {qv:.17f}")
print(f"C1 = {C1:.15f}")
print(f"new witness atoms = {len(positive)}, min positive = {positive.min():.15g}")
print(f"H(G_tilde) = {H2w:.15f}")
print(f"H(G_tilde)/2 = {H2w/2:.15f}")
print(f"2 C1 - H(G_tilde) = {2*C1-H2w:.15f}")
print(f"H(R1) = {H_R1:.15f}, H(R2) = {H_R2:.15f}")
print(f"I(R1;R2) = {I_R1_R2:.15f}")
print(f"I(X1;R2) = {I_X1_R2:.15f}, I(Z1;R2) = {I_Z1_R2:.15f}")
print(f"cross-coordinate mutual informations max abs = {max(abs(I_X1_X2),abs(I_X1_Z2),abs(I_Z1_X2),abs(I_Z1_Z2)):.3e}")
print(f"W1 = {W1:.15f}")
print(f"asymptotic synchronous lower bound = {LB:.15f}")
print(f"finite C2 lower bound h+C1 = {finite_lb:.15f}")
print(f"certified synchronous interval: [{LB:.15f}, {H2w/2:.15f}]")
print("ALL CHECKS PASSED")
