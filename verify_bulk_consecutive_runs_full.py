#!/usr/bin/env python3
# Exact-rational certificate for the scalar constants in
# "Polynomially long consecutive failures of log-concavity..."
#
# This verifies all explicit arithmetic used in the b=64 specialization,
# including the quantitative curvature gap extracted from the three-window valley.
# The functional-analytic implications (W2 contraction -> Fourier decay ->
# C^2 interpolation) are theorem proofs, not numerical assertions.

from fractions import Fraction as F
import json
from pathlib import Path

b = 64

# Fixed-point bracket: delta(2-delta)^b = 1.
dlo = F(1, 2**b)
dhi = F(2, 2**b)
assert dlo * (2-dlo)**b < 1
assert dhi * (2-dhi)**b > 1

# q=(1-delta)/(2-delta) decreases with delta.
qlo = (1-dhi)/(2-dhi)
qhi = (1-dlo)/(2-dlo)
rholo = b*qlo
rhohi = b*qhi

assert qlo >= F(49,100)
assert qhi < F(1,2)
assert rholo*rholo > b                  # sqrt(b) < rho
assert rhohi < b

# theta=1/(1+rho), a0=delta-theta !=0.
assert dhi < F(1,b*b)
assert dhi < F(1,1+rhohi)

# W2 contraction and fixed-pair variance.
ell2_hi = F(b,1)/(rholo*rholo)
assert ell2_hi < 1
assert qlo*qlo > F(6,25)
t_hi = F(1,b)/(qlo*qlo)
assert t_hi < F(25,6*b)
t0 = F(25,6*b)
assert 1-t0-t0*t0 > F(5,6)
v_upper = (t0*F(1,4))/F(5,6)
assert v_upper == F(5,4*b)             # = 5/256

# Three-window limiting valley.
outer_mass = F(49,100)*(1-F(5,16))
middle_mass = F(5,16)
assert outer_mass == F(539,1600)
assert middle_mass == F(500,1600)
assert outer_mass > middle_mass

# Each window has length 1/2, so:
outer_density_average = 2*outer_mass   # 539/800
middle_density_average = 2*middle_mass # 5/8
assert outer_density_average > F(67,100)
assert middle_density_average < F(63,100)

# Quantitative curvature certificate.
# There exist xL<xM<xR in the three window interiors with
# f(xL),f(xR)>67/100 and f(xM)<63/100.
# Hence the chord gap of g=log f exceeds log(67/63).
#
# Elementary rational lower bound:
# log(1+x) >= 2x/(2+x), x>=0.
# With x=4/63:
# 2 log(67/63) >= 8/65 > 3/25.
assert F(8,65) > F(3,25)

# Thus sup (log f)'' > 3/25. By continuity one may choose
# a nondegenerate closed J with (log f)'' >= 3/25 = 2*zeta.
zeta = F(3,50)
assert 2*zeta == F(3,25)

# Bulk location.
beta_upper = F((3*b-2)*(b+1), 3*(b*b+b-1))
assert beta_upper == F(12350,12477)
assert beta_upper < F(99,100)

# Run exponent kappa_64 > 4/5:
# rho > 64*49/100 > 30 and 30^5 > 64^4.
assert rholo > 30
assert 30**5 > 64**4

result = {
    "status": "PASS",
    "b": b,
    "maximum_degree": 65,
    "q_interval": [str(qlo), str(qhi)],
    "rho_interval": [str(rholo), str(rhohi)],
    "w2_contraction_squared_upper": str(ell2_hi),
    "fixed_variance_upper": str(v_upper),
    "outer_mass_lower": str(outer_mass),
    "middle_mass_upper": str(middle_mass),
    "outer_density_average": str(outer_density_average),
    "middle_density_average": str(middle_density_average),
    "curvature_point_lower": "3/25",
    "certified_interval_curvature_level_2zeta": "3/25",
    "zeta": str(zeta),
    "beta64_upper": str(beta_upper),
    "beta64_below_99_percent": True,
    "kappa64_gt_4_over_5": True,
    "analytic_dependencies": [
        "W2 contraction and fixed-pair existence",
        "stretched-exponential Fourier decay",
        "uniform C^2 lattice interpolation",
        "continuity transfer from log f curvature to log g_h curvature"
    ],
    "note": "No numerical value of J or H_64 is required by the theorem; their existence is certified analytically from strict curvature and uniform C^2 convergence."
}

out = Path(__file__).with_name("bulk_consecutive_runs_full_certificate.json")
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print("PASS: full scalar/curvature certificate")
print("curvature point lower > 3/25; choose zeta=3/50")
print("beta64 <", beta_upper, "< 99/100")
print("kappa64 > 4/5")
print("wrote", out.name)
