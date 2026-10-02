"""Integer-only scalar audit for the connectivity-tax lower bound.

This verifier supports the universal proof in connectivity_tax_certificate.tex.
It does NOT enumerate trees and it is not a substitute for the combinatorial proof.
Every theorem-critical comparison below is an integer comparison; no floating-point
arithmetic and no numerical logarithm evaluation is used.

The proved constant is
    c + rho/4,
where
    c   = log(34/5)/(6 log 2),
    rho = log(1458/1445)/(12 log 2).
The manuscript proves for every nonempty finite tree T
    mu(T)/alpha(T) >= c + rho/4.
This script checks the scalar inequalities used to make that proof uniform and the
exact comparison c+rho/4 > 461/1000.
"""


def check(name, cond):
    if not cond:
        raise AssertionError(name)
    print("PASS", name)


# Scalar separation away from d=4.
# d=5: 2 log(40/33) < 4 log(10/9).
check(
    "d=5 scalar separation",
    40**2 * 9**4 < 33**2 * 10**4,
)

# d>=6 follows from 2 log(5/4) < 5 log(10/9).
check(
    "d>=6 scalar separation",
    5**2 * 9**5 < 4**2 * 10**5,
)

# theta = 867 rho < 1 is equivalent to
# 289 log(1458/1445) < 4 log 2.
check(
    "theta=867*rho < 1",
    1458**289 < 16 * 1445**289,
)

# gamma >= 7 rho, where gamma=log(20/17)/(6 log 2), is equivalent to
# 2 log(20/17) >= 7 log(1458/1445).
check(
    "gamma > 7*rho",
    20**2 * 1445**7 > 17**2 * 1458**7,
)

# Exact target comparison:
#   [8 log(34/5) + log(1458/1445)]/(48 log 2) > 461/1000.
# Exponentiating and clearing denominators gives the integer comparison below.
LEFT_BASE = 34**8 * 1458
RIGHT_BASE = 5**8 * 1445
check(
    "c + rho/4 > 461/1000",
    LEFT_BASE**1000 > 2**22128 * RIGHT_BASE**1000,
)

# Pure rational identities behind the cancellation in the proof.
from fractions import Fraction as F
check("alpha coefficient = rho/4", F(867, 6 * 578) == F(1, 4))
check("constant penalty = rho", F(867 * 2, 3 * 578) == 1)
check("edge penalty = 7*rho", F(867 * 14, 3 * 578) == 7)

print("PASS: integer-only scalar audit complete")
print("Certified theorem target: a_*(1) >= c + rho/4 > 0.461")
