#!/usr/bin/env python3
"""
WHO PAYS THE BILL?
Exact rational viability certificates for the R=2 zero-repair tree.

NO MONTE CARLO. NO FLOATING-POINT PROOF.

The script verifies two support-independent lower certificates for any
INFINITE zero-repair history-indexed buffer tree.

1. CDF envelope:
      F(0)<=1/9, F(1)<=1/3, F(2)<=2/3.
   This alone gives E[J] >= 17/9.

2. One more viability generation yields an exact LP-dual certificate
   proving
      E[J] >= 19/9.

The dual is written for one parent B and children L,H satisfying the
five first-generation invariant inequalities:

  A0: b0 <= 1/9
  A1: b0+b1 <= 1/3
  A2: b0+b1+b2 <= 2/3
  C1: b0 + 3/4(b1+b2) <= 1/2
  C2: b0+b1 + 3/4(b2+b3) <= 3/4.

The zero-repair recurrence is
  .75 B(j) shifted by 3 + .25 B(j) shifted by 5
    = .25 L shifted by 2 + .75 H shifted by 4.

All checks below use fractions exactly and cover ALL j>=0, including the
infinite tail of the buffer law.
"""

from fractions import Fraction as F


# ----------------------------------------------------------------------
# First stationary CDF envelope
# ----------------------------------------------------------------------
QSTAR = {0: F(1,9), 1: F(1,3), 2: F(2,3)}


def mean_lower_from_cdf_caps():
    # E[J] = sum_{j>=0} P(J>j); using only j=0,1,2.
    return sum(F(1)-QSTAR[j] for j in range(3))


# ----------------------------------------------------------------------
# First combined child-viability inequalities
# ----------------------------------------------------------------------
# Necessary for every node of an infinite tree:
#   C1 = b0 + 3/4(b1+b2) <= 1/2
#   C2 = b0+b1 + 3/4(b2+b3) <= 3/4.
# Together with A0,A1,A2 these imply E[J] >= 55/27.


def check_55_over_27_dual():
    # Dual identity for all j >= 0:
    # j >= 4
    #      -(2/3) 1_{j<=0}
    #      -(2/3) 1_{j<=1}
    #      -(4/3) C1_coeff(j)
    #      -(4/3) C2_coeff(j).
    def c1(j):
        return {0:F(1), 1:F(3,4), 2:F(3,4)}.get(j,F(0))
    def c2(j):
        return {0:F(1), 1:F(1), 2:F(3,4), 3:F(3,4)}.get(j,F(0))

    for j in range(100):
        rhs = F(4)
        if j <= 0:
            rhs -= F(2,3)
        if j <= 1:
            rhs -= F(2,3)
        rhs -= F(4,3)*c1(j)
        rhs -= F(4,3)*c2(j)
        assert F(j) >= rhs, (j,F(j),rhs)

    bound = (
        F(4)
        - F(2,3)*F(1,9)
        - F(2,3)*F(1,3)
        - F(4,3)*F(1,2)
        - F(4,3)*F(3,4)
    )
    assert bound == F(55,27)
    return bound


# ----------------------------------------------------------------------
# Exact support-independent dual for 19/9
# ----------------------------------------------------------------------
# Active primal inequalities:
# Parent: A0,A1
# Low child: A2,C2
# High child: A1,C1
# plus normalization of parent and high child and the zero-repair equations.


def lam(u):
    # Multiplier of the total-information equality at level u.
    if u == 3:
        return F(-20,9)
    if u == 4:
        return F(-16,9)
    if u == 5:
        return F(-4,3)
    if u == 6:
        return F(0)
    if u >= 7:
        return F(4,3)
    return F(0)


def parent_dual_coeff(j):
    v = F(8,3) + F(3,4)*lam(j+3) + F(1,4)*lam(j+5)
    # Parent A0 multiplier -1/3
    if j == 0:
        v -= F(1,3)
    # Parent A1 multiplier -1/3
    if j <= 1:
        v -= F(1,3)
    return v


def low_dual_coeff(j):
    v = -F(1,4)*lam(j+2)
    # Low A2 multiplier -1/9
    if j <= 2:
        v -= F(1,9)
    # Low C2 multiplier -4/9
    c2 = {0:F(1),1:F(1),2:F(3,4),3:F(3,4)}.get(j,F(0))
    v -= F(4,9)*c2
    return v


def high_dual_coeff(j):
    v = F(1) - F(3,4)*lam(j+4)
    # High A1 multiplier -1
    if j <= 1:
        v -= F(1)
    # High C1 multiplier -4/3
    c1 = {0:F(1),1:F(3,4),2:F(3,4)}.get(j,F(0))
    v -= F(4,3)*c1
    return v


def check_19_over_9_dual():
    # Dual feasibility for every nonnegative buffer level j.
    for j in range(10000):
        assert parent_dual_coeff(j) <= F(j), ("parent",j,parent_dual_coeff(j))
        assert low_dual_coeff(j) <= 0, ("low",j,low_dual_coeff(j))
        assert high_dual_coeff(j) <= 0, ("high",j,high_dual_coeff(j))

    # The tail is explicit: for j>=5 parent coefficient is 4,
    # low coefficient is -1/3, high coefficient is 0.
    assert parent_dual_coeff(1000000) == F(4)
    assert low_dual_coeff(1000000) == F(-1,3)
    assert high_dual_coeff(1000000) == F(0)

    # Exact dual objective.
    value = (
        F(8,3)      # parent normalization
        + F(1)      # high-child normalization
        - F(1,3)*F(1,9)  # parent A0
        - F(1,3)*F(1,3)  # parent A1
        - F(1,9)*F(2,3)  # low A2
        - F(4,9)*F(3,4)  # low C2
        - F(1)*F(1,3)    # high A1
        - F(4,3)*F(1,2)  # high C1
    )
    assert value == F(19,9)
    return value


def main():
    print("="*86)
    print("EXACT ZERO-REPAIR VIABILITY CERTIFICATES — R=2")
    print("NO MONTE CARLO / EXACT FRACTION ARITHMETIC")
    print("="*86)
    print()

    b0 = mean_lower_from_cdf_caps()
    print("Stationary CDF-envelope lower:")
    print("  E[J] >=", b0, "=", float(b0))
    assert b0 == F(17,9)

    b1 = check_55_over_27_dual()
    print("One-generation combined-tail lower:")
    print("  E[J] >=", b1, "=", float(b1))

    b2 = check_19_over_9_dual()
    print("Two-generation exact dual lower:")
    print("  E[J] >=", b2, "=", float(b2))

    print()
    print("CONCLUSION")
    print("----------")
    print("Any infinite zero-repair history-indexed R=2 buffer tree must satisfy")
    print("    E[J_root] >= 19/9 > 2.")
    print()
    print("Therefore the conjectural finite limit B_infinity = 2 is impossible.")
    print("This does NOT yet prove B_n -> infinity; a finite limit >=19/9 remains possible.")


if __name__ == "__main__":
    main()
